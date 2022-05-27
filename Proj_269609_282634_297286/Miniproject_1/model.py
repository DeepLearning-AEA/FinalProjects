import torch
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import *
from torchvision.transforms.functional import *
from torch.nn import *
from torch.optim import *
from torch import nn
from torch.fft import *
import os
import pickle
from datetime import datetime

class Model():
    def __init__(self) -> None:
        # Fix the seed for reproducible results
        self.seed = 42
        # Fix the seed for reproducibility purposes
        torch.manual_seed(self.seed)

        # Training parameters
        self.learning_rate = 1e-3
        self.betas = (0.9, 0.99)
        self.batch_size = 400
        self.train_ratio = 0.9

        # Variable parameters for training
        self.MAX_EPOCHS = 25
        self.use_data_augmentation = False

        ## instantiate model + optimizer + loss function + any other stuff we need
        self.criterion = MSELoss()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = Noise2NoiseNetwork()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas)
        # Scheduler that will decrease the learning rate by a factor 0.1 every 10-step
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Transfer the model to either the GPU (if available) or the CPU
        self.model.to(self.device)

        # List that will store metrics encounter during training
        self.train_losses = []
        self.test_losses = []
        self.psnrs = []

    def set_model(self, model):
        """
        Set the model for training
        :param model: the new model we should use for training
        """
        # set the model and move it to the correct device
        self.model = model
        self.model.to(self.device)
        # Reset learning optimizers to fit the new given model
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def load_pretrained_model(self) -> None:
        """
        Load the weights of the best model to the current model.
        """
        # Load the weights from the best model in our model
        self.model.load_state_dict(torch.load("Miniproject_1/bestmodel.pth", map_location=self.device))
        # move the model to the currently used device
        self.model = self.model.to(self.device)

    def train(self, train_input, train_target, num_epochs) -> None:
        """
        Train the network with the given input
        :param train_input: the training input data
        :param train_target: the training target data
        :param num_epochs: The number of epochs to train the network. The exact number of epochs will be the minimum between the given num_epochs and the one set in our model paramters.
        """
        # Fix the seed for repoducibility purposes
        torch.manual_seed(self.seed)

        # : train_input : tensor of size (N, C, H, W) containing a noisy version of the images : train_target :
        # tensor of size (N, C, H, W) containing another noisy version of thesame images , which only differs from
        # the input by their noise .
        print("Normalizing the images between 0 and 1...")
        train_input = normalizing_data(train_input)
        train_target = normalizing_data(train_target)

        print("Split datasets...")
        # Split the dataset based on the parameters
        training_set, val_test = load_and_split_dataset(train_input, train_target, self.batch_size, self.train_ratio,
                                                        self.use_data_augmentation)

        print("Training is starting...")

        # Train the network for the given number of epochs
        for e in range(1, 1 + min(num_epochs, self.MAX_EPOCHS)):
            # Set the model in train mode
            self.model.train()
            train_acc_loss = 0
            # We optimize using mini-batches
            for batch_input, batch_target in training_set:
                # Move the data to the correct device
                train, test = batch_input.to(self.device), batch_target.to(self.device)

                # Compute the prediction of the network for the current mini batch 
                output = self.model(train)

                # Compute the loss w.r.t to the expected noisy output
                loss = self.criterion(output, test)
                train_acc_loss += loss.item()

                # Apply backpropagation and weight update
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Compute the current performance if we have a validation set
            if val_test is not None:
                # Compute and save metrics for this iteration
                test_loss, psnr = evaluate_performance(self.model, val_test, self.criterion, e, train_acc_loss,
                                                       self.device)
                self.test_losses.append(test_loss)
                self.psnrs.append(psnr)

            # Save the training loss in lists
            self.train_losses.append(train_acc_loss)

        print()
        print("End of training")
        print("Save trained model")
        # Save model weights on disk to be reloaded later on
        save_model(self.model, self.train_losses, self.test_losses, self.psnrs)

    def train_parameters(self, train_input, train_target, val_input, val_target, num_epochs):
        """
        This function plays the same role as the train function except that it allows us to define more settings
        Train the network with the given input :param train_input: the training input data :param train_target: the
        training target data :param val_input: the validation input set to compute performance :param val_target: the
        validation target set to compute performance :param num_epochs: The number of epochs to train the network.
        The exact number of epochs will be the minimum between the given num_epochs and the one set in our model
        paramters.
        """

        # : train_input : tensor of size (N, C, H, W) containing a noisy version of the images : train_target :
        # tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from
        # the input by their noise .
        print("Normalizing the images between 0 and 1...")

        # Compute data normalization
        train_input, train_target, val_input, val_target = normalizing_data(train_input), normalizing_data(
            train_target), normalizing_data(val_input), normalizing_data(val_target)

        print("Split datasets...")
        # Split the dataset into a train and a validation set
        training_set, val_test = load_and_split_dataset(train_input, train_target, self.batch_size, 1.0,
                                                        self.use_data_augmentation)
        # Set again the validation set to the given test value
        val_test = DataLoader(TensorDataset(val_input, val_target), batch_size=self.batch_size, shuffle=False,
                              pin_memory=False)

        # For visualisation purposes, we choose one image to see how it is denoised during training
        display_imgs = next(iter(val_test))
        train_img, train_img2 = display_imgs[0][0].unsqueeze(0).to(self.device), display_imgs[1][0].unsqueeze(0).to(
            self.device)

        print("Training is starting...")
        # Print the number of paramers in the network
        get_and_print_model_parameters(self.model)
        print()

        # Train the network for the given number of epochs
        for e in range(1, 1 + num_epochs):
            # Set the model in train mode
            self.model.train()
            train_acc_loss = 0
            # We optimize using mini-batches
            for batch_input, batch_target in training_set:
                # Move the data to the correct device
                train, test = batch_input.to(self.device), batch_target.to(self.device)

                # Compute the prediction of the network for the current mini batch 
                output = self.model(train)

                # Compute the loss w.r.t to the expected noisy output
                loss = self.criterion(output, test)
                train_acc_loss += loss.item()

                # Apply backpropagation and weight update
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

                # Compute the current performance
            test_loss, psnr = evaluate_performance(self.model, val_test, self.criterion, e, train_acc_loss, self.device)

            if e % 10 == 0:
                # Every 10 epoch, save how the training denoise the saved image
                show(train_img, train_img2, self.model(train_img))

            # Save the results for this epoch in lists
            self.train_losses.append(train_acc_loss)
            self.test_losses.append(test_loss)
            self.psnrs.append(psnr)

        print()
        print("End of training")
        print("Save trained model")
        # Save the model for later use
        save_model(self.model, self.train_losses, self.test_losses, self.psnrs)

    def predict(self, test_input) -> torch.Tensor:
        """
        Make the predictions for every data point in the test dataset
        :param test_input: the tesz dataset
        :return: the predicted/denoised images
        """
        # Set the model in evaluation mode
        self.model.eval()

        # Normalize the data to match network expectation
        test_input = normalizing_data(test_input)

        # move the data to the correct device
        test_input = test_input.to(self.device)

        # define a tensor on the CPU that will contain the update of the network (for some methods, batch size of 400 on the GPU is too big Noise2Noise HLS)
        output = torch.zeros(test_input.shape)
        for i in range(0, len(test_input), self.batch_size):
            # Apply the forward pass of the whole to a mini-batch, The denormalize step has the role of clipping 
            output[i:i + self.batch_size, :, :, :] = denormalize_data(
                self.model(test_input[i:i + self.batch_size, :, :, :])).cpu()

        # Return the predictions
        return output.to(self.device)



def psnr(x, y, max_range=1.0):
    """
    Compute the PSNR from the predictions and the targets

    :param x: The denoised version of the images (output of the network)
    :param y: The ground truth
    :param max_range: the maximal range of the value
    :return: the psnr
    """
    assert x.shape == y.shape and x.ndim == 4
    # Implementation from the test.py file
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


def psnr_mse(mse):
    """
    Compute the PSNR from the given MSE

    :param mse: The computed MSE from the data
    :return: the psnr
    """
    return -10 * torch.log10(mse + 10 ** -8)


def load_and_split_dataset(train_input, train_target, batch_size, ratio, data_augmentation=False):
    """
    Load the data into a DataLoader object and split into train and test set. ratio*100% of the data will go in the training set

    :param train_input: The training input for the network
    :param train_target: The training target we want to achieve
    :param batch_size: The number of images procssed simultaneously
    :param ratio: The ratio of training data going into the test set
    :param data_augmentation: true if we need to perform data_augmentation or not
    """
    # Create datasets
    train_input, train_target, val_input, val_target = split_data(train_input, train_target, ratio)

    if data_augmentation:
        # If we need to do data_augmentation, perform 3 random crop of the input images and resize it to their original size
        original_dimension = train_input[0].shape[1], train_input[0].shape[2]

        # set default paramerters
        train_input_ = train_input
        train_target_ = train_target

        # Generate random crops
        for i in range(3):
            # Get random crop params of size 32x32
            i, j, h, w = RandomCrop.get_params(train_input,
                                               output_size=(original_dimension[0] // 2, original_dimension[1] // 2))
            # Crop and store the different crops
            train_input_ = torch.cat(
                [train_input_, resized_crop(train_input, i, j, h, w, [original_dimension[0], original_dimension[1]])],
                0)
            train_target_ = torch.cat(
                [train_target_, resized_crop(train_target, i, j, h, w, [original_dimension[0], original_dimension[1]])],
                0)
        # Reassign the value
        train_target = train_target_
        train_input = train_input_

    print(f"Training set size: {train_target.size()}")
    print(f"Validation set size: {val_target.size()}")
    # Create datasets from tensors
    train = TensorDataset(train_input, train_target)
    val = TensorDataset(val_input, val_target)

    # Convert dataset to dataloader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, pin_memory=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=False) if ratio != 1.0 else None

    return train_loader, val_loader


def split_data(x, y, ratio):
    """
    Split the given dataset into 2 different datasets (local train/validation) according to the given ratio
    :param x: The training input for the network
    :param y: The training target we want to achieve
    :param ratio: The ratio of training data going into the test set
    :return: A 4-tuple: (training input, training target, validation input, validation target) each as a Subset object
    """
    # Compute the number of training and validation data
    nb_data = x.size(0)
    nb_train = int(ratio * nb_data)

    return x[:nb_train], y[:nb_train], x[nb_train:], y[nb_train:]


def normalizing_data(x):
    """
    Perform max-normalization so that the input on each channel is between 0 and 1
    :param x: the data we want to normalise
    :return: the normalized data
    """
    # max_s = torch.amax(x, dim=(1,2,3))[:, None, None, None]
    x_norm = x / 255

    # Check that the max_normalization indeed works
    assert (torch.max(x_norm) <= 1.0)
    assert (torch.min(x_norm) >= 0.0)

    return x_norm


def denormalize_data(x):
    """
    Undo the max-normalization and clip so that the input on each channel is between 0 and 255
    :param x: the data we want to denormalize
    :return: the denormalizerd data
    """
    x_denorm = x * 255
    return torch.clip(x_denorm, min=0.0, max=255.0)


def evaluate_performance(model, test_set, criterion, epoch, train_loss, device):
    """
    This function evaluates the performance of the model on the test set and return the test loss as well as the Peak Signal to Noise ratio

    :param model: The current state of the model
    :param test_set: the test set on which to test the performance
    :param criterion: the criterion to compute the loss
    :param epoch: the current epoch of the network
    :param train_loss: the training loss for this epoch
    :param device: the device on which the training is made
    :return: A tuple containing the loss accuracy and the Peak Signal to Noise Ratio
    """
    test_acc_loss = 0.0
    model.eval()
    for batch_input, batch_target in test_set:
        # Move data to correct device
        input_data, target = batch_input.to(device), batch_target.to(device)
        # Compute output of the network and compute the loss
        output = model(input_data)
        test_acc_loss += criterion(output, target).item()
        torch.cuda.empty_cache()
    # Get the PSNR
    psnr = psnr_mse(torch.tensor(test_acc_loss) / len(test_set))

    if epoch % 10 == 0:
        # Every 10 epochs, print the current value of the training
        print(f"Epochs {epoch}:")
        print(f"\tTraining loss: {train_loss:.2f}")
        print(f"\tTest loss: {test_acc_loss:.2f}")
        print(f"\tPSNR:{psnr:.2f} (db)")
        print("=======================")

    return test_acc_loss, psnr


def plot_losses(model):
    """
    Plot the losses of the given model during training
    :param model: The model for which we want to plot the losses
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(model.test_losses)), model.test_losses, 'b', lw=4, label="Test loss")
    plt.plot(range(0, len(model.train_losses)), model.train_losses, 'orange', lw=4, label="Train loss")
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Evolution of train and test losses during training', fontsize=20)
    plt.legend(loc=2)
    plt.grid()
    plt.show()


def plot_psnr_validation_training(model):
    """
    Plot the performance of the given model during training
    :param model: The model for which we want to plot the performance
    """
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, len(model.psnrs)), model.psnrs, 'b', lw=4)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel("PSNR", fontsize=20)
    plt.title('Evolution of PSNR during training', fontsize=20)
    plt.grid()
    plt.show()


def save_model(model, train_losses, test_losses, psnrs):
    """
    Save the model weights and performance to the disk

    :param model: The current state of the model
    :param train_losses: the losses encoutered during training
    :param test_losses: the losses encoutered during testing
    :param psnrs: the Peak Signal to Noise Ratio encountered during training
    """
    now = datetime.now()
    time = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(model.name + "/" + time)
    time_folder = model.name + "/" + time + "/"
    # save model weights
    torch.save(model.state_dict(), time_folder + "weights.pth")

    # Save the train losses
    with open(time_folder + "train_loss.pkl", "wb") as fp:
        pickle.dump(train_losses, fp)

    # save the test losses
    with open(time_folder + "test_loss.pkl", "wb") as fp:
        pickle.dump(test_losses, fp)

    # save the training psnrs
    with open(time_folder + "psnr.pkl", "wb") as fp:
        pickle.dump(psnrs, fp)


def get_model_from_name(name, device):
    """
    This function get the correct depending on the requested model.
    :param name: the name of the model we want to get
    :param device: the device on which we should place the model
    """
    if name == "noise2noise":
        return Noise2NoiseNetwork()
    elif name == "noise2noisedeep":
        return Noise2NoiseDeepNetwork()
    elif name == "noise2noiseinception":
        return Noise2NoiseInception()
    elif name == "noise2noisefft":
        return Noise2NoiseFFT(device=device)
    elif name == "noise2noisehls":
        return Noise2NoiseNetworkHLS()
    elif name == "noise2noise3":
        return Noise2NoiseNetwork3()
    else:
        raise ValueError("No network with this name")


# Function to make a full train of the network
def run_experiment(model_to_use, train=True, data_augmentation=False, epochs=20, seed=42):
    import Model
    """
    Make a full experiment on the network, i.e. load data, train network and test it
    :param model_to_use: An instance of a model to train
    :param train: If we need to train the network or simply load bestmodel.pth in the model
    :param data_augmentation: if we should perform data augmentation or not
    :param epochs: the number of epochs for which we should train the network
    :param seed: the seed to fix for random generator
    :return: return the model object to perform subsequent analysis
    """

    # Set the model to the given parameters
    model = Model.Model()
    # Fix the seed for repoducibility purposes
    torch.manual_seed(seed)
    model.set_model(get_model_from_name(model_to_use, model.device))
    model.use_data_augmentation = data_augmentation

    # Load the validation set
    noisy_imgs, clean_imgs = torch.load(DATA_FOLDER + 'val_data.pkl')

    # split the validation set into two different set so that we do not overfit the testing set when we compare performance
    val_imgs, val_clean_imgs = noisy_imgs[0:100], clean_imgs[0:100]
    test_imgs, test_clean_imgs = noisy_imgs[100:], clean_imgs[100:]

    if (train):
        # If we need to train the network
        # Record the start time
        start = time.perf_counter()
        # Load the training set
        noisy_imgs_1, noisy_imgs_2 = torch.load(DATA_FOLDER + 'train_data.pkl')

        # Launch the training
        model.train_parameters(noisy_imgs_1, noisy_imgs_2, val_imgs, val_clean_imgs, epochs)
        stop = time.perf_counter()
        # Print the training time
        print(f"Training duration on {model.device} : {(stop - start) // 60}min{(stop - start) % 60:.2f}")
    else:
        # Load the pre-trained model
        model.load_pretrained_model()

    # Move the testing set to the correct device
    test_clean_imgs = test_clean_imgs.to(model.device)

    # Compute the predictions on the testing set
    pred = model.predict(test_imgs)

    print(f"PSNR on test set : {psnr(normalizing_data(pred), normalizing_data(test_clean_imgs)):.4f}")
    return model


def get_and_print_model_parameters(model):
    """
    Compute the number and the size of the paramerts in the given model
    :param model: the model to analyse
    """
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Get the number of aprameters and ram needed for the parameters.
    # https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    print(f"Memory consumption {mem} Bytes")


class Noise2NoiseNetwork(nn.Module):
    """
    Network aiming at reflecting as much as possible the network from the original Noise2Noise paper given the time constraints and size of the images.
    The networks is as follows:
        - 5 downsampling layers consisting of two convolutions, Leaky relu and maxpooling to downsample the image
        - 5 upsampling layer conssits of nearest neighbour upsample, followed by two convolutions and leaky relu
        - Use of skip connection when we upsample to keep a maximum of information
    :param in_channels: the number of channels in the input images
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.name = "Noise2NoiseNetwork"
        # Depth of the different layers
        self.in_channels = in_channels
        self.channel_depth_1 = 48
        self.channel_depth_2 = 48
        self.channel_depth_3 = 48
        self.channel_depth_4 = 48
        self.channel_depth_5 = 48

        # Downsample block 1
        self.down1 = MultipleConvs(
            in_channels=[self.in_channels, self.channel_depth_1],
            out_channels=[self.channel_depth_1] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="down",
            nb_conv=2)
        # Downsample block 2
        self.down2 = MultipleConvs(
            in_channels=[self.channel_depth_1, self.channel_depth_2],
            out_channels=[self.channel_depth_2] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="down",
            nb_conv=2)
        # Downsample block 3
        self.down3 = MultipleConvs(
            in_channels=[self.channel_depth_2, self.channel_depth_3],
            out_channels=[self.channel_depth_3] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="down",
            nb_conv=2)
        # Downsample block 4
        self.down4 = MultipleConvs(
            in_channels=[self.channel_depth_3, self.channel_depth_4],
            out_channels=[self.channel_depth_4] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="down",
            nb_conv=2)
        # Downsample block 5
        self.down5 = MultipleConvs(
            in_channels=[self.channel_depth_4, self.channel_depth_5],
            out_channels=[self.channel_depth_5] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="down",
            nb_conv=2)

        # Nearest neighbour upsample (by a factor of 2)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        # Upsample block 4
        self.conv_up4 = MultipleConvs(
            in_channels=[self.channel_depth_5 + self.channel_depth_4, self.channel_depth_4],
            out_channels=[self.channel_depth_4] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="up",
            nb_conv=2)
        # Upsample block 3
        self.conv_up3 = MultipleConvs(
            in_channels=[self.channel_depth_3 + self.channel_depth_4, self.channel_depth_3],
            out_channels=[self.channel_depth_3] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="up",
            nb_conv=2)
        # Upsample block 2
        self.conv_up2 = MultipleConvs(
            in_channels=[self.channel_depth_2 + self.channel_depth_3, self.channel_depth_2],
            out_channels=[self.channel_depth_2] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="up",
            nb_conv=2)
        # Upsample block 1
        self.conv_up1 = MultipleConvs(
            in_channels=[self.channel_depth_1 + self.channel_depth_2, self.channel_depth_1],
            out_channels=[self.channel_depth_1] * 2,
            kernel_sizes=[3] * 2,
            paddings=[1] * 2,
            mode="up",
            nb_conv=2)

        # Convolution for the middle layer
        self.middle_conv = Conv2d(in_channels=self.channel_depth_5,
                                  out_channels=self.channel_depth_5,
                                  kernel_size=1)

        # 3 Convolutions for the final concatenation
        self.conv_final_1 = Conv2d(in_channels=self.channel_depth_1 + self.in_channels,
                                   out_channels=64,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode='replicate')

        self.conv_final_2 = Conv2d(in_channels=64,
                                   out_channels=32,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode='replicate')

        self.conv_final_3 = Conv2d(in_channels=32,
                                   out_channels=self.in_channels,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode='replicate')

        self.non_linear = LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # Downsample the images and save all the intermediate states
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        # Convolutions for the middle part
        mid = self.non_linear(self.middle_conv(down5))

        # Start by upsampling the first layer
        up5 = self.upsampler(mid)

        # Apply the upsampling to the concatenated layers
        up4 = self.upsampler(self.conv_up4(torch.cat((up5, down4), 1)))
        up3 = self.upsampler(self.conv_up3(torch.cat((up4, down3), 1)))
        up2 = self.upsampler(self.conv_up2(torch.cat((up3, down2), 1)))
        up1 = self.upsampler(self.conv_up1(torch.cat((up2, down1), 1)))

        # Perform the 3 final convolutions + non linearity activations
        conv_final_1 = self.non_linear(self.conv_final_1(torch.cat((up1, x), 1)))
        conv_final_2 = self.non_linear(self.conv_final_2(conv_final_1))
        return self.conv_final_3(conv_final_2)




"""
This file contains helper classes (subnets) for our different neural network architectures
"""


class MultipleConvs(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, mode="down", relu_type="leaky", nb_conv=2):
        """
        Constructor for a class representing a given number of convolutions followed by non-linearity
        :param in_channels: list telling the number of input channels for each convolutions
        :param out_channels: list containing the number of output channels for each convolution
        :param kernel_sizes: list containing the kernel size for each convolution
        :param paddings: list containing the padding for each convolution
        :param mode: parameter telling us if we are in downsample or in upsample
        :param relu_type: the type of ReLU we want to follow our convolution (either leaky or classical ReLU)
        :param nb_conv: the number of sequential convolution we consider.
        """
        super().__init__()
        self.nb_conv = nb_conv
        # Define the nonlinearity
        self.non_linearity = LeakyReLU(negative_slope=0.1) if relu_type == "leaky" else ReLU()
        # Generate a list of convulations
        self.convs = ModuleList([Conv2d(
            in_channels=in_channels[i],
            out_channels=out_channels[i],
            kernel_size=kernel_sizes[i],
            padding=paddings[i],
            padding_mode='replicate')
            for i in range(nb_conv)])
        # Define the pooling kernel
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.mode = mode

    def forward(self, x):
        for i in range(self.nb_conv):
            # For the forward pass, simply apply convolution followed by the chosen non-linearity
            x = self.non_linearity(self.convs[i](x))
        # If we are in "down" mode, we need to downsample the image, so apply the MaxPooling layer.
        return self.pool(x) if self.mode == "down" else x


class RFFT_IRFFT(nn.Module):
    def __init__(self, device, high_freq_nb=3):
        """
        Constructor for a class separating images in high and low frequencies using the FFT.
        :param device: the device on which the network is currently stored
        :param high_freq_nb: the number of components we keep for the low frequency image
        """
        super().__init__()
        # Store the given attributes.
        self.high_freq_nb = high_freq_nb
        self.device = device

    def forward(self, x):
        # This method supposed that the given images are square images
        x_size = x.shape[2]
        assert (0 <= self.high_freq_nb < x.shape[
            2] // 2)  # Ensure that the given number of components is smaller than half the diemnsion of the images
        # Define delimiters
        x_lower_high = self.high_freq_nb
        x_higher_high = x.shape[2] - self.high_freq_nb

        # Compute the fast fourier transform and move it to the correct device, for each channel
        fft_cpt = fft2(x, norm="forward").to(self.device)
        # Define zero tensors that will contains the high and low frequency coefficients of the Fourier tranform.
        fft_low = torch.zeros(x.shape).to(self.device)
        fft_high = torch.zeros(x.shape).to(self.device)

        # Select only the "border" of the FFT for the low frequency
        fft_low[:, :, :x_lower_high, :] = fft_cpt[:, :, :x_lower_high, :]
        fft_low[:, :, x_higher_high:, :] = fft_cpt[:, :, x_higher_high:, :]
        fft_low[:, :, x_lower_high:x_higher_high:, :x_lower_high] = fft_cpt[:, :, x_lower_high:x_higher_high:,
                                                                    :x_lower_high]
        fft_low[:, :, x_lower_high:x_higher_high:, x_higher_high:] = fft_cpt[:, :, x_lower_high:x_higher_high:,
                                                                     x_higher_high:]

        # Select only the center of the FFT for the high-frequency
        fft_high[:, :, x_lower_high:x_higher_high, x_lower_high:x_higher_high] = fft_cpt[:, :,
                                                                                 x_lower_high:x_higher_high,
                                                                                 x_lower_high:x_higher_high]

        # Concatenate the original signal, with the inverse FFT with only low and high frequencies
        return torch.cat(
            [x, ifft(fft_low, norm="forward", dim=1).float(), ifft(fft_high, norm="forward", dim=1).float()], dim=1)


class InceptionModule(nn.Module):
    """
    Class representing an inception module, i.e. a module that perform pooling, 1x1, 3x3 and 5x5 convolutions and let the model chooses what it needs
    :param in_channel: the number of channel as input
    :param out_channel: the number of channel required as output
    :param mode: parameter telling us if we are in downsample or upsample mode
    """

    def __init__(self, in_channel, out_channel, mode="down"):
        super().__init__()
        self.mode = mode

        # Only one 1x1 convolution for the first branch
        self.branch1x1 = Conv2d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=1)

        # For the 3x3 branch, we start by a 1x1 convolution
        self.branch3x3_1 = Conv2d(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=1)
        # Then, we use a 3x3 one
        self.branch3x3 = Conv2d(in_channels=out_channel,
                                out_channels=out_channel,
                                kernel_size=3,
                                padding=1,
                                padding_mode="replicate")

        # For the 5x5 branch, we start by a 1x1 convolution
        self.branch5x5_1 = Conv2d(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=1)
        # Then, we use a 5x5 one
        self.branch5x5 = Conv2d(in_channels=out_channel,
                                out_channels=out_channel,
                                kernel_size=5,
                                padding=2,
                                padding_mode="replicate")
        # Pooling branch is done via average pooling followed by a convolution
        self.poolbranch = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.poolbranchconv = Conv2d(in_channels=in_channel,
                                     out_channels=out_channel,
                                     kernel_size=1)
        # The nonlinearity is a leaky relu
        self.non_linear = LeakyReLU(negative_slope=0.1)

        # Convolution for the concatenated result
        self.final_conv_1 = Conv2d(in_channels=out_channel * 3 + in_channel, out_channels=out_channel, kernel_size=3,
                                   padding=1)
        # Pooling block if we have to downsample the image
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Perform the forward pass according to the inception module definition
        :param x: the tensor to which we want to compute the forward pass on
        """
        # Compute the results of the different branches
        branch1x1 = self.non_linear(self.branch1x1(x))
        branch3x3 = self.non_linear(self.branch3x3(self.non_linear(self.branch3x3_1(x))))
        branch5x5 = self.non_linear(self.branch5x5(self.non_linear(self.branch5x5_1(x))))
        pool = self.non_linear(self.poolbranch(x))
        # Concatenate all filters
        concat_filters = torch.concat([branch1x1, branch3x3, branch5x5, pool], dim=1)
        # Convolve the output and apply the select non-linearity
        x = self.non_linear(self.final_conv_1(concat_filters))
        # If we are in downsample mode, perform max pooling to reduce size by two, otherwise return the already computed result
        return self.pool(x) if self.mode == "down" else x
