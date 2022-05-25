import torch
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from others.Noise2NoiseNetwork import *
from net_helper import evaluate_performance, save_model, get_and_print_model_parameters

from others.DataLoading import load_and_split_dataset, normalizing_data, denormalize_data
from others.plots import show


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
        self.model.load_state_dict(torch.load("bestmodel.pth", map_location=self.device))
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
                scheduler.step()

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
