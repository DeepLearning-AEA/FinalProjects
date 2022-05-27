import matplotlib.pyplot as plt
import torch
from datetime import datetime
import pickle
import time
import os

#from others.plots import *
#from others.metrics import *
#from others.DataLoading import load_and_split_dataset, normalizing_data, denormalize_data

from others.Noise2NoiseNetwork import *
#from others.Noise2NoiseFFT import *
#from others.Noise2NoiseInception import *
#from others.Noise2NoiseDeepNetwork import *
#from others.Noise2NoiseHLS import *
#from others.Noise2Noise3 import *

DATA_FOLDER = "others/data/"


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
