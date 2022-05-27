from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import *
from torchvision.transforms.functional import *

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
    train_input, train_target, val_input, val_target = split_data(train_input,train_target,ratio)
   
    if data_augmentation:
        # If we need to do data_augmentation, perform 3 random crop of the input images and resize it to their original size
        original_dimension = train_input[0].shape[1], train_input[0].shape[2]
        
        # set default paramerters
        train_input_ = train_input
        train_target_ = train_target
        
        # Generate random crops
        for i in range(3):
            # Get random crop params of size 32x32
            i, j, h, w = RandomCrop.get_params(train_input, output_size=(original_dimension[0] // 2, original_dimension[1] // 2))
            # Crop and store the different crops
            train_input_ = torch.cat([train_input_, resized_crop(train_input, i, j, h, w, [original_dimension[0],original_dimension[1]])],0)
            train_target_ = torch.cat([train_target_, resized_crop(train_target, i, j, h, w,[original_dimension[0],original_dimension[1]])],0)
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
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=False) if ratio!=1.0 else None
    
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
    nb_train = int(ratio*nb_data)
    
    return x[:nb_train], y[:nb_train], x[nb_train:], y[nb_train:]

def normalizing_data(x):
    """
    Perform max-normalization so that the input on each channel is between 0 and 1
    :param x: the data we want to normalise
    :return: the normalized data
    """
    #max_s = torch.amax(x, dim=(1,2,3))[:, None, None, None]
    x_norm = x/255
    
    # Check that the max_normalization indeed works
    assert(torch.max(x_norm)<=1.0)
    assert(torch.min(x_norm)>=0.0)
    
    return x_norm

def denormalize_data(x):
    """
    Undo the max-normalization and clip so that the input on each channel is between 0 and 255
    :param x: the data we want to denormalize
    :return: the denormalizerd data
    """
    x_denorm = x*255
    return torch.clip(x_denorm, min=0.0, max=255.0)

