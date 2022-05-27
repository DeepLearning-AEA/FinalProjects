
def normalizing_data(x):
    """
    Perform max-normalization so that the input on each channel is between 0 and 1
    :param x: the data we want to normalise
    :return: the normalized data
    """
    #max_s = torch.amax(x, dim=(1,2,3))[:, None, None, None]
    x_norm = x/255
    
    # Check that the max_normalization indeed works
    assert(x_norm.max()<=1.0)
    assert(x_norm.min()>=0.0)
    
    return x_norm

def denormalize_data(x):
    """
    Undo the max-normalization and clip so that the input on each channel is between 0 and 255
    :param x: the data we want to denormalize
    :return: the denormalizerd data
    """
    x_denorm = x*255
    return x_denorm.clip(min=0.0, max=255.0)

