import torch

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
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

def psnr_mse(mse):
    """
    Compute the PSNR from the given MSE
    
    :param mse: The computed MSE from the data
    :return: the psnr 
    """
    return -10*torch.log10(mse+10**-8)