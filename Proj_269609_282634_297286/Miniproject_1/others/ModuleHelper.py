import torch
from torch.nn import *
from torch.optim import *
from torch import nn
from torch.fft import *

"""
This file contains helper classes (subnets) for our different neural network architectures
"""
class MultipleConvs(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, mode="down", relu_type="leaky", nb_conv = 2):
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
        self.pool = MaxPool2d(kernel_size = 2, stride=2)
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
        self.device= device
        
    def forward(self,x):
        # This method supposed that the given images are square images
        x_size = x.shape[2]
        assert(0<=self.high_freq_nb < x.shape[2]//2) # Ensure that the given number of components is smaller than half the diemnsion of the images
        # Define delimiters
        x_lower_high = self.high_freq_nb
        x_higher_high = x.shape[2]-self.high_freq_nb
        
        # Compute the fast fourier transform and move it to the correct device, for each channel
        fft_cpt = fft2(x, norm="forward").to(self.device)
        # Define zero tensors that will contains the high and low frequency coefficients of the Fourier tranform. 
        fft_low = torch.zeros(x.shape).to(self.device)
        fft_high = torch.zeros(x.shape).to(self.device)
        
        
        # Select only the "border" of the FFT for the low frequency
        fft_low[:,:,:x_lower_high,:]= fft_cpt[:,:,:x_lower_high,:]
        fft_low[:,:,x_higher_high:,:]= fft_cpt[:,:,x_higher_high:,:]
        fft_low[:,:,x_lower_high:x_higher_high:,:x_lower_high]= fft_cpt[:,:,x_lower_high:x_higher_high:,:x_lower_high]
        fft_low[:,:,x_lower_high:x_higher_high:,x_higher_high:]= fft_cpt[:,:,x_lower_high:x_higher_high:,x_higher_high:]
        
        # Select only the center of the FFT for the high-frequency
        fft_high[:,:,x_lower_high:x_higher_high,x_lower_high:x_higher_high]= fft_cpt[:,:, x_lower_high:x_higher_high, x_lower_high:x_higher_high]
        
        # Concatenate the original signal, with the inverse FFT with only low and high frequencies
        return torch.cat([x, ifft(fft_low, norm="forward", dim=1).float(), ifft(fft_high, norm="forward", dim=1).float()], dim=1)
              
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
        self.poolbranch = AvgPool2d(kernel_size=3,stride=1, padding=1)
        self.poolbranchconv = Conv2d(in_channels=in_channel,
                                out_channels=out_channel,
                                kernel_size=1)
        # The nonlinearity is a leaky relu
        self.non_linear = LeakyReLU(negative_slope=0.1)
        
        # Convolution for the concatenated result
        self.final_conv_1 = Conv2d(in_channels= out_channel*3 +in_channel,out_channels=out_channel, kernel_size=3, padding=1)
        # Pooling block if we have to downsample the image
        self.pool = MaxPool2d(kernel_size=2,stride=2)
        
    def forward(self,x):
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
        concat_filters = torch.concat([branch1x1,branch3x3, branch5x5, pool],dim=1)
        # Convolve the output and apply the select non-linearity
        x = self.non_linear(self.final_conv_1(concat_filters))
        # If we are in downsample mode, perform max pooling to reduce size by two, otherwise return the already computed result
        return self.pool(x) if self.mode=="down" else x
        