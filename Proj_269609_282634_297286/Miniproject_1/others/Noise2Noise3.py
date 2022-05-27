import torch
from torch.nn import *
from torch.optim import *
from torch import nn
from others.ModuleHelper import *

class Noise2NoiseNetwork3(nn.Module):
    """
    Network aiming at reflecting as much as possible the network from the original Noise2Noise paper given the time constraints and size of the images.
    The networks is as follows: 
        - 3 downsampling layers consisting of two convolutions, Leaky relu and maxpooling to downsample the image
        - 3 upsampling layer conssits of nearest neighbour upsample, followed by two convolutions and leaky relu
        - Use of skip connection when we upsample to keep a maximum of information
    :param in_channels: the number of channels in the input images 
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.name = "Noise2NoiseNetwork3"
        # Depth of the different layers
        self.in_channels = in_channels
        self.channel_depth_1 = 48
        self.channel_depth_2 = 48
        self.channel_depth_3 = 48
        
        # Downsample block 1
        self.down1 = MultipleConvs(
                        in_channels=[self.in_channels, self.channel_depth_1],
                        out_channels=[self.channel_depth_1]*2,
                        kernel_sizes = [3]*2,
                        paddings=[1]*2,
                        mode="down",
                        nb_conv=2)
        # Downsample block 2
        self.down2 = MultipleConvs(
                        in_channels=[self.channel_depth_1, self.channel_depth_2],
                        out_channels=[self.channel_depth_2]*2,
                        kernel_sizes = [3]*2,
                        paddings=[1]*2,
                        mode="down",
                        nb_conv=2)
        # Downsample block 3
        self.down3 = MultipleConvs(
                        in_channels=[self.channel_depth_2, self.channel_depth_3],
                        out_channels=[self.channel_depth_3]*2,
                        kernel_sizes = [3]*2,
                        paddings=[1]*2,
                        mode="down",
                        nb_conv=2)
        
        
        # Nearest neighbour upsample (by a factor of 2)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        # Upsample block 4
       
    
        # Upsample block 2
        self.conv_up2 = MultipleConvs(
                            in_channels=[self.channel_depth_2+self.channel_depth_3, self.channel_depth_2],
                            out_channels=[self.channel_depth_2]*2,
                            kernel_sizes = [3]*2,
                            paddings=[1]*2,
                            mode="up",
                            nb_conv=2)
        # Upsample block 1
        self.conv_up1 = MultipleConvs(
                            in_channels=[self.channel_depth_1+self.channel_depth_2, self.channel_depth_1],
                            out_channels=[self.channel_depth_1]*2,
                            kernel_sizes = [3]*2,
                            paddings=[1]*2,
                            mode="up",
                            nb_conv=2)
        
        # Convolution for the middle layer
        self.middle_conv = MultipleConvs(
                            in_channels=[self.channel_depth_3, self.channel_depth_3],
                            out_channels=[self.channel_depth_3]*2,
                            kernel_sizes = [3]*2,
                            paddings=[1]*2,
                            mode="up",
                            nb_conv=2)
        
        # 3 Convolutions for the final concatenation
        self.conv_final_1 = Conv2d(in_channels = self.channel_depth_1 + self.in_channels,
                         out_channels = 64,
                         kernel_size=3,
                         padding=1,
                         padding_mode='replicate')
        
        self.conv_final_2 = Conv2d(in_channels = 64,
                         out_channels = 32,
                         kernel_size=3,
                         padding=1,
                         padding_mode='replicate')
        
        self.conv_final_3 = Conv2d(in_channels = 32 ,
                         out_channels = self.in_channels,
                         kernel_size=3,
                         padding=1,
                         padding_mode='replicate')
        
        self.non_linear = LeakyReLU(negative_slope=0.1)
        
    def forward(self, x):
        # Downsample the images and save all the intermediate states
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        
        # Convolutions for the middle part
        mid = self.non_linear(self.middle_conv(down3))
        
        # Start by upsampling the first layer 
        up3 = self.upsampler(mid)
        
        # Apply the upsampling to the concatenated layers
        up2 =  self.upsampler(self.conv_up2(torch.cat((up3,down2),1)))
        up1 =  self.upsampler(self.conv_up1(torch.cat((up2,down1),1)))
        
        # Perform the 3 final convolutions + non linearity activations
        conv_final_1 = self.non_linear(self.conv_final_1(torch.cat((up1, x),1)))
        conv_final_2 = self.non_linear(self.conv_final_2(conv_final_1))
        return self.conv_final_3(conv_final_2)
     