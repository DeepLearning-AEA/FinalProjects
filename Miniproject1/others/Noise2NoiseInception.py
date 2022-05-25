import torch
from torch.nn import *
from torch.optim import *
from torch import nn
from others.ModuleHelper import *

class Noise2NoiseInception(nn.Module):
    def __init__(self, in_channels=3):
        """
        Network inspired by the Noise2Noise original network, with less channel during the first 3 downsampling and using inception module in each downsampling/upsampling block.
            - 4 downsampling layers consisting inception module, Leaky relu and maxpooling to downsample the image
            - 4 upsampling layer again using inception module
            - Use of skip connection when we upsample to keep a maximum of information
        :param in_channels: the number of channels in the input images 
        """
        super().__init__()
        self.name = "Noise2NoiseInception"
        self.in_channels = in_channels
        self.channel_depth_1 = 32
        self.channel_depth_2 = 48
        self.channel_depth_3 = 48
        self.channel_depth_4 = 48
        
        self.inception_down1 = InceptionModule(self.in_channels, self.channel_depth_1)
        self.inception_down2 = InceptionModule(self.channel_depth_1, self.channel_depth_2)
        self.inception_down3 = InceptionModule(self.channel_depth_2, self.channel_depth_3)
        self.inception_down4 = InceptionModule(self.channel_depth_4, self.channel_depth_4)
        
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
    
        self.inception_up3 = InceptionModule(self.channel_depth_4+self.channel_depth_3, self.channel_depth_3, mode="up")
        self.inception_up2 = InceptionModule(self.channel_depth_2+self.channel_depth_3, self.channel_depth_2, mode="up")
        self.inception_up1 = InceptionModule(self.channel_depth_1+self.channel_depth_2, self.channel_depth_1, mode="up")
        
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
        down1 = self.inception_down1(x)
        down2 = self.inception_down2(down1)
        down3 = self.inception_down3(down2)
        down4 = self.inception_down4(down3)
        
        up4 = self.upsampler(down4)
        
        # Apply the upsampling to the concatenated layers
        up3 = self.upsampler(self.inception_up3(torch.cat((up4,down3),1)))
        up2 =  self.upsampler(self.inception_up2(torch.cat((up3,down2),1)))
        up1 =  self.upsampler(self.inception_up1(torch.cat((up2,down1),1)))
        
        return self.conv_final_3(self.non_linear(self.conv_final_2(self.non_linear(self.conv_final_1(torch.cat((up1, x),1))))))