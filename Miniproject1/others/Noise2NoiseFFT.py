import torch
from torch.nn import *
from torch.optim import *
from torch import nn
from others.ModuleHelper import *

class Noise2NoiseFFT(nn.Module):
    """
    Noise2Noise model using FFT to separate low from high frequencies
    4x downsampling operation with FFT
    4x upsampling block
    """
    def __init__(self, in_channels=3, device=None):
        super().__init__()
        self.name = "Noise2NoiseFFT"
        self.in_channels = in_channels
        self.channel_depth_1 = 48
        self.channel_depth_2 = 48
        self.channel_depth_3 = 48
        self.channel_depth_4 = 48
        
        self.down1 = MultipleConvs(
                        in_channels=[3*self.in_channels, self.channel_depth_1,self.channel_depth_1],
                        out_channels=[self.channel_depth_1]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="down",
                        nb_conv=2)
        
        self.down2 = MultipleConvs(
                        in_channels=[3*self.channel_depth_1, self.channel_depth_2,self.channel_depth_2],
                        out_channels=[self.channel_depth_2]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="down",
                        nb_conv=2)
        
        self.down3 = MultipleConvs(
                        in_channels=[3*self.channel_depth_2, self.channel_depth_3,self.channel_depth_3],
                        out_channels=[self.channel_depth_3]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="down",
                        nb_conv=2)
        
        self.down4 = MultipleConvs(
                        in_channels=[3*self.channel_depth_3, self.channel_depth_4,self.channel_depth_4],
                        out_channels=[self.channel_depth_4]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="down",
                        nb_conv=2)
   
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
    
        self.middle = Conv2d(in_channels = self.channel_depth_4,
                         out_channels = self.channel_depth_4,
                         kernel_size=3,
                         padding=1,
                         padding_mode='replicate')
        self.non_linearity = LeakyReLU(negative_slope=0.1)
    
        self.conv_up3 = MultipleConvs(
                            in_channels=[self.channel_depth_3+self.channel_depth_4, self.channel_depth_3],
                            out_channels=[self.channel_depth_3]*2,
                            kernel_sizes = [3]*2,
                            paddings=[1]*2,
                            mode="up",
                            nb_conv=2)
        self.conv_up2 = MultipleConvs(
                            in_channels=[self.channel_depth_2+self.channel_depth_3, self.channel_depth_2],
                            out_channels=[self.channel_depth_2]*2,
                            kernel_sizes = [3]*2,
                            paddings=[1]*2,
                            mode="up",
                            nb_conv=2)
        
        self.conv_up1 = MultipleConvs(
                            in_channels=[self.channel_depth_1+self.channel_depth_2, self.channel_depth_1],
                            out_channels=[self.channel_depth_1]*2,
                            kernel_sizes = [3]*2,
                            paddings=[1]*2,
                            mode="up",
                            nb_conv=2)
        
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
        
        self.conv_final_3 = Conv2d(in_channels = 32,
                         out_channels = self.in_channels,
                         kernel_size=3,
                         padding=1,
                         padding_mode='replicate')
        # The fourier transform block
        self.fft = RFFT_IRFFT(device)
        
        
    def forward(self, x):
        down1 = self.down1(self.fft(x))
        down2 = self.down2(self.fft(down1))
        down3 = self.down3(self.fft(down2))
        down4 = self.down4(self.fft(down3))
        
        up4 = self.upsampler(self.non_linearity(self.middle(down4)))
        # Apply the upsampling to the concatenated layers
        up3 = self.upsampler(self.conv_up3(torch.cat((up4,down3),1)))
        up2 =  self.upsampler(self.conv_up2(torch.cat((up3,down2),1)))
        up1 =  self.upsampler(self.conv_up1(torch.cat((up2,down1),1)))
        conv_1 = self.non_linearity(self.conv_final_1(torch.cat((up1, x),1)))
        conv_2 = self.non_linearity(self.conv_final_2(conv_1))
        
        return self.conv_final_3(conv_2)