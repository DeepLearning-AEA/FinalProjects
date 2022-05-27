import torch
from torch.nn import *
from torch.optim import *
from torch import nn
from others.ModuleHelper import *

class Noise2NoiseNetworkHLS(nn.Module):
    def __init__(self, in_channels=3):
        """
        Network using a max-pooling with stride 1 instead of stride 2 to keep more information about the input
        5x downsampling blocks
        5x upsamplin blocks
        """
        super().__init__()
        self.name = "Noise2NoiseNetworkHLS"
        # Depth of the different layers
        self.in_channels = in_channels
        self.channel_depth_1 = 32
        self.channel_depth_2 = 32
        self.channel_depth_3 = 48
        self.channel_depth_4 = 48
        self.channel_depth_5 = 48
        
        # Downsample block 1
        self.down1 = MultipleConvs(
                        in_channels=[self.in_channels, self.channel_depth_1, self.channel_depth_1],
                        out_channels=[self.channel_depth_1]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="up",
                        nb_conv=2)
        # Downsample block 2
        self.down2 = MultipleConvs(
                        in_channels=[self.channel_depth_1, self.channel_depth_2, self.channel_depth_2],
                        out_channels=[self.channel_depth_2]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="up",
                        nb_conv=2)
        
        # Downsample block 3
        self.down3 = MultipleConvs(
                        in_channels=[self.channel_depth_2, self.channel_depth_3, self.channel_depth_3],
                        out_channels=[self.channel_depth_3]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="up",
                        nb_conv=2)
        # Downsample block 4
        self.down4 = MultipleConvs(
                        in_channels=[self.channel_depth_3, self.channel_depth_4, self.channel_depth_4],
                        out_channels=[self.channel_depth_4]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="up",
                        nb_conv=2)
        # Downsample block 5
        self.down5 = MultipleConvs(
                        in_channels=[self.channel_depth_4, self.channel_depth_5, self.channel_depth_5],
                        out_channels=[self.channel_depth_5]*3,
                        kernel_sizes = [3]*3,
                        paddings=[1]*3,
                        mode="up",
                        nb_conv=2)
        
        # Nearest neighbour upsample (by a factor of 2)
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        # Upsample block 4
        self.conv_up4 = MultipleConvs(
                                in_channels=[self.channel_depth_5+self.channel_depth_4, self.channel_depth_4, self.channel_depth_4],
                                out_channels=[self.channel_depth_4]*3,
                                kernel_sizes = [3]*3,
                                paddings=[1]*3,
                                mode="up",
                                nb_conv=2)
        # Upsample block 3
        self.conv_up3 = MultipleConvs(
                            in_channels=[self.channel_depth_3+self.channel_depth_4, self.channel_depth_3, self.channel_depth_3],
                            out_channels=[self.channel_depth_3]*3,
                            kernel_sizes = [3]*3,
                            paddings=[1]*3,
                            mode="up",
                            nb_conv=2)
        # Upsample block 2
        self.conv_up2 = MultipleConvs(
                            in_channels=[self.channel_depth_2+self.channel_depth_3, self.channel_depth_2, self.channel_depth_2],
                            out_channels=[self.channel_depth_2]*3,
                            kernel_sizes = [3]*3,
                            paddings=[1]*3,
                            mode="up",
                            nb_conv=2)
        # Upsample block 1
        self.conv_up1 = MultipleConvs(
                            in_channels=[self.channel_depth_1+self.channel_depth_2, self.channel_depth_1, self.channel_depth_1],
                            out_channels=[self.channel_depth_1]*3,
                            kernel_sizes = [3]*3,
                            paddings=[1]*3,
                            mode="up",
                            nb_conv=2)
        
        self.middle_conv = MultipleConvs(
                            in_channels=[self.channel_depth_5, self.channel_depth_5, self.channel_depth_5],
                            out_channels=[self.channel_depth_5]*3,
                            kernel_sizes = [3]*3,
                            paddings=[1]*3,
                            mode="up",
                            nb_conv=2)
        
        
        # max pooling 
        self.pool = MaxPool2d(kernel_size=3,stride=1)
        self.t_conv1 = ConvTranspose2d(in_channels=self.channel_depth_1, out_channels=self.channel_depth_1, stride=1, kernel_size=3)
        self.t_conv2 = ConvTranspose2d(in_channels=self.channel_depth_2, out_channels=self.channel_depth_1, stride=1, kernel_size=3)
        self.t_conv3 = ConvTranspose2d(in_channels=self.channel_depth_3, out_channels=self.channel_depth_2, stride=1, kernel_size=3)
        self.t_conv4 = ConvTranspose2d(in_channels=self.channel_depth_4, out_channels=self.channel_depth_4, stride=1, kernel_size=3)
        self.t_conv5 = ConvTranspose2d(in_channels=self.channel_depth_5, out_channels=self.channel_depth_5, stride=1, kernel_size=3)
        
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
        down1 = self.pool(self.down1(x))
        down2 = self.pool(self.down2(down1))
        down3 = self.pool(self.down3(down2))
        down4 = self.pool(self.down4(down3))
        down5 = self.pool(self.down5(down4))
        # Convolutions for the middle part
        mid = self.non_linear(self.middle_conv(down5))
        
        # Start by upsampling the first layer 
        up5 = self.t_conv5(mid)
        # Apply the upsampling to the concatenated layers
        up4 = self.t_conv4(self.conv_up4(torch.cat((up5,down4),1)))
        up3 = self.t_conv3(self.conv_up3(torch.cat((up4,down3),1)))
        up2 =  self.t_conv2(self.conv_up2(torch.cat((up3,down2),1)))
        up1 =  self.t_conv1(self.conv_up1(torch.cat((up2,down1),1)))
        
        # Perform the 3 final convolutions + non linearity activations
        conv_final_1 = self.non_linear(self.conv_final_1(torch.cat((up1, x),1)))
        conv_final_2 = self.non_linear(self.conv_final_2(conv_final_1))
        return self.conv_final_3(conv_final_2)
     