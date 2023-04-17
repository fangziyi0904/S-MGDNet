

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):


    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.conv(x1)


        return x1



class Siam(nn.Module):


    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=2,bias=False,dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=2,bias=False,dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.backbone(x)


class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear

        self.siamese_backbone = Siam()
 

        self.siamese_down1 = Down(64, 128)
        self.siamese_down2 = Down(128, 256)




        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)



        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.conv_seg = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
        )
        self.conv_denoise = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )


    def forward(self, x, mask):
        x1 = self.siamese_backbone(x)
        mask1 = self.siamese_backbone(mask)


        x1 = self.siamese_down1(x1)
        x1 = self.siamese_down2(x1)


        mask1 = self.siamese_down1(mask1)
        mask1 = self.siamese_down2(mask1)


        x_cat = torch.cat((x1,mask1), dim = 1)

        x_cat = self.up1(x_cat)
        x_cat = self.up2(x_cat)

        out = self.conv1(x_cat)
        seg_out = self.conv_seg(out)
        denoise_out = self.conv_denoise(out)
        return seg_out,denoise_out
