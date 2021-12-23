
"""SD-UNET Architecture"""
"""A Novel Segmentation framework for CT Images of Lung Infections"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math
from .feature_denosing import NonLocalBlockND
from.yin_net import DenseASPP


"""  conv block ""
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32,ch_out),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32,ch_out),
            nn.LeakyReLU(0.2,inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

""" SA moudule """
class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        #self.conv_res=nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)
       # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # print(x.shape)
        #x_in=self.conv_res(x)
        x_res = self.conv(x)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.conv_atten(y)
        # print(y.shape)
        y = self.upsample(y)
        # print(y.shape, x_res.shape)
        return (y * x_res) + y


####################################
class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.block = SqueezeAttentionBlock(in_channels, out_channels)
        #self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #x = self._conv(x)
        x = self.block(x)
        return x

"""  down-sampling  """
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.block = SqueezeAttentionBlock(in_channels, out_channels)
        self.maxpool =nn.MaxPool2d(kernel_size=2,stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.conv1=nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.down = nn.Sequential(
            # 指定输出为输入的多少倍数。mode=nearest
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(32,in_channels),
            nn.LeakyReLU(0.2, inplace=True))
        #self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1=self.maxpool(x)
        x2=self.avg_pool(x)
        x = torch.cat([x2, x1], dim=1)
        x=self.down(x)
        x = self.block(x)
        return x

"""up-sampling"""
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.block = SqueezeAttentionBlock(in_channels, out_channels)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 指定输出为输入的多少倍数。mode=nearest
            nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.GroupNorm(32,out_channels),
            nn.LeakyReLU(0.2,inplace=True))
        #self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self,x,x2):
        x = self.up(x)
        x = torch.cat([x2, x], dim=1)
        #x=self._conv(x)
        x = self.block(x)
        return x
""" output the semantic segmentation map """
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # self.conv5a = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        #                             nn.BatchNorm2d(in_channels),
        #                             nn.ReLU(inplace=True))
        #
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        #                             nn.BatchNorm2d(in_channels),
        #                             nn.ReLU(inplace=True))
        #
        # self.sa = PAM_Module(in_channels)
        # self.sc = CAM_Module(in_channels)
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=True)


    def forward(self, x):
        # featsa = self.conv5a(x)
        # featsc = self.conv5c(x)
        # sa_feat = self.sa(featsa)
        # sc_feat = self.sc(featsc)
        # feat_sum = sc_feat+sa_feat

        x = self._conv(x)

        return x

"""SD-UNET"""
class SD-UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(DualNorm_Unet, self).__init__()

        # self.non_local4 = NonLocalBlockND(1024, inter_channels=512, dimension=2)
        # self.non_local3 = NonLocalBlockND(256, inter_channels=128, dimension=2)
        # self.non_local2 = NonLocalBlockND(128, inter_channels=64, dimension=2)
        #self.non_local1 = NonLocalBlockND(64, inter_channels=32, dimension=2)

        self.inc = InConv(img_ch, 32)  #try lager number of channels for better accuarcy
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.as_pp = DenseASPP(512,atrous_rates=[6,12,18]) #改为3，6，9
        # self.mid = Mid(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, output_ch)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5_1 = self.down4(x4)
        x5_1 = self.as_pp(x5_1)

        #x5_1=self.non_local4(x5_1)
        x = self.up1(x5_1, x4)
        #x3_1 = self.non_local3(x3)
        x = self.up2(x, x3)
        #x2_1 = self.non_local2(x2)
        x = self.up3(x, x2)
        #x1_1 = self.non_local1(x1)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x





