import torch
import torch.nn as nn
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F


@HEADS.register_module()
class UnetHead(BaseDecodeHead):
    def __init__(self, decoder_channel=[1024, 512, 256, 128, 64], ca=True, resPath=False, **kwargs):
        super(UnetHead, self).__init__(**kwargs)
        self.up1 = Up(decoder_channel[0], int(decoder_channel[0] / 4), resPath, resLength=1, atrous=False, dilation=1)
        self.up2 = Up(decoder_channel[1], int(decoder_channel[1] / 4), resPath, resLength=2, atrous=False, dilation=2)
        self.up3 = Up(decoder_channel[2], int(decoder_channel[2] / 4), resPath, resLength=3, atrous=False, dilation=4)
        self.up4 = Up(decoder_channel[3], decoder_channel[4], ca, resPath, resLength=4, atrous=False, dilation=6)


    def forward(self, inputs):
        out = self.up1(inputs[4], inputs[3])
        out = self.up2(out, inputs[2])
        out = self.up3(out, inputs[1])
        out = self.up4(out, inputs[0])
        output = self.cls_seg(out)
        return output


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, resPath=False, resLength=0, atrous=False, dilation=1, mac=False):
        # 定义了self.up的方法
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)  # // 除以的结果向下取整

        self.coord = CoordAtt(in_ch, in_ch)

        self.resPath = resPath
        if self.resPath:
            self.resSkip = Respath(in_ch//2, in_ch//2, resLength)

        self.atrous = atrous
        if self.atrous:
            padding = (dilation * (3 - 1)) // 2  # 计算填充值
            self.atrous_conv = nn.Conv2d(in_ch//2, in_ch//2, kernel_size=3, padding=padding, dilation=dilation)

        if mac:
            self.conv = MAC(in_ch, out_ch)
        else:
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.resPath:
            x2 = self.resSkip(x2)

        if self.atrous:
            x2 = x2 + self.atrous_conv(x2)

        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # 融合
        x = torch.cat([x2, x1], dim=1)  # 将两个tensor拼接在一起 dim=1：在通道数（C）上进行拼接
        x = self.coord(x) + x
        # 卷积
        x = self.conv(x)
        return x

class MAC(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(MAC, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), dilation=(1, 1), padding="same"),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), dilation=(1, 1), padding="same"),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), dilation=(1, 2), padding="same"),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), dilation=(2, 1), padding="same"),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), dilation=(1, 3), padding="same"),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), dilation=(3, 1), padding="same"),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))


    def forward(self, x):
        x1 = self.conv1_2(self.conv1_1(x))
        x2 = self.conv2_2(self.conv2_1(x))
        x3 = self.conv3_2(self.conv3_1(x))
        return x1 + x2 + x3

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SeBlock(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h


# 残差路径
class Respath(torch.nn.Module):

    def __init__(self, num_in_filters, num_out_filters, respath_length):

        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if (i == 0):
                self.shortcuts.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))
            else:
                self.shortcuts.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):

        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
        return x


class Conv2d_batchnorm(torch.nn.Module):

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                                     stride=stride, padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        else:
            return x

