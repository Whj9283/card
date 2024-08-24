import torch
from torch import nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead
import itertools
import torch.nn.functional as F


@HEADS.register_module()
class LevitUnet(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=512,  ## aping: should change
                 patch_size=16,
                 in_chans=3,
                 num_classes=4,
                 embed_dim=[384, 512, 768],
                 key_dim=[32] * 3,
                 depth=[4, 4, 4],
                 num_heads=[6, 9, 12],
                 attn_ratio=[2, 2, 2],
                 mlp_ratio=[2, 2, 2],
                 hybrid_backbone=None,
                 down_ops=[
                     ['Subsample', 32, 12, 4, 2, 2],
                     ['Subsample', 32, 16, 4, 2, 2],
                 ],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=False,
                 drop_path=0.1, **kwargs):
        super(LevitUnet, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        # self.patch_embed = hybrid_backbone ## aping: CNN  # aping num_channel
        n = 384  ## cnn num channel
        activation = torch.nn.Hardswish

        # 初始化的第一层卷积 最后不能忽略第一层
        self.conv_init = torch.nn.Sequential(Conv2d_BN(in_chans, n // 8, 3, 1, pad=1), activation())

        # 第一次下采样卷积 (H/2,W/2)
        self.cnn_b1 = torch.nn.Sequential(
            Conv2d_BN(n // 8, n // 8, 3, 2, 1, resolution=img_size), activation())

        # 并行分支 进行残差卷积 保留高分辨率特征 保留(H/2,W/2)
        self.atrous_conv1 = torch.nn.Sequential(
            Conv2d_BN(n // 8, n // 4, 3, dilation=2, pad=2, resolution=img_size), activation())
        self.combine1 = torch.nn.Sequential(Conv2d_BN(n // 4, n // 4, 3, 2, 1), activation())
        self.atrous_conv2 = torch.nn.Sequential(
            Conv2d_BN(n // 4, n // 2, 3, dilation=3, pad=3, resolution=img_size), activation())
        self.combine2 = torch.nn.Sequential(Conv2d_BN(n // 2, n // 2, 3, 2, 1), activation(),
                                            Conv2d_BN(n // 2, n // 2, 3, 2, 1), activation())
        self.atrous_conv3 = torch.nn.Sequential(
            Conv2d_BN(n // 2, n, 3, dilation=4, pad=4, resolution=img_size), activation())
        self.combine3 = torch.nn.Sequential(Conv2d_BN(n, n, 3, 2, 1), activation(),
                                            Conv2d_BN(n, n, 3, 2, 1), activation(),
                                            Conv2d_BN(n, n, 3, 2, 1), activation())

        self.cnn_b2 = torch.nn.Sequential(
            Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=img_size // 2), activation())
        self.cnn_b3 = torch.nn.Sequential(
            Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=img_size // 4), activation())
        self.cnn_b4 = torch.nn.Sequential(
            Conv2d_BN(n // 2, n, 3, 2, 1, resolution=img_size // 8))

        self.decoderBlock_1 = DecoderBlock(2048, 512)  # ->att+cnn:384+512+768+384=2048==1280+256
        self.decoderBlock_2 = DecoderBlock(704, 256)  # ->28->56, 512+9+192=713
        self.decoderBlock_3 = DecoderBlock(352, 128)  # ->56->112, 256+9+96=361
        self.decoderBlock_4 = DecoderBlock(176, 64)  
        # ->56->112, 256+9+96=361
        self.segmentation_head = SegmentationHead(112, self.num_classes, kernel_size=3, upsampling=2)

        self.blocks = []  ## attention block
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
        self.blocks = torch.nn.Sequential(*self.blocks)
        ## divid the blocks
        self.block_1 = self.blocks[0:8]
        self.block_2 = self.blocks[8:18]
        self.block_3 = self.blocks[18:28]

        del self.blocks

        ## aping: upsampling
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):

        x_cnn_0 = self.conv_init(x)  # (n/8,H,W)
        x_cnn_1 = self.cnn_b1(x_cnn_0)  # (n/8,H/2,H/2)

#         # 高分辨率卷积结果
#         x_atrous2 = self.atrous_conv1(x_cnn_1)  # (n/4,H/2,H/2)

#         x_atrous3 = self.atrous_conv2(x_atrous2)  # (n/2,H/2,H/2)

#         x_atrous4 = self.atrous_conv3(x_atrous3)  # (n,H/2,H/2)


        # 融合高分辨率
        x_cnn_2 = self.cnn_b2(x_cnn_1)  # (n/4,H/4,H/4)
        # x_cnn_2 = self.combine1(x_atrous2) + x_cnn_2
        x_cnn_3 = self.cnn_b3(x_cnn_2)  # (n/2,H/2,H/2)
        # x_cnn_3 = self.combine2(x_atrous3) + x_cnn_3
        x_cnn = self.cnn_b4(x_cnn_3)  # (n,H/2,H/2)
        # x_cnn = self.combine3(x_atrous4) + x_cnn

        x = x_cnn.flatten(2).transpose(1, 2)  # torch.Size([4, 196, 256])

        ## aping
        x = self.block_1(x)  # torch.Size([4, 196, 384])-->Nx256x14x14
        x_num, x_len = x.shape[0], x.shape[1]

        x_r_1 = x.reshape(x_num, int(x_len ** 0.5), int(x_len ** 0.5), -1)
        x_r_1 = x_r_1.permute(0, 3, 1, 2)

        x = self.block_2(x)  # downsample + att  torch.Size([4, 49, 384])
        x_num, x_len = x.shape[0], x.shape[1]

        x_r_2 = x.reshape(x_num, int(x_len ** 0.5), int(x_len ** 0.5), -1)
        x_r_2 = x_r_2.permute(0, 3, 1, 2)
        ## upsampling
        x_r_2_up = self.up(x_r_2)

        x = self.block_3(x)  # torch.Size([4, 16, 512])
        x_num, x_len = x.shape[0], x.shape[1]

        x_r_3 = x.reshape(x_num, int(x_len ** 0.5), int(x_len ** 0.5), -1)
        x_r_3 = x_r_3.permute(0, 3, 1, 2)
        ## upsampling
        x_r_3_up = self.up(x_r_3)
        x_r_3_up = self.up(x_r_3_up)

        ## aping: resize the feature maps
        if (x_r_2_up.shape != x_r_3_up.shape):
            x_r_3_up = F.interpolate(x_r_3_up, size=x_r_2_up.shape[2:], mode="bilinear", align_corners=True)
        att_all = torch.cat([x_r_1, x_r_2_up, x_r_3_up], dim=1)  # 384+512+768

        x_att_all = torch.cat([x_cnn, att_all], dim=1)  ## torch.Size([4, 1408, 32, 32])
        decoder_feature = self.decoderBlock_1(x_att_all)  # x_att_all: ([4, 1408, 32, 32])->512
        decoder_feature = torch.cat([decoder_feature, x_cnn_3], dim=1)  #:(640+9)x64x64
        decoder_feature = self.decoderBlock_2(decoder_feature)

        decoder_feature = torch.cat([decoder_feature, x_cnn_2], dim=1)  # ([4, (320+9), 128, 128])
        decoder_feature = self.decoderBlock_3(decoder_feature)

        decoder_feature = torch.cat([decoder_feature, x_cnn_1], dim=1)  # ([4, 169, 256, 256])
        decoder_feature = self.decoderBlock_4(decoder_feature)

        decoder_feature = torch.cat([decoder_feature, x_cnn_0], dim=1)
        logits = self.segmentation_head(decoder_feature)  ## torch.Size([4, 2, 224, 224])

        return logits


# 卷积bn模块
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(1, n // 8, 3, 2, 1, resolution=resolution),  # aping num_channel
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
        1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
        1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            # skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            # in_channels + skip_channels,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


