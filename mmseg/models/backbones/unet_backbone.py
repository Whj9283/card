from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import SyncBatchNorm
from ..builder import BACKBONES
import torch
import torch.nn as nn


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            Hswish())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


@BACKBONES.register_module()
class UnetBackbone(BaseModule):
    def __init__(self, in_channels=3, channel_list=[64, 128, 256, 512], context_layer=None, coord_att=False,
                 transformer_block=False, conv_down=False, **kwargs):
        super(UnetBackbone, self).__init__(**kwargs)
        if conv_down:
            self.inc = MACInConv(in_channels, channel_list[0])
            self.down1 = Conv_Down(channel_list[0], channel_list[1])
            self.down2 = Conv_Down(channel_list[1], channel_list[2])
            self.down3 = Conv_Down(channel_list[2], channel_list[3])
            self.down4 = Conv_Down(channel_list[3], channel_list[3])

        else:
            self.inc = InConv(in_channels, channel_list[0])
            self.down1 = Down(channel_list[0], channel_list[1], coord_att=coord_att)
            self.down2 = Down(channel_list[1], channel_list[2], coord_att=coord_att)
            self.down3 = Down(channel_list[2], channel_list[3], coord_att=coord_att)
            self.down4 = Down(channel_list[3], channel_list[3], coord_att=coord_att)

        self.context_layer = context_layer
        self.coord_att = coord_att
        self.transformer_block = transformer_block
        if self.context_layer == "seLayer":
            self.context_layer1_1 = SELayer(channel=channel_list[0])
            self.context_layer2_1 = SELayer(channel=channel_list[1])
            self.context_layer3_1 = SELayer(channel=channel_list[2])

        if self.transformer_block:
            self.aspp4 = TransformerBlock(c1=512, c2=512, num_heads=2, num_layers=4)
            self.aspp5 = TransformerBlock(c1=512, c2=512, num_heads=2, num_layers=4)

    def forward(self, x):
        x1 = self.inc(x)
        if self.context_layer:
            res_x1 = self.context_layer1_1(x1)
            x1 = x1 + res_x1    
        x2 = self.down1(x1)
        if self.context_layer:
            res_x2 = self.context_layer2_1(x2)
            x2 = x2 + res_x2
        x3 = self.down2(x2)

        if self.context_layer:
            res_x3 = self.context_layer3_1(x3)
            x3 = x3 + res_x3
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.transformer_block:
            x4 = self.aspp4(x4) + x4
            x5 = self.aspp5(x5) + x5
        return [x1, x2, x3, x4, x5]


class KernelSelectAttention(nn.Module):
    def __init__(self, channel=512, kernels=[3, 5, 7, 9], reduction=1, group=1, L=32):
        super(KernelSelectAttention, self).__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                    SyncBatchNorm(channel),
                    nn.ReLU()
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, coord_att=False):
        super(Down, self).__init__()
        self.coord_att = coord_att
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


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

#  多尺度融合 加卷积的下采样
class Conv_Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_Down, self).__init__()
        self.down_conv = nn.Sequential(
            # 使用卷积进行下采样而不是最大池化层 
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, stride=2, padding=0),
            MAC(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.down_conv(x)
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


class MACInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MACInConv, self).__init__()
        self.conv = MAC(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


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
        self.bn1 = SyncBatchNorm(mip)
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


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=0, groups=g, dilation=d, bias=False)
        self.bn = SyncBatchNorm(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            # 如果TransformerBlock，即ViT模块输入和输出通道不同，提前通过一个卷积层让通道相同
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, self.out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding https://github.com/iscyy/yoloair
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv
