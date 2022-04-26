import torch
import torch.nn as nn
import torch.nn.functional as F
import kindle
from kindle import Model
from kindle.utils.torch_utils import count_model_params
from kindle.generator import GeneratorAbstract
import numpy as np
from typing import Any, Dict, List, Union, Tuple
from models.instance_whitening import InstanceWhitening, SyncSwitchWhiten2d, instance_whitening_loss, get_covariance_matrix
from models.cov_settings import CovMatrix_ISW, CovMatrix_IRW

class Acon(nn.Module):
    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(width, max(r, width // r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(max(r, width // r))
        self.fc2 = nn.Conv2d(max(r, width // r), width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width)

        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))

    def forward(self, x):
        beta = torch.sigmoid(
            self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, channels, stride, padding, output_padding, groups=1):
        if isinstance(stride, int):
            stride = (stride, stride)

        kernel_size = (2 * stride[0] - 1, 2 * stride[1] - 1)
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False)

    def reset_parameters(self):
        # nn.init.constant(self.bias, 0)
        nn.init.constant(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.stride)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(stride):
        num_dims = len(stride)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        for channel in range(num_dims):
            channel_stride = stride[channel]
            kernel_size = 2 * channel_stride - 1
            delta = torch.arange(1 - channel_stride, channel_stride)
            channel_filter = (1 - torch.abs(delta / channel_stride))
            shape = [1] * num_dims
            shape[channel] = kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel

class Encoding(nn.Module):
    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels)**0.5)
        self.codewords = nn.Parameter(
            torch.empty(num_codes, channels,
                        dtype=torch.float).uniform_(-std, std),
            requires_grad=True)
        self.scale = nn.Parameter(
            torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0),
            requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    def _drop(self):
        if self.training:
            self.scale.data.uniform_(-1, 0)
        else:
            self.scale.data.zero_().add_(-0.5)

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)

        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        batch_size = x.size(0)
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()

        # self._drop()
        assignment_weights = F.softmax(
            self.scaled_l2(x, self.codewords, self.scale), dim=2)
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        # self._drop()

        return encoded_feat

class EncModule(nn.Module):
    def __init__(self, in_channels, num_codes):#, conv_cfg, norm_cfg, act_cfg):
        super(EncModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm1d(num_codes)
        self.linear = nn.Linear(in_channels, in_channels)
        self.linearForSELoss = nn.Linear(in_channels, 2)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.Mish(inplace=True)
        self.encoding = Encoding(in_channels, num_codes)

    def forward(self, x):
        enc = self.conv1(x)
        enc = self.norm1(enc)
        enc = self.act(enc)
        enc = self.encoding(enc)
        enc = self.norm2(enc)
        enc = self.act(enc)
        enc = enc.mean(1)

        b, c, _, _ = x.size()
        gamma = self.sigmoid(self.linear(enc))
        y = gamma.view(b, c, 1, 1)
        outputs = [F.mish(x + x * y, inplace=True)]#, inplace=True)]
        outputs.append(self.linearForSELoss(enc))

        return outputs

class Input(nn.Module):
    def __init__(self, in_channels, out_channels, robust):
        super(Input, self).__init__()
        bn_mom = 0.1
        self.robust = robust
        print('Robust?', robust, in_channels, out_channels)
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(3, out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)

        if robust:
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=False, momentum=bn_mom)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=False, momentum=bn_mom)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels, momentum=bn_mom)
            self.norm2 = nn.BatchNorm2d(out_channels, momentum=bn_mom)

        self.act = nn.Mish(inplace=True)

    def forward(self, x):
        w_arr = []

        out = self.conv1(x)
        out = self.norm1(out)
        w = out
        w_arr.append(w)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        w = out
        w_arr.append(w)
        out = self.act(out)

        if self.robust == False:
            w_arr = []
        
        return [out, w_arr, self.robust]

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias = False, use_act = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act = nn.Mish(inplace=True)
        self.use_act = use_act

    def forward(self, x):
        if self.use_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))

class BasicBlock(nn.Module):

    def __init__(self, in_channels: int, channels: int, stride: int, padding: int, scale: int, k_size: int,
                 downsample: bool, use_relu: bool, residual: bool, use_in: bool) -> None:
        super(BasicBlock, self).__init__()
        bn_mom = 0.1
        expansion = 1
        padding = 1
        if k_size == 5:
            padding += 1
        self.use_relu = use_relu
        self.residual = residual
        self.use_in = use_in

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size = k_size, stride = stride, padding = padding, bias = False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=bn_mom)

        self.conv2 = nn.Conv2d(channels, channels*scale, kernel_size = k_size, stride = 1, padding = padding, bias = False)
        self.bn2 = nn.BatchNorm2d(channels*scale, momentum=bn_mom)

        self.instance_norm_layer = nn.InstanceNorm2d(channels * scale, affine=False)
        self.act = nn.Mish(inplace=True)

        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels * expansion, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(channels * expansion, momentum=bn_mom))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        robust = x[2]
        w_arr = x[1]
        x = x[0]

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual:
            out += residual

        if robust and self.use_in:
            out = self.instance_norm_layer(out)
            w = out
            w_arr.append(w)

        if self.use_relu:
            out = self.act(out)
        
        return [out, w_arr, robust]

class Bottleneck(nn.Module):

    def __init__(self, in_channels: int, channels: int, stride: int, padding: int,
                 downsample: bool, use_relu: bool, residual: bool, use_in: bool) -> None:
        super(Bottleneck, self).__init__()
        bn_mom=0.1
        expansion = 2
        self.use_relu = use_relu
        self.use_in = use_in

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=bn_mom)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=bn_mom)

        self.conv3 = nn.Conv2d(channels, channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * expansion, momentum=bn_mom)

        self.instance_norm_layer = nn.InstanceNorm2d(channels * expansion, affine=False)
        self.act = nn.Mish(inplace=True)

        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels * expansion, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(channels * expansion, momentum=bn_mom))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        robust = x[2]
        w_arr = x[1]
        x = x[0]

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual

        if robust and self.use_in:
            out = self.instance_norm_layer(out)
            w = out
            w_arr.append(w)

        if self.use_relu:
            out = self.act(out)
        
        return [out, w_arr, robust]

class Layer(nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: int, stride: int, k_size: int,
                 bottleneck: bool, new_w: bool, use_in: bool) -> None:
        super(Layer, self).__init__()

        self.new_w = new_w
        self.layers = []
        self.downsample = False
        self.use_in = use_in

        if self.use_in == False:
            in_channels = int(in_channels/2)
            channels = int(channels/2)
        expansion = 1
        if bottleneck:
            expansion = 2
        self.expansion = expansion

        channels_expanded = channels * expansion

        self.cv1 = Conv(in_channels, channels_expanded, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels_expanded*2, momentum=0.1)
        self.act = nn.Mish(inplace=True)
        self.cv2 = nn.Conv2d(in_channels, channels_expanded, 1, stride)
        self.cv3 = nn.Conv2d(channels_expanded, channels_expanded, 1, 1)
        self.cv4 = Conv(channels_expanded*2, channels_expanded*2, 1, 1, use_act=False)

        if stride != 1 or in_channels != channels * expansion:
            self.downsample = True

        if bottleneck:
            self.layers.append(Bottleneck(channels_expanded, channels, stride=stride, padding=0, downsample=self.downsample,
                                          use_relu=True, residual=True, use_in=use_in))

            for i in range(1, blocks):
                if i == (blocks-1):
                    self.layers.append(Bottleneck(channels_expanded, channels, stride=1, padding=0, downsample=None,
                                                  use_relu=False, residual=True, use_in=use_in))
                else:
                    self.layers.append(Bottleneck(channels_expanded, channels, stride=1, padding=0, downsample=None,
                                                  use_relu=True, residual=True, use_in=use_in))

        else:
            self.layers.append(BasicBlock(channels_expanded, channels, 
                                          stride=stride, padding=0, scale=1, k_size=k_size, 
                                          downsample=self.downsample, residual=True, use_relu=False, use_in=use_in))

            for i in range(1, blocks):
                if i == (blocks-1):
                    self.layers.append(BasicBlock(channels_expanded, channels,
                                                  stride=1, padding=0, scale=1, k_size=3,
                                                  downsample=None, use_relu=False, residual=True, use_in=use_in))
                else:
                    self.layers.append(BasicBlock(channels_expanded, channels,
                                                  stride=1, padding=0, scale=1, k_size=3,
                                                  downsample=None, use_relu=True, residual=True, use_in=use_in))
        
        self.convs = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        robust = x[2]
        w_arr = x[1]
        x = x[0]
        if self.new_w:
            w_arr = []

        # out = self.convs([x, w_arr, robust])
        if self.use_in == False:
            x = torch.tensor_split(x, 2, dim=1)
            x1 = self.cv1(x[0])
            x1 = self.convs([x1, w_arr, robust])
            w_arr = x1[1]
            x1 = self.cv3(x1[0])

            x2 = self.cv2(x[1])

            out = self.cv4(self.act(self.bn1(torch.cat((x1, x2), dim=1))))
        else:
            x = self.convs([x, w_arr, robust])
            out = x[0]
            w_arr = x[1]

        return [out, w_arr, robust]


class Compression(nn.Module):
    def __init__(self, in_channels: int, channels: int, scale_factor: int) -> None:
        super(Compression, self).__init__()
        bn_mom=0.1
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        if scale_factor == 4:
            padding = 2
            output_padding = 1
        elif scale_factor == 2:
            padding = 1
            output_padding = 1

        self.bn1 = nn.BatchNorm2d(channels, momentum=bn_mom)
        self.act = nn.Mish(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv(x[0]))
        height = int(out.shape[-2] * self.scale_factor)
        width = int(out.shape[-1] * self.scale_factor)

        return [F.interpolate(out, size=[height, width], mode='bilinear', align_corners=False), x[1], x[2]]

class Downsample(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int, stride: int, padding: int, extra_layers: bool) -> None:
        super(Downsample, self).__init__()
        bn_mom = 0.1
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(channels, momentum=bn_mom)

        self.extra_layers = extra_layers
        if extra_layers:
            self.act = nn.Mish(inplace=True)
            self.conv_extra = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn_extra = nn.BatchNorm2d(channels * 2, momentum=bn_mom)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv(x[0]))
        if self.extra_layers:
            out = self.act(out)
            out = self.bn_extra(self.conv_extra(out))
        return [out, x[1], x[2]]

class DAPPM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, branch_channels: int, scale: int) -> None:
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale = scale
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.Mish(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.Mish(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.Mish(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.Mish(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale0 = nn.Sequential(nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.Mish(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.process1 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.Mish(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))
        self.process2 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.Mish(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))
        self.process3 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.Mish(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))
        self.process4 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.Mish(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))        
        self.compression = nn.Sequential(nn.BatchNorm2d(branch_channels * 5, momentum=bn_mom),
                                         nn.Mish(inplace=True),
                                         nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False))
        self.shortcut = nn.Sequential(nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                      nn.Mish(inplace=True),
                                      nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        robust = x[2]
        w_arr = x[1]
        x = x[0]

        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        shape = out.shape
        w = shape[2]*self.scale
        h = shape[3]*self.scale

        return [F.interpolate(out, size=[w,h], mode='bilinear', align_corners=False), w_arr, robust]

class SegmentHead(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, inter_channels: int, scale_factor: int, calc_enc: bool) -> None:
        super(SegmentHead, self).__init__()
        bn_mom = 0.1
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.Mish(inplace=True)
        self.scale_factor = scale_factor
        self.calc_enc = calc_enc

        if calc_enc:
            head_size = 32
            self.bn2 = nn.BatchNorm2d(head_size, momentum=bn_mom)
            self.conv1 = nn.Conv2d(in_channels, head_size, kernel_size=3, stride=1, padding=1, bias=False)

            self.enc = EncModule(head_size, 32)

            self.dropout = nn.Dropout(0.1, False)
            self.conv6 = nn.Conv2d(head_size, out_channels, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        robust = x[2]
        w_arr = x[1]
        x = x[0]
        if w_arr == -1:
            w_arr = []

        out = self.conv1(self.act(self.bn1(x)))

        if self.calc_enc:
            out = self.act(self.bn2(out))

            out = self.enc(out)
            out[0] = self.conv6(self.dropout(out[0]))

        if self.scale_factor != -1:
            if isinstance(out, list):
                height = out[0].shape[-2] * self.scale_factor
                width = out[0].shape[-1] * self.scale_factor
                out[0] = F.interpolate(out[0],
                                    size=[height, width],
                                    mode='bilinear', align_corners=False)

            else:
                height = out.shape[-2] * self.scale_factor
                width = out.shape[-1] * self.scale_factor
                out = F.interpolate(out,
                                    size=[height, width],
                                    mode='bilinear', align_corners=False)

        return [out, w_arr, robust]

class Sum(nn.Module):

    def __init__(self, keep: bool, append: bool) -> None:
        super(Sum, self).__init__()
        self.keep = keep
        self.append = append

    def forward(
        self, x: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
    ) -> torch.Tensor:

        robust = x[0][2]
        out = x[0][0]

        out = out + x[1][0]

        if self.keep:
            w_arr = x[0][1]
            if self.append:
                w_arr += x[1][1]
        else:
            w_arr = x[1][1]
            if self.append:
                w_arr += x[0][1]
        return [out, w_arr, robust]

class Activation(nn.Module):

    def __init__(self, channels: int) -> None:
        super(Activation, self).__init__()
        self.act = nn.Mish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return [self.act(x[0]), x[1], x[2]]

class Covariance(nn.Module):
    def __init__(self):
        super(Covariance, self).__init__()

        self.eps = 1e-5
        self.cov_matrix_layer = []
        self.in_channel_list = [64, 64, 64, 64, 64, 64]
        self.check = False

        for i in range(len(self.in_channel_list)):
            self.cov_matrix_layer.append(CovMatrix_ISW(dim=self.in_channel_list[i], relax_denom=0.0, clusters=3))

    def forward(self, x: List) -> torch.Tensor:
        x1 = x[0]
        x2 = x[1]
        wt_loss = torch.tensor([0.0]).cuda()

        robust = x1[2]
        w_arr = x1[1]

        # for w in w_arr: #for testing robust
            # print(w.shape)

        if robust == False:
            return [[x1[0][0], x2[0], x1[0][1]], wt_loss, robust]
        
        if x1[0][0].shape[0] == 2:
            for index, f_map in enumerate(w_arr):
                B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
                HW = H * W
                f_map_temp = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
                eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
                f_cor = torch.bmm(f_map_temp, f_map_temp.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
                off_diag_elements = f_cor * reverse_eye
                self.cov_matrix_layer[index].set_variance_of_covariance(torch.var(off_diag_elements, dim=0))

            self.check = True
            eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[-1].get_mask_matrix()
            return 0
        

        if self.check:
            for index, f_map in enumerate(w_arr):
                eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                wt_loss = instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov)

        wt_loss /= len(w_arr)
        return [[x1[0][0], x2[0], x1[0][1]], wt_loss, robust]

class ActivationGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.in_channels[self.from_idx]

    @property
    def in_channel(self) -> int:
        # return self.out_channel
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return size

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(Activation, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [Activation(**self.kwargs) for _ in range(repeat)]
        else:
            module = Activation(**self.kwargs)

        return self._get_module(module)



class BasicBlockGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(BasicBlock, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [BasicBlock(**self.kwargs) for _ in range(repeat)]
        else:
            module = BasicBlock(**self.kwargs)

        return self._get_module(module)

class LayerGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(Layer, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [Layer(**self.kwargs) for _ in range(repeat)]
        else:
            module = Layer(**self.kwargs)

        return self._get_module(module)

class DownsampleGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(Downsample, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [Downsample(**self.kwargs) for _ in range(repeat)]
        else:
            module = Downsample(**self.kwargs)

        return self._get_module(module)

class CompressionGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(Compression, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [Compression(**self.kwargs) for _ in range(repeat)]
        else:
            module = Compression(**self.kwargs)

        return self._get_module(module)

class DAPPMGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]*2

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(DAPPM, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [DAPPM(**self.kwargs) for _ in range(repeat)]
        else:
            module = DAPPM(**self.kwargs)

        return self._get_module(module)

class SegmentHeadGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.args[1]

    @property
    def in_channel(self) -> int:
        #if isinstance(self.from_idx, list):
        #    raise Exception("from_idx can not be a list.")
        #return self.in_channels[self.from_idx]
        return self.args[0]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[2:]]
        kwargs = self._get_kwargs(SegmentHead, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [SegmentHead(**self.kwargs) for _ in range(repeat)]
        else:
            module = SegmentHead(**self.kwargs)

        return self._get_module(module)

class SumGenerator(GeneratorAbstract):
    """Add module generator."""

    def __init__(self, *args, **kwargs) -> None:
        """Add module generator."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        if isinstance(self.from_idx, int):
            raise Exception("Add must have more than 2 inputs.")

        return self.in_channels[self.from_idx[0]]

    @property
    def in_channel(self) -> int:
        return self.out_channel

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._get_kwargs(Sum, self.args)

    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return list(size[0])

    def __call__(self, repeat: int = 1) -> nn.Module:
        module = Sum(**self.kwargs)

        return self._get_module(module)

class CovarianceGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        # if isinstance(self.from_idx, int):
            # raise Exception("Add must have more than 2 inputs.")

        return self.in_channels[self.from_idx[0]]

    @property
    def in_channel(self) -> int:
        return self.out_channel

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return size

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(Covariance, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [Covariance(**self.kwargs) for _ in range(repeat)]
        else:
            module = Covariance(**self.kwargs)

        return self._get_module(module)

class InputGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.args[1]

    @property
    def in_channel(self) -> int:
        return self.args[0]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return size

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[2:]]
        kwargs = self._get_kwargs(Input, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        module = Input(**self.kwargs)

        return self._get_module(module)
