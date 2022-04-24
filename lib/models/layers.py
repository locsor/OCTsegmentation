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

class Input(nn.Module):
    def __init__(self, in_channels, out_channels, robust):
        super(Input, self).__init__()
        self.robust = False
        print('Robust?', robust)
        self.out_channels = out_channels
    def forward(self, x):
        return [x, [], self.robust]

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):

    def __init__(self, in_channels: int, channels: int, stride1: int, stride2: int, padding: int, 
                 downsample: bool, use_relu: bool, residual: bool, scale: int, use_in: bool) -> None:
        super(BasicBlock, self).__init__()
        bn_mom = 0.1
        expansion = 1
        self.use_relu = use_relu
        self.residual = residual
        self.use_in = use_in

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size = 3, stride = stride1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=bn_mom)

        self.conv2 = nn.Conv2d(channels, channels*scale, kernel_size = 3, stride = stride2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(channels*scale, momentum=bn_mom)

        # self.instance_norm_layer = InstanceWhitening(channels * expansion)
        self.instance_norm_layer = nn.InstanceNorm2d(channels * expansion, affine=False)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels * expansion, kernel_size=1, stride=stride1, bias=False),
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
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual:
            out += residual

        if robust and self.use_in:
            out = self.instance_norm_layer(out)
            w = out
            w_arr.append(w)

        if self.use_relu:
            out = self.relu(out)
        
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

        # self.instance_norm_layer = InstanceWhitening(channels * expansion)
        self.instance_norm_layer = nn.InstanceNorm2d(channels * expansion, affine=False)
        self.relu = nn.ReLU(inplace=False)

        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels * expansion, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(channels * expansion, momentum=bn_mom))
        else:
            self.downsample = None


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
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual

        if robust and self.use_in:
            out = self.instance_norm_layer(out)
            w = out
            w_arr.append(w)
        if self.use_relu:
            out = self.relu(out)
        
        return [out, w_arr, robust]

class Layer(nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: int, stride: int,
                 bottleneck: bool, new_w: bool, use_in: bool) -> None:
        super(Layer, self).__init__()

        self.new_w = new_w
        self.layers = []
        self.downsample = False
        self.use_in = use_in

        in_channels = int(in_channels/2)
        channels = int(channels/2)
        expansion = 1

        if bottleneck:
            expansion = 2

        in_channels_expanded = channels * expansion

        self.cv1 = nn.Conv2d(in_channels, channels, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(in_channels, channels, 1, stride, bias=False)
        self.cv3 = nn.Conv2d(channels+in_channels_expanded, in_channels_expanded*2, 1, 1, bias=False)

        if stride != 1 or in_channels != channels * expansion:
            self.downsample = True

        if bottleneck:
            self.layers.append(Bottleneck(channels, channels, stride=stride, padding=0, downsample=self.downsample,
                                          use_relu=True, residual=True, use_in=self.use_in))

            for i in range(1, blocks):
                if i == (blocks-1):
                    self.layers.append(Bottleneck(in_channels_expanded, channels, stride=1, padding=0, downsample=None,
                                                  use_relu=False, residual=True, use_in=self.use_in))
                else:
                    self.layers.append(Bottleneck(in_channels_expanded, channels, stride=1, padding=0, downsample=None,
                                                  use_relu=True, residual=True, use_in=False))

        else:
            self.layers.append(BasicBlock(channels, channels, stride1=stride, stride2=1,
                                          padding=0, downsample=self.downsample, residual=True,
                                          use_relu=False, scale=1,
                                          use_in=False))

            for i in range(1, blocks):
                if i == (blocks-1):
                    self.layers.append(BasicBlock(in_channels_expanded, channels, stride1=1, stride2=1, padding=0,
                                                  downsample=None, use_relu=False, residual=True, scale=1,
                                                  use_in=self.use_in))
                else:
                    self.layers.append(BasicBlock(in_channels_expanded, channels, stride1=1, stride2=1, padding=0,
                                                  downsample=None, use_relu=True, residual=True, scale=1,
                                                  use_in=False))
        
        self.convs = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        robust = x[2]
        w_arr = x[1]
        x = x[0]
        #print('Layer in ', x.shape, self.downsample)
        if self.new_w:
            w_arr = []

        # y1 = self.convs([self.cv1(x), w_arr, robust])
        # w_arr = y1[1]
        # y1 = y1[0]
        # y2 =  self.cv2(x)

        # out = self.cv3(torch.cat((y1, y2), dim=1))

        x = torch.tensor_split(x, 2, dim=1)
        out = self.convs([self.cv1(x[0]), w_arr, robust])
        x_out = out[0]
        w_arr = out[1]
        out = self.cv3(torch.cat((x_out, self.cv2(x[1])), dim=1))
        #print('Layer out ', out.shape)
        return [out, w_arr, robust]


class Compression(nn.Module):
    def __init__(self, in_channels: int, channels: int, scale_factor: int) -> None:
        super(Compression, self).__init__()
        bn_mom=0.1
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels, momentum=bn_mom)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv(x[0]))
        height = int(out.shape[-2] * self.scale_factor)
        width = int(out.shape[-1] * self.scale_factor)

        # print('Compression', out.shape, self.scale_factor)
        return [F.interpolate(out, size=[height, width], mode='bilinear'), x[1], x[2]]

class Downsample(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super(Downsample, self).__init__()
        bn_mom = 0.1
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(channels, momentum=bn_mom)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return [self.bn(self.conv(x[0])), x[1], x[2]]

class DAPPM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, branch_channels: int) -> None:
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.scale0 = nn.Sequential(nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False))
        self.process1 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))
        self.process2 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))
        self.process3 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))
        self.process4 = nn.Sequential(nn.BatchNorm2d(branch_channels, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False))        
        self.compression = nn.Sequential(nn.BatchNorm2d(branch_channels * 5, momentum=bn_mom),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False))
        self.shortcut = nn.Sequential(nn.BatchNorm2d(in_channels, momentum=bn_mom),
                                      nn.ReLU(inplace=True),
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
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)

        return [F.interpolate(out, size=[80, 120], mode='bilinear'), w_arr, robust]

class SegmentHead(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, inter_channels: int, scale_factor: int) -> None:
        super(SegmentHead, self).__init__()
        bn_mom = 0.1
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        robust = x[2]
        w_arr = x[1]
        x = x[0]
        if w_arr == -1:
            w_arr = []
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor != -1:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

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
        #print('Sum', x[0][0].shape, x[1][0].shape)
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

    def __init__(self) -> None:
        super(Activation, self).__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return [self.relu(x[0]), x[1], x[2]]

class Covariance(nn.Module):
    def __init__(self):
        super(Covariance, self).__init__()

        self.eps = 1e-5
        self.cov_matrix_layer = []
#        self.in_channel_list = [64, 64, 128, 128, 64, 32, 32]
        self.in_channel_list = [32, 32, 64, 32, 16, 32]
        self.check = False

        for i in range(len(self.in_channel_list)):
            self.cov_matrix_layer.append(CovMatrix_ISW(dim=self.in_channel_list[i], relax_denom=0.0, clusters=3))

    def forward(self, x: List) -> torch.Tensor:
        x1 = x[0]
        x2 = x[1]
        wt_loss = torch.FloatTensor([0]).cuda()

        robust = x1[2]
        w_arr = x1[1]

        # for w in w_arr:
        #     print(w.shape)

        if robust == False:
            return [[x1[0], x2[0]], wt_loss, robust]
        
        if x1[0].shape[0] == 2:
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
            print('ISW')
            for index, f_map in enumerate(w_arr):
                eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                wt_loss = instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov)

        wt_loss /= len(w_arr)
        return [[x1[0], x2[0]], wt_loss, robust]

class ActivationGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self.in_channels[self.from_idx]

    @property
    def in_channel(self) -> int:
        return self.out_channel

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
        return self.in_channels[self.from_idx]

    @property
    def in_channel(self) -> int:
        return self.out_channel

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        return size

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(Input, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        module = Input(**self.kwargs)

        return self._get_module(module)
