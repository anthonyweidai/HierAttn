import math

from typing import Any, Optional, Callable

import torch
from torch import nn, Tensor
from timm.models.layers import drop_path

from lib.utils import pair


class BaseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int]=1,
        padding: Optional[int]=None,
        groups: Optional[int]=1,
        bias: Optional[bool]=None,
        BNorm: bool=False,
        # norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
        ActLayer: Optional[Callable[..., nn.Module]]=None,
        dilation: int=1,
        Momentum: Optional[float]=0.1,
        **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)
            
        if bias is None:
            bias = not BNorm
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        self.Conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)
        
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()
        
        if ActLayer:
            self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

    def profileModule(self, Input: Tensor):
        if Input.dim() != 4:
            print('Conv2d requires 4-dimensional Input (BxCxHxW). Provided Input has shape: {}'.format(Input.size()))

        BatchSize, in_channels, in_h, in_w = Input.size()
        assert in_channels == self.in_channels, '{}!={}'.format(in_channels, self.in_channels)

        k_h, k_w = pair(self.kernel_size)
        stride_h, stride_w = pair(self.stride)
        pad_h, pad_w = pair(self.padding)
        groups = self.groups

        out_h = (in_h - k_h + 2 * pad_h) // stride_h + 1
        out_w = (in_w - k_w + 2 * pad_w) // stride_w + 1

        # compute MACs
        MACs = (k_h * k_w) * (in_channels * self.out_channels) * (out_h * out_w) * 1.0
        MACs /= groups

        if self.bias:
            MACs += self.out_channels * out_h * out_w

        # compute parameters
        Params = sum([p.numel() for p in self.parameters()])

        Output = torch.zeros(size=(BatchSize, self.out_channels, out_h, out_w), dtype=Input.dtype, device=Input.device)
        # print(MACs)
        return Output, Params, MACs


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Input = self.forward(Input)
        return Input, 0.0, 0.0    


class Globalpooling(nn.Module):
    def __init__(self, PoolType='mean', KeepDim=False):
        super(Globalpooling, self).__init__()
        self.PoolType = PoolType
        self.KeepDim = KeepDim
        
    def globalPool(self, x):
        assert x.dim() == 4, "Got: {}".format(x.shape)
        if self.PoolType == 'rms':
            x = x ** 2
            x = torch.mean(x, dim=[-2, -1], keepdim=self.KeepDim)
            x = x ** -0.5
        elif self.PoolType == 'abs':
            x = torch.mean(x, dim=[-2, -1], keepdim=self.KeepDim)
        else:
            # same as AdaptiveAvgPool
            x = torch.mean(torch.abs(x), dim=[-2, -1], keepdim=self.KeepDim)# use default method "mean"
        
        return x
        
    def forward(self, x: Tensor) -> Tensor:
        return self.globalPool(x)
    
    def profileModule(self, Input: Tensor):
        Input = self.forward(Input)
        return Input, 0.0, 0.0
    
    
class Linearlayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: Optional[bool]=True,
                 **kwargs
                 ) -> None:
        """
            Applies a linear transformation to the Input data

            :param in_features: size of each Input sample
            :param out_features:  size of each output sample
            :param bias: Add bias (learnable) or not
        """
        super(Linearlayer, self).__init__()
        self.weight = nn.Parameter(Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x

    def profileModule(self, Input: Tensor):
        out_size = list(Input.shape)
        out_size[-1] = self.out_features
        Params = sum([p.numel() for p in self.parameters()])
        MACs = Params
        output = torch.zeros(size=out_size, dtype=Input.dtype, device=Input.device)
        return output, Params, MACs
    

class MyLayernorm(nn.Module):
    # 3.1 Layernorm(LN) is applied before every block
    def __init__(self, DimEmb, Fn=None):
        super(MyLayernorm, self).__init__()
        self.LayerNorm = nn.LayerNorm(DimEmb)
        self.Fn = Fn
        
    def forward(self, x, **kwargs):
        x = self.LayerNorm(x)
        if self.Fn:
            x = self.Fn(x, **kwargs)
        return x
    
    def profileModule(self, Input: Tensor):
        # Since normalization layers can be fused, we do not count their operations
        Params = sum([p.numel() for p in self.parameters()])
        return Input, Params, 0.0
    

class StochasticDepth(nn.Module):
    def __init__(self, DProb: float) -> None:
        super().__init__()
        self.DProb = DProb

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.DProb, self.training)
    
    def profileModule(self, Input: Tensor):
        _, in_channels, in_h, in_w = Input.size()
        MACs = in_channels * in_h * in_w # one multiplication for each element
        return Input, 0.0, MACs
    
    
class Dropout(nn.Dropout):
    def __init__(self, p: float=0.5, inplace: bool=False):
        super(Dropout, self).__init__(p=p, inplace=inplace)

    def profileModule(self, Input: Tensor):
        Input = self.forward(Input)
        return Input, 0.0, 0.0


def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    B, C, H, W = Feature.shape
    if Mode == 1:
        Feature = Feature.flatten(2)
        Feature = Feature[:, :, torch.randperm(Feature.shape[-1])]
        Feature = Feature.reshape(B, C, H, W)
    else:
        Feature = Feature[:, :, torch.randperm(H)]
        Feature = Feature[:, :, :, torch.randperm(W)]
    return Feature


def initWeight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)