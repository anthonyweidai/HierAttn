import math
import numpy as np
from einops import rearrange
from typing import Any, List, Dict, Tuple, Callable, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .baseBlock import BaseConv2d, StochasticDepth, AdaptiveAvgPool2d, \
    Linearlayer, MyLayernorm, Dropout, shuffleTensor
from .baseBlock import BaseConv2d, Linearlayer, MyLayernorm, StochasticDepth, Dropout
from ..misc import moduleProfile
from ...utils import makeDivisible, pair


class HierAvgPooling(nn.Module):
    def __init__(self, BranchPool) -> None:
        assert len(BranchPool) == 3, "Got {} branches".format(len(BranchPool))
        super().__init__()
        
        self.AdtPooling1 = AdaptiveAvgPool2d(BranchPool[0])
        self.AdtPooling2 = AdaptiveAvgPool2d(BranchPool[1])
        self.AdtPooling3 = AdaptiveAvgPool2d(BranchPool[2])

    def forward(self, x: Tensor) -> Tensor:
        return self.AdtPooling1(x), self.AdtPooling2(x), self.AdtPooling3(x)
    

class BranchFolding(nn.Module):
    def __init__(self, BranchPool, OutChannels, MCRandom) -> None:
        super().__init__()
        self.OutChannel = np.sum(OutChannels)
        self.MCRandom = MCRandom
        self.OutputHW = int(math.sqrt((sum([ i ** 2 for i in BranchPool]) + BranchPool[0] ** 2)))
        
        self.HierPooling = HierAvgPooling(BranchPool)
        
    def branchEnsemble(self, InB1: Tensor, InB2: Tensor, InB3: Tensor) -> Tensor:
        x11, x12, x13 = self.HierPooling(InB1)
        x21, x22, x23 = self.HierPooling(InB2)
        x31, x32, x33 = self.HierPooling(InB3)
        
        HierPool1 = torch.cat((x11, x21, x31), dim=1) # [B, C, H, W] -> [B, C', H, W]
        HierPool2 = torch.cat((x12, x22, x32), dim=1)
        HierPool3 = torch.cat((x13, x23, x33), dim=1)
        
        HierPool1 = HierPool1.flatten(2) # [B, C', H, W] -> [B, C', HW]
        HierPool2 = HierPool2.flatten(2)
        HierPool3 = HierPool3.flatten(2)
        
        HierPool = torch.cat((HierPool1, HierPool1, HierPool2, HierPool3), dim=2) # [B, C', HW] -> [B, C', H'W']
        FeatureMap = HierPool.reshape(x11.shape[0], self.OutChannel, self.OutputHW, self.OutputHW) # [B, C', H'W'] -> [B, C', H', W']
        
        if self.MCRandom and self.training:
            FeatureMap = shuffleTensor(FeatureMap, Mode=1)
        return FeatureMap
        
    def forward(self, InB1: Tensor, InB2: Tensor, InB3: Tensor) -> Tensor:
        return self.branchEnsemble(InB1, InB2, InB3)
        
    def profileModule(self, InB1: Tensor, InB2: Tensor, InB3: Tensor):
        return self.branchEnsemble(InB1, InB2, InB3), 0.0, 0.0
    
    
class SameChannelAttn(nn.Module):
    def __init__(
        self,
        ScaleAct: Callable[..., nn.Module]=nn.Sigmoid,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.Avgpool = AdaptiveAvgPool2d(1)
        self.ScaleAct = ScaleAct()

    def forward(self, x: Tensor) -> Tensor:
        Feature = self.Avgpool(x)
        return x * self.ScaleAct(Feature) # attention mechanisms, use global info
    
    def profileModule(self, Input: Tensor):
        return Input, 0.0, 0.0
    
    
class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self,
                 ExpRatio: float, Kernel: int, Stride: int,
                 InChannels: int, OutChannels: int, NumLayers: int, 
                 UseSPA: Optional[bool]=True,
                 WidthMult: float=1.0, 
                 DepthMult: float=1.0) -> None:
        self.ExpRatio = ExpRatio
        self.Kernel = Kernel
        self.Stride = Stride
        self.InChannels = self.adjust_channels(InChannels, WidthMult)
        self.OutChannels = self.adjust_channels(OutChannels, WidthMult)
        self.NumLayers = self.adjust_depth(NumLayers, DepthMult)
        self.UseSPA = UseSPA

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'ExpRatio={ExpRatio}'
        s += ', Kernel={Kernel}'
        s += ', Stride={Stride}'
        s += ', InChannels={InChannels}'
        s += ', OutChannels={OutChannels}'
        s += ', NumLayers={NumLayers}'
        s += ')'
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, WidthMult: float, min_value: Optional[int] = None) -> int:
        return makeDivisible(channels * WidthMult, 8, min_value)

    @staticmethod
    def adjust_depth(NumLayers: int, DepthMult: float):
        return int(math.ceil(NumLayers * DepthMult))


class MBConv(nn.Module):
    # Mobile inverted bottleneck MBConv, in section 4
    def __init__(self, Cnf: MBConvConfig, SDProb: float=0., 
                #  norm_layer: Optional[Callable[..., nn.Module]]=None,
                SELayer: Callable[..., nn.Module]=SameChannelAttn,
                SEAct: Callable[..., nn.Module]=nn.SiLU,
                SamePool: Optional[int]=1,
                **kwargs: Any) -> None:
        super().__init__()
        if not (1 <= Cnf.Stride <= 2):
            raise ValueError('illegal stride value')
        
        InChannels = Cnf.InChannels
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d

        self.ResConnect = Cnf.Stride == 1 and InChannels == Cnf.OutChannels

        Layers: List[nn.Module] = []
            
        # expand
        ExpChannels = Cnf.adjust_channels(InChannels, Cnf.ExpRatio)
        if ExpChannels != InChannels:
            Layers.append(BaseConv2d(InChannels, ExpChannels, 1, BNorm=True, ActLayer=SEAct))

        # depthwise
        Layers.append(BaseConv2d(ExpChannels, ExpChannels, Cnf.Kernel, Cnf.Stride, 
                                 groups=ExpChannels, BNorm=True, ActLayer=SEAct))

        # same channel attention
        if Cnf.UseSPA:
            Layers.append(SELayer(SamePool=SamePool, **kwargs))
                
        # project
        Layers.append(BaseConv2d(ExpChannels, Cnf.OutChannels, 1, BNorm=True))
        self.Block = nn.Sequential(*Layers)
            
        self.Dropout = StochasticDepth(SDProb)
        self.OutChannels = Cnf.OutChannels

    def forward(self, x: Tensor) -> Tensor:
        Result = self.Block(x)
        
        if self.ResConnect:
            Result = self.Dropout(Result)
            Result += x
        return Result
    
    def profileModule(self, Input: Tensor):
        Params = MACs = 0.0
        
        Output, ParamsIB, MACsIB = moduleProfile(module=self.Block, x=Input)
        Params += ParamsIB
        MACs += MACsIB
        
        Output, ParamsSD, MACsSD = moduleProfile(module=self.Dropout, x=Output)
        Params += ParamsSD
        MACs += MACsSD
        return Output, Params, MACs
    
    

class MLP(nn.Module):
    # 3.1 The MLP contains two layers with a GELU non-linearity 
    def __init__(self, DimEmb, MLPSize, DropRate1=0., DropRate2=0., bias=False):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            Linearlayer(DimEmb, MLPSize, bias),
            nn.GELU(),
            Dropout(DropRate1),
            Linearlayer(MLPSize, DimEmb, bias),
            Dropout(DropRate2),
        )
        
    def forward(self, x):
        return self.MLP(x)
    
    def profileModule(self, Input: Tensor):
        return moduleProfile(module=self.MLP, x=Input)
        

class Attention(nn.Module):
    # Multi-head Self-attention mechanism
    def __init__(self, DimEmb, NumHead=8, DimHead=64, DropRate=0., bias=False):
        super(Attention, self).__init__()
        DimModel = NumHead * DimHead # d_k = d_v = d_model/h = 64 (by default)
        
        self.NumHead = NumHead
        self.Scale = DimHead ** -0.5 # 1 / sqrt(d_k)
        
        self.Softmax = nn.Softmax(dim=-1)
        self.toQKV = Linearlayer(DimEmb, DimModel * 3, bias)
        
        self.toOut = nn.Sequential(
            Linearlayer(DimModel, DimEmb, bias),
            Dropout(DropRate)
        ) if not (NumHead == 1 and DimHead == DimEmb) else nn.Identity()
        
    def forward(self, x):
        qkv = self.toQKV(x).chunk(3, dim=-1)
        Query, Key, Value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.NumHead), qkv)
        
        x = torch.matmul(Query, Key.transpose(-1, -2)) * self.Scale # scaled dot product
        x = self.Softmax(x)
        
        x = torch.matmul(x, Value)
        x = rearrange(x, 'b h n d -> b n (h d)')
        return self.toOut(x)
    
    def profileModule(self, Input):
        BatchSize, SeqLen, InChannels = Input.shape
        Params = MACs = 0.0

        _, p, m = moduleProfile(module=self.toQKV, x=Input)
        Params += p
        MACs += (m * SeqLen * BatchSize)

        # number of operations in QK^T
        m_qk = (SeqLen * InChannels * InChannels) * BatchSize
        MACs += m_qk

        # number of operations in computing weighted sum
        m_wt = (SeqLen * InChannels * InChannels) * BatchSize
        MACs += m_wt

        _, p, m = moduleProfile(module=self.toOut, x=Input)
        Params += p
        MACs += (m * SeqLen * BatchSize)

        return Input, Params, MACs
    
    
class Transformer(nn.Module):
    def __init__(self, DimEmb, Depth, NumHead, DimHead, MLPSize, AttDropRate=0., MLPDropRate1=0., MLPDropRate2=0., bias=False):
        super(Transformer, self).__init__()
        self.Depth = Depth
        
        self.Layers = nn.ModuleList([])
        for _ in range(Depth):
            self.Layers.append(nn.ModuleList([
                nn.Sequential(MyLayernorm(DimEmb),
                              Attention(DimEmb, NumHead, DimHead, AttDropRate, bias)),
                nn.Sequential(MyLayernorm(DimEmb),
                              MLP(DimEmb, MLPSize, MLPDropRate1, MLPDropRate2, bias))
            ])) # Trainable
            
    def forward(self, x):
        # 3.1 Residual connections after every block
        for AttBlock, MLPBlock in self.Layers:
            x = AttBlock(x) + x # Eq(2)
            x = MLPBlock(x) + x # Eq(3)
        return x
    
    def profileModule(self, Input: Tensor):
        MACs, Params = 0, 0
        BatchSize, SeqLen = Input.shape[:2]

        for AttBlock, MLPBlock in self.Layers:
            ## The input shape doesn't change from each block
            _, p_mha, m_mha = moduleProfile(module=AttBlock, x=Input)

            _, p_ffn, m_ffn = moduleProfile(module=MLPBlock, x=Input)
            
            m_ffn = (m_ffn * BatchSize * SeqLen)
            
            MACs += m_mha + m_ffn
            Params += p_mha + p_ffn

        return Input, Params, MACs
    

class ConvViTBlock(nn.Module):
    """
        Inspired by MobileViT block: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self, InChannel: int, PatchRes: int, DimEmb: int, MLPSize: int,
                 Depth: int=2, DimHead: int=32,
                 AttDropRate: Optional[float]=0., MLPDropRate1: float=0., MLPDropRate2: float=0.1,
                 ConvKSize: int=3, NoFusion: Optional[bool]=False, 
                 FusionRes: bool=False, SDProb: float=0, 
                 **kwargs: Any):
        self.FusionRes = FusionRes
        self.HPatch, self.WPatch = pair(PatchRes)
        self.PatchArea = self.HPatch * self.WPatch
        
        # Padding = int((ConvKSize - 1) / 2)
        Conv3X3In = BaseConv2d(InChannel, InChannel, ConvKSize, 1, BNorm=True, ActLayer=nn.SiLU) # Standard conv
        Conv1x1In = BaseConv2d(InChannel, DimEmb, 1, 1, BNorm=False, bias=False) # Point-wise conv
        Conv1X1Out = BaseConv2d(DimEmb, InChannel, 1, 1, BNorm=True, ActLayer=nn.SiLU) # Point-wise conv
        Conv3X3Out = BaseConv2d(2 * InChannel, InChannel, ConvKSize, 1, BNorm=True, ActLayer=nn.SiLU) if not NoFusion else nn.Identity()  # Standard conv
        
        super(ConvViTBlock, self).__init__()
        self.LocalRep = nn.Sequential() # 3.1 Local representation
        self.LocalRep.add_module(name='conv_3x3', module=Conv3X3In) # Encode local spatial information
        self.LocalRep.add_module(name='conv_1x1', module=Conv1x1In) # Porject tensor to a high dimensional space
        
        assert DimEmb % DimHead == 0
        NumHead = DimEmb // DimHead
        
        GlobalRep = [Transformer(DimEmb, Depth, NumHead, DimHead, MLPSize, 
                                    AttDropRate, MLPDropRate1, MLPDropRate2, bias=True)]
        
        GlobalRep.append(MyLayernorm(DimEmb))
        self.GlobalRep = nn.Sequential(*GlobalRep) # Encode global spatial information
        
        self.ConvProj = Conv1X1Out # Project folded tensor to C dimension
        self.Fusion = Conv3X3Out # Fuse local and global features

        self.Dropout = StochasticDepth(SDProb) if FusionRes else nn.Identity()
        
    def unfolding(self, FeatureMap: Tensor) -> Tuple[Tensor, Dict]:
        BatchSize, InChannel, HOrig, WOrig = FeatureMap.shape 
        
        # H and W should be multiple by patch dimensions
        Hnew = int(math.ceil(HOrig/ self.HPatch) * self.HPatch)
        WNew = int(math.ceil(WOrig/ self.WPatch) * self.WPatch)
        
        Interpolate = False # Bilinear interpolation
        if Hnew != HOrig or WNew != WOrig:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            # Resize the feature map.
            FeatureMap = F.interpolate(FeatureMap, size=(Hnew, WNew), mode="bilinear", align_corners=False)
            Interpolate = True
            
        # number of patches along width and height
        NumWPatch = WNew // self.WPatch # n_w
        NumHpatch = Hnew // self.HPatch # n_h
        NumPatches = NumHpatch * NumWPatch # N

        ## Keep the right output tensor dimension order
         # C is channel of tensor. In Figure, C means d
         # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        ReshapedFm = FeatureMap.reshape(BatchSize * InChannel * NumHpatch, self.HPatch, NumWPatch, self.WPatch)
         # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        TransposedFm = ReshapedFm.transpose(1, 2)
         # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        ReshapedFm = TransposedFm.reshape(BatchSize, InChannel, NumPatches, self.PatchArea)
         # [B, C, N, P] --> [B, P, N, C]
        TransposedFm = ReshapedFm.transpose(1, 3)
         # [B, P, N, C] --> [BP, N, C]
        Patches = TransposedFm.reshape(BatchSize * self.PatchArea, NumPatches, -1)

        InfoDict = {
            "orig_size": (HOrig, WOrig),
            "batch_size": BatchSize,
            "interpolate": Interpolate,
            "total_patches": NumPatches,
            "num_patches_w": NumWPatch,
            "num_patches_h": NumHpatch
        }
        
        return Patches, InfoDict
    
    def folding(self, Patches: Tensor, InfoDict: Dict) -> Tensor:
        NumDim = Patches.dim()
        assert NumDim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(Patches.shape)
        
        # [BP, N, C] --> [B, P, N, C]
        Patches = Patches.contiguous().view(InfoDict["batch_size"], self.PatchArea, InfoDict["total_patches"], -1)

        BatchSize, _, _, Channels = Patches.size()
        NumHPatch = InfoDict["num_patches_h"]
        NumWPatch = InfoDict["num_patches_w"]

        ## Keep the right output tensor dimension order
         # [B, P, N, C] --> [B, C, N, P]
        Patches = Patches.transpose(1, 3)
         # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        FeatureMap = Patches.reshape(BatchSize * Channels * NumHPatch, NumWPatch, self.HPatch, self.WPatch)
         # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        FeatureMap = FeatureMap.transpose(1, 2)
         # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        FeatureMap = FeatureMap.reshape(BatchSize, Channels, NumHPatch * self.HPatch, NumWPatch * self.WPatch)
        if InfoDict["interpolate"]:
            FeatureMap = F.interpolate(FeatureMap, size=InfoDict["orig_size"], mode="bilinear", align_corners=False)
        return FeatureMap
    
    def forward(self, x: Tensor) -> Tensor:
        LocalFeatureMap = self.LocalRep(x)  # [B x C x HImg x WImg] --> [B x C x Patches x Patch] 
        
        # Convert feature map to patches
        Patches, InfoDict = self.unfolding(LocalFeatureMap)  # [B x C x Patches x Patch] --> [4B x (Patch x Patches / 4) x C]
        # Learn global representations
        Patches = self.GlobalRep(Patches) # [4B x (Patch x Patches / 4) x C] --> [4B x (Patch x Patches / 4) x C]
        
        FeatureMap = self.folding(Patches, InfoDict) # [4B x (Patch x Patches / 4) x C] --> [B x C x Patches x Patch]
        
        FeatureMap = self.ConvProj(FeatureMap) # [B x C x Patches x Patch] --> [B x C x HImg x WImg]
        
        FeatureMap = self.Fusion(torch.cat((x, FeatureMap), dim=1))
        
        if self.FusionRes:
            FeatureMap = self.Dropout(FeatureMap)
            Output = x + FeatureMap
        else:
            Output = FeatureMap

        return Output
    
    def profileModule(self, Input: Tensor):
        Params = MACs = 0.0

        res = Input
        out, p, m = moduleProfile(module=self.LocalRep, x=Input)
        Params += p
        MACs += m

        Patches, InfoDict = self.unfolding(FeatureMap=out)

        Patches, p, m = moduleProfile(module=self.GlobalRep, x=Patches)
        Params += p
        MACs += m

        fm = self.folding(Patches=Patches, InfoDict=InfoDict)

        out, p, m = moduleProfile(module=self.ConvProj, x=fm)
        Params += p
        MACs += m

        if self.Fusion is not None:
            out, p, m = moduleProfile(module=self.Fusion, x=torch.cat((out, res), dim=1))
            Params += p
            MACs += m

        return res, Params, MACs