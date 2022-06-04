import numpy as np
from functools import partial
from typing import Optional, Callable, Any, List, Dict

from torch import nn, Tensor

from . import registerClsModels
from .config import getConfiguration
from .base import BaseHierAttn
from .baseBlock import BaseConv2d, Linearlayer, Globalpooling, Dropout, initWeight
from .myLayers import MBConv, MBConvConfig, BranchFolding, SameChannelAttn, ConvViTBlock


class HierAttentionNet(BaseHierAttn):
    def __init__(
        self,
        HierAttnConfig,
        NumClasses: int=1000, 
        SELayer: Callable[..., nn.Module]=SameChannelAttn,
        FusionRes: bool=False,
        SDFlag: int=1,
        TotalBlocks: Optional[int]=None,
        BranchPool: list=[1, 3, 5],
        MCRandom: Optional[bool]=True,
        InitWeights: Optional[bool]=True,
        **kwargs: Any
        ):
        OutChannel = 16
        super().__init__()
        self.FirstOuChannel = OutChannel
        
        self.BlockID = 0
        self.SDFlag = SDFlag
        self.FusionRes = FusionRes
        self.SELayer = SELayer
        if not TotalBlocks:
            self.TotalBlocks = 6 if self.FusionRes else 3
            
        self.ModelConfigDict = dict()
        self.Conv1 = BaseConv2d(3, OutChannel, 3, 2, BNorm=True, ActLayer=nn.SiLU)
        
        self.ModelConfigDict['conv1'] = {'in': 3, 'out': OutChannel}
        
        InChannel = OutChannel
        self.Layer1, OutChannel = self.makeLayer(InChannel=InChannel, Cfg=HierAttnConfig['layer1'], **kwargs)
        self.ModelConfigDict['layer1'] = {'in': InChannel, 'out': OutChannel}
        
        InChannel = OutChannel
        self.Layer2, OutChannel = self.makeLayer(InChannel=InChannel, Cfg=HierAttnConfig['layer2'], **kwargs)
        self.ModelConfigDict['layer2'] = {'in': InChannel, 'out': OutChannel}
        
        InChannel = OutChannel
        self.Layer3, OutChannel = self.makeLayer(InChannel=InChannel, Cfg=HierAttnConfig['layer3'],  **kwargs)
        self.ModelConfigDict['layer3'] = {'in': InChannel, 'out': OutChannel}
        self.BranchInChannel1 = OutChannel
        
        InChannel = OutChannel
        self.Layer4, OutChannel = self.makeLayer(InChannel=InChannel, Cfg=HierAttnConfig['layer4'], **kwargs)
        self.ModelConfigDict['layer4'] = {'in': InChannel, 'out': OutChannel}
        self.BranchInChannel2 = OutChannel
        
        InChannel = OutChannel
        self.Layer5, OutChannel = self.makeLayer(InChannel=InChannel, Cfg=HierAttnConfig['layer5'], **kwargs)
        self.ModelConfigDict['layer5'] = {'in': InChannel, 'out': OutChannel}
        
        # Add one learnable layer
        BranchInChannels = []
        for Layer in range(2, 5):
            # Only MViT block branch
            Temp = HierAttnConfig['layer' + str(Layer + 1)].get("out_channels", 0)
            BranchInChannels.append(Temp)
        
        ExpFactor = HierAttnConfig['last_layer_exp_factor']
        BranchOutChannels = BranchInChannels.copy()

        self.BranchFolding = BranchFolding(BranchPool, BranchOutChannels, MCRandom)
        InChannel = np.sum(BranchOutChannels)
        ExpChannel = InChannel  * ExpFactor
        self.Conv1x1Hier = BaseConv2d(InChannel, ExpChannel, 1, 1, BNorm=True, ActLayer=nn.SiLU) # MCRandom influence this layer
        
        # Global pool to linear
        self.Classifier = nn.Sequential()
        self.Classifier.add_module(name='global_pool', module=Globalpooling(PoolType='mean'))
        self.Classifier.add_module(name='dropout', module=Dropout(0.1, inplace=True))
        self.Classifier.add_module(name='fc', module=Linearlayer(ExpChannel, NumClasses, bias=True))
        
        if InitWeights:
            self.apply(initWeight)
        
    def makeLayer(self, InChannel: int, Cfg: Dict, **kwargs: Any):
        BlockType = Cfg.get('block_type', 'stageattn')
        if BlockType.lower() == 'stageattn':
            return self.makeStageAttnblcok(InChannel, Cfg, **kwargs)
        else:
            return self.makeSCADWblcok(InChannel, Cfg, **kwargs)
    
    def makeStageAttnblcok(self, InChannel: int, Cfg: Dict, **kwargs: Any):
        Block = []
        Stride = Cfg.get("stride", 1)
        OutChannel = Cfg.get("out_channels")
        ExpandRatio = Cfg.get("mv_expand_ratio", 4)
        MBAttnType = Cfg.get("mb_attn", 'mck3')
        KeepRes = Cfg.get("keep_res", 1)
        KeepRes2 = Cfg.get("keep_res_2", 1)
        KeepRes3 = Cfg.get("keep_res_3", 1)
        DepthStage = Cfg.get("depth_stage", 1)
        
        if Stride == 2:
            WidthMult = 1.0
            DepthMult = 1.0
            BneckConf = partial(MBConvConfig, WidthMult=WidthMult, DepthMult=DepthMult)
            # expand_ratio, kernel, stride, input_channels, out_channels, num_layers
            Cfn = BneckConf(ExpandRatio, 3, Stride, InChannel, OutChannel, 1)
            Layer = MBConv(Cfn, SELayer=self.SELayer, MBAttnType=MBAttnType, DepthStage=DepthStage, 
                            KeepRes=KeepRes, KeepRes2=KeepRes2, KeepRes3=KeepRes3, **kwargs)    
            Block.append(Layer)
            InChannel = OutChannel
        
        DimHead = Cfg.get("dim_head", 32)
        DimEmb = Cfg["transformer_channels"]
        MLPSize = Cfg.get("ffn_dim")
        if DimHead is None:
            NumHeads = Cfg.get("num_head", 4)
            if NumHeads is None:
                NumHeads = 4
            DimHead = DimEmb // NumHeads
        
        self.BlockID += 1
        SDProb = 0.2 * float(self.BlockID) * self.SDFlag / self.TotalBlocks
        
        Block.append(
            ConvViTBlock(
                InChannel,
                (Cfg.get("patch_h", 2), Cfg.get("patch_w", 2)),
                DimEmb,
                MLPSize,
                Cfg.get("transformer_blocks", 1),
                DimHead,
                ConvKSize=3,
                FusionRes=self.FusionRes,
                SDProb=SDProb,
                **kwargs))
        
        return nn.Sequential(*Block), InChannel
    
    def makeSCADWblcok(self, InChannel: int, Cfg: Dict, **kwargs: Any):
        OutChannel = Cfg.get("out_channels")
        NumBlocks = Cfg.get("num_blocks", 2)
        ExpandRatio = Cfg.get("expand_ratio", 4)
        MBAttnType = Cfg.get("mb_attn", 'se')
        KeepRes = Cfg.get("keep_res", 1)
        KeepRes2 = Cfg.get("keep_res_2", 1)
        KeepRes3 = Cfg.get("keep_res_3", 1)
        DepthStage = Cfg.get("depth_stage", 1)
        Block: List[nn.Module] = []
        
        for i in range(NumBlocks):
            Stride = Cfg.get("stride", 1) if i == 0 else 1 # The first one (64 x 64) is perform down-sampling
            if Stride == 1:
                self.BlockID += 1
            SDProb = 0.2 * float(self.BlockID) * self.SDFlag / self.TotalBlocks # 0.1 is initial value
            WidthMult = 1.0
            DepthMult = 1.0
            BneckConf = partial(MBConvConfig, WidthMult=WidthMult, DepthMult=DepthMult)
            Cfn = BneckConf(ExpandRatio, 3, Stride, InChannel, OutChannel, 1)
            Layer = MBConv(Cfn, SDProb, SELayer=self.SELayer, MBAttnType=MBAttnType, DepthStage=DepthStage, 
                            KeepRes=KeepRes, KeepRes2=KeepRes2, KeepRes3=KeepRes3, **kwargs)
            Block.append(Layer)
            InChannel = OutChannel
        return nn.Sequential(*Block), InChannel
        
    def forwardFeatures(self, x: Tensor) -> Tensor:
        InB1, InB2, InB3 = self.branchFeatures(x)
        x = self.BranchFolding(InB1, InB2, InB3)
        return self.Conv1x1Hier(x)
       
    def forward(self, x: Tensor) -> Tensor: 
        x = self.forwardFeatures(x)
        return self.Classifier(x)
    

@registerClsModels("hierattnxs")
def hierAttnNetxs(**kwargs: Any):
    Mode = 'x_small'
    DimHead = None
    NumberofHead = 4
    MLPDropRate2 = 0.05
    HierAttnConfig = getConfiguration(Mode, DimHead, NumberofHead)
    return HierAttentionNet(HierAttnConfig, MLPDropRate2=MLPDropRate2, **kwargs)


@registerClsModels("hierattns")
def hierAttnNets(**kwargs: Any):
    Mode = 'small'
    DimHead = None
    NumberofHead = 4
    MLPDropRate2 = 0.1
    HierAttnConfig = getConfiguration(Mode, DimHead, NumberofHead)
    return HierAttentionNet(HierAttnConfig, MLPDropRate2=MLPDropRate2, **kwargs)


@registerClsModels("hierattnm")
def hierAttnNetm(**kwargs: Any):
    Mode = 'm_small'
    DimHead = None
    NumberofHead = 4
    MLPDropRate2 = 0.1
    HierAttnConfig = getConfiguration(Mode, DimHead, NumberofHead)
    return HierAttentionNet(HierAttnConfig, MLPDropRate2=MLPDropRate2, **kwargs)