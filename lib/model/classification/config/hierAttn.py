from typing import Dict


def getConfiguration(Mode, DimHead, NumberofHead) -> Dict:
    Mode = Mode.lower()
    if Mode == 'x_small':
        DWExpFactor = 2
        # Fig. 1, 1 + 3 depth-wise seperable block, and then 3 stageattn block
        Config = {
            "layer1": {
                "out_channels": 16,
                "expand_ratio": DWExpFactor,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "dws",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": DWExpFactor,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "dws",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 64,
                "block_type": "stageattn",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 96,
                "block_type": "stageattn",
            },
            "layer5": {  # 7x7
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 224,
                "block_type": "stageattn",
            },
            "last_layer_exp_factor": 4
        }
    elif Mode == 'small':
        DWExpFactor = 4
        Config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": DWExpFactor,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "dws",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": DWExpFactor,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "dws",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 80,
                "block_type": "stageattn",
            },
            "layer4": {  # 14x14
                "out_channels": 80,
                "transformer_channels": 120,
                "ffn_dim": 240,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 128,
                "block_type": "stageattn",
            },
            "layer5": {  # 7x7
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 448,
                "block_type": "stageattn",
            },
            "last_layer_exp_factor": 4
        }
    elif Mode == "m_small":
        DWExpFactor = 4
        Config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": DWExpFactor,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "dws",
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": DWExpFactor,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "dws",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 128,
                "block_type": "stageattn",
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 192,
                "block_type": "stageattn",
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": DWExpFactor,
                "dim_head": DimHead,
                "num_head": NumberofHead,
                "branch_channels": 448,
                "block_type": "stageattn",
            },
            "last_layer_exp_factor": 4
        }
    else:
        raise NotImplementedError
        
    return Config