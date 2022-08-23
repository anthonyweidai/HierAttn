# HierAttn: Deeply Supervised Skin Lesions Diagnosis with Stage and Branch Attention

[**Deeply Supervised Skin Lesions Diagnosis with Stage and Branch Attention**](https://arxiv.org/abs/2205.04326)  
Wei Dai, Rui Liu, Tianyi Wu, Min Wang, Jianqin Yin, Jun Liu        
In arxiv, 2022. [[Paper](https://arxiv.org/abs/2205.04326)]

<p align="left"> <img src=readme/Architecture.pdf align="center" width="1080px">




## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Benchmark Evaluation and Training

Please refer to [DATA.md](readme/DATA.md) for dataset preparation. 

We used transfer learning to partly initialize the tunable weights of HierAttn and SOTA models. Please refer to [PREMODEL.md](readme/PREMODEL.md) for pretrained models download.

### *Skin lesions classification in dermoscopy dataset*  

|                    IHISIC20000 Val                    | # Parameters/M | Top-1 Accuracy/%↑ |
| :---------------------------------------------------: | :------------: | :---------------: |
|                      MobileNetV2                      |      2.23      |       93.45       |
|                      MobileViT_s                      |      4.94      |       94.72       |
|                   MobileNetV3_Large                   |      4.21      |       94.77       |
|                    ShuffleNetV2_1×                    |      2.28      |       95.23       |
|                      MnasNet1.0                       |      3.11      |       95.45       |
|                    EfficientNet_b0                    |      4.02      |       95.48       |
| [HierAttn_xs(Ours)](https://arxiv.org/abs/2205.04326) |    **1.08**    |       96.15       |
| [HierAttn_s(Ours)](https://arxiv.org/abs/2205.04326)  |      2.14      |     **96.70**     |

### *Skin lesions classification in smartphone dataset*  

|                     IHPAD3000 Val                     | # Parameters/M | Top-1 Accuracy/%↑ |
| :---------------------------------------------------: | :------------: | :---------------: |
|                      MobileNetV2                      |      2.23      |       87.44       |
|                    ShuffleNetV2_1×                    |      2.28      |       87.89       |
|                      MobileViT_s                      |      4.94      |       88.22       |
|                   MobileNetV3_Large                   |      4.21      |       88.78       |
| [HierAttn_xs(Ours)](https://arxiv.org/abs/2205.04326) |    **1.08**    |       90.11       |
|                    EfficientNet_b0                    |      4.02      |       90.22       |
|                      MnasNet1.0                       |      3.11      |       90.33       |
| [HierAttn_s(Ours)](https://arxiv.org/abs/2205.04326)  |      2.13      |     **91.22**     |

## Citation

If you find it useful in your research, please consider citing our paper as follows:

    @article{dai2022hierattn,
      title={Deeply Supervised Skin Lesions Diagnosis with Stage and Branch Attention},
      author={Dai, Wei and Liu, Rui and Wu, Tianyi and Wang, Min and Yin, Jianqin and Liu, Jun},
      journal={arXiv preprint arXiv:2205.04326},
      year={2022}
    }

## Acknowledgment
Many thanks to **[ml-cvnets](https://github.com/apple/ml-cvnets)** authors for their great framework!
