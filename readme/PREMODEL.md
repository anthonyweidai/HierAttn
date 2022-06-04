# Pre-trained models preparation

## MobileViT models

- Download the models from [Training MobileViT Models](https://github.com/apple/ml-cvnets/blob/main/examples/README-mobilevit.md).
- Move the model to folder *savemodel*.
- The output data structure should be:

```
${HierAttn_ROOT}
|-- savemodel
`-- |-- mobilevit_xxs.pt
    |-- mobilevit_xs.pt
    |-- mobilevit_s.pt
```

## Other SOTA models

- Download the models from [Torchvision](https://pytorch.org/vision/stable/index.html).
- Move the model to folder *savemodel*.
- The output data structure should be:

```
${HierAttn_ROOT}
|-- savemodel
`-- |-- efficientnetb0.pth
    |-- mnasnet1.0.pth
    |-- mobilenetv2.pth
    |-- mobilenetv3_large.pth
    |-- shufflenetv2_x1.pth
```

# Reference

Please cite the corresponding references if you use the pretrained models.

```
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}

@incollection{NEURIPS2019_9015,
title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {8024--8035},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}
```

