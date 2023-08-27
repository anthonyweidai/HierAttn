# Dataset preparation

## ISIC2019 Dataset

The data of ISIC2019 is dermoscopy skin lesion image. 

- Download and unzip the dataset from [Skin Lesion Images for Melanoma Classification](https://www.kaggle.com/datasets/andrewmvd/isic-2019).

- Rearrange the images with same class (each class use a folder with the class name).

- Random oversample using `lib/tools/oversampling.py`.

- IH undersample using `lib/tools/undersampling.py`.

- Create the 10 split subset for cross-validation (the augmented data and its original image should be on the same subset).

- The output data structure should be:

  ~~~
  ${HierAttn_ROOT}
  |-- dataset
  `-- |-- IHISIC20000
      `-- |--- split1
          |   |--- ack_1.jpg
          |   |--- ack_2.jpg
          |   |--- ...
          |   |--- bcc_1.jpg
          |   |--- ...
          |--- split2
          |   |--- ack_1.jpg
          |   |--- ack_2.jpg
          |   |--- ...
          |   |--- bcc_1.jpg
          |   |--- ...
          |--- ...
  ~~~

## PAD-UFES-20 (PAD20) Dataset

The data of PAD20 is smartphone skin lesion image. 

- Download and unzip the dataset from [PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones](https://data.mendeley.com/datasets/zr7vgbcyr2/1).

- Rearrange the images with same class (each class use a folder with the class name).

- Random oversample using `lib/tools/oversampling.py`.

- IH undersample using `lib/tools/undersampling.py`.

- Create the 10 split subset for cross-validation (the augmented data and its original image should be on the same subset).

- The output data structure should be:

  ~~~
  ${HierAttn_ROOT}
  |-- dataset
  `-- |-- IHPAD3000
      `-- |--- split1
          |   |--- ack_1.jpg
          |   |--- ack_2.jpg
          |   |--- ...
          |   |--- bcc_1.jpg
          |   |--- ...
          |--- split2
          |   |--- ack_1.jpg
          |   |--- ack_2.jpg
          |   |--- ...
          |   |--- bcc_1.jpg
          |   |--- ...
          |--- ...
  ~~~

## References

Please cite the corresponding references if you use the datasets and our data pre-processing codes.

~~~
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  number={1},
  pages={1--9},
  year={2018},
  publisher={Nature Publishing Group}
}

@inproceedings{codella2018skin,
  title={Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic)},
  author={Codella, Noel CF and Gutman, David and Celebi, M Emre and Helba, Brian and Marchetti, Michael A and Dusza, Stephen W and Kalloo, Aadi and Liopyris, Konstantinos and Mishra, Nabin and Kittler, Harald and others},
  booktitle={2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018)},
  pages={168--172},
  year={2018},
  organization={IEEE}
}

@article{combalia2019bcn20000,
  title={Bcn20000: Dermoscopic lesions in the wild},
  author={Combalia, Marc and Codella, Noel CF and Rotemberg, Veronica and Helba, Brian and Vilaplana, Veronica and Reiter, Ofer and Carrera, Cristina and Barreiro, Alicia and Halpern, Allan C and Puig, Susana and others},
  journal={arXiv preprint arXiv:1908.02288},
  year={2019}
}

@article{pacheco2020pad,
  title={PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones},
  author={Pacheco, Andre GC and Lima, Gustavo R and Salom{\~a}o, Amanda S and Krohling, Breno and Biral, Igor P and de Angelo, Gabriel G and Alves Jr, F{\'a}bio CR and Esgario, Jos{\'e} GM and Simora, Alana C and Castro, Pedro BC and others},
  journal={Data in brief},
  volume={32},
  pages={106221},
  year={2020},
  publisher={Elsevier}
}

@ARTICLE{dai2023deeply,
  author={Dai, Wei and Liu, Rui and Wu, Tianyi and Wang, Min and Yin, Jianqin and Liu, Jun},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Deeply Supervised Skin Lesions Diagnosis with Stage and Branch Attention}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/JBHI.2023.3308697}}
~~~