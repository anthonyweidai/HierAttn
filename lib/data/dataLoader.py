import os
from glob import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold

from torch.utils import data
from torchvision import transforms

from ..utils import pair


class Transforms(transforms.Compose):
    def __init__(self, ResizeRes=256):        
        # Set up transforms
        ImgHeight, ImgWidth = pair(ResizeRes)
        self.TF = {
            "train": transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((ImgHeight, ImgWidth)),
                transforms.ToTensor(), # divided by 255. This is how it is forces the network to be between 0 and 1.
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                # transforms.Normalize((0.5, 0.5, 0.5), [0.5, 0.5, 0.5])
            ]),
            "test": transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(ResizeRes),
                transforms.Resize((ImgHeight, ImgWidth)),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                # transforms.Normalize((0.5, 0.5, 0.5), [0.5, 0.5, 0.5])
            ]),
        }
    
    def retrunItem(self):
        return self.TF


class Mydataset(data.Dataset):
    # For CPU dataloader
    def __init__(self, ImgPaths, ClassNames, ResizeRes, SetType="train", Transform=None, TargetTransform=None):
        self.ImgPaths =  np.asarray(ImgPaths)
        self.NumClasses = len(ClassNames)
        self.ClassNames = ClassNames
        if not Transform:
            self.Transform = Transforms(ResizeRes).TF[SetType]
        self.TargetTransform = TargetTransform
        self.getLabel()
        # self.ClassSenFactor = self.getSenFactor()
        print('The amount of ' + SetType + ' data:', self.__len__())
    
    def getLabel(self):
        self.Labels = np.zeros((len(self.ImgPaths), 1), dtype=np.int64)
        for i, ImgPath in enumerate(self.ImgPaths):
            for j, spec in enumerate(self.ClassNames):
                if spec in ImgPath:
                    self.Labels[i] = j
                    break

    def __getitem__(self, index):
        ImgPath = str(self.ImgPaths[index])
        Label = self.Labels[index]  # Convert the data type of label to long int type
        Img = Image.open(ImgPath)
        # Change Image channels
        if Img.mode == "RGBA":
            r, g, b, _ = Img.split()
            Img = Image.merge("RGB", (r, g, b))
        if Img.mode != "RGB":
            Img = Img.convert("RGB")
        
        ImgData = self.Transform(Img)
        if self.TargetTransform:
            Label = self.TargetTransform(Label)
        return ImgData, Label

    def __len__(self):
        return len(self.ImgPaths)
    
    def getNumEachClass(self):
        NumEachClass = np.zeros((self.NumClasses, 1)) # prevent some class doen't has data
        Unique, Counts = np.unique(self.Labels, return_counts=True)
        ClassCount = dict(zip(Unique, Counts))
        for i in range(self.NumClasses):
            NumEachClass[i] = ClassCount[i]
        return NumEachClass
    
    def getSenFactor(self):
        # for data imbalanced dataset
        NumEachClass = self.getNumEachClass()
        Reciprocal = np.reciprocal(NumEachClass)
        return Reciprocal / max(Reciprocal)


def getImgPath(DatasetPath, NumSplit, Mode=1, Shuffle=True):
    # Put images into train set or test set
    if Mode == 1:
        '''
        root/split1/dog_1.png
        root/split1/dog_2.png
        root/split2/cat_1.png
        root/split2/cat_2.png
        '''
        TrainSet, TestSet = [], []
        for i in range(1, NumSplit + 1):
            TestSet.append(glob(DatasetPath + '/' + 'split{}'.format(i) + '/*'))
            
            TrainImgs = []
            for j in range(1, NumSplit + 1):
                if j != i:
                    TrainImgs.extend(glob(DatasetPath + '/' + 'split{}'.format(j) + '/*'))
            TrainSet.append(TrainImgs)
                
    elif Mode == 2:
        '''
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
        '''
        TrainSet, TestSet = [[] for _ in range(NumSplit)], [[] for _ in range(NumSplit)]
        ClassNames = os.listdir(DatasetPath)
        Kf = KFold(n_splits=NumSplit, shuffle=Shuffle)
        
        for ClassName in ClassNames:
            ImagePath = glob(DatasetPath + '/' + ClassName + '/*')
            IndexList = range(0, len(ImagePath))

            Kf.get_n_splits(IndexList)
            
            for idx, (TrainIndexes, TestIdexes) in enumerate(Kf.split(IndexList)):
                [TrainSet[idx].append(ImagePath[i]) for i in TrainIndexes]
                [TestSet[idx].append(ImagePath[j]) for j in TestIdexes]
            
    return TrainSet, TestSet