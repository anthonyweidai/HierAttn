import glob
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import DataLoader

from .instanceHardnessThreshold import InstanceHardnessThreshold
from ..data import Mydataset
from ..utils import CUDA_AVAI


if __name__ == "__main__":
    # downsampling to balance data by iht
    ResizeRes = 256
    DownSampleList = ['bcc', 'bkl', 'mel', 'nv'] # ['bcc', 'ack'] for PAD20
    HomePath = r"D:\dataset\Skin Disease\ISIC2019"
    
    OriDataPath = HomePath + '/data'
    
    OriImgsPath = []
    for ClassName in DownSampleList:
        OriImg = glob.glob(OriDataPath + '/' + ClassName + '/*')
        OriImgsPath.extend(OriImg)
        
    NumImgs = len(OriImgsPath)

    TrainImgs = Mydataset(OriImgsPath, DownSampleList, ResizeRes, SetType="train")  
    TrainDL = DataLoader(TrainImgs, 1, num_workers=4)
    
    Images = np.zeros((NumImgs, 3, ResizeRes, ResizeRes))
    Labels = np.zeros(NumImgs)
    Count = 0
    for x, y in TrainDL:
        Images[Count, :, :, :] = x.numpy() if CUDA_AVAI else x.cpu().numpy()
        Labels[Count] = y.numpy() if CUDA_AVAI else y.cpu().numpy()
        Count += 1
    
    Images = Images.reshape((NumImgs, 3 * ResizeRes * ResizeRes))
    Labels = Labels.astype(int) # Convert data type from long int to int

    IHThred = InstanceHardnessThreshold(random_state=0,
                            estimator=RandomForestClassifier(
                                max_depth=None, n_estimators=100, max_features='auto')) # max_depth, max_features
    # IHThred = InstanceHardnessThreshold(random_state=0,
    #                         estimator=LogisticRegression(solver='lbfgs', multi_class='auto')) # max_depth, max_features

    
    _, Ysampled, SampleIdxs = IHThred.fit_resample(Images, Labels)
    
    # print(str(SampleIdxs))
    print('Original dataset shape %s' % Counter(Labels))
    print('Resampled dataset shape %s' % Counter(Ysampled))
    
    with tqdm(total=len(SampleIdxs), colour='blue', ncols=60) as t:
        for i in range(Ysampled.shape[0]):
            SampleImgPath = OriImgsPath[SampleIdxs[i]]
            ClassName = DownSampleList[Ysampled[i]]
            DestPath = HomePath + '/downsample_iht/' + ClassName
            Path(DestPath).mkdir(parents=True, exist_ok=True)
            shutil.copy(SampleImgPath, DestPath)
        
            t.update()