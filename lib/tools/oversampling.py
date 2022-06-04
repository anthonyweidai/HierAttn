import os
import glob
import shutil
from pathlib import Path
from tqdm import tqdm

from .dataAugment import dataAugment

if __name__ == "__main__":
    # upsampling to balance data
    NumSamples = 2500 # 500 for PAD 20
    UpSampleList = ['ack', 'df', 'scc', 'vasc'] # ['mel', 'nv', 'scc', 'sek'] for PAD20
    HomePath = r"D:\dataset\Skin Disease\ISIC2019"
    
    OriDataPath = HomePath + '/data'

    for ClassName in UpSampleList:
        OriImg = glob.glob(OriDataPath + '/' + ClassName + '/*')
        NumImgs = len(OriImg)
        NomberofSample = NumSamples - NumImgs

        print("Upsampling {} images from ".format(NomberofSample) + ClassName)

        DestPath = HomePath + '/Upsample_random{}/'.format(NumSamples) + ClassName
        if os.path.exists(DestPath) and os.path.isdir(DestPath):
            shutil.rmtree(DestPath)
        Path(DestPath).mkdir(parents=True)
        
        UpsampleSingle = [NomberofSample // NumImgs + (1 if x < NomberofSample % NumImgs else 0)  
                            for x in range (NumImgs)]
        
        with tqdm(total=NumImgs, colour='blue', ncols=60) as t:
            for j in range(NumImgs):
                SampleImgPath = OriImg[j]
                dataAugment(SampleImgPath, DestPath, UpsampleSingle[j])
                shutil.copy(SampleImgPath, DestPath) # copy the original image
                
                t.update()