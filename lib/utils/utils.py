import os
import importlib
from pathlib import Path

import re
import csv
import random
import numpy as np
from typing import Any, Optional

import torch
from torch import optim

from .device import CUDA_AVAI


TextColors = {
    'logs': '\033[34m',  # 033 is the escape code and 34 is the color code
    'info': '\033[32m',
    'warning': '\033[33m',
    'error': '\033[31m',
    'bold': '\033[1m',
    'end_color': '\033[0m',
    'light_red': '\033[36m'
}


def colorText(in_text: str) -> str:
    return TextColors['light_red'] + in_text + TextColors['end_color']


def seedSetting(RPMode, Seed=999):
    # Set random seed for reproducibility
    if RPMode:
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", Seed)
        np.random.seed(Seed)
        random.seed(Seed)
        torch.manual_seed(Seed)
    return


def importModule(CurrentPath, RelativePath, SubFold=''):
    # Automatically import the modules
    ModulesDir = os.path.dirname(CurrentPath) + SubFold
    if os.path.isdir(ModulesDir):
        for file in os.listdir(ModulesDir):
            path = os.path.join(ModulesDir, file)
            if (
                    not file.startswith("_")
                    and not file.startswith(".")
                    and (file.endswith(".py") or os.path.isdir(path))
            ):
                ModuleName = file[: file.find(".py")] if file.endswith(".py") else file
                _ = importlib.import_module(RelativePath + ModuleName)


def getSubdirectories(Dir):
    return [SubDir for SubDir in os.listdir(Dir)
            if os.path.isdir(os.path.join(Dir, SubDir))]


def expFolderCreator(ExpType, Mode=0):
    # Count the number of exsited experiments
    FolderPath = './exp/' + ExpType
    Path(FolderPath).mkdir(parents=True, exist_ok=True)
    
    ExpList = getSubdirectories(FolderPath)
    if len(ExpList) == 0:
        ExpCount = 1
    else:
        MaxNum = 0
        for idx in range(len(ExpList)):
            temp = int(re.findall('\d+', ExpList[idx])[0]) + 1
            if MaxNum < temp:
                MaxNum = temp
        ExpCount = MaxNum if Mode == 0 else MaxNum - 1
    
    DestPath = '%s/exp%s/' % (FolderPath, str(ExpCount))
    Path(DestPath).mkdir(parents=True, exist_ok=True)
    Path(DestPath + '/model').mkdir(parents=True, exist_ok=True)
    
    return DestPath, ExpCount


def writeCsv(DestPath, FieldName, FileData, NewFieldNames=[], DictMode=False):
    Flag = 0 if os.path.isfile(DestPath) else 1
    
    with open(DestPath, 'a', encoding='UTF8', newline='') as f:
        if DictMode:
            writer = csv.DictWriter(f, fieldnames=FieldName)
            if Flag == 1:
                writer.writeheader()
            writer.writerows(FileData) # write data
        else:
            writer = csv.writer(f)
            if Flag == 1:
                if NewFieldNames != []:
                    _ = [FieldName.append(FiledName) for FiledName in NewFieldNames]
                writer.writerow(FieldName) # write the header
            writer.writerow(FileData) # write data


def pair(Res):
    return Res if isinstance(Res, tuple) else (Res, Res)


def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def wightFrozen(Model, WeightFreeze, TransferFlag=0,  PreTrained=1):
    # ModelDict = Model.state_dict()
    if WeightFreeze == 0:
        print("Skin freezing layers")
        return Model
    else:
        Idx = 0
        for Name, Param in Model.named_parameters():
            if WeightFreeze == 1: 
                # if 'features' in Name:
                # Judger = 'classifier' not in Name.lower()
                # if PreTrained == 2:
                #     Judger = Judger and idx < len(ModelDict.keys()) - 7
                    
                if 'classifier' not in Name.lower():
                    Param.requires_grad = False
                else:
                    print(Name, Param.requires_grad)      
            elif WeightFreeze == 2:
                # Frize the layers without transferred weight
                while 1:
                    if TransferFlag[Idx] == 1:
                        Param.requires_grad = False
                        break
                    elif TransferFlag[Idx] == 2:
                        Idx += 1
                    else:
                        print(Name, Param.requires_grad)
                        break
                Idx += 1
            elif WeightFreeze == 3:
                # For step weight freezing
                Param.requires_grad = True
            elif WeightFreeze == 4:
                Param.requires_grad = False
            else:
                print(Name, Param.requires_grad)
                
        if WeightFreeze == 3:
            print("Unfreeze all layers")
        elif WeightFreeze == 4:
            print("Freeze all layers")
            
        return Model


def ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew, FalseFlag=0):
    if any(s in NewKey for s in IgnoreStrList):
        TransferFlag[IdxNew] = 2
    else:
        TransferFlag[IdxNew] = FalseFlag
    return TransferFlag


def loadModelWeight(Model, WeightFreeze, PreTrainedWeight, 
                    PreTrained=1, DropLast=False):
    print('Knowledge transfer from: %s' %(PreTrainedWeight))
    
    if CUDA_AVAI:
        PretrainedDict = torch.load(PreTrainedWeight)
    else:
        PretrainedDict = torch.load(PreTrainedWeight, map_location=torch.device('cpu'))
    
    IgnoreStrList = ["running_mean", "running_var", "num_batches_tracked"] # pytorch 1.10
    
    ModelDict = Model.state_dict()
    TransferFlag = np.zeros((len(ModelDict), 1))
    if PreTrained == 1 or PreTrained == 3:
        # Get weight if pretrained weight has the same dict
        if PreTrained == 1:
            PretrainedDict = {k: v for k, v in PretrainedDict.items() if k in ModelDict and 'classifier' not in k.lower()}
        else:
            PretrainedDict = {k.replace('module.',''): v for k, v in PretrainedDict.items()}
            TransferFlag.fill(1)
    elif PreTrained == 2 or PreTrained == 4:
        # Get weight if pretrained weight has the partly same structure but diffetnent keys
        # initialize keys and values to keep the original order
        OldDictKeys = list(PretrainedDict.keys())
        OldValues = list(PretrainedDict.values())
        NewDictKeys = list(ModelDict.keys())
        NewValues = list(ModelDict.values())
        
        LenFlag =  len(PretrainedDict) > len(ModelDict)
        MaxLen = max(len(PretrainedDict), len(ModelDict))
        
        Count = IdxNew = IdxOld = 0
        for _ in range(MaxLen):
            OldKey = OldDictKeys[IdxOld]
            OldVal = OldValues[IdxOld]
            NewKey = NewDictKeys[IdxNew]
            NewVal = NewValues[IdxNew]
            
            TransferFlag = ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew)
            
            if not PreTrained == 4 and \
                ('classifier' in OldKey.lower() or 'classifier' in NewKey.lower()):
                if NewKey in OldKey:
                    PretrainedDict.pop(OldKey)
                    IdxNew += 1
                    IdxOld += 1
                continue
            
            if OldVal.shape == NewVal.shape:
                PretrainedDict[NewKey] = PretrainedDict.pop(OldKey)
                TransferFlag = ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew, 1)
                Count += 1
            elif LenFlag:
                IdxNew -= 1
            else:
                IdxOld -= 1
            IdxNew += 1
            IdxOld += 1

            if DropLast:
                if LenFlag and IdxOld == len(OldDictKeys) - 2:
                    break
                elif IdxNew == len(NewDictKeys) - 2:
                    break
    
        print('The number of transferred layers: %d' %(Count))
 
    ModelDict.update(PretrainedDict)
    Model.load_state_dict(ModelDict, strict=False)
    
    Model = wightFrozen(Model, WeightFreeze, TransferFlag, PreTrained)

    return Model


def optimizerChoice(NetParam, lr, Choice='Adam', **kwargs: Any):
    
    # OptimChoices = ['Adam', 'AdamW', 'Adamax', 'SparseAdam', 'SGD', 'ASGD',
    #                'RMSprop', 'Rprop', 'LBFGS', 'Adadelta', 'Adagrad']
    
    CallDict = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adamax': optim.Adamax,
    'sparseadam': optim.SparseAdam,
    'sgd': optim.SGD,
    'asgd': optim.ASGD,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'lbfgs': optim.LBFGS,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    }
    
    Optimizer = CallDict[Choice](NetParam, lr=lr, **kwargs)
        
    return Optimizer