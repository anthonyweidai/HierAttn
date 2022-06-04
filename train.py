## Import module
 # path manager
from pathlib import Path
 # data processing
import csv
import time
import numpy as np
import pandas as pd
from datetime import datetime
 # torch module
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
 # my module
from lib.model import getModel
from lib.optim import getSchedular
from lib.data import Mydataset, getImgPath
from lib.utils import Device, DataParallel, CnnTrainer, \
    expFolderCreator, writeCsv, loadModelWeight, wightFrozen, optimizerChoice, workerManager


class Train(object):
    def __init__(
        self,
        DatasetPath, NumSplit, ClassNames, 
        NumRepeat=0, LrRate=0.0002, Beta1=0.9, Beta2=0.999, WeightDecay=0, 
        Epochs=500, BatchSize=32, StopStation=50, ResizeRes=224, 
        MyOptimizer='Adam', ModelName='Alexnet', SetName='ITG', 
        LrDecay=False, SchedularFnName='base', PreTrained=0, PreTrainedWeight='', WeightFrize=0,
        ) -> None:
       
        Tick0 = time.perf_counter()
        
        self.IndicatorType = 'accuracy'
        self.TrainSet, self.TestSet = getImgPath(DatasetPath, NumSplit, Mode=1)
            
        self.NumClasses = len(ClassNames)
        print('The number of classes:', self.NumClasses)
        
        print("We use {} device".format(Device))
        
        self.NumSplit = NumSplit
        self.ClassNames = ClassNames
        self.NumRepeat = NumRepeat
        self.LrRate = LrRate
        self.Beta1 = Beta1
        self.Beta2 = Beta2
        self.WeightDecay = WeightDecay
        self.Epochs = Epochs
        self.BatchSize = BatchSize
        self.StopStation = StopStation
        self.ResizeRes = ResizeRes
        self.MyOptimizer = MyOptimizer
        self.ModelName = ModelName
        self.SetName = SetName
        self.LrDecay = LrDecay
        self.SchedularFnName = SchedularFnName
        self.PreTrained = PreTrained
        self.PreTrainedWeight = PreTrainedWeight
        self.WeightFrize = WeightFrize
        
        self.filePathInit()
        self.logFieldInit()
        
        self.StopEpochList = np.zeros((NumSplit, 1), dtype=int)
        self.training()
        self.AvgStopEpoch = (sum(self.StopEpochList) / NumRepeat)[0]
        
        TimeCost = time.perf_counter() - Tick0
        print('Finish training using: %.4f minutes' % (TimeCost / 60))
        
        self.writeLogFile(TimeCost)
    
    def filePathInit(self):
        DestPath, self.ExpCount = expFolderCreator(ExpType='train_cnn')
        
        self.ExpLogPath = './exp/train_cnn/log.csv' # No need to change
        self.InputLogPath = DestPath + 'input_param.csv'
        self.ModelSavePath =  DestPath + 'model'  # Save weight
        self.TrainRecordPath = DestPath + '/metrics/record'  # Save indicators during training
        self.TrainMetricsPath =  DestPath + 'best_metrics.csv'  # Save metrics
        
    def logFieldInit(self):
        # save input params
        self.LogField = [
            'exp', 'date', 'Model', 'Dataset', 
            'Optimizer', 'LrDecay', 'Schedular', 
            'PreTrained', 'WeightFrize',
            'NumberofSplit', 'NumRepeat', 'LrRate', 'Beta1', 'Beta2', 
            'Epochs', 'BatchSize', 'StopStation', 'ResizeResolution'
            ] # Define header
        self.LogInfo = [
            self.ExpCount, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            self.ModelName, self.SetName,
            self.MyOptimizer, self.LrDecay, self.SchedularFnName, 
            self.PreTrained, self.WeightFrize,
            self.NumSplit, self.NumRepeat, self.LrRate, self.Beta1, self.Beta2, 
            self.Epochs, self.BatchSize, self.StopStation, self.ResizeRes
            ]

        writeCsv(self.InputLogPath, self.LogField, self.LogInfo)
    
    def training(self):
        for Split in range(self.NumSplit):
            self.Split = Split
            
            Model = getModel(self.ModelName, self.NumClasses, 
                             ResizeRes=self.ResizeRes, BatchSize=self.BatchSize)
            Model.to(Device)
            
            if DataParallel:
                Model = torch.nn.DataParallel(Model)
                
            if PreTrained:
                if WeightFrize == 3:
                    Model = loadModelWeight(Model, 2, self.PreTrainedWeight, self.PreTrained)
                else:
                    Model = loadModelWeight(Model, 0, self.PreTrainedWeight, self.PreTrained)

            Optimizer = optimizerChoice(Model.parameters(), lr=self.LrRate, Choice=self.MyOptimizer, 
                                        betas=(self.Beta1, self.Beta2), weight_decay=self.WeightDecay)
            
            Milestones = 30 if self.Epochs > 100 else 10
            if 'cosine' in self.SchedularFnName:
                Scheduler = getSchedular(self.SchedularFnName, Optimizer=Optimizer, 
                                        Milestones=Milestones, MaxEpochs=self.Epochs, MinLrRate=self.LrRate)
            else:
                Scheduler = MultiStepLR(Optimizer, milestones=[Milestones], gamma=0.1) 
        
            TrainImgs = Mydataset(self.TrainSet[self.Split], self.ClassNames, self.ResizeRes, SetType="train")
            TestImgs = Mydataset(self.TestSet[self.Split], self.ClassNames, self.ResizeRes, SetType="test")
            PinMemory, NumWorkers = workerManager(self.BatchSize)
            TrainDL = DataLoader(TrainImgs, self.BatchSize, num_workers=NumWorkers, shuffle=True, pin_memory=PinMemory)
            TestDL = DataLoader(TestImgs, self.BatchSize, num_workers=NumWorkers, shuffle=True, pin_memory=PinMemory)
            
            Temp = []
            [Temp.append([]) for _ in range(9)]
            self.TrainLossList, self.TrainAccuList, self.TestLossList, self.TestAccuList, \
                self.AvgRecallList, self.AvgSpecificityList, self.AvgPrecisionList, self.AvgF1ScoreList, \
                    self.LrRateList = tuple(Temp)

            # Start tranining
            Temp = 0
            BestEpoch = 0
            BestStopIndicator = 10.0  if self.IndicatorType == 'loss' else 0 # Best metric indicator
            print('***************Validation {}*****************'.format(self.Split + 1))
            for Epoch in range(self.Epochs):
                time.sleep(0.5)  # To prevent possible deadlock during epoch transition
                Tick1 = time.perf_counter()
                
                if Epoch == Milestones and self.WeightFrize == 3 and self.PreTrained:
                    Model = wightFrozen(Model, self.WeightFrize) # From Milestones + 1, stop all layers' weight frizing
                
                TCnn = CnnTrainer(Device, self.NumClasses, Optimizer, Model, TrainDL, TestDL) # training class
                
                AvgRecall = np.mean(TCnn.Recall)
                AvgPrecision = np.mean(TCnn.Precision)
                AvgF1Score = np.mean(TCnn.F1Score)
                AvgSpecificity = np.mean(TCnn.Specificity)
                
                StopIndicator = TCnn.TestLoss if self.IndicatorType == 'loss' else TCnn.TestAcc
                if (StopIndicator < BestStopIndicator and self.IndicatorType == 'loss') or \
                (StopIndicator > BestStopIndicator and self.IndicatorType != 'loss'):
                    BestEpoch = Epoch
                    BestStopIndicator = StopIndicator
                    torch.save(Model.state_dict(), '%s/%s_val%d.pth' % (self.ModelSavePath, self.ModelName, self.Split + 1))
                    Temp = 0
                else:
                    Temp += 1
                    
                self.TrainLossList.append(TCnn.TrainLoss)
                self.TrainAccuList.append(TCnn.TrainAcc)
                self.TestLossList.append(TCnn.TestLoss)
                self.TestAccuList.append(TCnn.TestAcc)
                self.AvgRecallList.append(AvgRecall)
                self.AvgPrecisionList.append(AvgPrecision)
                self.AvgF1ScoreList.append(AvgF1Score)
                self.AvgSpecificityList.append(AvgSpecificity)
                
                if self.LrDecay:
                    if "cosine" in self.SchedularFnName:
                        Scheduler.step(Epoch)
                    else:
                        Scheduler.step()
                    CurrentLrRate = Scheduler._last_lr[0] 
                else:
                    CurrentLrRate = self.LrRate
                self.LrRateList.append(CurrentLrRate)
                    
                print('Epoch: [%d/%d] \tCrossValid: [%d/%d] \tBest: %.5f  Accuracy: %.4f  ValAcc: %.4f'
                        % (Epoch + 1, Epochs, self.Split + 1, self.NumSplit, BestStopIndicator, TCnn.TrainAcc, TCnn.TestAcc),
                        '\nComplet [%d/%d]  AvgRecall: %.4f  AvgPrecis: %.4f  AvgF1Score: %.4f  AvgSpec: %.4f'
                        % (Epoch - BestEpoch, self.StopStation, AvgRecall, AvgPrecision, AvgF1Score, AvgSpecificity), 
                        '\nTime cost: %.2f seconds \tLearning rate: %.6f \tLoss: %.4f \tValLoss: %.4f'
                        % (time.perf_counter() - Tick1, CurrentLrRate, TCnn.TrainLoss, TCnn.TestLoss))
                
                if Temp == self.StopStation:
                    break
                
            self.StopEpochList[self.Split] = Epoch + 1
            
            ## Writing results
            self.writeAvgMetrics()
            self.writeBestMetrics()
            
            if self.Split >= self.NumRepeat - 1:
                self.writeAvgBestMetrics()
                break
          
    def writeAvgMetrics(self):
        # Write avrage training metrics record
        Path(self.TrainRecordPath).mkdir(parents=True, exist_ok=True)
        OutputExcel = {
            'LrRate': self.LrRateList, 'BatchSize': self.BatchSize,
            'StopStation': self.StopStation, 'ResizeResolution': self.ResizeRes, 
            'TrainLoss': self.TrainLossList, 'TrainAccuracy': self.TrainAccuList, 
            'TestLoss': self.TestLossList, 'TestAccuracy': self.TestAccuList, 
            'AvgRecall': self.AvgRecallList, 'AvgPrecision': self.AvgPrecisionList, 
            'AvgF1Score': self.AvgF1ScoreList, 'AvgSpecificity': self.AvgSpecificityList,
            }
        Output = pd.DataFrame(OutputExcel)
        
        OutputFieldNames = [
            'LrRate', 'BatchSize', 'StopStation', 'ResizeResolution', 
            'TrainLoss', 'TrainAccuracy', 'TestLoss', 'TestAccuracy', 
            'AvgRecall', 'AvgPrecision', 'AvgF1Score', 'AvgSpecificity'
            ]
        Output.to_csv(self.TrainRecordPath + '_val{}.csv'.format(self.Split + 1), 
                      columns=OutputFieldNames, encoding='utf-8')
    
    def writeBestMetrics(self):
        # Export and write the best result
        if self.IndicatorType == 'loss':
            idx = self.TestLossList.index(min(self.TestLossList))
        else:
            idx = self.TestAccuList.index(max(self.TestAccuList))
        
        BestAccuracy = self.TestAccuList[idx]
        BestRecall = self.AvgRecallList[idx]
        BestSpecificity = self.AvgSpecificityList[idx]
        BestPrecision = self.AvgPrecisionList[idx]
        BestF1Score = self.AvgF1ScoreList[idx]
        
        DfBest = ['val_{}'.format(self.Split + 1), BestAccuracy, BestRecall, BestSpecificity, BestPrecision, BestF1Score]
        self.MetricsFieldNames = ['K-Fold', 'BestAccuracy', 'BestRecall', 'BestSpecificity', 'BestPrecision', 'BestF1Score']    
        writeCsv(self.TrainMetricsPath, self.MetricsFieldNames, DfBest)
        
    def writeAvgBestMetrics(self):
        # Compute and write mean values of metrics in k-fold validation
        MetricsReader = csv.reader(open(self.TrainMetricsPath, 'r'))
        BestMetrics = []
        for Row in MetricsReader:
            BestMetrics.append(Row)
        BestMetrics.pop(0) # remove the header/title in the first row
        RowNum = len(BestMetrics)
        ColNum = len(BestMetrics[0])
        
        Values = np.zeros((ColNum - 1, 1))
        
        Flag = 0
        for i in range(RowNum):
            if 'Average' in BestMetrics[i][0]:
                Flag = 1
                continue
            for j in range(ColNum - 1):
                Values[j] += float(BestMetrics[i][j + 1])
        Values /= RowNum

        self.AvgBestMetric = ['Average'] if Flag == 0 else ['Average_Sup']
        self.AvgBestMetric.extend(list(Values.flatten()))
        writeCsv(self.TrainMetricsPath, self.MetricsFieldNames, self.AvgBestMetric)

    def writeLogFile(self, TimeCost):
        # Write input and output param in log file
        self.LogInfo.extend([TimeCost, self.AvgStopEpoch])
        self.LogInfo.extend(self.AvgBestMetric[1:])
        self.LogInfo[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        NewFieldNames = ['TimeCost', 'AvgStopEpoch', 'Accuracy',  'Recall', 'Specificity', 'Precision', 'F1Score']
        writeCsv(self.ExpLogPath, self.LogField, self.LogInfo, NewFieldNames)


if __name__ == "__main__":
    Beta1 = 0.9
    Beta2 = 0.999
    MyOptimizer = 'adamw'
    WeightDecay = 1.e-2 if 'adamw' in MyOptimizer else 0
    PreTrained = 2
    WeightFrize = 3
    Epochs = 500
    BatchSize = 64 # 8 multiple
    ResizeRes = 256
    BestMode = False # Best model fully transfer learning
    NumSplit = 10
    
    LrDecay = True # True, False
    SchedularFnName = 'mycosine'
    LrRate = 0.0002 if 'cosine' in SchedularFnName else 0.002
    StopStation = Epochs if 'cosine' in SchedularFnName else 100
    NumRepeat = NumSplit
    SetNamePool = ['IHISIC20000']
    '''
    'IHISIC20000', 'RandISIC2500',
    'IHPAD3000', 'RandPAD3000'
    '''
    
    ModelNames = ['hierattns']
    WeightPool = ['mobilevit_xs.pt']
    
    for idx, ModelName in enumerate(ModelNames):
        ModelName = ModelName.lower() # lowercase model name
        if BestMode:
            PreTrainedWeight = './savemodel/best/' + WeightPool[idx]
        else:
            PreTrainedWeight = './savemodel/' + WeightPool[idx]
            
        for SetName in SetNamePool:
            DatasetPath = './dataset/' + SetName
            if 'PAD' in SetName:
                ClassNames = ['ack', 'bcc', 'bkl', 'mel', 'nv', 'scc']
            else:
                ClassNames = ['ack', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']
            
            Train(DatasetPath, NumSplit, ClassNames, 
                  NumRepeat, LrRate, Beta1, Beta2, WeightDecay, 
                  Epochs, BatchSize, StopStation, ResizeRes, 
                  MyOptimizer, ModelName, SetName, 
                  LrDecay, SchedularFnName, PreTrained, PreTrainedWeight, WeightFrize)