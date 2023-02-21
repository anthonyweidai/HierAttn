import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn


class CnnTrainer(object):
    def __init__(self, Device, NumClasses, Optim, Model, TrainDL, TestDL) -> None:
            
        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0

        TruePositive, FalsePositive, TrueNegative, FalseNegative = np.zeros((4, NumClasses), dtype=int)
        Recall, Precision, Specificity, F1Score = np.zeros((4, NumClasses), dtype=float)
        
        LossFn = nn.CrossEntropyLoss().to(Device)
        Model.train()  # Normalization is different in trainning and evaluation
        for x, y in tqdm(TrainDL, ncols=60, colour='magenta'):  # iterate x, y in dataloader (one batch data)
            Optim.zero_grad()  # Initialize gradient, preventing accumulation
            
            x, y = x.to(Device), y.to(Device)
            y = torch.squeeze(y)

            YPred = Model(x)  # prediction
            Loss = LossFn(YPred, y)

            Loss.backward()  # backpropagation
            Optim.step()  # optimize model's weight
            
            with torch.no_grad():
                YPred = torch.argmax(YPred, dim=1)
                NumCorrect += (torch.eq(YPred, y)).sum().item()
                NumTotal += y.size(0)
                RunningLoss += Loss.item()
        
        with torch.no_grad():
            TrainLoss = RunningLoss / NumTotal
            TrainAcc = NumCorrect / NumTotal

        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0

        Model.eval()
        with torch.no_grad():
            for x, y in tqdm(TestDL, ncols=60, colour='magenta'):
                x, y = x.to(Device), y.to(Device)
                y = torch.squeeze(y)
                
                YPred = Model(x)  # Evaluation
                Loss = LossFn(YPred, y)
                
                YPred = torch.argmax(YPred, dim=1)
                NumCorrect += (torch.eq(YPred, y)).sum().item()
                BatchSize = y.size(0) # It could be unequal to the batchsize you use, if dataloader without dropping last
                NumTotal += BatchSize
                RunningLoss += Loss.item()

                # Get true/false and positive/negative samples
                for i in range(BatchSize):
                    for k in range(NumClasses):
                        if y[i].item() == k:
                            if YPred[i] == y[i]:
                                TruePositive[k] += 1
                            else:
                               FalseNegative[k] += 1
                        else: 
                            if YPred[i].item() == k:
                                FalsePositive[k] += 1
                            else:
                                TrueNegative[k] += 1

        TestLoss = RunningLoss / NumTotal
        TestAcc = NumCorrect / NumTotal

        # Compute metrics: Recall, Precision, F-score, Specificity
        for k in range(NumClasses):
            PositiveAll = TruePositive[k] + FalseNegative[k]
            TPAndFP = TruePositive[k] + FalsePositive[k]
            NegativeAll = TrueNegative[k] + FalsePositive[k]
            if PositiveAll != 0:
                Recall[k] = TruePositive[k] / PositiveAll
            if TPAndFP != 0:
                Precision[k] = TruePositive[k] / TPAndFP
            if (Recall[k] + Precision[k]) != 0:
                F1Score[k] = 2 * Recall[k] * Precision[k] / (Recall[k] + Precision[k])
            if NegativeAll != 0:
                Specificity[k] = TrueNegative[k] / NegativeAll

        self.TrainLoss = TrainLoss
        self.TrainAcc = TrainAcc
        self.TestLoss = TestLoss
        self.TestAcc = TestAcc
        self.Recall = Recall
        self.Precision = Precision
        self.F1Score = F1Score
        self.Specificity = Specificity