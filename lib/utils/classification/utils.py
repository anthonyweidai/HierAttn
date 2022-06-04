import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

class CnnTrainer(object):
    def __init__(self, Device, NumClasses, Optim, Model, TrainDL, TestDL) -> None:
            
        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0

        TruePositives, FalsePositives, TrueNegatives, FalseNegatives = np.zeros((4, NumClasses), dtype=int)
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
            
            YPred = torch.argmax(YPred, dim=1)
            NumCorrect += (torch.eq(YPred, y)).sum().item()
            NumTotal += y.size(0)
            RunningLoss += Loss.item()
                
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
                                TruePositives[k] += 1
                            else:
                                TrueNegatives[k] += 1
                        else: 
                            if YPred[i].item() == k:
                                FalsePositives[k] += 1
                            else:
                                FalseNegatives[k] += 1

        TestLoss = RunningLoss / NumTotal
        TestAcc = NumCorrect / NumTotal

        # Compute metrics: Recall, Precision, F-score, Specificity
        for k in range(NumClasses):
            TrueAll = TruePositives[k] + TrueNegatives[k]
            PositivesAll = TruePositives[k] + FalsePositives[k]
            TNAndFP = TrueNegatives[k] + FalsePositives[k]
            if TrueAll != 0:
                Recall[k] = TruePositives[k] / TrueAll
            if PositivesAll != 0:
                Precision[k] = TruePositives[k] / PositivesAll
            if (Recall[k] + Precision[k]) != 0:
                F1Score[k] = 2 * Recall[k] * Precision[k] / (Recall[k] + Precision[k])
            if TNAndFP != 0:
                Specificity[k] = TrueNegatives[k] / TNAndFP

        self.TrainLoss = TrainLoss
        self.TrainAcc = TrainAcc
        self.TestLoss = TestLoss
        self.TestAcc = TestAcc
        self.Recall = Recall
        self.Precision = Precision
        self.F1Score = F1Score
        self.Specificity = Specificity