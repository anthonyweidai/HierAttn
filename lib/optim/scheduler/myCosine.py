import math
from typing import Any

from . import registerScheduler


@registerScheduler("mycosine")
class MyCosineScheduler(object):
    def __init__(
        self,
        Optimizer,
        Milestones,
        MaxEpochs=400,
        MinLrRate=2e-4,
        **kwargs: Any,
        ) -> None:
        super(MyCosineScheduler, self).__init__()
        self.EpochTemp = 0
        self.Optimizer = Optimizer
        self.Milestones = Milestones
        
        self.MinLrRate = MinLrRate
        self.MaxLrRate = MinLrRate * 10

        self.WarmupEpoches = max(Milestones // 2, 0)
        if self.WarmupEpoches > 0:
            self.WarmupLrRate = MinLrRate
            self.warmup_step = (self.MaxLrRate - self.WarmupLrRate) / self.WarmupEpoches

        self.Period = Milestones
        self.MaxEpochs = MaxEpochs

    def get_lr(self, Epoch: int) -> float:
        if Epoch == self.Milestones:
            self.EpochTemp = Epoch - 1
            self.WarmupEpoches += Epoch
            self.Period = self.MaxEpochs
        
        if Epoch < self.WarmupEpoches:
            Currlr = self.WarmupLrRate + (Epoch - self.EpochTemp) * self.warmup_step
        elif Epoch + 1 < self.Period:
            Currlr = self.MinLrRate + 0.5 * (self.MaxLrRate - self.MinLrRate) * (1 + math.cos(math.pi * Epoch / self.Period))
        else:
            Currlr = self.MinLrRate
        return max(0.0, Currlr)
    
    def step(self, Epoch: int):
        Values = self.get_lr(Epoch)
        self.Optimizer.param_groups[0]['lr'] = Values

        self._last_lr = [group['lr'] for group in self.Optimizer.param_groups]