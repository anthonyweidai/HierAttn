from typing import Any
from ...utils import colorText, importModule


SchedularRegistry = {}


def registerScheduler(Name: str):
    def registerSchedulerClass(Cls):
        if Name in SchedularRegistry:
            raise ValueError("Cannot register duplicate scheduler ({})".format(Name))
        SchedularRegistry[Name] = Cls
        return Cls
    return registerSchedulerClass


def buildScheduler(SchedulerName, **kwargs: Any):
    LrScheduler = None
    if SchedulerName in SchedularRegistry:
        LrScheduler = SchedularRegistry[SchedulerName](**kwargs)
    else:
        SuppList = list(SchedularRegistry.keys())
        SuppStr = "LR Scheduler ({}) not yet supported. \n Supported schedulers are:".format(SchedulerName)
        for i, m_name in enumerate(SuppList):
            SuppStr += "\n\t {}: {}".format(i, colorText(m_name))

    return LrScheduler


# Automatically import the LR schedulers
importModule(__file__, RelativePath="lib.optim.scheduler.")