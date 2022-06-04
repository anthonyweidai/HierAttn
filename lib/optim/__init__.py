from typing import Any
from .scheduler import buildScheduler


SUPPORTED_TASKS = ["segmentation", "classification", "detection"]

def getSchedular(SchedularName, DatasetCategory='classification', **kwargs: Any):
    Schedular = None
    if DatasetCategory == "classification":
        Schedular = buildScheduler(SchedularName, **kwargs)
    # elif dataset_category == "segmentation":
    #     Schedular = build_segmentation_LossFn(opts=opts)
    # elif dataset_category == "detection":
    #     Schedular = build_detection_LossFn(opts=opts)
    else:
        task_str = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                   '\nSupported tasks are:'.format(DatasetCategory)
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
    print("We use %s schedular" %(SchedularName))
    return Schedular