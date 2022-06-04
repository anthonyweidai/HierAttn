from typing import Any
from .classification import buildClassificationModel


SUPPORTED_TASKS = ["segmentation", "classification", "detection"]

def getModel(ModelName, NumClasses, DatasetCategory='classification', **kwargs: Any):
    
    Model = None
    if DatasetCategory == "classification":
        Model = buildClassificationModel(ModelName, NumClasses, **kwargs)
    # elif dataset_category == "segmentation":
    #     Model = build_segmentation_model(opts=opts)
    # elif dataset_category == "detection":
    #     Model = build_detection_model(opts=opts)
    else:
        task_str = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                   '\nSupported tasks are:'.format(DatasetCategory)
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
    # print("We use %s as our training and validation model" %(ModelName))
    return Model