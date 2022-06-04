from typing import Any
from ...utils import colorText, importModule


ClsModelRegistry = {}


def registerClsModels(Name):
    def registerModelClass(Cls):
        if Name in ClsModelRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        ClsModelRegistry[Name] = Cls
        return Cls
    return registerModelClass


def buildClassificationModel(ModelName, NumClasses, **kwargs: Any):
    Model = None
    if ModelName in ClsModelRegistry:
        Model = ClsModelRegistry[ModelName](NumClasses=NumClasses, **kwargs)
    else:
        SupportedModels = list(ClsModelRegistry.keys())
        SuppModelStr = "Supported models are:"
        for i, Name in enumerate(SupportedModels):
            SuppModelStr += "\n\t {}: {}".format(i, colorText(Name))
            
    # for key in ClsModelRegistry.keys():
    #     print(key)
    return Model


# Automatically import the models
importModule(__file__, RelativePath="lib.model.classification.")
importModule(__file__, RelativePath="lib.model.classification.classic.", SubFold='/classic/')
importModule(__file__, RelativePath="lib.model.classification.myothers.", SubFold='/myothers/')