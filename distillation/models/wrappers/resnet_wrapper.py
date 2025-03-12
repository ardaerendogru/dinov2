from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import ResNet, BasicStem, make_resnet_stages
from torchvision import models

versions = {
    "resnet_18": {"depth": 18},
    "resnet_34": {"depth": 34},
    "resnet_50": {"depth": 50},
    "resnet_101": {"depth": 101},
}


def list_resnet_configs():
    return list(versions.keys())


def get_resnet_config(name):
    if name in versions:
        return versions[name]
    else:
        raise NotImplementedError(
            f"There is not a ResNet config called {name}."
        )

class ResNetWrapper(BaseModel):
    """Wrapper class for ResNet models with configurable depth and output features.
    
    Args:
        model_id (str): ID of the ResNet model to use.
        out_features (Optional[List[str]]): List of feature levels to output.
            Possible values: ['res2', 'res3', 'res4', 'res5']. Default: all levels
        freeze_at (int): Freeze layers up to this index. 0 means no freezing. Default: 0
        norm_type (str): Normalization type to use ('BN' for BatchNorm). Default: 'BN'
        
    Attributes:
        model (ResNet): The underlying ResNet model
        feature_channels (dict): Dictionary mapping feature levels to their channel counts
    """
    def __init__(
        self,
        model_name: str = "resnet_50",
        out_features: Optional[List[str]] = None,

    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        
        # Get ResNet config
        config = get_resnet_config(model_name)
        depth = config["depth"]
        
        # Create ResNet model
        stem = BasicStem(in_channels=3, out_channels=64, norm='BN')
        stages = make_resnet_stages(
            depth=depth,
            norm='BN'
        )
        
        self.model = ResNet(
            stem=stem,
            stages=stages,
            num_classes=None,
            out_features=out_features,
            freeze_at=0
        )

    def get_features(self, x):
        return self.model(x)
    
    @property
    def feature_channels(self):
        return self.model._out_feature_channels