from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from ..backbones import PResNet

versions = {
    "presnet_18": {"depth": 18, "variant": "d"},
    "presnet_34": {"depth": 34, "variant": "d"},
    "presnet_50": {"depth": 50, "variant": "d"},
    "presnet_101": {"depth": 101, "variant": "d"},
}


def list_presnet_configs():
    return list(versions.keys())


def get_presnet_config(name):
    if name in versions:
        return versions[name]
    else:
        raise NotImplementedError(
            f"There is not a PResNet config called {name}."
        )

class PResNetWrapper(BaseModel):
    """Wrapper class for PResNet models.
    
    Args:
        model_id (str): ID of the PResNet model to use.
        out_features (Optional[List[str]]): List of feature levels to output.
            Possible values: ['res2', 'res3', 'res4', 'res5']. Default: all levels
    
    Attributes:
        model (PResNet): The underlying PResNet model.
        feature_channels (dict): Dictionary mapping feature levels to their channel counts.
    """
    def __init__(
        self,
        model_name: str = "presnet_50",
        out_features: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        self._out_features = out_features
        
        # Get PResNet config
        config = get_presnet_config(model_name)
        depth = config["depth"]
        variant = config["variant"]
        
        # Create PResNet model
        self.model = PResNet(
            depth=depth,
            variant=variant,
            num_stages=4,
            act='relu',
            freeze_at=-1,
            freeze_norm = False,
            pretrained=False,
        )

        # Filter feature channels based on out_features
        self._out_feature_channels = {
            f"res{i+2}": self.model.out_channels[i]
            for i in range(len(out_features))
        }

    def get_features(self, x):
        outs = self.model(x)
        return {
            f"res{i+2}": outs[i] for i in range(len(self._out_features))
        }
    
    @property
    def feature_channels(self):
        return self._out_feature_channels