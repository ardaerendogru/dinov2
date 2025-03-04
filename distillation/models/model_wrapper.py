import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from .resnet_wrapper import ResNetWrapper
from .stdc_wrapper import STDCWrapper
import torch.nn.functional as F
import logging

logger = logging.getLogger("dinov2_distillation")

class ModelWrapper(nn.Module):
    MODEL_MAP = {
        'resnet': ResNetWrapper,
        'stdc': STDCWrapper,
    }
    def __init__(
        self,
        model_name: str,
        n_patches: int = 256,
        target_feature: list[str] = ['res5', 'res4'],
        checkpoint_path: Optional[str] = None,
        **model_kwargs
    ):
        super().__init__()
        
        # Validate model type
        if model_name.lower() not in ['resnet', 'stdc']:
            logger.error(f"Unsupported model type: {model_name}")
            raise ValueError(f"Unsupported model type: {model_name}")
        
        # Create model
        self.model = self.MODEL_MAP[model_name.lower()](**model_kwargs)
        
        # Load checkpoint if specified
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.n_patches = n_patches
        self.target_features = target_feature
        self.patch_size = int(np.sqrt(self.n_patches))



    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        features = self.model.get_features(x)
        matched_features = {}
        for feat in self.target_features:
            if feat in features:
                target_feature = features[feat]
                
                # Interpolate to match patch size
                matched_features[feat] = torch.nn.functional.interpolate(
                    target_feature,
                    size=(self.patch_size, self.patch_size),
                    mode='bilinear',
                    align_corners=False
                )
        return matched_features

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint and log details."""
        logger.info(f"Loading student checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            result = self.load_state_dict(checkpoint, strict=False)
            logger.info(f"Checkpoint loaded successfully. Missing keys: {result.missing_keys}, Unexpected keys: {result.unexpected_keys}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def print_model_summary(self) -> None:
        """Print a summary of the model architecture."""
        logger.info("Model Summary:")
        for name, param in self.named_parameters():
            logger.info(f"{name}: {param.size()}")
    @property
    def feature_channels(self):
        return self.model.feature_channels  # Access feature_channels directly from self.model