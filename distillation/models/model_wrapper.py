import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from .resnet_wrapper import ResNetWrapper
from .stdc_wrapper import STDCWrapper
from .feature_matcher import FeatureMatcher

class ModelWrapper(nn.Module):
    def __init__(
        self,
        model_type: str,
        n_patches: int = 256,
        target_feature: list[str] = ['res5'],
        feature_matcher_config: Optional[Dict[str, Any]] = None,
        **model_kwargs
    ):
        super().__init__()
        
        # Create model
        if model_type.lower() == 'resnet':
            self.model = ResNetWrapper(**model_kwargs)
        elif model_type.lower() == 'stdc':
            self.model = STDCWrapper(**model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.n_patches = n_patches
        self.target_features = target_feature
        
        # Create feature matchers if config provided
        if feature_matcher_config:
            # Convert dictionary to ModuleDict for proper registration
            self.feature_matchers = nn.ModuleDict()
            for feat in self.target_features:
                # Get correct input channels from model
                in_channels = self.model.feature_channels[feat]
                
                # Create new config with correct in_channels
                matcher_config = {**feature_matcher_config}
                matcher_config['in_channels'] = in_channels
                
                self.feature_matchers[feat] = FeatureMatcher(**matcher_config)
        else:
            self.feature_matchers = nn.ModuleDict()
    
    def forward(self, x):
        # Get features from model
        features = self.model.get_features(x)
        
        # Process target features if matchers exist
        matched_features = {}
        if self.feature_matchers:
            for feat in self.target_features:
                if feat in features:
                    target_feature = features[feat]
                    
                    # Interpolate to match patch size
                    patch_size = int(np.sqrt(self.n_patches))
                    interpolated = torch.nn.functional.interpolate(
                        target_feature,
                        size=(patch_size, patch_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    matched_features[feat] = self.feature_matchers[feat](interpolated)
        
        return matched_features