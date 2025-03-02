import lightning as L
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import tempfile
import torch.nn as nn
import logging
from losses import ScaleKD, DinoiserLoss
# Create a temporary directory in your home or storage
USER_TMP = '/storage/disk0/arda/tmp'
os.makedirs(USER_TMP, exist_ok=True)

# Set multiple environment variables to ensure temp files go to the right place
os.environ['TMPDIR'] = USER_TMP
os.environ['TEMP'] = USER_TMP
os.environ['TMP'] = USER_TMP
tempfile.tempdir = USER_TMP

LOSS_REGISTRY = {
    'scalekd': ScaleKD,
    'dinoiser': DinoiserLoss
}

class DistillationModule(L.LightningModule):
    def __init__(
        self,
        student,
        teacher,
        cfg
    ):
        super().__init__()
        
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self._initialize_models(student, teacher)
        self._initialize_loss()
        
        self.loss_mse = nn.MSELoss(reduction='sum')
    def _initialize_models(self, student, teacher):
        """Initialize models with gradient verification."""
        self.student = student
        self.teacher = teacher
        self.register_module('student', self.student)
        self.register_module('teacher', self.teacher)



        
        # Freeze teacher
        self._freeze_teacher()
        
        # Load checkpoint if specified
        if self.cfg.student.get('checkpoint_path', None):
            self._load_student_checkpoint(self.cfg.student.checkpoint_path)
                # Assert all student parameters are trainable
        for name, param in self.student.named_parameters():
            assert param.requires_grad, f"Parameter {name} in student model must be trainable"
            

    def _freeze_teacher(self):
        """Freeze teacher model parameters."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        

    def _initialize_loss(self):
        """Initialize compound loss function."""
        self.losses = nn.ModuleDict()  # ModuleDict automatically registers modules
        self.loss_weights = {}
        
        for loss_spec in self.cfg.loss['losses']:
            loss_type = loss_spec['type']
            weight = loss_spec['weight']
            kwargs = loss_spec['kwargs']
            
            # Get loss name based on type or kwargs
            name = kwargs.get('name', loss_type)
            
            # Get loss class from registry and add it to ModuleDict
            loss_fn = LOSS_REGISTRY[loss_type](**kwargs)
            self.losses[name] = loss_fn  # This automatically registers the module
            self.loss_weights[name] = weight
        self.register_module('losses', self.losses)
        # # Debug: Print all registered modules and their parameters
        # for name, module in self.named_modules():
        #     if isinstance(module, nn.Module):
        #         params = [p for p in module.parameters() if p.requires_grad]
        #         if params:
        #             print(f"Module {name} has {len(params)} trainable parameters")

    def _forward_specific_stage(self, feat, percentage):
        #AVG of patch tokens
        """Forward through specific stages of teacher model."""
        n_total_blocks = len(self.teacher.model.blocks)
        target_block = int(n_total_blocks*percentage)
        for i in range(target_block, n_total_blocks):
            feat = self.teacher.model.blocks[i](feat)
        return feat

    def _compute_losses(self, features, *args, **kwargs):
        """Compute compound loss with affinity map loss."""
        total_loss = 0
        loss_dict = {}

        scalekd_n_loss = self.losses['scalekd_n']
        scalekd_last_loss = self.losses['scalekd_last']
        scalekd_n_weight = self.loss_weights['scalekd_n']
        scalekd_last_weight = self.loss_weights['scalekd_last']

        # Project student features and forward through teacher stages
        feat_S_s3_spat = scalekd_n_loss.project_feat_spat(features['student']['res4'], query=None)
        feat_S_s3_spat = self._forward_specific_stage(feat_S_s3_spat, 0.75)

        feat_S_s3_freq = scalekd_n_loss.project_feat_freq(features['student']['res4'], query=None)
        feat_S_s3_freq = self._forward_specific_stage(feat_S_s3_freq, 0.75)

        # Existing ScaleKD losses
        scalekd_n_spat = scalekd_n_loss.get_spat_loss(feat_S_s3_spat, features['teacher'])
        scalekd_n_freq = scalekd_n_loss.get_freq_loss(feat_S_s3_freq, features['teacher'])
        scalekd_last_dict = scalekd_last_loss(features['student']['res5'], features['teacher'], 
                                            query_s=feat_S_s3_spat, 
                                            query_f=feat_S_s3_freq)

        # Compute affinity maps
        N,C,H,W = features['teacher'].shape
        scalekd_last_spat_feat = scalekd_last_loss.project_feat_spat(features['student']['res5'], query = feat_S_s3_spat).permute(0, 2, 1).contiguous().view(N, C, H, W)
        teacher_affinity = scalekd_n_loss._compute_affinity_map(features['teacher'])
        student_affinity = scalekd_n_loss._compute_affinity_map(scalekd_last_spat_feat)

        # Compute affinity loss (e.g., MSE)
        affinity_loss = self.loss_mse(student_affinity, teacher_affinity)/scalekd_last_spat_feat.shape[0]
        affinity_weight = 15  # Configurable via cfg.loss.losses

        # Update total loss
        total_loss += (scalekd_n_spat[0] + scalekd_n_freq) * scalekd_n_weight
        total_loss += scalekd_last_dict['loss'] * scalekd_last_weight
        total_loss += affinity_loss * affinity_weight

        # Update loss dictionary
        loss_dict['loss_scalekd_n_spat'] = scalekd_n_spat[0] * scalekd_n_weight
        loss_dict['loss_scalekd_n_freq'] = scalekd_n_freq * scalekd_n_weight
        loss_dict['loss_scalekd_n_similarity'] = scalekd_n_spat[1] * scalekd_n_weight
        loss_dict['loss_scalekd_last'] = scalekd_last_dict['loss'] * scalekd_last_weight
        loss_dict['loss_scalekd_last_similarity'] = scalekd_last_dict['cosine_similarity'] * scalekd_last_weight
        loss_dict['loss_scalekd_last_spatial_loss'] = scalekd_last_dict['spatial_loss'] * scalekd_last_weight
        loss_dict['loss_scalekd_last_frequency_loss'] = scalekd_last_dict['frequency_loss'] * scalekd_last_weight
        loss_dict['loss_affinity'] = affinity_loss * affinity_weight

        loss_dict['loss'] = total_loss
        return loss_dict
    def training_step(self, batch, batch_idx):
        """Training step with detailed gradient debugging."""

        
        # Get features with gradient checking
        features = self._extract_features(batch)

        # Compute losses with gradient tracking
        losses = self._compute_losses(features)
        
        self._log_training_metrics(losses, features)

        return losses['loss']
    

    def validation_step(self, batch, batch_idx):
        features = self._extract_features(batch)
        losses = self._compute_losses(features)
        self._log_validation_metrics(losses, features)

    def _extract_features(self, batch):
        """Extract features from both models."""
        global_crops = batch["collated_global_crops"]
        
        with torch.no_grad():
            teacher_output = self.teacher(global_crops)
            teacher_features = teacher_output[self.cfg.teacher.teacher_key]

        student_output = self.student(global_crops)
        return {
            'student': student_output,
            'teacher': teacher_features
        }

    def _log_training_metrics(self, losses, features):
        """Log training metrics."""
        # Log all your training losses and metrics here
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, sync_dist=True)
        

    def _log_validation_metrics(self, losses, features):
        """Log validation metrics."""
        # Log all your validation losses and metrics here
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, sync_dist=True)


    @staticmethod
    def _compute_feature_similarity(feat1, feat2):
        """Compute cosine similarity between feature vectors."""
        feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
        similarity = F.cosine_similarity(feat1, feat2, dim=1)
        return similarity.mean()

    def _load_student_checkpoint(self, checkpoint_path):
        """Load student checkpoint, update state dict and log load info via the Lightning logger."""
        checkpoint = torch.load(checkpoint_path)
        
        # Process the checkpoint based on model name configuration.
        if self.cfg.student.model_name == 'stdc':
            checkpoint = {f"model.model.{k.replace('cp.backbone.', '')}": v for k, v in checkpoint.items()}
            result = self.student.load_state_dict(checkpoint, strict=False)
        elif 'resnet' in self.cfg.student.model_name:
            checkpoint = {f"model.model.{k}": v for k, v in checkpoint.items()}
            result = self.student.load_state_dict(checkpoint, strict=False)
        else:
            # Default loading behavior if no specific model name match.
            result = self.student.load_state_dict(checkpoint, strict=False)
        
        # Log checkpoint load details using the Lightning logger.
        if self.logger:
            self.logger.info(f"Loading student checkpoint from: {checkpoint_path}")
            self.logger.info(f"Missing keys: {result.missing_keys}")
            self.logger.info(f"Unexpected keys: {result.unexpected_keys}")
        else:
            print("Logger is not configured. Falling back to print:")
            print(f"Loading student checkpoint from: {checkpoint_path}")
            print(f"Missing keys: {result.missing_keys}")
            print(f"Unexpected keys: {result.unexpected_keys}")
        
            
    def configure_optimizers(self):
        """Configure optimizers with flexible optimizer and scheduler options."""
        # Collect parameters from both student and losses
        param_groups = []
        
        # Student parameters
        student_params = list(self.student.parameters())
        param_groups.append({
            'params': student_params,
            'name': 'student'
        })
        
        # Loss function parameters
        for loss_name, loss_module in self.losses.items():
            loss_params = list(loss_module.parameters())
            if loss_params:  # Only add if there are parameters
                param_groups.append({
                    'params': loss_params,
                    'name': f'loss_{loss_name}'
                })
                # print(f"Added {len(loss_params)} parameters from {loss_name} loss")

        optimizer = getattr(torch.optim, self.cfg['optimizer']['type'])(
            param_groups,
            **self.cfg['optimizer'].get('kwargs', {})
        )
        
        # Configure scheduler if specified
        if 'scheduler' in self.cfg['optimizer']:
            scheduler = getattr(torch.optim.lr_scheduler, 
                            self.cfg['optimizer']['scheduler']['type'])(
                optimizer,
                **self.cfg['optimizer']['scheduler'].get('kwargs', {})
            )
            
            return {
                "optimizer": optimizer,
                
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg['optimizer']['scheduler'].get('monitor', 'val_loss'),
                    "interval": self.cfg['optimizer']['scheduler'].get('interval', 'epoch'),
                    "frequency": self.cfg['optimizer']['scheduler'].get('frequency', 1)
                }
            }
        
        return optimizer

