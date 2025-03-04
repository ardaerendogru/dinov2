import warnings
import sys
import os
from typing import Dict, Any
from dataclasses import dataclass
import argparse
sys.path.append('../')
import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from models import DINOv2ViT
from models.model_wrapper import ModelWrapper
from train.distillation_module import DistillationModule
from distillation.datasets.CustomDataset import CustomDataModule
from dinov2.data.augmentations import DataAugmentationDINO
import tempfile
import wandb
import logging

os.environ["NCCL_P2P_DISABLE"] = "1"
# Create a temporary directory in your home or storage
USER_TMP = '/storage/disk0/arda/tmp'
os.makedirs(USER_TMP, exist_ok=True)

# Set multiple environment variables to ensure temp files go to the right place
os.environ['TMPDIR'] = USER_TMP
os.environ['TEMP'] = USER_TMP
os.environ['TMP'] = USER_TMP
tempfile.tempdir = USER_TMP


logger = logging.getLogger("dinov2_distillation")

@dataclass
class TrainingConfig:
    """Configuration for training setup."""
    max_epochs: int
    precision: int
    learning_rate: float


class DistillationTrainer:
    """Handles the training pipeline for knowledge distillation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = self._handle_config(config)
        self.training_config = self._setup_training_config()
        
        # Initialize components
        self.transform = self._create_transform()
        self.data_module = self._create_data_module()
        self.teacher, self.student = self._create_models()
        self.distillation_module = self._create_distillation_module()
        self.trainer = self._create_trainer()
        self.checkpoint_path = self.cfg.train.get('resume_from_checkpoint', None)
    

    def _handle_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        teacher_dims = {
            'dinov2_vits14' : 384,
            'dinov2_vitb14' : 768,
            'dinov2_vitl14' : 1024,
            'dinov2_vitg14' : 1536

        }

        config.teacher.out_dim = teacher_dims[config.teacher.model_name]
        config.teacher.teacher_key = 'feature_map'
        config.teacher.n_patches = int((config.data_transform.global_crops_size[0]//14)**2)
        config.student.kwargs = {}

        for loss in config.loss.losses:
            if loss.type == 'scalekd':
                # loss['kwargs']['student_dims'] = len(config.student.kwargs.out_features) * 512  # Example calculation
                loss.kwargs.teacher_dims = config.teacher.out_dim
                loss.kwargs.teacher_dims = config.teacher.out_dim
                loss.kwargs.pos_dims = config.teacher.out_dim
                loss.kwargs.pos_hw = [int(config.data_transform.global_crops_size[0]//14),int(config.data_transform.global_crops_size[0]//14)]
                loss.kwargs.query_hw = [int(config.data_transform.global_crops_size[0]//14),int(config.data_transform.global_crops_size[0]//14)]
        if config.student.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            config.student.model_name = 'resnet'
            config.student.kwargs.depth = int(config.student.model_name.replace('resnet', '')) 
            config.student.kwargs.out_features = ['res2', 'res3', 'res4', 'res5']
            config.student.kwargs.freeze_at = 0
            config.student.kwargs.norm_type = 'BN'
        elif config.student.model_name in ['stdc1', 'stdc2']:
            config.student.model_name = 'stdc'
            config.student.kwargs.base_channels = 64
            config.student.kwargs.layers = [2,2,2] if config.student.model_name == 'stdc1' else [4,5,3]
            config.student.kwargs.block_num = 4
            config.student.kwargs.block_type = 'cat'
            config.student.kwargs.use_conv_last = False


        return config
        

    def _setup_training_config(self) -> TrainingConfig:
        """Setup training configuration."""
        return TrainingConfig(
            max_epochs=self.cfg['train']['max_epochs'],
            precision=self.cfg.get('precision', 16),
            learning_rate=self.cfg['optimizer']['kwargs']['lr'],
        )

    def _create_transform(self) -> DataAugmentationDINO:
        """Create data transformation pipeline."""
        return DataAugmentationDINO(
            global_crops_scale=tuple(self.cfg['data_transform']['global_crops_scale']),
            local_crops_scale=tuple(self.cfg['data_transform']['local_crops_scale']),
            local_crops_number=self.cfg['data_transform']['n_local_crops'],
            global_crops_size=tuple(self.cfg['data_transform']['global_crops_size']),
            local_crops_size=tuple(self.cfg['data_transform']['local_crops_size']),
        )

    def _create_data_module(self) -> CustomDataModule:
        """Create data module."""
        return CustomDataModule(
            train_data_dir=self.cfg['data_loader'].get('train_dir', '/home/arda/data/train2017'),
            val_data_dir = self.cfg['data_loader'].get('val_dir', None),
            transform=self.transform,
            batch_size=self.cfg['data_loader']['batch_size'],
            num_workers=self.cfg['data_loader']['num_workers']
        )

    def _create_models(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Create teacher and student models."""
        teacher = DINOv2ViT(
            model_name=self.cfg['teacher']['model_name'],
        )
        student = ModelWrapper(
            model_name=self.cfg['student']['model_name'],
            n_patches=self.cfg.teacher.n_patches,
            target_feature=self.cfg['student']['student_keys'],
            checkpoint_path=self.cfg.student.get('checkpoint_path', None)
            **self.cfg['student']['kwargs']
        )
        for loss in self.cfg.loss.losses:
            if loss.type == 'scalekd':
                loss.kwargs.student_dims = int(student.feature_channels[loss.kwargs.name.split('_')[1]] )
                

        return teacher, student

    def _create_distillation_module(self) -> DistillationModule:
        """Create distillation module."""
        return DistillationModule(
            student=self.student,
            teacher=self.teacher,
            cfg=self.cfg
        )

        
    def _create_trainer(self) -> L.Trainer:
        """Create Lightning trainer."""
        experiment_dir = f"logs/{self.cfg.train.name}"
        wandb_config = OmegaConf.to_container(self.cfg, resolve=True)
        wandb.init(
            config=wandb_config,  # Log config to wandb
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            tags=self.cfg.wandb.tags,
            notes=self.cfg.wandb.notes,
            sync_tensorboard=True
        )
        wandb.define_metric("global_step")


        logger = TensorBoardLogger(experiment_dir, name="distillation",default_hp_metric=False )
        logger.log_hyperparams(self.cfg)

        # Log tensorboard directory to wandb for easy access        
        # Set up checkpoint callback to save in the same experiment directory
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            filename="{epoch}-{val_cosine_similarity:.2f}",
            monitor=self.cfg.checkpoints.monitor,
            mode=self.cfg.checkpoints.mode,
            save_top_k=self.cfg.checkpoints.save_top_k,
            save_last=True
        )

        return L.Trainer(
            default_root_dir='/storage/disk0/arda/tmp',  # Set default root dir
            max_epochs=self.training_config.max_epochs,
            accelerator=self.cfg.train.accelerator,
            devices=self.cfg.train.devices,
            num_nodes=self.cfg.train.num_nodes,
            strategy=self.cfg.train.strategy,
            precision=self.training_config.precision,
            callbacks=[checkpoint_callback],
            logger=logger,
            num_sanity_val_steps=0,
            gradient_clip_val=1.0,  # Example: Clip gradients to a maximum norm of 1.0
            gradient_clip_algorithm="norm", # Optional: "norm" (default) or "value"
            # resume_from_checkpoint=self.cfg.train.get('resume_from_checkpoint', None)
            # accumulate_grad_batches=2

        )

    def train(self):
        """Execute training pipeline."""
        if self.checkpoint_path:
            print(f"Resuming training from checkpoint: {self.checkpoint_path}")
            self.trainer.fit(self.distillation_module, self.data_module, ckpt_path=self.checkpoint_path)
        else:
            print("Starting training from scratch.")
            self.trainer.fit(self.distillation_module, self.data_module)

def setup_environment():
    """Setup environment configurations."""
    # Add the project root to Python path
    sys.path.append('../')
    
    # Configure warnings
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set precision for Tensor Cores
    torch.set_float32_matmul_precision('high')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Training script for distillation')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to the config file'
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration from YAML
    with open(args.config, "r") as f:
        config = OmegaConf.load(f)
    
    trainer = DistillationTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()