import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
from torchvision import transforms
import lightning as L
from torch.utils.data import DataLoader, random_split

def collate_data_and_cast(samples_list):
    n_global_crops = len(samples_list[0]["global_crops"])
    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    return {
        "collated_global_crops": collated_global_crops,
    }

class CustomDataset(Dataset):
    def __init__(self, img_dir: str = "/home/arda/.cache/kagglehub/datasets/ardaerendoru/gtagta/versions/1/GTA5/GTA5/images", transform: Optional[transforms.Compose] = None):

        self.img_dir = img_dir
        self.transform = transform
        
        # Get all image files
        self.images = []
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                self.images.append(os.path.join(self.img_dir, img_name))
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image
    


class CustomDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        transform,
        val_data_dir: Optional[str] = None,  # Make val_data_dir optional
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.99
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir  # Store val_data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        train_dataset = CustomDataset(img_dir=self.train_data_dir, transform=self.transform)
        
        if self.val_data_dir is None:  # Check if val_data_dir is not provided
            train_size = int(self.train_val_split * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, 
                [train_size, val_size]
            )
        else:
            self.val_dataset = CustomDataset(img_dir=self.val_data_dir, transform=self.transform)  # Use val_data_dir
            self.train_dataset = train_dataset  # Direct assignment

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_data_and_cast
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data_and_cast
        )
