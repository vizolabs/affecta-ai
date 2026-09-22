"""Dataset loading and preprocessing for facial expression recognition."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

from src.core.config import EMOTIONS, DATA_DIR


class FERDataset(Dataset):
    """Facial Expression Recognition dataset.
    
    Supports FER2013, RAF-DB, and similar datasets with
    folder-based structure:
        data/
            train/
                angry/
                disgust/
                fear/
                happy/
                sad/
                surprise/
                neutral/
            val/
                ...
            test/
                ...
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            image_size: Target image size
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.image_size = image_size
        
        # Default transforms
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:
            self.transform = transform
        
        # Load samples
        self.samples = []
        self.emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
        
        for emotion in EMOTIONS:
            emotion_dir = self.data_dir / emotion
            if emotion_dir.exists():
                for img_path in emotion_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.emotion_to_idx[emotion]))
                for img_path in emotion_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.emotion_to_idx[emotion]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample.
        
        Returns:
            (image_tensor, emotion_label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = False,
    augmentation: str = 'baseline',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (disable for MPS)
        augmentation: 'baseline' (legacy weak) or 'strong' (RandAugment etc.)
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from ml.data.augmentation import build_train_transform, build_eval_transform

    if augmentation == 'baseline':
        train_transform = None  # use FERDataset's built-in weak aug
    else:
        train_transform = build_train_transform(image_size, augmentation=augmentation)

    train_dataset = FERDataset(
        data_dir, split='train', image_size=image_size, transform=train_transform
    )
    val_dataset = FERDataset(data_dir, split='val', image_size=image_size)
    test_dataset = FERDataset(data_dir, split='test', image_size=image_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_sample_weights(dataset: FERDataset) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.
    
    Returns:
        Class weights tensor
    """
    class_counts = np.zeros(len(EMOTIONS))
    
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(EMOTIONS)
    
    return torch.FloatTensor(weights)
