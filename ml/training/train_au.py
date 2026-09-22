"""Train Action Unit detector model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import json
from datetime import datetime
from typing import Tuple, List

from src.core.config import ACTION_UNITS, MODELS_DIR
from src.models.au_detector import AUDetector


class AUDataset(Dataset):
    """Action Unit detection dataset.
    
    Expected structure:
        data/
            images/
                subject_001_frame_0001.jpg
            labels/
                subject_001_frame_0001.json
                    {"AU4": 0, "AU6": 1, ...}
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load samples
        self.samples = []
        labels_dir = self.data_dir / 'labels'
        
        if labels_dir.exists():
            for label_file in labels_dir.glob('*.json'):
                with open(label_file) as f:
                    labels = json.load(f)
                
                img_path = self.data_dir / 'images' / f"{label_file.stem}.jpg"
                if img_path.exists():
                    au_labels = [labels.get(au, 0) for au in ACTION_UNITS]
                    self.samples.append((str(img_path), au_labels))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, au_labels = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = torch.FloatTensor(image.transpose(2, 0, 1))
        
        # AU labels
        au_tensor = torch.FloatTensor(au_labels)
        
        return image, au_tensor


def train_au_model(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3
) -> dict:
    """Train AU detector.
    
    Args:
        data_dir: Path to AU dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Training results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    train_dataset = AUDataset(data_dir, split='train')
    val_dataset = AUDataset(data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = AUDetector(pretrained=True).to(device)
    
    # Loss (BCE for multi-label)
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    print(f"\nTraining AU detector")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for images, au_labels in train_loader:
            images = images.to(device)
            au_labels = au_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, au_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, au_labels in val_loader:
                images = images.to(device)
                au_labels = au_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, au_labels)
                
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(au_labels.cpu())
        
        val_loss /= len(val_loader)
        
        # Compute F1 per AU
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        f1_scores = []
        for i in range(len(ACTION_UNITS)):
            tp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum()
            fp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
            fn = ((all_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            f1_scores.append(f1.item())
        
        mean_f1 = np.mean(f1_scores)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(mean_f1)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f}, F1={mean_f1:.4f}")
        
        # Save best model
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            save_path = MODELS_DIR / "au_detector_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    print(f"\nTraining complete! Best F1: {best_val_f1:.4f}")
    
    results = {
        'best_val_f1': best_val_f1,
        'f1_per_au': dict(zip(ACTION_UNITS, f1_scores)),
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/au"
    results = train_au_model(data_dir)
