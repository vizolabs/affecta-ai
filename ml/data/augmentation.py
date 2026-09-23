"""Strong augmentation pipeline for accuracy-first fine-tuning.

baseline : legacy weak aug (flip / color jitter / rotate) - matches EXP-004/005.
strong   : RandAugment + RandomResizedCrop + RandomErasing + flip/jitter.
mixup / cutmix operate on batch tensors inside the training loop (see
train_expression.train_one_epoch); they return soft-target metadata.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int, augmentation: str = 'baseline') -> transforms.Compose:
    if augmentation == 'baseline':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if augmentation == 'strong':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    raise ValueError(f'Unknown augmentation preset: {augmentation}')


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _TenCropFlipViews:
    """Picklable 10-view TTA transform: 5-crop + horizontal flips.

    Implemented as a module-level class so it survives multiprocessing with
    spawn workers (nested closures cannot be pickled).
    """

    def __init__(self, image_size: int):
        self.larger = int(round(image_size * 1.14))
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, img):
        resized = transforms.Resize((self.larger, self.larger))(img)
        # torchvision TenCrop may yield 10 (incl. vertical flips); keep standard
        # 5 geometric crops (TL,TR,BL,BR,C) + their hflips = 10 views.
        crops = list(transforms.TenCrop(self.image_size)(resized))[:5]
        tensors = [self.to_tensor(c) for c in crops]
        views = [self.normalize(t) for t in tensors]
        views += [self.normalize(transforms.functional.hflip(t)) for t in tensors]
        return torch.stack(views, dim=0)


def ten_crop_flip_views(image_size: int) -> _TenCropFlipViews:
    """Return transform producing a (10, 3, S, S) tensor: 5-crop + horizontal flips.

    TenCrop yields 5 PIL crops; we append their hflips for 10 total views.
    """
    return _TenCropFlipViews(image_size)


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp: returns (mixed_images, target_a, target_b, lam)."""
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch = images.size(0)
    index = torch.randperm(batch, device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix: returns (mixed_images, target_a, target_b, lam) with lam = area ratio of image A."""
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch, _, h, w = images.shape
    index = torch.randperm(batch, device=images.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda to exact cut area so the loss weighting matches pixels.
    cut_area = (y2 - y1) * (x2 - x1)
    lam = 1.0 - cut_area / float(h * w)
    return mixed, targets, targets[index], float(lam)
