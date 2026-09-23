"""Production model zoo: load trained classifiers with a unified class order.

All models exposed through :func:`load_production` emit logits in the project
class order ``src.core.config.EMOTIONS`` (alphabetical), regardless of the
architecture's native training order:

- DDAMFN++ native: [neutral, happy, sad, surprise, fear, disgust, angry]
- DAN native:      [surprise, fear, disgust, happy, sad, anger, neutral]
- timm/torchvision models: already alphabetical (trained by this repo)

Checkpoint paths default to the local experiments/ + data/raw layout and can
be overridden per call.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.core.config import EMOTIONS, NUM_EMOTIONS, PROJECT_ROOT

# Native (training-time) class order per architecture.
DDAMFN_NATIVE = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']
DAN_NATIVE = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']

DEFAULT_CKPTS = {
    'ddamfn': str(PROJECT_ROOT / 'experiments/EXP-009-DDAMFN-PP/checkpoints/rafdb_epoch20_acc0.9204_bacc0.8617.pth'),
    'dan': str(PROJECT_ROOT / 'data/raw/dan_weights/RAF-DB.pth'),
    'vit_small': str(PROJECT_ROOT / 'experiments/EXP-008-VIT-SMALL/checkpoints/best.pt'),
    'efficientnet_b3': str(PROJECT_ROOT / 'experiments/EXP-007-EFFICIENTNET-B3/checkpoints/best.pt'),
}

INPUT_SIZES: Dict[str, int] = {
    'ddamfn': 112,
    'dan': 224,
    'vit_small': 224,
    'efficientnet_b3': 300,
}


def _gather(native_order) -> torch.Tensor:
    """Indices such that logits_project = logits_native[:, gather]."""
    lookup = {name.replace('anger', 'angry'): i for i, name in enumerate(native_order)}
    return torch.tensor([lookup[emotion] for emotion in EMOTIONS], dtype=torch.long)


class ProductionClassifier(nn.Module):
    """Wraps a native model; forward returns logits in EMOTIONS order.

    Returns a plain tensor (not a tuple) so Grad-CAM / LRP engines and the
    standard CE loss work unchanged.
    """

    def __init__(self, native: nn.Module, native_order, input_size: int, name: str):
        super().__init__()
        self.native = native
        self.input_size = input_size
        self.name = name
        self.num_classes = NUM_EMOTIONS
        self.register_buffer('gather', _gather(native_order))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.native(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out[:, self.gather]


def _load_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'model', 'net'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(f'Unrecognized checkpoint structure: {ckpt_path}')


def load_production(
    name: str,
    ckpt_path: Optional[str] = None,
    device: Optional[str] = None,
) -> ProductionClassifier:
    """Load a production classifier by name: ddamfn | dan | vit_small | efficientnet_b3."""
    name = name.lower()
    if name not in DEFAULT_CKPTS:
        raise ValueError(f'Unknown model {name!r}; choose from {sorted(DEFAULT_CKPTS)}')
    ckpt_path = ckpt_path or DEFAULT_CKPTS[name]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    if name == 'ddamfn':
        from ml.models.ddam import DDAMNet
        native = DDAMNet(num_class=7, num_head=2, pretrained=False)
        native.load_state_dict(_load_state(ckpt_path), strict=True)
        model = ProductionClassifier(native, DDAMFN_NATIVE, 112, name)
    elif name == 'dan':
        from ml.models.dan import DAN
        native = DAN(num_class=7, num_head=4, pretrained=False)
        native.load_state_dict(_load_state(ckpt_path), strict=True)
        model = ProductionClassifier(native, DAN_NATIVE, 224, name)
    else:
        from ml.models.backbones import build_expression_model
        backbone = 'vit_small_patch16_224' if name == 'vit_small' else name
        native = build_expression_model(backbone, pretrained=False, init_weights=ckpt_path)
        model = ProductionClassifier(native, list(EMOTIONS), INPUT_SIZES[name], name)

    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else (
            'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


def build_transform(name: str):
    """Evaluation transform matching each architecture's training recipe."""
    from torchvision import transforms
    size = INPUT_SIZES[name.lower()]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
