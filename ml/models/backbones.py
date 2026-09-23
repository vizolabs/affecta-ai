"""Unified backbone loader: torchvision baselines + timm ImageNet-pretrained models.

Backwards-compatible factory for the existing staged trainer API:
    - model.features.blocks : nn.ModuleList of 5 stage groups (0=early .. 4=late)
    - model.head            : classifier parameters
    - freeze_backbone / unfreeze_block / get_features

timm models are grouped into 5 stages so STAGES in train_expression.py works
unchanged. Weights download once on first use and are cached (torch hub /
HuggingFace); after that runs are fully offline.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from src.core.config import NUM_EMOTIONS
from src.models.expression_classifier import get_expression_model


# Input resolution per backbone (used by trainer + latency bench).
INPUT_SIZES = {
    'vgg16': 224,
    'resnet50': 224,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'vit_small_patch16_224': 224,
    'vit_base_patch16_224': 224,
}
DEFAULT_INPUT_SIZE = 224

TORCHVISION_BACKBONES = ('vgg16', 'resnet50', 'efficientnet_b2')


def input_size_for(backbone: str) -> int:
    return INPUT_SIZES.get(backbone, DEFAULT_INPUT_SIZE)


def _group_modules(modules: List[nn.Module], n_groups: int = 5) -> List[nn.Sequential]:
    """Split an ordered module list into n_groups near-equal Sequential groups."""
    n = len(modules)
    if n == 0:
        return [nn.Sequential() for _ in range(n_groups)]
    if n <= n_groups:
        groups: List[nn.Sequential] = []
        idx = 0
        for i in range(n_groups):
            take = 1 if idx < n else 0
            groups.append(nn.Sequential(*modules[idx:idx + take]))
            idx += take
        return groups
    base, extra = divmod(n, n_groups)
    groups = []
    idx = 0
    for i in range(n_groups):
        take = base + (1 if i < extra else 0)
        groups.append(nn.Sequential(*modules[idx:idx + take]))
        idx += take
    return groups


class TimmStagedBackbone(nn.Module):
    """Ordered stage groups with the same freeze API as StagedBackbone."""

    def __init__(self, groups: List[nn.Sequential]):
        super().__init__()
        self.blocks = nn.ModuleList(groups)
        for i, g in enumerate(self.blocks):
            setattr(self, f'block{i + 1}', g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for g in self.blocks:
            x = g(x)
        return x

    def freeze_block(self, block_idx: int):
        if 0 <= block_idx < len(self.blocks):
            for p in self.blocks[block_idx].parameters():
                p.requires_grad = False

    def unfreeze_block(self, block_idx: int):
        if 0 <= block_idx < len(self.blocks):
            for p in self.blocks[block_idx].parameters():
                p.requires_grad = True

    def freeze_all(self):
        for g in self.blocks:
            for p in g.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for g in self.blocks:
            for p in g.parameters():
                p.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TimmExpressionClassifier(nn.Module):
    """timm backbone + linear head, staged into 5 fine-tuning groups.

    forward_features path is reconstructed as: stem -> stage groups -> pool
    -> head so that freezing stages matches the existing trainer contract.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_classes: int = NUM_EMOTIONS,
        dropout: float = 0.5,
        init_weights: Optional[str] = None,
    ):
        super().__init__()
        import timm

        net = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = int(net.num_features)

        post = nn.Identity()
        if hasattr(net, 'blocks') and hasattr(net, 'conv_stem'):
            # ConvNet family (EfficientNet-style stem + blocks).
            # timm >=1.0: bn1 is BatchNormAct2d (activation baked in, no .act1).
            # conv_head+bn2 sit after blocks in forward_features and expand to
            # num_features (e.g. 384 -> 1536 for EffNet-B3) — must run post-blocks.
            stem_mods = [net.conv_stem]
            if hasattr(net, 'bn1'):
                stem_mods.append(net.bn1)
            if hasattr(net, 'act1') and net.act1 is not None:
                stem_mods.append(net.act1)
            stem = nn.Sequential(*stem_mods)
            raw = list(net.blocks)
            groups = _group_modules(raw, 5)
            groups[0] = nn.Sequential(stem, groups[0])
            pool = net.global_pool
            # Append conv_head+bn2 to the LAST stage so freeze/unfreeze covers them.
            if hasattr(net, 'conv_head') and hasattr(net, 'bn2'):
                groups[-1] = nn.Sequential(*groups[-1], net.conv_head, net.bn2)
            post = nn.Identity()
        elif hasattr(net, 'blocks') and hasattr(net, 'patch_embed'):
            # ViT family: patch_embed is stage-0 stem; pos_embed added via wrapper
            stem = _ViTStem(net.patch_embed, getattr(net, 'pos_embed', None),
                            getattr(net, 'cls_token', None))
            groups = _group_modules(list(net.blocks), 5)
            groups[0] = nn.Sequential(stem, groups[0])
            pool = nn.Identity()  # ViT blocks keep token dim; flatten in forward
            norm = getattr(net, 'norm', None)
            if norm is not None and not isinstance(norm, nn.Identity):
                groups[-1] = nn.Sequential(*groups[-1], norm)
            post = nn.Identity()
            self._vit = True
        else:
            raise ValueError(
                f'Unsupported timm architecture for staging: {model_name} '
                '(need .blocks with conv stem or patch_embed)'
            )

        if not hasattr(self, '_vit'):
            self._vit = False

        self.model_name = model_name
        self.features = TimmStagedBackbone(groups)
        self.pool = pool
        self.post = post
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self._intermediate_features = None

        # Drop the source net so stage groups are the sole owners (no dup state_dict keys).
        del net

        if init_weights:
            self.load_partial(init_weights)

    def load_partial(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('model_state_dict', 'state_dict', 'model', 'net'):
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
        if not isinstance(ckpt, dict):
            raise ValueError(f'Unrecognized checkpoint structure at {path}')
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        print(
            f'[backbones] init_weights {path}: '
            f'loaded {len(ckpt)} tensors, missing={len(missing)}, unexpected={len(unexpected)}'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.get_features(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self._vit:
            if x.dim() == 3:
                x = x[:, 0]
            x = torch.flatten(x, 1) if x.dim() > 2 else x
        else:
            x = self.pool(x)
            x = torch.flatten(x, 1)
        self._intermediate_features = x
        return x

    def freeze_backbone(self):
        self.features.freeze_all()
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_block(self, block_idx: int):
        self.features.unfreeze_block(block_idx)

    def get_block_params(self, block_idx: int):
        if 0 <= block_idx < len(self.features.blocks):
            return self.features.blocks[block_idx].parameters()
        return []

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _stage_out_channels(seq: nn.Sequential) -> int:
    """Best-effort output channel count of a Sequential stage (for probes)."""
    ch = None
    for m in seq.modules():
        if isinstance(m, nn.Conv2d):
            ch = m.out_channels
    return ch or 1


class _ViTStem(nn.Module):
    """patch_embed + optional cls_token/pos_embed broadcast (runs inside stage 0)."""

    def __init__(self, patch_embed: nn.Module, pos_embed=None, cls_token=None):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.cls_token = cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        return x


def get_timm_expression_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = NUM_EMOTIONS,
    dropout: float = 0.5,
    init_weights: Optional[str] = None,
) -> nn.Module:
    return TimmExpressionClassifier(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        init_weights=init_weights,
    )


def _extract_state_dict(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        raise ValueError('Checkpoint is not a dict')
    for key in ('model_state_dict', 'state_dict', 'model', 'net'):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # Flat state dict (all tensor values)
    if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError('Could not find state dict in checkpoint')


def build_expression_model(
    backbone: str,
    pretrained: bool = True,
    dropout: float = 0.5,
    init_weights: Optional[str] = None,
) -> nn.Module:
    """Single entry point for all backbones (torchvision baselines + timm)."""
    if backbone in TORCHVISION_BACKBONES:
        model = get_expression_model(backbone=backbone, pretrained=pretrained, dropout=dropout)
        if init_weights:
            sd = _extract_state_dict(torch.load(init_weights, map_location='cpu', weights_only=False))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(
                f'[backbones] init_weights {init_weights}: '
                f'missing={len(missing)}, unexpected={len(unexpected)}'
            )
        return model
    return get_timm_expression_model(
        backbone,
        pretrained=pretrained,
        dropout=dropout,
        init_weights=init_weights,
    )
