"""Expression classifier models - VGG16 baseline and candidates."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List

from src.core.config import NUM_EMOTIONS, VGG16_INPUT_SIZE, RESNET_INPUT_SIZE


class StagedBackbone(nn.Module):
    """Container grouping a backbone's layers into named blocks for staged fine-tuning.

    Exposes the same API as VGG16Features so the staged trainer works across
    backbones:
        - .blocks: nn.ModuleList of stage blocks (block1..block5)
        - .forward: feeds input through all blocks in order
        - freeze_all / unfreeze_block
    """

    def __init__(self, blocks: List[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        for i, block in enumerate(self.blocks):
            setattr(self, f'block{i + 1}', block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def freeze_block(self, block_idx: int):
        if 0 <= block_idx < len(self.blocks):
            for param in self.blocks[block_idx].parameters():
                param.requires_grad = False

    def unfreeze_block(self, block_idx: int):
        if 0 <= block_idx < len(self.blocks):
            for param in self.blocks[block_idx].parameters():
                param.requires_grad = True

    def freeze_all(self):
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ExpressionHead(nn.Module):
    """Classification head for expression recognition.
    
    Standard VGG-style head:
        Linear(25088) -> ReLU -> Dropout -> Linear(4096) -> ReLU -> Dropout -> Linear(num_classes)
    """
    
    def __init__(self, in_features: int = 25088, hidden_dim: int = 4096, num_classes: int = NUM_EMOTIONS, dropout: float = 0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class VGG16Features(nn.Module):
    """VGG16 feature extractor with named blocks.
    
    Block structure (matching standard VGG16):
        block1: Conv(3->64) -> ReLU -> Conv(64->64) -> ReLU -> MaxPool
        block2: Conv(64->128) -> ReLU -> Conv(128->128) -> ReLU -> MaxPool
        block3: Conv(128->256) -> ReLU -> Conv(256->256) -> ReLU -> Conv(256->256) -> ReLU -> MaxPool
        block4: Conv(256->512) -> ReLU -> Conv(512->512) -> ReLU -> Conv(512->512) -> ReLU -> MaxPool
        block5: Conv(512->512) -> ReLU -> Conv(512->512) -> ReLU -> Conv(512->512) -> ReLU -> MaxPool
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained VGG16
        if pretrained:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = models.vgg16(weights=None)
        
        self.features = vgg.features
        
        # Define named blocks (using standard VGG16 block boundaries)
        self.block1 = self.features[0:5]   # Conv(3->64)x2 + MaxPool
        self.block2 = self.features[5:10]  # Conv(64->128)x2 + MaxPool
        self.block3 = self.features[10:17] # Conv(128->256)x3 + MaxPool
        self.block4 = self.features[17:24] # Conv(256->512)x3 + MaxPool
        self.block5 = self.features[24:31] # Conv(512->512)x3 + MaxPool
        
        # Store for easy access
        self.blocks = nn.ModuleList([
            self.block1, self.block2, self.block3, self.block4, self.block5
        ])
        
        # Hook for LRP
        self._feature_hook = None
        self._intermediate_features = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through all blocks."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    
    def get_intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features after block5 (for LRP)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        self._intermediate_features = x
        return x
    
    def freeze_block(self, block_idx: int):
        """Freeze a specific block (0-indexed: 0=block1, 4=block5)."""
        if 0 <= block_idx < len(self.blocks):
            for param in self.blocks[block_idx].parameters():
                param.requires_grad = False
    
    def unfreeze_block(self, block_idx: int):
        """Unfreeze a specific block."""
        if 0 <= block_idx < len(self.blocks):
            for param in self.blocks[block_idx].parameters():
                param.requires_grad = True
    
    def freeze_all(self):
        """Freeze all blocks."""
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all blocks."""
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VGG16ExpressionClassifier(nn.Module):
    """VGG16-based expression classifier with explicit head and named blocks.
    
    Architecture:
        VGG16Features (5 blocks) -> ExpressionHead -> 7-class logits
    
    Supports staged unfreezing for transfer learning.
    """
    
    def __init__(self, num_classes: int = NUM_EMOTIONS, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        self.features = VGG16Features(pretrained=pretrained)
        self.head = ExpressionHead(
            in_features=25088,
            hidden_dim=4096,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # For LRP
        self._intermediate_features = None
        self._register_lrp_hook()
    
    def _register_lrp_hook(self):
        """Register hook on block5 for LRP."""
        def hook(module, input, output):
            self._intermediate_features = output
        
        # Hook on the last block
        self.features.block5.register_forward_hook(hook)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
        
        Returns:
            Logits [B, 7]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features (after block5, before head) for LRP."""
        x = self.features.get_intermediate_features(x)
        return x
    
    def freeze_backbone(self):
        """Freeze all convolutional blocks."""
        self.features.freeze_all()
        # Head remains trainable
        for param in self.head.parameters():
            param.requires_grad = True
    
    def unfreeze_block(self, block_idx: int):
        """Unfreeze a specific block (0=block1, 4=block5)."""
        self.features.unfreeze_block(block_idx)
    
    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze last n blocks (from block5 backwards)."""
        for i in range(5-n, 5):
            self.features.unfreeze_block(i)
    
    def get_head_params(self):
        """Get head parameters for optimizer."""
        return self.head.parameters()
    
    def get_backbone_params(self):
        """Get all backbone parameters."""
        return self.features.parameters()
    
    def get_block_params(self, block_idx: int):
        """Get parameters for a specific block."""
        if 0 <= block_idx < 5:
            return self.features.blocks[block_idx].parameters()
        return []


class ResNet50ExpressionClassifier(nn.Module):
    """ResNet50-based expression classifier with staged fine-tuning (candidate).

    Block structure (0-indexed, matching the staged trainer):
        block1: conv1 + bn1 + relu + maxpool   (input stem)
        block2: layer1
        block3: layer2
        block4: layer3
        block5: layer4                          (last residual stage)
        head  : Linear(2048 -> num_classes)
    """

    def __init__(self, num_classes: int = NUM_EMOTIONS, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = models.resnet50(weights=None)

        stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.features = StagedBackbone([
            stem, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(backbone.fc.in_features, num_classes)

        # For LRP (populated by get_features)
        self._intermediate_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features after the final residual stage (for LRP)."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self._intermediate_features = x
        return x

    def freeze_backbone(self):
        """Freeze all convolutional blocks (head remains trainable)."""
        self.features.freeze_all()
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_block(self, block_idx: int):
        """Unfreeze a specific block (0=block1, 4=block5)."""
        self.features.unfreeze_block(block_idx)

    def get_block_params(self, block_idx: int):
        if 0 <= block_idx < 5:
            return self.features.blocks[block_idx].parameters()
        return []

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientNetB2ExpressionClassifier(nn.Module):
    """EfficientNet-B2 expression classifier with staged fine-tuning (candidate).

    Block structure (0-indexed, grouping torchvision's sequential features):
        block1: features[0]                      (stem conv)
        block2: features[1]                      (first MBConv stage)
        block3: features[2:4]                    (stages 2-3)
        block4: features[4:6]                    (stages 4-5)
        block5: features[6:9]                    (stages 6-7 + final conv)
        head  : Linear(1408 -> num_classes)
    """

    def __init__(self, num_classes: int = NUM_EMOTIONS, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()

        if pretrained:
            backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
            )
        else:
            backbone = models.efficientnet_b2(weights=None)

        feats = backbone.features
        self.features = StagedBackbone([
            feats[0],
            feats[1],
            nn.Sequential(feats[2], feats[3]),
            nn.Sequential(feats[4], feats[5]),
            nn.Sequential(feats[6], feats[7], feats[8]),
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(backbone.classifier[1].in_features, num_classes)

        # For LRP (populated by get_features)
        self._intermediate_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features after the final conv block (for LRP)."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self._intermediate_features = x
        return x

    def freeze_backbone(self):
        """Freeze all convolutional blocks (head remains trainable)."""
        self.features.freeze_all()
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_block(self, block_idx: int):
        """Unfreeze a specific block."""
        self.features.unfreeze_block(block_idx)

    def get_block_params(self, block_idx: int):
        if 0 <= block_idx < 5:
            return self.features.blocks[block_idx].parameters()
        return []

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_expression_model(backbone: str = "vgg16", pretrained: bool = True, **kwargs) -> nn.Module:
    """Factory function to get expression classifier.
    
    Args:
        backbone: One of 'vgg16', 'resnet50', 'efficientnet_b2'
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments (dropout, etc.)
    
    Returns:
        Expression classifier model
    """
    models_dict = {
        'vgg16': VGG16ExpressionClassifier,
        'resnet50': ResNet50ExpressionClassifier,
        'efficientnet_b2': EfficientNetB2ExpressionClassifier,
    }
    
    if backbone not in models_dict:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(models_dict.keys())}")
    
    return models_dict[backbone](pretrained=pretrained, **kwargs)