"""Action Unit detector model."""

import torch
import torch.nn as nn
import torchvision.models as models

from src.core.config import NUM_AUS


class AUDetector(nn.Module):
    """Multi-label Action Unit detector.
    
    Architecture:
        ResNet-18 (pretrained) → Linear → Sigmoid (per AU)
    
    Input: [B, 3, 224, 224]
    Output: [B, 6] sigmoid activations (one per AU)
    
    Detected AUs:
        AU4: Brow Lowerer
        AU6: Cheek Raiser
        AU7: Lid Tightener
        AU9: Nose Wrinkler
        AU10: Upper Lip Raiser
        AU12: Lip Corner Puller
    """
    
    def __init__(self, num_aus: int = NUM_AUS, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-18
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_aus)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
        
        Returns:
            AU activations [B, 6] (sigmoid, 0-1 per AU)
        """
        logits = self.backbone(x)
        activations = torch.sigmoid(logits)
        return activations
    
    def get_evidence(self, activations: torch.Tensor, emotion: str) -> float:
        """Compute AU evidence score for a specific emotion.
        
        Args:
            activations: AU activations [B, 6]
            emotion: Emotion label
        
        Returns:
            Evidence score (0-1)
        """
        from src.core.config import AU_TO_EMOTION
        
        au_weights = AU_TO_EMOTION.get(emotion, {})
        if not au_weights:
            return 0.0
        
        evidence = 0.0
        au_list = ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12']
        
        for au_name, weight in au_weights.items():
            au_idx = au_list.index(au_name)
            evidence += weight * activations[0, au_idx].item()
        
        return evidence


def get_au_detector(pretrained: bool = True) -> AUDetector:
    """Factory function to get AU detector."""
    return AUDetector(pretrained=pretrained)
