"""Expression intensity estimator."""

import torch
import torch.nn as nn

from src.core.config import NUM_AUS


class IntensityEstimator(nn.Module):
    """Expression intensity estimator.
    
    Estimates the strength of a facial expression (0-100),
    independent of classification confidence.
    
    Input: [expression_features, AU_activations]
    Output: Scalar intensity (0-100)
    """
    
    def __init__(self, feature_dim: int = 512, num_aus: int = NUM_AUS):
        super().__init__()
        
        input_dim = feature_dim + num_aus
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, au_activations: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Expression features [B, feature_dim]
            au_activations: AU activations [B, num_aus]
        
        Returns:
            Intensity score [B, 1] (0-100 after scaling)
        """
        x = torch.cat([features, au_activations], dim=1)
        intensity = self.fc(x) * 100
        return intensity
    
    @staticmethod
    def get_category(intensity: float) -> str:
        """Get intensity category.
        
        Args:
            intensity: Intensity score (0-100)
        
        Returns:
            Category: 'low', 'medium', or 'high'
        """
        if intensity < 33:
            return 'low'
        elif intensity < 67:
            return 'medium'
        else:
            return 'high'


def get_intensity_estimator(feature_dim: int = 512) -> IntensityEstimator:
    """Factory function to get intensity estimator."""
    return IntensityEstimator(feature_dim=feature_dim)
