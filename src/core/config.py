"""Core configuration for AFFECTA AI."""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Emotion classes (alphabetical order matching directory structure)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_EMOTIONS = len(EMOTIONS)

# Action Units
ACTION_UNITS = ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12']
NUM_AUS = len(ACTION_UNITS)

# AU-to-Emotion evidence mapping
AU_TO_EMOTION = {
    'angry': {'AU4': 0.4, 'AU7': 0.4, 'AU6': 0.2},
    'disgust': {'AU9': 0.5, 'AU10': 0.3, 'AU6': 0.2},
    'fear': {'AU4': 0.5, 'AU6': 0.3, 'AU12': 0.2},
    'happy': {'AU6': 0.5, 'AU12': 0.5},
    'neutral': {},
    'sad': {'AU4': 0.7, 'AU6': 0.3},
    'surprise': {'AU6': 0.4, 'AU12': 0.4, 'AU4': 0.2},
}

# Facial regions for LRP aggregation
FACIAL_REGIONS = ['mouth', 'cheeks', 'eyes', 'eyebrows', 'nose']

# Model input sizes
VGG16_INPUT_SIZE = 224
EFFICIENTNET_INPUT_SIZE = 260
RESNET_INPUT_SIZE = 224


@dataclass
class ModelConfig:
    """Configuration for model training."""
    backbone: str = "vgg16"
    num_classes: int = NUM_EMOTIONS
    pretrained: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for real-time inference."""
    face_detection_confidence: float = 0.5
    max_faces: int = 10
    xai_frame_interval: int = 10
    intensity_change_threshold: float = 20.0
    temporal_smoothing_alpha: float = 0.3
    margin_threshold: float = 0.15
    probability_threshold: float = 0.30
