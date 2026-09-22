"""Unit tests for ML models."""

import pytest
import torch
import numpy as np

from src.core.config import EMOTIONS, ACTION_UNITS, NUM_EMOTIONS, NUM_AUS
from src.models.expression_classifier import (
    VGG16ExpressionClassifier,
    ResNet50ExpressionClassifier,
    EfficientNetB2ExpressionClassifier,
    get_expression_model,
)
from src.models.au_detector import AUDetector, get_au_detector
from src.models.ranking_engine import RankingEngine, get_ranking_engine
from src.models.intensity_estimator import IntensityEstimator, get_intensity_estimator
from src.models.uncertainty_engine import UncertaintyEngine, get_uncertainty_engine
from src.models.temporal_engine import TemporalEngine, get_temporal_engine


class TestExpressionClassifier:
    """Tests for expression classifier models."""
    
    def test_vgg16_forward(self):
        model = VGG16ExpressionClassifier(num_classes=NUM_EMOTIONS)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, NUM_EMOTIONS)
    
    def test_resnet50_forward(self):
        model = ResNet50ExpressionClassifier(num_classes=NUM_EMOTIONS)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, NUM_EMOTIONS)
    
    def test_efficientnet_forward(self):
        model = EfficientNetB2ExpressionClassifier(num_classes=NUM_EMOTIONS)
        x = torch.randn(1, 3, 260, 260)
        output = model(x)
        assert output.shape == (1, NUM_EMOTIONS)
    
    def test_get_expression_model(self):
        for backbone in ['vgg16', 'resnet50', 'efficientnet_b2']:
            model = get_expression_model(backbone=backbone, pretrained=False)
            assert model is not None
    
    def test_output_is_logits(self):
        model = VGG16ExpressionClassifier(num_classes=NUM_EMOTIONS)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        # Should be raw logits, not probabilities
        assert output.min() < 0  # Logits can be negative


class TestAUDetector:
    """Tests for AU detector model."""
    
    def test_au_detector_forward(self):
        model = AUDetector(num_aus=NUM_AUS)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, NUM_AUS)
    
    def test_au_output_is_sigmoid(self):
        model = AUDetector(num_aus=NUM_AUS)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        # Output should be in [0, 1] (sigmoid)
        assert output.min() >= 0
        assert output.max() <= 1
    
    def test_get_au_detector(self):
        model = get_au_detector(pretrained=False)
        assert model is not None


class TestRankingEngine:
    """Tests for ranking engine."""
    
    def test_preliminary_ranking(self):
        engine = get_ranking_engine()
        
        probs = torch.tensor([0.1, 0.05, 0.05, 0.6, 0.1, 0.05, 0.05])
        aus = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0, 0.8])
        
        scores = engine.compute_preliminary(probs, aus)
        assert 'ranked_emotions' in scores
        assert 'evidence_scores' in scores
    
    def test_ranking_order(self):
        engine = get_ranking_engine()
        
        # Happy should rank highest
        probs = torch.tensor([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1])
        aus = torch.tensor([0.0, 0.8, 0.0, 0.0, 0.0, 0.9])
        
        scores = engine.compute_preliminary(probs, aus)
        top_emotion = scores['ranked_emotions'][0]
        assert top_emotion == 'happy'


class TestIntensityEstimator:
    """Tests for intensity estimator."""
    
    def test_intensity_estimator_forward(self):
        model = get_intensity_estimator()
        features = torch.randn(1, 512)
        aus = torch.randn(1, NUM_AUS)
        output = model(features, aus)
        assert output.shape == (1, 1)
    
    def test_intensity_range(self):
        model = get_intensity_estimator()
        features = torch.randn(1, 512)
        aus = torch.randn(1, NUM_AUS)
        output = model(features, aus)
        # Output should be in [0, 100]
        assert output.item() >= 0
        assert output.item() <= 100
    
    def test_get_category(self):
        assert IntensityEstimator.get_category(10) == 'low'
        assert IntensityEstimator.get_category(50) == 'medium'
        assert IntensityEstimator.get_category(90) == 'high'


class TestUncertaintyEngine:
    """Tests for uncertainty engine."""
    
    def test_certain_detection(self):
        engine = get_uncertainty_engine()
        probs = [0.05, 0.05, 0.05, 0.8, 0.02, 0.02, 0.01]
        state, margin = engine.compute(probs)
        assert state == 'certain'
        assert margin > 0.15
    
    def test_ambiguous_detection(self):
        engine = get_uncertainty_engine()
        probs = [0.05, 0.05, 0.05, 0.35, 0.30, 0.10, 0.10]
        state, margin = engine.compute(probs)
        assert state == 'ambiguous'
    
    def test_insufficient_detection(self):
        engine = get_uncertainty_engine()
        probs = [0.15, 0.15, 0.15, 0.15, 0.15, 0.13, 0.12]
        state, margin = engine.compute(probs)
        assert state == 'insufficient'


class TestTemporalEngine:
    """Tests for temporal engine."""
    
    def test_smoothing(self):
        engine = get_temporal_engine()
        
        history = [
            {'happy': 0.6, 'sad': 0.1, 'neutral': 0.3},
            {'happy': 0.7, 'sad': 0.1, 'neutral': 0.2},
        ]
        current = {'happy': 0.5, 'sad': 0.2, 'neutral': 0.3}
        
        smoothed = engine.smooth(history, current)
        assert 'happy' in smoothed
        assert abs(sum(smoothed.values()) - 1.0) < 0.01
    
    def test_transition_detection(self):
        engine = get_temporal_engine()
        
        transition = engine.detect_transition('happy', 'sad')
        assert transition is not None
        assert transition['from'] == 'happy'
        assert transition['to'] == 'sad'
        
        no_transition = engine.detect_transition('happy', 'happy')
        assert no_transition is None
