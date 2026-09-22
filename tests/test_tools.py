"""Regression tests for prep tools: LRP, calibration, RAF-DB converter, evaluate."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ml.xai.lrp import get_lrp_engine
from ml.xai.lrp_epsilon import get_epsilon_lrp
from ml.benchmarks.calibration import (
    compute_ece,
    fit_temperature,
    apply_temperature,
    _softmax,
)
from scripts.prepare_rafdb import parse_labels, RAF_TO_EMOTION


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, 7)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


class TestCalibration:
    def test_softmax_valid(self):
        logits = np.random.randn(100, 7)
        probs = _softmax(logits)
        assert probs.shape == logits.shape
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert probs.min() > 0

    def test_perfect_calibration_zero_ece(self):
        n = 300
        targets = np.random.randint(0, 7, n)
        logits = np.eye(7)[targets] * 1e6
        metrics = compute_ece(logits, targets)
        assert metrics['accuracy'] == 1.0
        assert metrics['ece'] < 1e-4

    def test_temperature_positive_and_probs_valid(self):
        logits = np.random.randn(500, 7)
        targets = np.random.randint(0, 7, 500)
        T = fit_temperature(logits, targets, steps=50)
        assert np.isfinite(T) and T > 0
        probs = apply_temperature(logits, T)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestLRPEngine:
    def test_grad_input_map_shape_and_finite(self):
        model = TinyConv()
        eng = get_lrp_engine()
        x = torch.randn(1, 3, 32, 32)
        pred = model(x).argmax(1).item()
        r = eng.compute(model, x, target_class=pred)
        assert r.shape == (1, 1, 32, 32)
        assert bool(torch.isfinite(r).all())

    def test_gradcam_shape_and_finite(self):
        model = TinyConv()
        eng = get_lrp_engine()
        x = torch.randn(1, 3, 32, 32)
        cam = eng.compute_gradcam(model, x, 3)
        assert tuple(cam.shape) == (32, 32)
        assert bool(torch.isfinite(cam).all())

    def test_region_aggregation_sums_to_one(self):
        eng = get_lrp_engine()
        r = torch.randn(1, 1, 100, 100).abs()
        landmarks = np.tile(np.array([[50.0, 50.0]]), (68, 1)) + np.random.RandomState(1).randn(68, 2) * 5
        img_size = (100, 100)
        regions = eng.aggregate_regions(r, landmarks, image_size=img_size)
        assert set(regions) == {'mouth', 'cheeks', 'eyes', 'eyebrows', 'nose'}
        assert abs(sum(regions.values()) - 1.0) < 1e-6


class TestLRPEpsilon:
    def test_map_shape_and_conservation(self):
        torch.manual_seed(0)
        model = TinyConv()
        eng = get_epsilon_lrp()
        x = torch.randn(1, 3, 32, 32)
        pred = model(x).argmax(1).item()
        r = eng.compute(model, x, target_class=pred)
        assert r.shape == (1, 3, 32, 32)
        assert bool(torch.isfinite(r).all())
        relevance, rel_err = eng.compute_conservation(model, x, pred)
        assert rel_err < 200.0  # composite z-rule: approx conservation (guards vs structural explosions 1e3+)


class TestPrepareRafdb:
    def test_parse_three_column(self, tmp_path):
        f = tmp_path / 'list_patition_label.txt'
        f.write_text('a.jpg\ttrain\t1\nb.jpg\ttest\t4\n')
        rows = parse_labels(f)
        assert rows == [('a.jpg', 'train', 1), ('b.jpg', 'test', 4)]

    def test_parse_two_column_missing_partition(self, tmp_path):
        f = tmp_path / 'list_patition_label.txt'
        f.write_text('a.jpg\t3\nb.jpg\t7\n')
        rows = parse_labels(f)
        assert rows == [('a.jpg', None, 3), ('b.jpg', None, 7)]

    def test_label_mapping_complete_and_bijective(self):
        mapped = set(RAF_TO_EMOTION.values())
        assert mapped == {'surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral'}
        assert len(mapped) == 7


class TestEvaluateFn:
    def test_evaluate_metrics_shapes(self):
        from ml.evaluate import evaluate

        model = TinyConv()
        model.eval()
        inputs = torch.randn(40, 3, 32, 32)
        labels = torch.randint(0, 7, (40,))
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=8)
        metrics = evaluate(model, loader, nn.CrossEntropyLoss(), torch.device('cpu'))
        assert 0.0 <= metrics['accuracy'] <= 100.0
        assert 0.0 <= metrics['macro_f1'] <= 100.0
        assert set(metrics['per_class_f1']) == {'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'}
        assert len(metrics['confusion_matrix']) == 7
        assert len(metrics['confusion_matrix'][0]) == 7