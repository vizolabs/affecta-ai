"""Unit tests: production model zoo + webcam module (headless, no camera)."""

import numpy as np
import pytest
import torch

from src.core.config import EMOTIONS, NUM_EMOTIONS
from ml.models.production import (
    ProductionClassifier,
    _gather,
    DDAMFN_NATIVE,
    DAN_NATIVE,
    DEFAULT_CKPTS,
    INPUT_SIZES,
)
from ml.models.ddam import DDAMNet
from ml.models.dan import DAN


class TestClassOrder:
    def test_ddamfn_gather_is_permutation_of_seven(self):
        g = _gather(DDAMFN_NATIVE)
        assert sorted(g.tolist()) == list(range(7))

    def test_dan_gather_is_permutation_of_seven(self):
        g = _gather(DAN_NATIVE)
        assert sorted(g.tolist()) == list(range(7))

    def test_gather_maps_known_positions(self):
        # DDAMFN native index 6 = angry -> project index 0
        assert _gather(DDAMFN_NATIVE)[0].item() == 6
        # DDAMFN native index 0 = neutral -> project index 4
        assert _gather(DDAMFN_NATIVE)[4].item() == 0
        # DAN native index 0 = surprise -> project index 6
        assert _gather(DAN_NATIVE)[6].item() == 0

    def test_anger_renamed_to_angry(self):
        # DAN uses 'anger' where the project uses 'angry'
        g = _gather(DAN_NATIVE)
        assert g[0].item() == 5  # angry sits at DAN native index 5


class _TupleNet(torch.nn.Module):
    """Mimics native models returning (logits, feat, heads)."""

    def forward(self, x):
        n = x.shape[0]
        logits = torch.zeros(n, 7)
        logits[:, 1] = 5.0  # native index 1 wins
        return logits, x, None


class TestProductionWrapper:
    def test_wrapper_reorders_to_project_space(self):
        wrap = ProductionClassifier(_TupleNet(), DDAMFN_NATIVE, 112, 'fake')
        out = wrap(torch.zeros(2, 3, 8, 8))
        assert out.shape == (2, NUM_EMOTIONS)
        # native winner index 1 = happy -> project index 3
        assert int(out.argmax(1)[0].item()) == EMOTIONS.index('happy')

    def test_wrapper_returns_tensor_not_tuple(self):
        wrap = ProductionClassifier(_TupleNet(), DDAMFN_NATIVE, 112, 'fake')
        out = wrap(torch.zeros(1, 3, 8, 8))
        assert isinstance(out, torch.Tensor)


class TestNativeArchitectures:
    def test_ddamfn_forward_shape(self):
        model = DDAMNet(num_class=7, num_head=2, pretrained=False)
        model.eval()
        with torch.no_grad():
            out, feat, heads = model(torch.randn(1, 3, 112, 112))
        assert out.shape == (1, 7)
        assert len(heads) == 2

    def test_dan_forward_shape(self):
        model = DAN(num_class=7, num_head=4, pretrained=False)
        model.eval()
        with torch.no_grad():
            out, feat, heads = model(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 7)


class TestCheckpointsOnDisk:
    """Skip when local (gitignored) checkpoints are absent."""

    @pytest.mark.skipif(not DEFAULT_CKPTS['dan'].endswith('RAF-DB.pth')
                        or not __import__('os').path.exists(DEFAULT_CKPTS['dan']),
                        reason='DAN checkpoint not present')
    def test_load_dan_production(self):
        from ml.models.production import load_production
        model = load_production('dan', device='cpu')
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 7)
        assert int(out.argmax(1)[0].item()) in range(7)

    @pytest.mark.skipif(not __import__('os').path.exists(DEFAULT_CKPTS['ddamfn']),
                        reason='DDAMFN++ checkpoint not present')
    def test_load_ddamfn_production(self):
        from ml.models.production import load_production
        model = load_production('ddamfn', device='cpu')
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(1, 3, 112, 112))
        assert out.shape == (1, 7)


class TestWebcamModule:
    def test_argparser_defaults(self):
        from src.pipeline.webcam import build_argparser
        args = build_argparser().parse_args([])
        assert args.model == 'dan'
        assert args.camera == 0
        assert 0.0 < args.ema <= 1.0

    def test_face_detector_loads(self):
        from src.pipeline.webcam import FaceDetector
        det = FaceDetector()
        assert not det.cascade.empty()

    def test_face_detector_on_synthetic(self):
        import cv2
        from src.pipeline.webcam import FaceDetector
        # 200x200 gray noise frame: cascade returns [] or boxes, must not crash
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        faces = FaceDetector().detect(frame)
        assert isinstance(faces, list)

    def test_predictor_ema_smoothing(self):
        from src.pipeline.webcam import EmotionPredictor
        pytest.importorskip('ml')
        try:
            from ml.models.production import DEFAULT_CKPTS
            import os
            if not os.path.exists(DEFAULT_CKPTS['dan']):
                pytest.skip('DAN checkpoint absent')
        except Exception:
            pytest.skip('production load unavailable')
        pred = EmotionPredictor('dan', ema_alpha=0.5, device=torch.device('cpu'))
        crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        p1 = pred.predict_probs(crop, reset=True)
        p2 = pred.predict_probs(crop)
        assert p1.shape == (7,)
        assert abs(float(p1.sum()) - 1.0) < 1e-4
        # identical crops -> EMA stays near the first prediction
        assert float(np.abs(p1 - p2).max()) < 0.2

    def test_annotate_draws_without_error(self):
        from src.pipeline.webcam import annotate
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        probs = np.full(7, 0.02, dtype=np.float32)
        probs[3] = 0.86  # happy
        out = annotate(frame, (100, 100, 120, 120), probs, fps=30.0)
        assert out.shape == (480, 640, 3)
        assert out.any()  # something was drawn

    def test_image_mode_headless(self, tmp_path):
        """--image path: no camera, writes annotated output + JSON."""
        import json as _json
        import subprocess
        import sys
        import os
        from ml.models.production import DEFAULT_CKPTS
        if not os.path.exists(DEFAULT_CKPTS['dan']):
            pytest.skip('DAN checkpoint absent')
        img = tmp_path / 'face.png'
        import cv2
        cv2.imwrite(str(img), np.random.randint(0, 255, (240, 240, 3), dtype=np.uint8))
        proc = subprocess.run(
            [sys.executable, '-m', 'src.pipeline.webcam', '--image', str(img)],
            capture_output=True, text=True, timeout=300,
            cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        payload = _json.loads(proc.stdout)
        assert payload['faces'], 'expected at least one face (full-frame fallback)'
        assert payload['faces'][0]['dominant'] in EMOTIONS
