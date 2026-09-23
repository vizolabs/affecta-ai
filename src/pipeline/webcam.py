"""Real-time webcam emotion detection for AFFECTA AI.

Runs a production classifier (DAN / DDAMFN++ / ViT / EffNet-B3) on a live
camera stream with Haar-cascade face detection, EMA-smoothed probabilities,
on-screen HUD (label, confidence, top-3 bars, FPS), optional Grad-CAM overlay,
and an ``--image`` mode for headless single-image inference.

Usage:
    python -m src.pipeline.webcam                      # default: DAN + camera 0
    python -m src.pipeline.webcam --model ddamfn       # DDAMFN++ (112px)
    python -m src.pipeline.webcam --image face.jpg     # headless, saves out_image.png
    python -m src.pipeline.webcam --cam                # Grad-CAM heatmap overlay

Keys (live mode): q/ESC = quit, c = toggle Grad-CAM, s = save snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.core.config import EMOTIONS, InferenceConfig
from ml.models.production import load_production, build_transform

EMOTION_COLORS = {
    'angry': (40, 40, 230), 'disgust': (0, 140, 0), 'fear': (150, 0, 150),
    'happy': (0, 220, 220), 'neutral': (200, 200, 200), 'sad': (230, 120, 0),
    'surprise': (0, 160, 255),
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class FaceDetector:
    """Haar-cascade frontal face detector (offline, no dlib)."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (40, 40)):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f'Failed to load Haar cascade: {cascade_path}')
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return faces as (x, y, w, h) sorted by area, descending."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=self.min_size, flags=cv2.CASCADE_SCALE_IMAGE,
        )
        faces = [tuple(int(v) for v in b) for b in (boxes if len(boxes) else [])]
        faces.sort(key=lambda b: b[2] * b[3], reverse=True)
        return faces


class EmotionPredictor:
    """Loads a production model; predicts with EMA smoothing across frames."""

    def __init__(self, model_name: str, ckpt_path: Optional[str] = None,
                 ema_alpha: float = 0.3, device: Optional[torch.device] = None):
        self.device = device or pick_device()
        self.model = load_production(model_name, ckpt_path, device=str(self.device))
        self.transform = build_transform(model_name)
        self.input_size = self.model.input_size
        self.ema_alpha = ema_alpha
        self._state: Optional[np.ndarray] = None

    @torch.inference_mode()
    def predict_probs(self, face_bgr: np.ndarray, reset: bool = False) -> np.ndarray:
        """Face crop (BGR) -> softmax probabilities in EMOTIONS order."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu().numpy()
        if reset or self._state is None:
            self._state = probs
        else:
            a = self.ema_alpha
            self._state = a * probs + (1.0 - a) * self._state
        return self._state

    def reset(self):
        self._state = None


class GradCamOverlay:
    """Grad-CAM heatmap for the production wrapper (logits-only forward)."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        from ml.xai.lrp import LRPEngine
        self.engine = LRPEngine()
        self.model = model
        self.device = device

    @torch.inference_mode()
    def __call__(self, face_bgr: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
        from PIL import Image
        from ml.models.production import build_transform  # reuse model transform
        size = self.model.input_size
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        if target_class is None:
            target_class = int(out.argmax(1).item())
        out[0, target_class].backward()
        grads = x.grad
        if grads is None:
            return np.zeros((size, size), dtype=np.float32)
        cam = (grads * x).mean(dim=1, keepdim=True).relu().squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, (face_bgr.shape[1], face_bgr.shape[0]))


def annotate(frame: np.ndarray, box: Tuple[int, int, int, int],
             probs: np.ndarray, fps: float, cam: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw box + label + confidence + top-3 bars (+ optional Grad-CAM blend)."""
    x, y, w, h = box
    top3 = np.argsort(probs)[::-1][:3]
    dominant = EMOTIONS[int(top3[0])]
    color = EMOTION_COLORS.get(dominant, (255, 255, 255))

    if cam is not None and cam.shape[:2] == (h, w):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        roi = frame[y:y + h, x:x + w]
        frame[y:y + h, x:x + w] = cv2.addWeighted(roi, 0.55, heatmap, 0.45, 0)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f'{dominant} {probs[int(top3[0])] * 100:.0f}%'
    cv2.putText(frame, label, (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    for rank, idx in enumerate(top3):
        by = y + h + 14 + rank * 14
        if by > frame.shape[0] - 4:
            break
        bar_w = int(probs[int(idx)] * 120)
        cv2.putText(frame, f'{EMOTIONS[int(idx)]}', (x, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x + 70, by - 9), (x + 70 + max(1, bar_w), by),
                      EMOTION_COLORS[EMOTIONS[int(idx)]], -1)

    cv2.putText(frame, f'FPS {fps:.0f}', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1, cv2.LINE_AA)
    return frame


def run_image_mode(args, predictor: EmotionPredictor, detector: FaceDetector) -> int:
    """Headless single-image inference; saves annotated copy, prints JSON."""
    frame = cv2.imread(args.image)
    if frame is None:
        print(f'error: cannot read {args.image}', file=sys.stderr)
        return 2
    faces = detector.detect(frame)
    predictor.reset()
    results = []
    if not faces:
        # fall back: treat whole frame as one face
        faces = [(0, 0, frame.shape[1], frame.shape[0])]
    cam_engine = GradCamOverlay(predictor.model, predictor.device) if args.cam else None
    for box in faces:
        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        probs = predictor.predict_probs(crop, reset=True)
        cam = cam_engine(crop) if cam_engine else None
        annotate(frame, box, probs, fps=0.0, cam=cam)
        top3 = np.argsort(probs)[::-1][:3]
        results.append({
            'box': [int(v) for v in box],
            'dominant': EMOTIONS[int(top3[0])],
            'confidence': round(float(probs[int(top3[0])]), 4),
            'probabilities': {e: round(float(p), 4) for e, p in zip(EMOTIONS, probs)},
            'top3': [EMOTIONS[int(i)] for i in top3],
        })
    out_path = Path(f'out_{Path(args.image).stem}.png')
    cv2.imwrite(str(out_path), frame)
    payload = {'image': args.image, 'model': args.model, 'faces': results,
               'annotated_saved': str(out_path)}
    print(json.dumps(payload, indent=2))
    return 0


def run_live(args, predictor: EmotionPredictor, detector: FaceDetector) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'error: cannot open camera {args.camera}', file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 30.0,
                                 (int(cap.get(3)), int(cap.get(4))))
    cam_engine = GradCamOverlay(predictor.model, predictor.device) if args.cam else None
    show_cam = args.cam
    fps_ema = 0.0
    print('AFFECTA webcam live — q/ESC quit, c toggle Grad-CAM, s snapshot')
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            faces = detector.detect(frame)[:args.max_faces]
            if not faces:
                predictor.reset()
            for i, box in enumerate(faces):
                x, y, w, h = box
                crop = frame[y:y + h, x:x + w]
                probs = predictor.predict_probs(crop, reset=(i == 0 and not faces))
                cam = None
                if show_cam and cam_engine is not None:
                    cam = cam_engine(crop)
                annotate(frame, box, probs, fps=fps_ema, cam=cam)
            dt = time.perf_counter() - t0
            inst = 1.0 / max(dt, 1e-6)
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            if writer is not None:
                writer.write(frame)
            cv2.imshow('AFFECTA AI', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                show_cam = not show_cam
            elif key == ord('s'):
                snap = f'snapshot_{int(time.time())}.png'
                cv2.imwrite(snap, frame)
                print('saved', snap)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='AFFECTA AI webcam emotion detection')
    p.add_argument('--model', default='dan', choices=['dan', 'ddamfn', 'vit_small', 'efficientnet_b3'],
                   help='production model (default: dan, 89.70%% RAF-DB test)')
    p.add_argument('--ckpt', default=None, help='override checkpoint path')
    p.add_argument('--camera', type=int, default=0, help='camera index')
    p.add_argument('--image', default=None, help='headless mode: run on a single image')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--ema', type=float, default=0.35, help='EMA smoothing (0..1, higher = more responsive)')
    p.add_argument('--max-faces', type=int, default=5)
    p.add_argument('--cam', action='store_true', help='Grad-CAM overlay')
    p.add_argument('--save', default=None, help='save live video to .mp4 path')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    device = pick_device()
    print(f'[webcam] model={args.model} device={device}', file=sys.stderr)
    predictor = EmotionPredictor(args.model, args.ckpt, ema_alpha=args.ema, device=device)
    detector = FaceDetector()
    if args.image:
        return run_image_mode(args, predictor, detector)
    return run_live(args, predictor, detector)


if __name__ == '__main__':
    raise SystemExit(main())
