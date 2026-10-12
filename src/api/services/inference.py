"""Analysis service + per-session pipeline.

AnalysisService  — app-level singleton: production model, face detector,
                   Grad-CAM. Stateless between calls.
SessionAnalysis  — per-connection state: face tracker, EMA smoothing,
                   evidence ranking, event engine, narrator, timeline.

Pipeline per frame: detect -> track -> classify -> rank (P/R/T evidence)
-> events -> narrate -> optional XAI (Grad-CAM -> geometric face regions
-> R_e cache for the ranking engine).

Honesty constraints (docs/05, ADRs):
  - AU detector is untrained: action_units=None, au_model_status="untrained",
    ranking weight w_a=0.
  - Intensity is a deterministic distribution-extremity proxy, not the
    untrained IntensityEstimator network.
  - Region evidence uses fixed geometric boxes (no landmark detector).
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.api.core.config import settings
from src.api.services.tracker import FaceTracker
from src.core.config import EMOTIONS
from src.models.event_engine import EventEngine
from src.models.narrator import Narrator
from src.models.ranking_engine import get_ranking_engine
from src.models.uncertainty_engine import get_uncertainty_engine
from src.pipeline.webcam import FaceDetector, pick_device

Box = Tuple[int, int, int, int]

# Geometric face regions as (y0, y1, x0, x1) fractions of the face crop.
REGION_BOXES: Dict[str, List[Tuple[float, float, float, float]]] = {
    "eyebrows": [(0.10, 0.28, 0.15, 0.85)],
    "eyes": [(0.25, 0.45, 0.15, 0.85)],
    "nose": [(0.40, 0.65, 0.35, 0.65)],
    "mouth": [(0.65, 0.95, 0.20, 0.80)],
    "cheeks": [(0.45, 0.75, 0.05, 0.30), (0.45, 0.75, 0.70, 0.95)],
}

# Prior: which regions are expected to matter for each emotion.
REGION_TO_EMOTION: Dict[str, Dict[str, float]] = {
    "happy": {"mouth": 0.50, "cheeks": 0.30, "eyes": 0.15, "eyebrows": 0.05, "nose": 0.0},
    "fear": {"eyes": 0.45, "eyebrows": 0.35, "mouth": 0.15, "nose": 0.05, "cheeks": 0.0},
    "disgust": {"nose": 0.50, "mouth": 0.30, "eyes": 0.10, "eyebrows": 0.05, "cheeks": 0.05},
    "surprise": {"eyes": 0.30, "eyebrows": 0.25, "mouth": 0.35, "nose": 0.05, "cheeks": 0.05},
    "sad": {"eyebrows": 0.45, "mouth": 0.30, "eyes": 0.15, "cheeks": 0.10, "nose": 0.0},
    "angry": {"eyebrows": 0.45, "nose": 0.25, "mouth": 0.20, "eyes": 0.10, "cheeks": 0.0},
    "neutral": {"eyes": 0.35, "mouth": 0.25, "nose": 0.20, "eyebrows": 0.10, "cheeks": 0.10},
}


def distribution_extremity(probs: np.ndarray) -> float:
    top1 = float(np.max(probs))
    uniform = 1.0 / len(probs)
    return max(0.0, (top1 - uniform) / (1.0 - uniform) * 100.0)


def region_percentages(cam: np.ndarray) -> Dict[str, float]:
    h, w = cam.shape[:2]
    total = float(cam.sum())
    out: Dict[str, float] = {}
    for region, boxes in REGION_BOXES.items():
        acc = 0.0
        for y0, y1, x0, x1 in boxes:
            ys, ye = int(y0 * h), max(int(y0 * h) + 1, int(y1 * h))
            xs, xe = int(x0 * w), max(int(x0 * w) + 1, int(x1 * w))
            acc += float(cam[ys:ye, xs:xe].sum())
        out[region] = (acc / total * 100.0) if total > 0 else 0.0
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def region_evidence(region_pct: Dict[str, float]) -> Dict[str, float]:
    frac = {r: p / 100.0 for r, p in region_pct.items()}
    return {
        emotion: sum(frac.get(region, 0.0) * weight for region, weight in prior.items())
        for emotion, prior in REGION_TO_EMOTION.items()
    }


def decode_jpeg(jpeg: bytes) -> Optional[np.ndarray]:
    if not jpeg:
        return None
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_heatmap_png(cam: np.ndarray) -> str:
    img = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
    ok_flag, buf = cv2.imencode(".png", np.ascontiguousarray(img))
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok_flag else ""


def _crop(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return crop


class AnalysisService:
    """App-level model holder: detection, classification, Grad-CAM."""

    def __init__(self, model_name: Optional[str] = None, warm: bool = True):
        self.model_name = model_name or settings.default_model
        self.device = pick_device()
        self.model: Optional[torch.nn.Module] = None
        self.transform = None
        self.load_error: Optional[str] = None
        self.detector = FaceDetector()
        if warm:
            self.load_model()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        if self.model is None:
            from ml.models.production import build_transform, load_production

            try:
                self.model = load_production(self.model_name, device=str(self.device))
                self.transform = build_transform(self.model_name)
                self.load_error = None
            except Exception as exc:  # corrupt/missing checkpoint
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Box]:
        # No slicing here: the tracker (single-person mode) picks the best
        # box itself (IoU-continuity first, largest second).
        return self.detector.detect(frame_bgr)

    def predict_probs(self, face_bgr: np.ndarray) -> np.ndarray:
        if self.model is None or self.transform is None:
            raise RuntimeError("model not loaded")
        from PIL import Image

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            x = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            logits = self.model(x)
            return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    def gradcam(self, face_bgr: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not loaded")
        from PIL import Image
        from torchvision import transforms

        size = self.model.input_size
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.enable_grad():
            x = tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            x.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            out = self.model(x)
            if target_class is None:
                target_class = int(out.argmax(1).item())
            out[0, target_class].backward()
            grads = x.grad
        if grads is None:
            return np.zeros(face_bgr.shape[:2], dtype=np.float32)
        cam = (grads * x).mean(dim=1, keepdim=True).relu().squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, (face_bgr.shape[1], face_bgr.shape[0]))


class SessionAnalysis:
    """Per-connection pipeline state (tracker + events + narration + timeline)."""

    def __init__(
        self,
        service: AnalysisService,
        *,
        xai_interval: Optional[int] = None,
        narrator_enabled: bool = True,
        narrator_detail: str = "normal",
        ema_alpha: float = 0.35,
    ):
        self.service = service
        # Clamp: negative/zero would make frame_id % xai_interval always 0
        # (Grad-CAM on every frame — a cheap CPU DoS from one URL param).
        raw_interval = xai_interval or settings.xai_frame_interval
        self.xai_interval = max(1, min(int(raw_interval), 120))
        self.tracker = FaceTracker(max_faces=settings.max_faces)
        self.event_engine = EventEngine(
            intensity_threshold=settings.intensity_change_threshold,
            confidence_drop_threshold=settings.confidence_drop_threshold,
            stabilization_seconds=settings.stabilization_seconds,
            face_exit_seconds=settings.face_exit_seconds,
            transition_debounce_seconds=settings.transition_debounce_seconds,
        )
        self.narrator = Narrator(enabled=narrator_enabled, detail=narrator_detail)
        self.ranker = get_ranking_engine(w_p=0.6, w_a=0.0, w_r=0.2, w_t=0.2)
        self.uncertainty = get_uncertainty_engine()
        self.ema_alpha = ema_alpha

        self._ema: Dict[int, np.ndarray] = {}
        self._raw: Dict[int, np.ndarray] = {}
        self._region_cache: Dict[int, Dict[str, float]] = {}
        self._re_cache: Dict[int, Dict[str, float]] = {}

        self.frame_id = 0
        self._t_prev: Optional[float] = None
        self.fps = 0.0
        self.last_frame: Optional[np.ndarray] = None
        self.last_tracks: Dict[int, Box] = {}

        self.timeline: List[dict] = []
        self._timeline_last = 0.0
        self._conf_sum = 0.0
        self._int_sum = 0.0
        self._obs = 0
        self.last_events: List[Any] = []

    @property
    def avg_confidence(self) -> Optional[float]:
        return round(self._conf_sum / self._obs, 4) if self._obs else None

    @property
    def avg_intensity(self) -> Optional[float]:
        return round(self._int_sum / self._obs, 4) if self._obs else None

    def _tick_fps(self, now: float) -> None:
        if self._t_prev is not None:
            dt = now - self._t_prev
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._t_prev = now

    def process_jpeg(
        self,
        jpeg: bytes,
        *,
        now: Optional[float] = None,
        fallback_full_frame: bool = False,
    ) -> dict:
        frame = decode_jpeg(jpeg)
        if frame is None:
            raise ValueError("could not decode frame")
        return self.process_frame(
            frame, now=now, fallback_full_frame=fallback_full_frame
        )

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        now: Optional[float] = None,
        fallback_full_frame: bool = False,
    ) -> dict:
        """Run the full pipeline on one already-decoded BGR frame."""
        now = time.time() if now is None else now

        self.frame_id += 1
        self._tick_fps(now)
        boxes = self.service.detect_faces(frame)
        if not boxes and fallback_full_frame:
            boxes = [(0, 0, int(frame.shape[1]), int(frame.shape[0]))]
        tracks = self.tracker.update(boxes)
        self.last_frame = frame
        self.last_tracks = dict(tracks)

        cams: Dict[int, np.ndarray] = {}
        states: Dict[int, dict] = {}
        face_ctx: Dict[int, dict] = {}
        face_payloads: List[dict] = []
        ranked_by_id: Dict[int, List[str]] = {}

        for tid, box in tracks.items():
            crop = _crop(frame, box)
            if crop is None:
                continue

            raw = np.asarray(self.service.predict_probs(crop), dtype=np.float64)
            prev = self._ema.get(tid)
            if prev is None:
                ema = raw
            else:
                a = self.ema_alpha
                ema = a * raw + (1.0 - a) * prev
            self._ema[tid] = ema
            self._raw[tid] = raw

            if tid not in self._region_cache and min(crop.shape[:2]) >= 16:
                try:
                    cam = self.service.gradcam(crop, int(np.argmax(raw)))
                    cams[tid] = cam
                    regions = region_percentages(cam)
                    self._region_cache[tid] = regions
                    self._re_cache[tid] = region_evidence(regions)
                except Exception:
                    pass

            temporal = {e: float(ema[i]) for i, e in enumerate(EMOTIONS)}
            scores = self.ranker.final_ranking(
                probs=[float(v) for v in raw],
                au_activations=[0.0] * 6,
                region_evidence=self._re_cache.get(tid),
                temporal_evidence=temporal,
            )
            ranked = list(scores.keys())
            dominant = ranked[0]
            confidence = float(raw[EMOTIONS.index(dominant)])
            # Certainty/margin from classifier probabilities (ADR-006), NOT
            # from the renormalised evidence scores — the ranker's output is
            # scaled by the evidence total, which biases margin upward and
            # everything looks "certain".
            certainty, margin = self.uncertainty.compute(
                [float(ema[i]) for i in range(len(EMOTIONS))]
            )
            intensity = distribution_extremity(raw)
            if intensity < 33:
                category = "low"
            elif intensity < 67:
                category = "medium"
            else:
                category = "high"

            states[tid] = {
                "emotion": dominant,
                "confidence": confidence,
                "certainty": certainty,
                "intensity": intensity,
            }
            ranked_by_id[tid] = ranked
            cached_regions: Optional[Dict[str, float]] = self._region_cache.get(tid)
            face_ctx[tid] = {
                "regions": cached_regions,
                "ranked_emotions": ranked,
                "margin": margin,
                "certainty": certainty,
            }

            x, y, w, h = box
            face_payloads.append({
                "face_id": tid,
                "box": [int(x), int(y), int(w), int(h)],
                "emotion": dominant,
                "confidence": round(confidence, 4),
                "probabilities": {e: round(float(raw[i]), 4) for i, e in enumerate(EMOTIONS)},
                "ranked_emotions": ranked,
                "evidence_scores": {e: round(float(scores[e]), 4) for e in EMOTIONS},
                "certainty": certainty,
                "margin": round(float(margin), 4),
                "intensity": round(float(intensity), 1),
                "intensity_category": category,
                "action_units": None,
                "au_model_status": "untrained",
                "regions": cached_regions,
            })
            self._conf_sum += confidence
            self._int_sum += intensity
            self._obs += 1

        events = self.event_engine.detect_frame(states, now=now)
        self.last_events = events

        interval_due = self.frame_id % self.xai_interval == 0
        xai_targets: set = set()
        if interval_due and tracks:
            xai_targets.add(next(iter(tracks)))
        for ev in events:
            if ev.event_type in ("transition", "ambiguity_enter") and ev.face_id in tracks:
                xai_targets.add(ev.face_id)

        xai_payload = None
        for tid in sorted(xai_targets):
            next_box = tracks.get(tid)
            if next_box is None:
                continue
            next_cam = cams.get(tid)
            if next_cam is None:
                crop = _crop(frame, next_box)
                if crop is None:
                    continue
                target = EMOTIONS.index(states[tid]["emotion"]) if tid in states else None
                try:
                    next_cam = self.service.gradcam(crop, target)
                except Exception:
                    continue
            cams[tid] = next_cam
            regions = region_percentages(next_cam)
            self._region_cache[tid] = regions
            self._re_cache[tid] = region_evidence(regions)
            face_ctx.setdefault(tid, {})["regions"] = regions
            if xai_payload is None:
                st = states.get(tid, {})
                xai_payload = {
                    "face_id": tid,
                    "method": "gradcam",
                    "emotion": st.get("emotion"),
                    "confidence": st.get("confidence"),
                    "heatmap": encode_heatmap_png(next_cam),
                    "regions": regions,
                }

        narration = self.narrator.narrate(events, face_ctx)

        timeline_sample = None
        if now - self._timeline_last >= 1.0:
            self._timeline_last = now
            timeline_sample = {
                "t": round(now, 1),
                "faces": [
                    {
                        "id": tid,
                        "emotion": st["emotion"],
                        "intensity": round(st["intensity"], 1),
                        "confidence": round(st["confidence"], 3),
                    }
                    for tid, st in states.items()
                ],
            }
            self.timeline.append(timeline_sample)
            # Cap: ~1 sample/sec is written forever; the whole array is
            # serialised into the session metadata JSON on every flush, so
            # an unbounded list turns an 8-hour session into multi-MB
            # rewrites every few seconds. Keep the most recent hour.
            if len(self.timeline) > 3600:
                del self.timeline[: len(self.timeline) - 3600]

        for tid in [t for t in self._region_cache if t not in self.tracker.alive_ids]:
            self._region_cache.pop(tid, None)
            self._re_cache.pop(tid, None)
            self._ema.pop(tid, None)
            self._raw.pop(tid, None)

        return {
            "session_id": None,
            "frame_id": self.frame_id,
            "faces": face_payloads,
            "events": [ev.as_dict() for ev in events],
            "narration": narration,
            "xai": xai_payload,
            "fps": round(self.fps, 1),
            "model": self.service.model_name,
            "timeline": timeline_sample,
            "timestamp": now,
        }

    def compute_xai(self, face_id: int) -> Optional[dict]:
        """On-demand XAI for a face from the last processed frame (no side effects)."""
        if self.last_frame is None:
            return None
        box = self.last_tracks.get(face_id)
        if box is None:
            return None
        crop = _crop(self.last_frame, box)
        if crop is None:
            return None
        raw = np.asarray(self.service.predict_probs(crop), dtype=np.float64)
        target = int(np.argmax(raw))
        try:
            cam = self.service.gradcam(crop, target)
        except Exception:
            return None
        regions = region_percentages(cam)
        self._region_cache[face_id] = regions
        self._re_cache[face_id] = region_evidence(regions)
        return {
            "face_id": face_id,
            "method": "gradcam",
            "emotion": EMOTIONS[target],
            "confidence": round(float(raw[target]), 4),
            "heatmap": encode_heatmap_png(cam),
            "regions": regions,
        }
