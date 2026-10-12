"""Offline video analysis for the Research Lab (photo + video in one place).

A video is decoded with OpenCV, sampled at a user-configurable rate, and each
sampled frame is pushed through the *same* SessionAnalysis pipeline used by the
live WebSocket stream (tracker, EMA, evidence ranking, event engine, narrator).
The result is an aggregate over the whole clip: emotion distribution,
transitions, timeline, narration and the Grad-CAM of the peak frame.

Nothing here persists sessions — video runs are ad-hoc research artifacts, not
recorded Observer sessions. Frames stay in memory and are never written to disk
beyond the temporary decode file (removed immediately after use).
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from typing import Optional

import cv2
import numpy as np

from src.api.core.config import settings
from src.api.services.inference import (
    SessionAnalysis,
    encode_heatmap_png,
    region_percentages,
)
from src.core.config import EMOTIONS

MAX_VIDEO_BYTES = 60 * 1024 * 1024  # 60 MB
MAX_DURATION_SECONDS = 120.0
MAX_SAMPLED_FRAMES = 400

# Extensions we hand to OpenCV; the suffix is only used to let the demuxer
# pick the right container.
SUPPORTED_SUFFIXES = (
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
)


class VideoDecodeError(ValueError):
    """Raised when the upload is not a decodable video."""


def _suffix_for(filename: str, content_type: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in SUPPORTED_SUFFIXES:
        return ext
    guess = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
        "video/mpeg": ".mpeg",
    }.get((content_type or "").split(";")[0].strip(), "")
    return guess


def analyze_video(
    service,
    data: bytes,
    *,
    filename: str = "",
    content_type: str = "",
    sample_fps: int = 5,
    max_frames: int = MAX_SAMPLED_FRAMES,
    debounce_seconds: Optional[float] = None,
) -> dict:
    """Decode + analyse an uploaded clip. Raises VideoDecodeError on bad input."""
    if len(data) == 0:
        raise VideoDecodeError("Empty upload")
    if len(data) > MAX_VIDEO_BYTES:
        raise VideoDecodeError("Video too large (max 60MB)")

    suffix = _suffix_for(filename, content_type)
    tmp_path = None
    cap = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="hfed_video_", suffix=suffix or ".mp4")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise VideoDecodeError(
                "Could not decode this video format — try MP4 (H.264) or MOV."
            )

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native_fps <= 0 or not np.isfinite(native_fps):
            native_fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            duration = total_frames / native_fps
        else:
            duration = 0.0

        if duration > MAX_DURATION_SECONDS:
            cap.release()
            cap = None
            raise VideoDecodeError(
                f"Video is {duration:.0f}s long — the limit is "
                f"{MAX_DURATION_SECONDS:.0f}s. Trim it and retry."
            )

        sample_fps = max(1, min(int(sample_fps or 5), 10))
        step = max(1, int(round(native_fps / sample_fps)))

        analysis = SessionAnalysis(service)
        # Keep the SAME transition semantics as the live stream: without the
        # hold, per-frame flicker (A→B→A) would be reported as real transitions.
        analysis.event_engine.transition_debounce_seconds = (
            settings.transition_debounce_seconds
            if debounce_seconds is None
            else max(0.0, float(debounce_seconds))
        )

        emotions: Counter = Counter()
        transitions: list = []
        narration_lines: list = []
        timeline: list = []
        samples = 0
        faces_seen = 0
        frames_with_face = 0
        conf_sum = 0.0
        int_sum = 0.0
        peak: Optional[dict] = None
        peak_frame_idx = -1
        peak_crop: Optional[np.ndarray] = None

        idx = -1
        while len(timeline) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step != 0:
                continue

            t = idx / native_fps
            result = analysis.process_frame(frame, now=t)
            samples += 1
            if result["faces"]:
                frames_with_face += 1

            for face in result["faces"]:
                faces_seen += 1
                emotions[face["emotion"]] += 1
                conf_sum += face["confidence"]
                int_sum += face["intensity"]
                if peak is None or face["confidence"] > peak["confidence"]:
                    peak = face
                    peak_frame_idx = idx
                    x, y, w, h = face["box"]
                    crop = frame[y:y + h, x:x + w]
                    peak_crop = crop if crop.size else None

            for ev in result["events"]:
                if ev.get("event_type") == "transition":
                    transitions.append(
                        {
                            "t": round(ev.get("timestamp", t), 2),
                            "from_emotion": ev.get("from_emotion"),
                            "to_emotion": ev.get("to_emotion"),
                            "confidence": ev.get("confidence"),
                        }
                    )
            for line in result["narration"]:
                text = line.get("text")
                if text and text not in narration_lines:
                    narration_lines.append(text)

            if result["timeline"]:
                timeline.append(result["timeline"])

        cap.release()
        cap = None

        if samples == 0:
            raise VideoDecodeError("No frames could be decoded from this video.")

        total_observations = sum(emotions.values())
        distribution = [
            {
                "emotion": label,
                "count": int(count),
                "share": round(count / total_observations, 4),
            }
            for label, count in emotions.most_common()
        ]
        dominant = distribution[0]["emotion"] if distribution else None

        # Grad-CAM for the single most confident frame of the clip, so the Lab
        # XAI panel is meaningful for video uploads too.
        xai_payload = None
        if peak is not None and peak_crop is not None and min(peak_crop.shape[:2]) >= 16:
            try:
                cam = service.gradcam(peak_crop, EMOTIONS.index(peak["emotion"]))
                xai_payload = {
                    "face_id": 0,
                    "method": "gradcam",
                    "emotion": peak["emotion"],
                    "confidence": peak["confidence"],
                    "heatmap": encode_heatmap_png(cam),
                    "regions": region_percentages(cam),
                }
            except Exception:
                xai_payload = None

        return {
            "kind": "video",
            "model": analysis.service.model_name,
            "frames_analyzed": samples,
            "native_fps": round(native_fps, 2),
            "sample_fps": sample_fps,
            "sample_every_n_frames": step,
            "total_frames": total_frames or None,
            "duration_seconds": round(
                duration or (samples * step / native_fps), 2
            ),
            "faces_per_frame_avg": round(faces_seen / samples, 2),
            "frames_with_face": frames_with_face,
            "frames_without_face": samples - frames_with_face,
            "face_coverage": round(frames_with_face / samples, 3),
            "emotion_distribution": distribution,
            "dominant_emotion": dominant,
            "avg_confidence": round(conf_sum / total_observations, 4)
            if total_observations
            else None,
            "avg_intensity": round(int_sum / total_observations, 1)
            if total_observations
            else None,
            "transitions": transitions[:100],
            "transition_count": len(transitions),
            "timeline": timeline,
            "narration": [{"text": t, "event_type": "summary"} for t in narration_lines[:12]],
            "peak_frame": {
                "index": peak_frame_idx,
                "t": round(peak_frame_idx / native_fps, 2),
                "emotion": peak["emotion"],
                "confidence": peak["confidence"],
                "intensity": peak["intensity"],
                "certainty": peak["certainty"],
                "probabilities": peak["probabilities"],
                "ranked_emotions": peak["ranked_emotions"],
                "evidence_scores": peak["evidence_scores"],
            }
            if peak
            else None,
            "xai": xai_payload,
        }
    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass