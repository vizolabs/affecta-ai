"""Greedy IoU face tracker: assigns stable track ids across frames.

Spec 12 (ml/tracker.py). Ghost tracks (temporarily missed faces) are kept
for `max_age` frames so a 1-2 frame detection dropout reuses the same id,
but ghosts are NOT returned as active — the event engine sees true absence
and can fire face_exit after its time threshold.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

Box = Tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """Stable per-face ids via greedy highest-IoU matching.

    max_faces=1 (product mode): exactly one track is ever kept, pinned to
    id 0. The tracked box prefers IoU-continuity with the current track
    (same human wins even if another face is larger), falling back to the
    largest box — so the narrator always sees a single "Person 01".
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 10,
        max_faces: Optional[int] = None,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_faces = max_faces
        self._next_id = 0
        self._tracks: Dict[int, dict] = {}

    @property
    def active(self) -> Dict[int, Box]:
        return {tid: t["box"] for tid, t in self._tracks.items() if t["missed"] == 0}

    @property
    def alive_ids(self) -> Set[int]:
        return set(self._tracks)

    def update(self, boxes: List[Box]) -> Dict[int, Box]:
        if self.max_faces == 1:
            return self._update_single(boxes)
        pairs = []
        for tid, track in self._tracks.items():
            for i, box in enumerate(boxes):
                score = iou(track["box"], box)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, i))
        pairs.sort(reverse=True)

        used_tracks: Set[int] = set()
        used_boxes: Set[int] = set()
        for _score, tid, i in pairs:
            if tid in used_tracks or i in used_boxes:
                continue
            self._tracks[tid]["box"] = boxes[i]
            self._tracks[tid]["missed"] = 0
            used_tracks.add(tid)
            used_boxes.add(i)

        for tid in list(self._tracks):
            if tid not in used_tracks:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_age:
                    del self._tracks[tid]

        for i, box in enumerate(boxes):
            if i in used_boxes:
                continue
            if self.max_faces is not None and len(self._tracks) >= self.max_faces:
                break
            self._tracks[self._next_id] = {"box": box, "missed": 0}
            self._next_id += 1

        return self.active

    def _update_single(self, boxes: List[Box]) -> Dict[int, Box]:
        if boxes:
            chosen: Optional[Box] = None
            if 0 in self._tracks:
                best = max(boxes, key=lambda b: iou(self._tracks[0]["box"], b))
                if iou(self._tracks[0]["box"], best) >= self.iou_threshold:
                    chosen = best
            if chosen is None:
                chosen = max(boxes, key=lambda b: b[2] * b[3])
            for tid in list(self._tracks):
                if tid != 0:
                    del self._tracks[tid]
            self._tracks[0] = {"box": chosen, "missed": 0}
            return {0: chosen}

        if 0 in self._tracks:
            self._tracks[0]["missed"] += 1
            if self._tracks[0]["missed"] > self.max_age:
                del self._tracks[0]
        return self.active
