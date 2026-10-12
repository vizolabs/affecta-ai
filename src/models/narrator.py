"""Narrator — evidence-grounded templated commentary (docs/20 §6 + spec "narrator").

Deterministic, fully offline. Consumes Event objects + optional face context
(ranked evidence, region evidence) and produces human-readable lines.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.models.event_engine import Event

# Display names for emotion labels
_PRETTY = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear",
    "happy": "Happy", "neutral": "Neutral", "sad": "Sad",
    "surprise": "Surprise",
}

# Which event types are spoken at each detail level. Everything is still
# recorded in analytics — this only gates the commentary.
_DETAIL_EVENTS = {
    "calm": {"face_enter", "face_exit", "transition"},
    "normal": {
        "face_enter", "face_exit", "transition",
        "intensity_change", "stabilization", "ambiguity_enter",
    },
    "detailed": {
        "face_enter", "face_exit", "transition",
        "intensity_change", "stabilization",
        "ambiguity_enter", "ambiguity_exit", "confidence_drop",
    },
}


def _pretty(emotion: Optional[str]) -> str:
    if not emotion:
        return "an unknown expression"
    return _PRETTY.get(emotion, emotion.capitalize())


def _person(face_id: int) -> str:
    # Single-subject product: the tracker pins every face to id 0, so the
    # narrator always speaks about one person. The face_id argument is kept
    # for API stability but intentionally ignored.
    return "Person 01"


class Narrator:
    """Turns events (+ optional per-face context) into narration lines.

    face_context: face_id -> {
        "regions": {region: pct},          # optional Grad-CAM region evidence
        "ranked_emotions": [...],          # optional
        "margin": float,                   # optional top1-top2 margin
        "certainty": str,                  # optional
    }
    """

    def __init__(self, enabled: bool = True, detail: str = "normal"):
        self.enabled = enabled
        self.detail = detail if detail in _DETAIL_EVENTS else "normal"
        self._allowed = _DETAIL_EVENTS[self.detail]
        self._line_budget = 3  # max lines returned per frame (anti-spam)

    def narrate(
        self,
        events: List[Event],
        face_context: Optional[Dict[int, dict]] = None,
    ) -> List[dict]:
        """Return [{'face_id', 'event_type', 'text'}] for the given events."""
        if not self.enabled or not events:
            return []
        face_context = face_context or {}
        lines: List[dict] = []
        for ev in events[: self._line_budget * 2]:
            text = self._render(ev, face_context.get(ev.face_id) or {})
            if text:
                lines.append(
                    {
                        "face_id": ev.face_id,
                        "event_type": ev.event_type,
                        "text": text,
                        "timestamp": ev.timestamp,
                    }
                )
            if len(lines) >= self._line_budget:
                break
        return lines

    def _render(self, ev: Event, ctx: dict) -> Optional[str]:
        if ev.event_type not in self._allowed:
            return None
        p = _person(ev.face_id)
        regions: dict = ctx.get("regions") or {}
        top_region = next(iter(regions), None)

        if ev.event_type == "transition":
            line = (
                f"{p} changed from {_pretty(ev.from_emotion)} "
                f"to {_pretty(ev.to_emotion)}."
            )
            conf = ev.confidence
            if conf is not None:
                line += f" Confidence {conf * 100:.0f}%."
            if top_region and top_region not in ("uniform", "other"):
                line += f" Strongest evidence: {_region_phrase(top_region)}."
            return line

        if ev.event_type == "ambiguity_enter":
            kind = "uncertain" if ev.certainty == "insufficient" else "ambiguous"
            return f"{p}'s expression became {kind}."

        if ev.event_type == "intensity_change":
            direction = (ev.metadata or {}).get("direction", "changed")
            if direction == "increased":
                strength = "grew stronger"
            elif direction == "decreased":
                strength = "grew weaker"
            else:
                return f"{p}'s expression changed in intensity."
            if ev.to_emotion:
                return f"{p}'s {_pretty(ev.to_emotion)} expression {strength}."
            return f"{p}'s expression {strength}."

        if ev.event_type == "face_enter":
            return f"{p} entered the scene."

        if ev.event_type == "face_exit":
            return f"{p} left the scene."

        if ev.event_type == "stabilization":
            held = (ev.metadata or {}).get("held_seconds", 0)
            return (
                f"{p} has held {_pretty(ev.to_emotion)} steady for "
                f"{held:.1f} seconds."
            )

        if ev.event_type == "confidence_drop":
            drop = (ev.metadata or {}).get("drop_pp", 0)
            return f"{p}'s confidence dropped {drop:.0f} points."

        if ev.event_type == "ambiguity_exit":
            return f"{p}'s expression became clear again."

        return None

    def describe_face(self, face_id: int, face: dict) -> Optional[str]:
        """One-line standing description for a face (no event needed)."""
        if not self.enabled:
            return None
        p = _person(face_id)
        emotion = face.get("emotion", "neutral")
        conf = float(face.get("confidence", 0))
        certainty = face.get("certainty", "certain")
        line = f"{p} shows {_pretty(emotion)} ({conf * 100:.0f}%)."
        if certainty != "certain":
            line += " Evidence is not yet decisive."
        regions = face.get("regions") or {}
        top_region = next(iter(regions), None)
        if top_region and top_region not in ("uniform", "other"):
            line += f" Model attention: {_region_phrase(top_region)}."
        return line


def _region_phrase(region: str) -> str:
    phrases = {
        "mouth": "mouth region",
        "eyes": "eye region",
        "eyebrows": "brow region",
        "nose": "nose region",
        "cheeks": "cheek region",
    }
    return phrases.get(region, f"{region} region")


def get_narrator(enabled: bool = True, detail: str = "normal") -> Narrator:
    return Narrator(enabled=enabled, detail=detail)
