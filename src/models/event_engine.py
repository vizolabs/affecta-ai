"""Event engine — detects significant state changes (docs/20_EVENT_ENGINE.md).

Event types:
    transition, ambiguity_enter, ambiguity_exit, intensity_change,
    face_enter, face_exit, confidence_drop, stabilization
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Event:
    event_type: str
    face_id: int
    timestamp: float
    from_emotion: Optional[str] = None
    to_emotion: Optional[str] = None
    confidence: Optional[float] = None
    intensity: Optional[float] = None
    from_intensity: Optional[float] = None
    to_intensity: Optional[float] = None
    certainty: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "event_type": self.event_type,
            "face_id": self.face_id,
            "timestamp": self.timestamp,
            "from_emotion": self.from_emotion,
            "to_emotion": self.to_emotion,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "certainty": self.certainty,
        }
        if self.from_intensity is not None:
            d["from_intensity"] = self.from_intensity
        if self.to_intensity is not None:
            d["to_intensity"] = self.to_intensity
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class EventEngine:
    """Stateful per-track event detector.

    Expected per-frame state dict per track:
        {
          "emotion": str,          # dominant emotion
          "confidence": float,     # 0..1
          "certainty": str,        # 'certain' | 'ambiguous' | 'insufficient'
          "intensity": float,      # 0..100
        }
    """

    def __init__(
        self,
        intensity_threshold: float = 20.0,
        confidence_drop_threshold: float = 20.0,
        stabilization_seconds: float = 3.0,
        face_exit_seconds: float = 2.0,
        transition_debounce_seconds: float = 0.0,
    ):
        self.intensity_threshold = intensity_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.stabilization_seconds = stabilization_seconds
        self.face_exit_seconds = face_exit_seconds
        self.transition_debounce_seconds = transition_debounce_seconds
        # track_id -> tracking dicts
        self.state: Dict[int, dict] = {}
        self.first_seen: Dict[int, float] = {}
        self.last_seen: Dict[int, float] = {}
        self.emotion_since: Dict[int, dict] = {}  # track -> {"emotion": e, "t": t}
        self.stabilization_fired: Dict[int, str] = {}
        self.exited: Dict[int, bool] = {}
        self.pending_transition: Dict[int, dict] = {}

    def detect_frame(
        self, track_states: Dict[int, dict], now: Optional[float] = None
    ) -> List[Event]:
        """Detect events for the current frame's track states.

        Args:
            track_states: track_id -> state dict (emotion/confidence/certainty/intensity)
            now: unix timestamp (defaults to time.time())

        Returns:
            List of Event objects (may be empty).
        """
        now = time.time() if now is None else now
        events: List[Event] = []

        # face_enter for new tracks / re-entry after a real exit
        for tid, st in track_states.items():
            if tid not in self.first_seen:
                self.first_seen[tid] = now
                self.exited[tid] = False
                events.append(
                    Event(
                        event_type="face_enter",
                        face_id=tid,
                        timestamp=now,
                        metadata={"emotion": st.get("emotion")},
                    )
                )
            elif self.exited.get(tid, False):
                self.exited[tid] = False
                # Treat the returning face as fresh: drop stale state so we
                # don't fire phantom transitions vs. the pre-exit frame.
                self.state.pop(tid, None)
                self.emotion_since.pop(tid, None)
                self.stabilization_fired.pop(tid, None)
                self.pending_transition.pop(tid, None)
                events.append(
                    Event(
                        event_type="face_enter",
                        face_id=tid,
                        timestamp=now,
                        metadata={"emotion": st.get("emotion"), "reentry": True},
                    )
                )

        # per-track comparisons against previous state
        for tid, st in track_states.items():
            prev = self.state.get(tid)
            self.last_seen[tid] = now

            if prev is not None:
                events.extend(self._diff(tid, prev, st, now))
            else:
                # first observation: start emotion clock
                self.emotion_since[tid] = {"emotion": st.get("emotion"), "t": now}

            # keep emotion_since clock in sync
            clock = self.emotion_since.get(tid, {"emotion": None, "t": now})
            if clock["emotion"] != st.get("emotion"):
                self.emotion_since[tid] = {"emotion": st.get("emotion"), "t": now}
                self.stabilization_fired.pop(tid, None)
            else:
                # stabilization check
                held = now - clock["t"]
                if (
                    held >= self.stabilization_seconds
                    and self.stabilization_fired.get(tid) != st.get("emotion")
                ):
                    self.stabilization_fired[tid] = st.get("emotion")
                    events.append(
                        Event(
                            event_type="stabilization",
                            face_id=tid,
                            timestamp=now,
                            to_emotion=st.get("emotion"),
                            confidence=st.get("confidence"),
                            intensity=st.get("intensity"),
                            certainty=st.get("certainty"),
                            metadata={"held_seconds": round(held, 2)},
                        )
                    )

            self.state[tid] = st

        # face_exit for tracks missing beyond threshold
        for tid in list(self.state.keys()):
            if tid in track_states:
                continue
            missing = now - self.last_seen.get(tid, now)
            if missing >= self.face_exit_seconds and not self.exited.get(tid, False):
                self.exited[tid] = True
                prev = self.state.get(tid, {})
                events.append(
                    Event(
                        event_type="face_exit",
                        face_id=tid,
                        timestamp=now,
                        from_emotion=prev.get("emotion"),
                        metadata={"missing_seconds": round(missing, 2)},
                    )
                )

        return events

    def _diff(self, tid: int, prev: dict, st: dict, now: float) -> List[Event]:
        events: List[Event] = []

        # transition (debounced: emotion must hold for transition_debounce_seconds)
        self._check_transition(tid, prev, st, now, events)

        # ambiguity enter/exit
        prev_c, cur_c = prev.get("certainty"), st.get("certainty")
        if prev_c != cur_c:
            if cur_c in ("ambiguous", "insufficient"):
                events.append(
                    Event(
                        event_type="ambiguity_enter",
                        face_id=tid,
                        timestamp=now,
                        confidence=st.get("confidence"),
                        certainty=cur_c,
                        metadata={"from": prev_c, "to": cur_c},
                    )
                )
            elif prev_c in ("ambiguous", "insufficient"):
                events.append(
                    Event(
                        event_type="ambiguity_exit",
                        face_id=tid,
                        timestamp=now,
                        confidence=st.get("confidence"),
                        certainty=cur_c,
                        metadata={"from": prev_c, "to": cur_c},
                    )
                )

        # intensity_change (threshold in 0..100 units)
        if prev.get("intensity") is not None and st.get("intensity") is not None:
            diff = abs(st["intensity"] - prev["intensity"])
            if diff > self.intensity_threshold:
                events.append(
                    Event(
                        event_type="intensity_change",
                        face_id=tid,
                        timestamp=now,
                        intensity=st.get("intensity"),
                        from_intensity=prev["intensity"],
                        to_intensity=st["intensity"],
                        to_emotion=st.get("emotion"),
                        metadata={"direction": "increased" if st["intensity"] > prev["intensity"] else "decreased"},
                    )
                )

        # confidence_drop (percentage points)
        if prev.get("confidence") is not None and st.get("confidence") is not None:
            drop = (prev["confidence"] - st["confidence"]) * 100.0
            if drop > self.confidence_drop_threshold:
                events.append(
                    Event(
                        event_type="confidence_drop",
                        face_id=tid,
                        timestamp=now,
                        confidence=st.get("confidence"),
                        metadata={
                            "drop_pp": round(drop, 1),
                            "from": round(prev["confidence"], 4),
                            "to": round(st["confidence"], 4),
                        },
                    )
                )

        return events

    def _check_transition(
        self, tid: int, prev: dict, st: dict, now: float, events: List[Event]
    ) -> None:
        prev_e, cur_e = prev.get("emotion"), st.get("emotion")
        if not cur_e:
            return
        pend = self.pending_transition.get(tid)
        if pend is not None:
            if cur_e == pend["to"]:
                if now - pend["since"] >= self.transition_debounce_seconds:
                    events.append(
                        self._transition_event(tid, pend["from"], pend["to"], st, now)
                    )
                    self.pending_transition.pop(tid, None)
                return
            # emotion moved away from the pending target: cancel and maybe restart
            self.pending_transition.pop(tid, None)
        if prev_e and cur_e != prev_e:
            if self.transition_debounce_seconds <= 0:
                events.append(self._transition_event(tid, prev_e, cur_e, st, now))
            else:
                self.pending_transition[tid] = {
                    "from": prev_e,
                    "to": cur_e,
                    "since": now,
                }

    @staticmethod
    def _transition_event(
        tid: int, from_e: str, to_e: str, st: dict, now: float
    ) -> Event:
        return Event(
            event_type="transition",
            face_id=tid,
            timestamp=now,
            from_emotion=from_e,
            to_emotion=to_e,
            confidence=st.get("confidence"),
            intensity=st.get("intensity"),
            certainty=st.get("certainty"),
        )

    def forget_track(self, track_id: int) -> None:
        for d in (self.state, self.first_seen, self.last_seen,
                  self.emotion_since, self.stabilization_fired, self.exited,
                  self.pending_transition):
            d.pop(track_id, None)


def get_event_engine(**kwargs) -> EventEngine:
    return EventEngine(**kwargs)
