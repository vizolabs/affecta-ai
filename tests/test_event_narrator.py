"""Tests for event engine + narrator (docs/20)."""

import time

from src.models.event_engine import EventEngine
from src.models.narrator import Narrator


def _state(emotion="neutral", confidence=0.9, certainty="certain", intensity=50.0):
    return {
        "emotion": emotion,
        "confidence": confidence,
        "certainty": certainty,
        "intensity": intensity,
    }


def test_face_enter_on_first_sight():
    eng = EventEngine()
    events = eng.detect_frame({0: _state()}, now=100.0)
    types = [e.event_type for e in events]
    assert "face_enter" in types


def test_no_duplicate_face_enter():
    eng = EventEngine()
    eng.detect_frame({0: _state()}, now=100.0)
    events = eng.detect_frame({0: _state()}, now=100.1)
    assert [e.event_type for e in events] == []


def test_transition_event():
    eng = EventEngine()
    eng.detect_frame({0: _state(emotion="neutral")}, now=100.0)
    events = eng.detect_frame({0: _state(emotion="happy")}, now=100.1)
    transitions = [e for e in events if e.event_type == "transition"]
    assert len(transitions) == 1
    assert transitions[0].from_emotion == "neutral"
    assert transitions[0].to_emotion == "happy"


def test_ambiguity_enter_exit():
    eng = EventEngine()
    eng.detect_frame({0: _state(certainty="certain")}, now=100.0)
    events = eng.detect_frame({0: _state(certainty="ambiguous")}, now=100.1)
    assert [e.event_type for e in events] == ["ambiguity_enter"]
    events = eng.detect_frame({0: _state(certainty="certain")}, now=100.2)
    assert [e.event_type for e in events] == ["ambiguity_exit"]


def test_intensity_change_threshold():
    eng = EventEngine(intensity_threshold=20.0)
    eng.detect_frame({0: _state(intensity=40.0)}, now=100.0)
    # small change: no event
    events = eng.detect_frame({0: _state(intensity=50.0)}, now=100.1)
    assert events == []
    # large change: event
    events = eng.detect_frame({0: _state(intensity=80.0)}, now=100.2)
    assert [e.event_type for e in events] == ["intensity_change"]


def test_confidence_drop():
    eng = EventEngine(confidence_drop_threshold=20.0)
    eng.detect_frame({0: _state(confidence=0.9)}, now=100.0)
    events = eng.detect_frame({0: _state(confidence=0.65)}, now=100.1)
    assert [e.event_type for e in events] == ["confidence_drop"]
    assert events[0].metadata["drop_pp"] == 25.0


def test_face_exit_after_timeout():
    eng = EventEngine(face_exit_seconds=2.0)
    eng.detect_frame({0: _state()}, now=100.0)
    # still missing but under threshold
    events = eng.detect_frame({}, now=101.0)
    assert events == []
    # beyond threshold
    events = eng.detect_frame({}, now=102.5)
    assert [e.event_type for e in events] == ["face_exit"]


def test_stabilization_after_held_time():
    eng = EventEngine(stabilization_seconds=3.0)
    eng.detect_frame({0: _state(emotion="sad")}, now=100.0)
    events = eng.detect_frame({0: _state(emotion="sad")}, now=104.0)
    stabs = [e for e in events if e.event_type == "stabilization"]
    assert len(stabs) == 1
    assert stabs[0].metadata["held_seconds"] == 4.0
    # does not re-fire while still held
    events = eng.detect_frame({0: _state(emotion="sad")}, now=105.0)
    assert [e for e in events if e.event_type == "stabilization"] == []


def test_multi_face_independent_tracks():
    eng = EventEngine()
    eng.detect_frame({0: _state(), 1: _state()}, now=100.0)
    events = eng.detect_frame({0: _state(emotion="happy"), 1: _state()}, now=100.1)
    transitions = [e for e in events if e.event_type == "transition"]
    assert len(transitions) == 1
    assert transitions[0].face_id == 0


def test_transition_debounce_blocks_flicker():
    """A->B->A oscillation must never produce a transition event."""
    eng = EventEngine(transition_debounce_seconds=1.0)
    eng.detect_frame({0: _state(emotion="neutral")}, now=100.0)
    t = 100.1
    for emo in ["happy", "neutral", "happy", "neutral", "happy", "neutral", "happy"]:
        events = eng.detect_frame({0: _state(emotion=emo)}, now=t)
        assert [e for e in events if e.event_type == "transition"] == []
        t += 0.1


def test_transition_debounce_fires_once_after_stable_hold():
    eng = EventEngine(transition_debounce_seconds=1.0)
    eng.detect_frame({0: _state(emotion="neutral")}, now=100.0)
    eng.detect_frame({0: _state(emotion="happy")}, now=100.2)  # start debounce
    events = eng.detect_frame({0: _state(emotion="happy")}, now=101.0)  # 0.8s < 1.0
    assert [e for e in events if e.event_type == "transition"] == []
    events = eng.detect_frame({0: _state(emotion="happy")}, now=101.3)  # 1.1s held
    assert [e.event_type for e in events if e.event_type == "transition"] == ["transition"]
    # stays fired exactly once
    events = eng.detect_frame({0: _state(emotion="happy")}, now=102.0)
    assert [e for e in events if e.event_type == "transition"] == []


def test_face_enter_fires_again_after_exit():
    eng = EventEngine(face_exit_seconds=2.0)
    eng.detect_frame({0: _state(emotion="happy")}, now=100.0)
    events = eng.detect_frame({}, now=103.0)
    assert [e.event_type for e in events] == ["face_exit"]
    events = eng.detect_frame({0: _state(emotion="neutral")}, now=104.0)
    enters = [e for e in events if e.event_type == "face_enter"]
    assert len(enters) == 1
    assert enters[0].metadata.get("reentry") is True
    # no stale transition from before the exit
    assert [e for e in events if e.event_type == "transition"] == []


# ---- Narrator ----


def _event(event_type="transition", face_id=0, **kw):
    from src.models.event_engine import Event

    ev = Event(event_type=event_type, face_id=face_id, timestamp=time.time(), **kw)
    return ev


def test_narrator_transition_template():
    nar = Narrator()
    ev = _event(from_emotion="neutral", to_emotion="happy", confidence=0.91)
    lines = nar.narrate([ev])
    assert len(lines) == 1
    assert lines[0]["text"] == "Person 01 changed from Neutral to Happy. Confidence 91%."


def test_narrator_single_subject_always():
    """Even for a foreign face_id, the narrator speaks about Person 01."""
    nar = Narrator()
    lines = nar.narrate([_event(event_type="face_enter", face_id=7)])
    assert lines[0]["text"] == "Person 01 entered the scene."


def test_narrator_face_enter_exit_templates():
    nar = Narrator()
    lines = nar.narrate([_event(event_type="face_enter", face_id=2)])
    assert lines[0]["text"] == "Person 01 entered the scene."
    lines = nar.narrate([_event(event_type="face_exit", face_id=0)])
    assert lines[0]["text"] == "Person 01 left the scene."


def test_narrator_intensity_template():
    nar = Narrator()
    ev = _event(event_type="intensity_change", to_emotion="angry",
                metadata={"direction": "increased"})
    lines = nar.narrate([ev])
    assert lines[0]["text"] == "Person 01's Angry expression grew stronger."
    ev = _event(event_type="intensity_change", to_emotion="sad",
                metadata={"direction": "decreased"})
    lines = nar.narrate([ev])
    assert lines[0]["text"] == "Person 01's Sad expression grew weaker."


def test_narrator_ambiguity_template():
    nar = Narrator()
    ev = _event(event_type="ambiguity_enter", certainty="ambiguous")
    lines = nar.narrate([ev])
    assert lines[0]["text"] == "Person 01's expression became ambiguous."


def test_narrator_skips_chatty_events():
    """confidence_drop and ambiguity_exit are analytics-only at normal detail."""
    nar = Narrator()
    lines = nar.narrate([
        _event(event_type="confidence_drop", metadata={"drop_pp": 25}),
        _event(event_type="ambiguity_exit", certainty="certain"),
    ])
    assert lines == []


def test_narrator_detail_levels():
    intensity = _event(event_type="intensity_change", to_emotion="angry",
                       metadata={"direction": "increased"})
    drop = _event(event_type="confidence_drop", metadata={"drop_pp": 25})
    trans = _event(from_emotion="neutral", to_emotion="happy")

    calm = Narrator(detail="calm")
    assert [l["event_type"] for l in calm.narrate([intensity, trans])] == ["transition"]

    normal = Narrator(detail="normal")
    assert len(normal.narrate([intensity, drop])) == 1  # intensity yes, drop no

    detailed = Narrator(detail="detailed")
    assert len(detailed.narrate([drop])) == 1

    # unknown detail falls back to normal
    fallback = Narrator(detail="loud")
    assert fallback.detail == "normal"


def test_narrator_stabilization_template():
    nar = Narrator()
    ev = _event(event_type="stabilization", to_emotion="sad",
                metadata={"held_seconds": 3.5})
    lines = nar.narrate([ev])
    assert lines[0]["text"] == "Person 01 has held Sad steady for 3.5 seconds."


def test_narrator_evidence_grounding_on_transition():
    nar = Narrator()
    ev = _event(from_emotion="neutral", to_emotion="happy", confidence=0.9)
    ctx = {0: {"regions": {"mouth": 55.0, "eyes": 20.0}}}
    lines = nar.narrate([ev], face_context=ctx)
    assert "Strongest evidence: mouth region." in lines[0]["text"]


def test_narrator_disabled_returns_empty():
    nar = Narrator(enabled=False)
    assert nar.narrate([_event()]) == []


def test_narrator_respects_line_budget():
    nar = Narrator()
    events = [_event(event_type="face_enter", face_id=i) for i in range(10)]
    lines = nar.narrate(events)
    assert len(lines) <= nar._line_budget * 2


def test_narrator_describe_face():
    nar = Narrator()
    line = nar.describe_face(0, {
        "emotion": "surprise", "confidence": 0.87, "certainty": "certain",
        "regions": {"eyes": 40.0},
    })
    assert line == "Person 01 shows Surprise (87%). Model attention: eye region."
