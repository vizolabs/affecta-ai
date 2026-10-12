"""API integration tests: auth, settings, WS stream, research-lab analyze,
sessions/analytics/export — with a fake inference service (no model load).
"""

import json
import os
import tempfile
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.tracker import FaceTracker, iou

NEUTRAL = np.array([0.05, 0.05, 0.05, 0.10, 0.60, 0.05, 0.10])
HAPPY = np.array([0.05, 0.05, 0.05, 0.70, 0.05, 0.05, 0.05])


class FakeService:
    model_name = "fake"
    loaded = True

    def __init__(self):
        self.detect_result = [(10, 10, 60, 60)]
        self.probs_script = []
        self.default_probs = NEUTRAL.copy()

    def detect_faces(self, frame):
        return list(self.detect_result)

    def predict_probs(self, face):
        if self.probs_script:
            return self.probs_script.pop(0).copy()
        return self.default_probs.copy()

    def gradcam(self, face, target_class=None):
        h, w = face.shape[:2]
        rng = np.random.default_rng(0)
        cam = rng.random((h, w)).astype(np.float32) * 0.3
        cam[int(h * 0.70):, int(w * 0.30):int(w * 0.70)] += 2.0
        return cam


@pytest.fixture()
def client():
    with TestClient(app) as c:
        c.app.state.service = FakeService()
        yield c


@pytest.fixture()
def fake(client):
    return client.app.state.service


def make_jpeg(h=160, w=160) -> bytes:
    img = np.full((h, w, 3), (180, 160, 140), np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def auth_headers(client, password="secret123"):
    email = f"{uuid.uuid4().hex[:12]}@user.com"
    r = client.post(
        "/api/auth/register", json={"email": email, "password": password}
    )
    assert r.status_code == 200, r.text
    token = r.json()["data"]["token"]
    return {"Authorization": f"Bearer {token}"}, token, email


# ---- health / auth ----


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["data"]["status"] == "ok"
    assert isinstance(body["data"]["model_loaded"], bool)


def test_register_login_me_logout(client):
    email = f"{uuid.uuid4().hex[:12]}@user.com"
    r = client.post("/api/auth/register", json={"email": email, "password": "secret123"})
    assert r.status_code == 200
    token = r.json()["data"]["token"]

    r = client.post("/api/auth/login", json={"email": email, "password": "wrong"})
    assert r.status_code == 401

    r = client.post("/api/auth/login", json={"email": email, "password": "secret123"})
    assert r.status_code == 200
    token = r.json()["data"]["token"]

    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["data"]["email"] == email

    r = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_register_duplicate_email(client):
    email = f"{uuid.uuid4().hex[:12]}@user.com"
    payload = {"email": email, "password": "secret123"}
    assert client.post("/api/auth/register", json=payload).status_code == 200
    assert client.post("/api/auth/register", json=payload).status_code == 409


def test_register_password_policy(client):
    """Spec 12: 8-char minimum on register; oversized passwords rejected."""
    email = f"{uuid.uuid4().hex[:12]}@user.com"
    r = client.post("/api/auth/register", json={"email": email, "password": "short7c"})
    assert r.status_code == 422
    # Login rejects oversized passwords before PBKDF2 hashing (DoS guard).
    r = client.post("/api/auth/login", json={"email": email, "password": "x" * 5000})
    assert r.status_code == 422


def test_settings_clamps(client):
    """Out-of-range settings (e.g. ema_alpha > 1) must be rejected, not stored."""
    email = f"{uuid.uuid4().hex[:12]}@user.com"
    r = client.post("/api/auth/register", json={"email": email, "password": "secret123"})
    headers = {"Authorization": f"Bearer {r.json()['data']['token']}"}

    r = client.put("/api/settings", json={"ema_alpha": 5.0}, headers=headers)
    assert r.status_code == 422

    r = client.put("/api/settings", json={"narrator_speed": 9.0}, headers=headers)
    assert r.status_code == 422

    r = client.put("/api/settings", json={"narrator_detail": "verbose"}, headers=headers)
    assert r.status_code == 422

    r = client.put("/api/settings", json={"ema_alpha": 0.4}, headers=headers)
    assert r.status_code == 200
    assert r.json()["data"]["ema_alpha"] == 0.4


def test_register_local_domain_email(client):
    """Local-first: internal domains like .local are accepted (no mail sent)."""
    email = f"{uuid.uuid4().hex[:12]}@civicpulse.local"
    r = client.post("/api/auth/register", json={"email": email, "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["data"]["user"]["email"] == email.lower()

    r = client.post("/api/auth/login", json={"email": email, "password": "secret123"})
    assert r.status_code == 200


def test_register_invalid_email_422(client):
    """FastAPI 422 detail is an array — shape asserted for frontend parsing."""
    r = client.post(
        "/api/auth/register", json={"email": "notanemail", "password": "secret123"}
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list) and len(detail) >= 1
    assert "@" in detail[0]["msg"]


def test_unauthenticated_rejected(client):
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/settings").status_code == 401
    assert client.get("/api/models").status_code == 401


def test_login_rate_limit(client):
    email = f"{uuid.uuid4().hex[:12]}@user.com"
    for _ in range(10):
        r = client.post("/api/auth/login", json={"email": email, "password": "nope"})
        assert r.status_code == 401
    r = client.post("/api/auth/login", json={"email": email, "password": "nope"})
    assert r.status_code == 429


# ---- settings ----


def test_settings_roundtrip(client):
    headers, _, _ = auth_headers(client)
    r = client.get("/api/settings", headers=headers)
    assert r.status_code == 200
    body = r.json()["data"]
    assert body["narrator_enabled"] is True
    assert body["narrator_detail"] == "normal"
    assert body["ema_alpha"] == 0.35
    assert body["analysis_fps"] == 8
    assert body["emotion_hold_seconds"] == 0.0
    assert body["analysis_mode"] == "live"
    assert body["analysis_interval_seconds"] == 120
    assert body["motion_trigger_enabled"] is True

    r = client.put(
        "/api/settings",
        headers=headers,
        json={
            "narrator_enabled": False,
            "xai_frame_interval": 5,
            "narrator_detail": "calm",
            "ema_alpha": 0.5,
            "analysis_fps": 3,
            "emotion_hold_seconds": 2.5,
            "analysis_mode": "periodic",
            "analysis_interval_seconds": 90,
            "motion_trigger_enabled": False,
        },
    )
    assert r.status_code == 200
    body = r.json()["data"]
    assert body["narrator_enabled"] is False
    assert body["xai_frame_interval"] == 5
    assert body["narrator_detail"] == "calm"
    assert body["ema_alpha"] == 0.5
    assert body["analysis_fps"] == 3
    assert body["emotion_hold_seconds"] == 2.5
    assert body["analysis_mode"] == "periodic"
    assert body["analysis_interval_seconds"] == 90
    assert body["motion_trigger_enabled"] is False

    # Persisted across a fresh read (round-trip through SQLite).
    r = client.get("/api/settings", headers=headers)
    assert r.json()["data"]["analysis_fps"] == 3
    assert r.json()["data"]["emotion_hold_seconds"] == 2.5
    assert r.json()["data"]["analysis_mode"] == "periodic"
    assert r.json()["data"]["analysis_interval_seconds"] == 90
    assert r.json()["data"]["motion_trigger_enabled"] is False


def test_settings_reject_out_of_range(client):
    headers, _, _ = auth_headers(client)
    # frame rate bounds: 1..15
    assert client.put("/api/settings", headers=headers, json={"analysis_fps": 0}).status_code == 422
    assert client.put("/api/settings", headers=headers, json={"analysis_fps": 99}).status_code == 422
    # hold bounds: 0..600 seconds (minute-scale holds allowed)
    assert (
        client.put("/api/settings", headers=headers, json={"emotion_hold_seconds": -1}).status_code
        == 422
    )
    assert (
        client.put("/api/settings", headers=headers, json={"emotion_hold_seconds": 601}).status_code
        == 422
    )
    # boundaries are accepted
    assert client.put("/api/settings", headers=headers, json={"analysis_fps": 1}).status_code == 200
    assert client.put("/api/settings", headers=headers, json={"analysis_fps": 15}).status_code == 200
    assert (
        client.put("/api/settings", headers=headers, json={"emotion_hold_seconds": 0}).status_code
        == 200
    )
    # minute-scale hold is representable
    assert (
        client.put(
            "/api/settings", headers=headers, json={"emotion_hold_seconds": 120}
        ).status_code
        == 200
    )


def test_settings_cadence_bounds(client):
    headers, _, _ = auth_headers(client)
    # only live/periodic are valid modes
    assert (
        client.put(
            "/api/settings", headers=headers, json={"analysis_mode": "realtime"}
        ).status_code
        == 422
    )
    assert (
        client.put("/api/settings", headers=headers, json={"analysis_mode": "periodic"}).status_code
        == 200
    )
    # interval bounds: 30s .. 600s
    assert (
        client.put(
            "/api/settings", headers=headers, json={"analysis_interval_seconds": 29}
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/settings", headers=headers, json={"analysis_interval_seconds": 601}
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/settings", headers=headers, json={"analysis_interval_seconds": 30}
        ).status_code
        == 200
    )
    assert (
        client.put(
            "/api/settings", headers=headers, json={"analysis_interval_seconds": 600}
        ).status_code
        == 200
    )


# ---- tracker ----


def test_tracker_stable_ids():
    tracker = FaceTracker()
    box = (10, 10, 60, 60)
    first = tracker.update([box])
    assert list(first.keys()) == [0]
    second = tracker.update([(12, 12, 60, 60)])
    assert list(second.keys()) == [0]
    third = tracker.update([(400, 400, 60, 60)])
    assert list(third.keys()) == [1]


def test_tracker_ghost_then_new():
    tracker = FaceTracker(max_age=2)
    tracker.update([(10, 10, 60, 60)])
    active = tracker.update([])
    assert active == {}
    tracker.update([(10, 12, 60, 60)])
    active = tracker.update([(10, 12, 60, 60)])
    assert list(active.keys()) == [0]


def test_tracker_single_face_mode():
    """Product mode: only one face is ever tracked, pinned to id 0."""
    tracker = FaceTracker(max_faces=1)

    # two faces -> the larger one wins, id 0
    active = tracker.update([(10, 10, 40, 40), (100, 100, 80, 80)])
    assert list(active.keys()) == [0]
    assert active[0] == (100, 100, 80, 80)

    # a different (smaller) face appears alone -> still id 0, never id 1
    active = tracker.update([(10, 10, 40, 40)])
    assert list(active.keys()) == [0]

    # dropout (ghost) then reappearance somewhere else -> still id 0
    assert tracker.update([]) == {}
    active = tracker.update([(300, 300, 50, 50)])
    assert list(active.keys()) == [0]

    # many faces in frame -> exactly one track, id 0
    active = tracker.update(
        [(10, 10, 60, 60), (500, 500, 90, 90), (200, 20, 30, 30)]
    )
    assert list(active.keys()) == [0]
    assert active[0] == (500, 500, 90, 90)


def test_tracker_single_face_keeps_same_human_via_iou():
    """Continuity: keep following the tracked human even if another face is larger."""
    tracker = FaceTracker(max_faces=1)
    tracker.update([(10, 10, 60, 60)])  # id 0 on the smaller face
    # a larger face appears elsewhere AND our face moves slightly (IoU ok):
    # IoU-continuity must keep our face, not jump to the larger one.
    active = tracker.update([(12, 12, 60, 60), (200, 200, 100, 100)])
    assert list(active.keys()) == [0]
    assert active[0] == (12, 12, 60, 60)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


# ---- WebSocket stream ----


def test_websocket_stream_events_and_narration(client, fake):
    headers, token, _ = auth_headers(client)
    fake.probs_script = [NEUTRAL, HAPPY]

    with client.websocket_connect(f"/ws/emotion-stream?token={token}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "session"
        session_id = hello["session_id"]

        ws.send_bytes(make_jpeg())
        out = ws.receive_json()
        assert out["type"] == "analysis"
        assert out["session_id"] == session_id
        assert out["model"] == "fake"
        assert len(out["faces"]) == 1
        face = out["faces"][0]
        assert face["emotion"] == "neutral"
        assert face["action_units"] is None
        assert face["au_model_status"] == "untrained"
        assert set(face["probabilities"]) == {
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
        }
        event_types = [e["event_type"] for e in out["events"]]
        assert "face_enter" in event_types

        ws.send_bytes(make_jpeg())
        out = ws.receive_json()
        event_types = [e["event_type"] for e in out["events"]]
        assert "transition" in event_types
        assert any("changed from Neutral to Happy" in line["text"]
                   for line in out["narration"])
        assert out["faces"][0]["emotion"] == "happy"

        ws.send_text(json.dumps({"action": "ping"}))
        assert ws.receive_json()["type"] == "pong"

        ws.send_text(json.dumps({"action": "xai", "face_id": 0}))
        xai = ws.receive_json()
        assert xai["type"] == "xai"
        assert xai["face_id"] == 0
        assert xai["method"] == "gradcam"
        assert "heatmap" in xai and xai["heatmap"]
        assert "mouth" in xai["regions"]

    r = client.get("/api/sessions", headers=headers)
    sessions = r.json()["data"]
    assert len(sessions) >= 1
    sess = sessions[0]
    assert sess["total_frames"] == 2
    assert sess["total_events"] >= 2
    assert sess["metadata"]["timeline"]
    assert sess["metadata"]["model"] == "fake"

    r = client.get(f"/api/sessions/{sess['id']}/events", headers=headers)
    events = r.json()["data"]
    types = [e["event_type"] for e in events]
    assert "face_enter" in types
    assert "transition" in types


def test_websocket_bad_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/emotion-stream?token=bogus.token.here") as ws:
            ws.receive_json()


def test_websocket_token_via_first_message(client, fake):
    headers, token, _ = auth_headers(client)
    with client.websocket_connect("/ws/emotion-stream") as ws:
        ws.send_text(token)
        hello = ws.receive_json()
        assert hello["type"] == "session"


def test_websocket_bad_frame_error(client):
    _, token, _ = auth_headers(client)
    with client.websocket_connect(f"/ws/emotion-stream?token={token}") as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(b"not-a-jpeg")
        err = ws.receive_json()
        assert err["type"] == "error"


# ---- Research Lab analyze ----


def test_analyze_upload_with_xai(client):
    headers, _, _ = auth_headers(client)
    jpeg = make_jpeg()
    r = client.post(
        "/api/analyze",
        headers=headers,
        files={"file": ("face.jpg", jpeg, "image/jpeg")},
        data={"xai": "true", "face_id": "0"},
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert len(data["faces"]) == 1
    assert data["xai"]["method"] == "gradcam"
    assert data["xai"]["regions"]


def test_analyze_rejects_garbage(client):
    headers, _, _ = auth_headers(client)
    r = client.post(
        "/api/analyze",
        headers=headers,
        files={"file": ("bad.jpg", b"nonsense", "image/jpeg")},
    )
    assert r.status_code == 400


def make_mp4(frames: int = 12, fps: float = 10.0, size: int = 160) -> bytes:
    """Build a tiny decodable mp4 in memory (OpenCV cannot encode to a buffer)."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
        )
        assert writer.isOpened()
        for i in range(frames):
            img = np.full((size, size, 3), (180, 160, 140), np.uint8)
            cv2.putText(img, str(i), (10, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2)
            writer.write(img)
        writer.release()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.remove(path)


def test_analyze_video_upload(client, fake):
    headers, _, _ = auth_headers(client)
    # Long enough (4s) for the live-style 1.0s transition debounce to elapse.
    fake.probs_script = [NEUTRAL] * 6 + [HAPPY] * 60
    clip = make_mp4(frames=40, fps=10.0)
    r = client.post(
        "/api/analyze/video",
        headers=headers,
        files={"file": ("clip.mp4", clip, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["kind"] == "video"
    assert data["frames_analyzed"] > 0
    assert data["duration_seconds"] > 0
    assert data["sample_fps"] >= 1
    # the sampled frames were actually classified
    assert data["emotion_distribution"]
    assert data["dominant_emotion"] == "happy"
    assert 0 <= data["avg_confidence"] <= 1
    assert data["face_coverage"] > 0
    # peak frame + its Grad-CAM make the XAI panel meaningful for video too
    assert data["peak_frame"]["emotion"] == "happy"
    assert data["xai"]["method"] == "gradcam"
    assert data["xai"]["regions"]


def test_analyze_video_suppresses_transition_flicker(client, fake):
    """A clip whose label alternates must not report a transition per sample."""
    headers, _, _ = auth_headers(client)
    alternating = [NEUTRAL if i % 2 else HAPPY for i in range(120)]
    fake.probs_script = alternating
    clip = make_mp4(frames=60, fps=10.0)
    r = client.post(
        "/api/analyze/video",
        headers=headers,
        files={"file": ("flicker.mp4", clip, "video/mp4")},
        # the test env zeroes the server debounce, so request it explicitly
        data={"debounce_seconds": "1.0"},
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    # Without debounce this would be ~25 transitions.
    assert data["transition_count"] <= 2


def test_analyze_video_rejects_bad_input(client):
    headers, _, _ = auth_headers(client)
    r = client.post(
        "/api/analyze/video",
        headers=headers,
        files={"file": ("bad.mp4", b"definitely not a video", "video/mp4")},
    )
    assert r.status_code == 400

    r = client.post(
        "/api/analyze/video",
        headers=headers,
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert r.status_code == 400


def test_analyze_video_requires_auth(client):
    r = client.post(
        "/api/analyze/video",
        files={"file": ("clip.mp4", make_mp4(frames=4), "video/mp4")},
    )
    assert r.status_code in (401, 403)


def test_models_registry(client):
    headers, _, _ = auth_headers(client)
    r = client.get("/api/models", headers=headers)
    assert r.status_code == 200
    rows = r.json()["data"]
    names = {m["model_name"] for m in rows}
    assert {"dan", "efficientnet_b3", "ensemble"} <= names
    by_name = {m["model_name"]: m for m in rows}
    # normalized metric keys (no internal shorthand leaked)
    assert "test_acc" in by_name["dan"]["metrics"]
    assert "macro_f1_or_bacc" not in (by_name["dan"]["metrics"] or {})
    # B3 self-heals to the finished EXP-007 run
    assert by_name["efficientnet_b3"]["metrics"]["test_acc"] == 66.49
    assert by_name["ensemble"]["metrics"]["test_acc"] == 89.99
    assert by_name["dan"]["status"] == "deployed"


def test_delete_account(client):
    headers, token, email = auth_headers(client)
    r = client.delete("/api/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["data"]["deleted"] is True
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    r = client.post("/api/auth/login", json={"email": email, "password": "secret123"})
    assert r.status_code == 401


def test_websocket_session_title(client, fake):
    headers, token, _ = auth_headers(client)
    with client.websocket_connect(
        f"/ws/emotion-stream?token={token}&title=Team+standup"
    ) as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(make_jpeg())
        ws.receive_json()
    r = client.get("/api/sessions", headers=headers)
    sess = r.json()["data"][0]
    assert sess["metadata"]["title"] == "Team standup"


def test_websocket_records_applied_fps(client, fake):
    headers, token, _ = auth_headers(client)
    with client.websocket_connect(
        f"/ws/emotion-stream?token={token}&fps=3"
    ) as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(make_jpeg())
        ws.receive_json()
    r = client.get("/api/sessions", headers=headers)
    sess = r.json()["data"][0]
    assert sess["metadata"]["applied_fps"] == 3
    assert "emotion_hold_seconds" in sess["metadata"]


def test_websocket_records_periodic_cadence(client, fake):
    headers, token, _ = auth_headers(client)
    with client.websocket_connect(
        f"/ws/emotion-stream?token={token}&mode=periodic&interval=90&motion=1"
    ) as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(make_jpeg())
        ws.receive_json()
    sess = client.get("/api/sessions", headers=headers).json()["data"][0]
    meta = sess["metadata"]
    assert meta["analysis_mode"] == "periodic"
    assert meta["applied_interval_seconds"] == 90
    assert meta["motion_trigger"] is True


def test_websocket_clamps_periodic_cadence(client, fake):
    """Out-of-range cadence values are clamped, never trusted blindly."""
    headers, token, _ = auth_headers(client)
    with client.websocket_connect(
        f"/ws/emotion-stream?token={token}&mode=periodic&interval=5&motion=0"
    ) as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(make_jpeg())
        ws.receive_json()
    meta = client.get("/api/sessions", headers=headers).json()["data"][0]["metadata"]
    assert meta["applied_interval_seconds"] == 30  # floor
    assert meta["motion_trigger"] is False


def test_websocket_live_mode_ignores_interval(client, fake):
    """Live sessions must not report a periodic interval."""
    headers, token, _ = auth_headers(client)
    with client.websocket_connect(
        f"/ws/emotion-stream?token={token}&mode=live&interval=300"
    ) as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(make_jpeg())
        ws.receive_json()
    meta = client.get("/api/sessions", headers=headers).json()["data"][0]["metadata"]
    assert meta["analysis_mode"] == "live"
    assert meta["applied_interval_seconds"] is None


def test_account_delete_during_live_session(client, fake):
    """Deleting the account mid-stream must not break session finalisation."""
    headers, token, _ = auth_headers(client)
    with client.websocket_connect(f"/ws/emotion-stream?token={token}") as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(make_jpeg())
        ws.receive_json()
        # Danger zone fires while the stream is still open.
        r = client.delete("/api/auth/me", headers=headers)
        assert r.status_code == 200
    # The stream closes cleanly afterwards; the session is gone with the account.
    assert client.get("/api/sessions", headers=headers).status_code == 401


# ---- analytics / export ----


def _recorded_session(client, headers, fake):
    _, token, _ = headers["Authorization"], None, None
    token = headers["Authorization"].split(" ", 1)[1]
    fake.probs_script = [NEUTRAL, HAPPY]
    with client.websocket_connect(f"/ws/emotion-stream?token={token}") as ws:
        assert ws.receive_json()["type"] == "session"
        ws.send_bytes(make_jpeg())
        ws.receive_json()
        ws.send_bytes(make_jpeg())
        ws.receive_json()
    r = client.get("/api/sessions", headers=headers)
    return r.json()["data"][0]


def test_analytics_summary_timeline_and_exports(client, fake):
    headers, _, _ = auth_headers(client)
    sess = _recorded_session(client, headers, fake)
    sid = sess["id"]

    r = client.get(f"/api/analytics/summary?session_id={sid}", headers=headers)
    assert r.status_code == 200
    summary = r.json()["data"]
    assert summary["transition_count"] >= 1
    assert summary["expression_distribution"]
    assert summary["model"] == "fake"

    r = client.get(f"/api/analytics/timeline?session_id={sid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["data"]["timeline"]

    r = client.get(f"/api/analytics/export/{sid}?format=csv", headers=headers)
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "transition" in r.text

    r = client.get(f"/api/analytics/export/{sid}?format=json", headers=headers)
    assert r.status_code == 200
    payload = json.loads(r.text)
    assert payload["summary"]["transition_count"] >= 1

    r = client.get(f"/api/analytics/export/{sid}?format=pdf", headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert r.content.rstrip().endswith(b"%%EOF")

    r = client.get(f"/api/analytics/export/{sid}?format=docx", headers=headers)
    assert r.status_code == 400


def test_session_ownership(client):
    headers_a, _, _ = auth_headers(client)
    headers_b, _, _ = auth_headers(client)
    sess = _recorded_session(client, headers_a, FakeService())

    r = client.get(f"/api/sessions/{sess['id']}", headers=headers_b)
    assert r.status_code == 404

    r = client.delete(f"/api/sessions/{sess['id']}", headers=headers_b)
    assert r.status_code == 404

    r = client.delete(f"/api/sessions/{sess['id']}", headers=headers_a)
    assert r.status_code == 200
    r = client.get(f"/api/sessions/{sess['id']}", headers=headers_a)
    assert r.status_code == 404
