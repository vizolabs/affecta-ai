"""Session summary + export builders (CSV / JSON / minimal PDF).

docs/15: PDF report sections, CSV export, analytics queries.
The PDF is a stdlib one-page-or-paginated text report (no external deps).
"""

from __future__ import annotations

import csv
import io
import json
from typing import Iterable, List, Tuple


def compute_summary(row, events, timeline) -> dict:
    dist: dict = {}
    per_person: dict = {}
    for sample in timeline or []:
        for face in sample.get("faces", []):
            emotion = face.get("emotion") or "unknown"
            dist[emotion] = dist.get(emotion, 0) + 1
            person = per_person.setdefault(
                str(face.get("id")), {"samples": 0, "emotions": {}}
            )
            person["samples"] += 1
            person["emotions"][emotion] = person["emotions"].get(emotion, 0) + 1
    for person in per_person.values():
        person["dominant"] = max(person["emotions"], key=person["emotions"].get)

    transitions = [e for e in events if e.event_type == "transition"]
    matrix: dict = {}
    for e in transitions:
        key = f"{e.from_emotion}->{e.to_emotion}"
        matrix[key] = matrix.get(key, 0) + 1

    type_counts: dict = {}
    for e in events:
        type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

    meta = row.meta or {}
    # Prefer live event rows over the denormalised session columns: those are
    # only written when the WS closes, so a still-open or server-killed
    # session would otherwise report "Events: 0" next to real transitions.
    return {
        "session_id": row.id,
        "duration_seconds": row.duration_seconds,
        "total_frames": row.total_frames if row.total_events is not None else len(events),
        "total_events": len(events) if events else (row.total_events or 0),
        "avg_confidence": row.avg_confidence,
        "avg_intensity": row.avg_intensity,
        "model": meta.get("model"),
        "expression_distribution": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
        "event_type_counts": type_counts,
        "transition_count": len(transitions),
        "transition_matrix": matrix,
        "uncertainty_events": type_counts.get("ambiguity_enter", 0)
        + type_counts.get("confidence_drop", 0),
        "per_person": per_person,
    }


def events_csv(events: Iterable) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "timestamp", "face_id", "event_type", "from_emotion", "to_emotion",
        "confidence", "intensity", "certainty",
    ])
    for e in events:
        writer.writerow([
            e.timestamp.isoformat() if hasattr(e.timestamp, "isoformat") else e.timestamp,
            e.face_id,
            e.event_type,
            e.from_emotion or "",
            e.to_emotion or "",
            "" if e.confidence is None else round(e.confidence, 4),
            "" if e.intensity is None else round(e.intensity, 2),
            e.certainty or "",
        ])
    return buf.getvalue()


def export_json(row, summary: dict, events: List, timeline: List) -> str:
    from src.api.schemas import event_out, session_out

    payload = {
        "session": session_out(row).model_dump(mode="json"),
        "summary": summary,
        "timeline": timeline or [],
        "events": [event_out(e).model_dump(mode="json") for e in events],
    }
    return json.dumps(payload, indent=2, default=str)


def _pdf_escape(text: str) -> str:
    cleaned = text.encode("latin-1", errors="replace").decode("latin-1")
    return cleaned.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(title: str, sections: List[Tuple[str, List[str]]]) -> bytes:
    display: List[Tuple[bool, str]] = [(True, title), (False, "")]
    for heading, lines in sections:
        display.append((True, heading))
        for line in lines:
            display.append((False, line))
        display.append((False, ""))

    per_page = 44
    pages = [display[i:i + per_page] for i in range(0, len(display), per_page)] or [[]]
    n_pages = len(pages)
    font_obj = 3 + 2 * n_pages

    bodies: List[bytes] = []
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    bodies.append("<< /Type /Catalog /Pages 2 0 R >>".encode())
    bodies.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())

    for i, page_lines in enumerate(pages):
        page_obj = 3 + 2 * i
        content_obj = page_obj + 1
        parts = ["BT", "/F1 10 Tf", "13 TL", "40 752 Td"]
        for is_heading, line in page_lines:
            font = "/F2 12 Tf" if is_heading else "/F1 10 Tf"
            parts.append(font)
            parts.append(f"({_pdf_escape(line)}) Tj T*")
            if is_heading:
                parts.append("/F1 10 Tf")
        parts.append("ET")
        stream = "\n".join(parts).encode("latin-1", errors="replace")
        bodies.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_obj} 0 R "
                f"/Resources << /Font << /F1 {font_obj} 0 R /F2 {font_obj + 1} 0 R >> >> >>"
            ).encode()
        )
        bodies.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    bodies.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    bodies.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)


def report_pdf(row, summary: dict, events: List, timeline: List) -> bytes:
    dist = summary.get("expression_distribution", {})
    dist_lines = [f"{k}: {v}" for k, v in dist.items()] or ["(no timeline samples)"]
    per_person_lines = [
        f"Person {int(pid) + 1:02d}: dominant={info.get('dominant')} "
        f"({info.get('samples')} samples)"
        for pid, info in summary.get("per_person", {}).items()
    ] or ["(no faces observed)"]
    matrix = summary.get("transition_matrix", {})
    transition_lines = [f"{k}: {v}" for k, v in matrix.items()] or ["(none)"]
    uncertainty = [e for e in events if e.event_type in ("ambiguity_enter", "confidence_drop")]
    uncertainty_lines = [
        f"{e.timestamp.isoformat()} face={e.face_id} {e.event_type}"
        for e in uncertainty[:10]
    ] or ["(none)"]
    recent = events[-20:]
    event_lines = [
        f"{e.timestamp.isoformat()} f{e.face_id} {e.event_type} "
        f"{e.from_emotion or ''}->{e.to_emotion or ''}".strip()
        for e in recent
    ] or ["(none)"]

    sections = [
        ("1. Session Overview", [
            f"Session: {row.id}",
            f"Started: {row.start_time.isoformat() if row.start_time else 'unknown'}",
            f"Model: {summary.get('model') or 'unknown'}  (RAF-DB test n=3,068)",
            f"Duration: {row.duration_seconds}s  Frames: {row.total_frames}  "
            f"Events: {row.total_events}",
            f"Avg confidence: {row.avg_confidence}  Avg intensity: {row.avg_intensity}",
            f"Transitions: {summary.get('transition_count')}  "
            f"Uncertainty events: {summary.get('uncertainty_events')}",
        ]),
        ("2. Expression Distribution", dist_lines),
        ("3. Per-Person Analysis", per_person_lines),
        ("4. Transition Matrix", transition_lines),
        ("5. Timeline (recent events)", event_lines),
        ("6. Uncertainty Events", uncertainty_lines),
        ("7. XAI Evidence", [
            "Grad-CAM snapshots computed on demand; region relevance stored per request.",
        ]),
        ("Notes — method & honesty", [
            "Single-person tracking: exactly one face per frame (largest /",
            "IoU-continuous), pinned to a single subject.",
            "Action-unit branch untrained: w_AU=0, action_units omitted.",
            "Intensity = distribution-extremity proxy (not a trained regressor).",
            "Region evidence = Grad-CAM mass over geometric face boxes",
            "(no landmark detector).",
            "Narration is deterministic templated text (fully offline).",
            "All processing runs locally; video frames are never stored.",
            "Human Facial Emotion Detection — ranking-based explainable facial emotion detection.",
        ]),
    ]
    return build_pdf("Human Facial Emotion Detection - Session Report", sections)
