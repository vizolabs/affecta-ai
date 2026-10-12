#!/usr/bin/env python3
"""Generate the official Human Facial Emotion Detection end-to-end project report (.docx).

Usage:  python scripts/generate_project_report.py [output.docx]
Requires: python-docx
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ACCENT = RGBColor(0x1F, 0x4E, 0x79)       # dark professional blue
ACCENT_HEX = "1F4E79"
HEADER_HEX = "D9E2F3"
LIGHT_HEX = "F2F2F2"
WARN_HEX = "FFF2CC"


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------

def shade_cell(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def add_field(paragraph, instruction: str, placeholder: str = "") -> None:
    run = paragraph.add_run()
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    f1.set(qn("w:dirty"), "true")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instruction
    f2 = OxmlElement("w:fldChar")
    f2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t")
    t.text = placeholder
    f3 = OxmlElement("w:fldChar")
    f3.set(qn("w:fldCharType"), "end")
    for el in (f1, it, f2, t, f3):
        run._r.append(el)


def set_cell_text(cell, text: str, bold: bool = False, size: int = 9,
                  color: RGBColor | None = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color


# --------------------------------------------------------------------------
# document scaffolding
# --------------------------------------------------------------------------

def new_document() -> Document:
    doc = Document()

    # page setup (A4, ~2 cm margins)
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)

    # base style
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    for name, size, color in (
        ("Heading 1", 16, ACCENT),
        ("Heading 2", 13, ACCENT),
        ("Heading 3", 11.5, ACCENT),
    ):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.font.bold = True
        style.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        style.paragraph_format.space_after = Pt(6)

    # header / footer
    section = doc.sections[0]
    hp = section.header.paragraphs[0]
    hp.text = "Human Facial Emotion Detection — End-to-End Project Report  |  Version 1.0  |  2026-09-23"
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in hp.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    fp = section.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("Page ")
    r.font.size = Pt(8)
    add_field(fp, "PAGE")
    r2 = fp.add_run(" of ")
    r2.font.size = Pt(8)
    add_field(fp, "NUMPAGES")
    r3 = fp.add_run("   ·   Academic use only — not a medical device")
    r3.font.size = Pt(8)
    r3.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    return doc


def h(doc, level: int, text: str) -> None:
    doc.add_heading(text, level=level)


def p(doc, text: str, *, bold=False, italic=False, size=11,
      align=None, space_after=6) -> None:
    para = doc.add_paragraph()
    if align is not None:
        para.alignment = align
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    para.paragraph_format.space_after = Pt(space_after)


def bullets(doc, items, level: int = 0) -> None:
    for item in items:
        para = doc.add_paragraph(style="List Bullet")
        if level:
            para.paragraph_format.left_indent = Cm(1.2 * (level + 1))
        if isinstance(item, tuple):
            lead, rest = item
            r1 = para.add_run(lead)
            r1.bold = True
            para.add_run(rest)
        else:
            para.add_run(item)


def numbered(doc, items) -> None:
    for item in items:
        para = doc.add_paragraph(style="List Number")
        if isinstance(item, tuple):
            lead, rest = item
            r1 = para.add_run(lead)
            r1.bold = True
            para.add_run(rest)
        else:
            para.add_run(item)


def table(doc, headers, rows, widths=None, font_size: int = 9) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0]
    repeat_header(hdr)
    for i, text in enumerate(headers):
        set_cell_text(hdr.cells[i], text, bold=True, size=font_size,
                      color=RGBColor(0xFF, 0xFF, 0xFF))
        shade_cell(hdr.cells[i], ACCENT_HEX)
    for r_idx, row in enumerate(rows):
        cells = t.add_row().cells
        for i, text in enumerate(row):
            set_cell_text(cells[i], str(text), size=font_size)
            if r_idx % 2 == 1:
                shade_cell(cells[i], LIGHT_HEX)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def callout(doc, title: str, body: str, fill: str = WARN_HEX) -> None:
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    shade_cell(cell, fill)
    cell.text = ""
    para = cell.paragraphs[0]
    r1 = para.add_run(title + "  ")
    r1.bold = True
    r1.font.size = Pt(10)
    r2 = para.add_run(body)
    r2.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def page_break(doc) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# --------------------------------------------------------------------------
# report content
# --------------------------------------------------------------------------

def build(output: Path) -> None:
    doc = new_document()

    # ---------------- cover ----------------
    for _ in range(4):
        doc.add_paragraph()
    p(doc, "Human Facial Emotion Detection", bold=True, size=26, align=WD_ALIGN_PARAGRAPH.CENTER,
      space_after=2)
    p(doc, "END-TO-END PROJECT REPORT", bold=True, size=18,
      align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    p(doc, "(Progress, Design, Results & Verification — Full Lifecycle Documentation)",
      size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    p(doc, "A Ranking-Based Explainable AI Model for Human Facial Emotion Detection",
      bold=True, italic=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)

    table(
        doc,
        ["Field", "Detail"],
        [
            ["Document ID", "AFFECTA-RPT-001"],
            ["Version / Status", "1.0 — Final for submission"],
            ["Report date", "23 September 2026"],
            ["Project identity (academic title)",
             "A Ranking-Based Explainable AI Model for Human Facial Emotion Detection"],
            ["Product name", "Human Facial Emotion Detection — Evidence-Ranked Facial Emotion Intelligence"],
            ["Governing specification",
             "docs/00_MASTER_SPECIFICATION.md, V1.0 — APPROVED (2026-09-10)"],
            ["Prepared by", "[Author name / Roll no. / Programme]"],
            ["Institution", "[Department / Institution]"],
            ["Submitted to", "[Reviewing authority / Examiner / Guide]"],
            ["Development environment",
             "macOS (Apple M4, 16 GB), Python 3.12.14, PyTorch 2.14 (MPS), "
             "Node 25.9 / npm 11.12, Next.js 14.2"],
            ["Classification", "Private — Academic Use Only"],
        ],
        widths=[5.0, 11.5],
        font_size=10,
    )
    p(doc,
      "Integrity statement: every metric in this report is produced from on-disk "
      "experiment artifacts in this repository (test_metrics.json / leaderboard "
      "JSON). Model provenance (trained-in-house vs. evaluated-released) is declared "
      "explicitly in Section 6.4. No result has been estimated or fabricated.",
      italic=True, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)

    page_break(doc)

    # ---------------- table of contents ----------------
    h(doc, 1, "Table of Contents")
    toc_p = doc.add_paragraph()
    add_field(toc_p, 'TOC \\o "1-3" \\h \\z \\u',
              "To build the contents list: right-click → Update Field "
              "(or press F9 in Microsoft Word).")
    page_break(doc)

    # ---------------- 1 executive summary ----------------
    h(doc, 1, "1. Executive Summary")
    p(doc,
      "Human Facial Emotion Detection is a fully local, real-time facial emotion detection and "
      "explainability system developed under the approved master specification "
      "“A Ranking-Based Explainable AI Model for Human Facial Emotion "
      "Detection”. The system analyses a webcam or uploaded image, detects the "
      "face, classifies the expression into seven emotions, and — unlike black-box "
      "classifiers — explains its decision through an evidence-ranking engine, "
      "Grad-CAM region attribution, calibrated confidence, an uncertainty flag, an "
      "intensity proxy, a temporal event engine, and a deterministic narrator that "
      "speaks what changed and why.")
    p(doc,
      "Machine-learning development followed an experiment-tracked, ADR-governed "
      "process on the RAF-DB official test split (n = 3,068). Headline outcome: a "
      "three-model probability-average ensemble (DAN + DDAMFN++ + ViT-Small) reaches "
      "89.99% accuracy / 82.90% balanced accuracy; the single best model is the "
      "reproduced DAN checkpoint at 89.70% / 82.75% (matching its published figure "
      "exactly on our split). In-house staged fine-tunes were trained for "
      "EfficientNet-B3 (66.49% acc / 59.13 macro-F1) and ViT-Small (76.63% raw, "
      "78.88% with 10-view test-time augmentation). Temperature scaling (val-fitted, "
      "T = 0.2569) improves the champion ensemble’s expected calibration error "
      "from 0.113 to 0.087.")
    p(doc,
      "The product layer is complete and verified: a FastAPI + WebSocket + SQLite "
      "backend, a Next.js 14 frontend with five product areas (Dashboard, Live "
      "Observer, Research Lab, Analytics, Settings), JWT-style local authentication, "
      "PDF/CSV/JSON exports with honesty footers, and a 104-test automated suite. "
      "Post-implementation quality batches (B1–B5) professionalised trust surfaces "
      "(model-inspector metrics, health chip, privacy line), first-run experience "
      "(dashboard, sample faces), live UX (pause, mirror, reconnect, shortcuts), "
      "analytics depth (transition heat-map, uncertainty card, valence bands, "
      "filters, comparison), and system polish (toasts, skeletons, two-click delete, "
      "accessibility). All checks are green: 104/104 unit & integration tests, lint "
      "clean, production build clean, and a 38-assertion end-to-end smoke run "
      "against the live servers.")
    callout(
        doc, "One-line status:",
        "ML research core COMPLETE (experiment artifacts on disk); product "
        "COMPLETE and running locally (API :8765, web :3000); 104 tests passing; "
        "remaining work is listed in Sections 12–14 (AU branch training, LRP "
        "production path, ranking ablation ADR-002, longer in-house fine-tunes, "
        "repository commit of the front-end layer).",
        fill="E2EFDA",
    )

    # ---------------- 2 introduction ----------------
    h(doc, 1, "2. Introduction")
    h(doc, 2, "2.1 Background & problem statement")
    p(doc,
      "Facial emotion recognition (FER) systems typically output a single label with "
      "a softmax confidence. In high-stakes, human-facing settings (research, "
      "accessibility, well-being tooling) a bare label is insufficient: users must be "
      "able to ask what evidence supported the decision, how strong the expression "
      "was, how sure the model actually is, and what changed over time. "
      "Commercial cloud FER also raises privacy concerns, as frames leave the device.")
    h(doc, 2, "2.2 Objectives")
    numbered(doc, [
        ("Accuracy — ", "build and evaluate expression classifiers on RAF-DB’s "
         "official split with experiment-tracked discipline."),
        ("Explainability — ", "answer seven user questions: WHAT (ranking), HOW SURE "
         "(calibrated confidence / certainty), HOW STRONG (intensity), WHY "
         "(Grad-CAM / LRP path), WHICH CUES (Action Units), WHICH REGIONS "
         "(attribution shares), WHAT CHANGED (temporal events)."),
        ("Ranking-based evidence — ", "combine probability, AU, regional and temporal "
         "evidence into a single ranked emotion list (the core research contribution)."),
        ("Real-time operation — ", "webcam streaming with bounded latency on consumer "
         "hardware (no cloud)."),
        ("Privacy by design — ", "everything local: frames processed in memory, never "
         "uploaded or persisted; local SQLite only; no identity recognition."),
        ("Honest AI — ", "surface uncertainty, disclose untrained branches, declare "
         "model provenance, never fabricate metrics."),
        ("Professional product — ", "three product modes (Live Observer, Research "
         "Lab, Analytics) plus dashboard/settings, with verified quality."),
    ])
    h(doc, 2, "2.3 Core research contribution")
    p(doc,
      "An integrated evidence-ranking framework that fuses classifier probabilities "
      "with regional attribution and temporal evidence into one interpretable score "
      "per emotion, complemented by separate intensity estimation, uncertainty "
      "detection, explanation-faithfulness tooling (LRP research kit), and a "
      "deterministic event narrator.")
    h(doc, 2, "2.4 Scope, ethics and explicit boundaries")
    bullets(doc, [
        ("No identity recognition. ", "The system classifies expressions only; it "
         "never identifies who the person is."),
        ("Single-person product mode. ", "The product deliberately tracks exactly one "
         "subject (largest / IoU-continuous face) with debounce-stable transitions."),
        ("Not a medical or psychological device. ", "Outputs are computational "
         "estimates for research/demo use; no clinical claim is made. Valence and "
         "intensity are labelled proxies."),
        ("Honest provenance. ", "Models trained in-house and models that were only "
         "evaluated (released checkpoints) are labelled distinctly everywhere "
         "(Section 6.4)."),
        ("No fabricated results. ", "All numbers come from stored JSON artifacts; "
         "stale documentation numbers are reconciled rather than repeated."),
        ("Privacy. ", "Camera frames are never stored; video buffers live only for "
         "the active session; only scalar analytics (labels, timings, summaries) are "
         "written to the local database."),
    ])

    # ---------------- 3 governance & work methodology ----------------
    h(doc, 1, "3. Project Governance & How the Work Was Done")
    h(doc, 2, "3.1 Specification-driven process")
    p(doc,
      "Work followed a written-governance model rather than ad-hoc coding:")
    bullets(doc, [
        ("Master specification — ", "docs/00_MASTER_SPECIFICATION.md (V1.0, APPROVED "
         "2026-09-10) fixes product and ML decisions as LOCKED / EXPERIMENTAL / "
         "IMPL-DEPENDENT."),
        ("Numbered design documents — ", "30 documents (specs 01–22 plus ADRs): "
         "architecture, dataset/training, evaluation, XAU/AU modules, XAI, event "
         "engine, narrator, backend, frontend, security, testing, reporting, "
         "experiment tracking, model registry."),
        ("Architecture Decision Records — ", "ADR-001…ADR-007 pose each open "
         "decision with options, criteria and an experiment ID; ADR-001 (backbone) "
         "is ACCEPTED on evidence."),
        ("Experiment tracking — ", "each run has an experiments/<ID>/ folder with "
         "config, environment, train history, checkpoints and test_metrics.json; "
         "results are only believed when the artifact exists."),
    ])
    h(doc, 2, "3.2 Timeline of work (phases)")
    table(
        doc,
        ["Phase", "Dates (attested)", "What was done", "Why"],
        [
            ["P0 — Specification & governance",
             "2026-09-10",
             "Master spec approved; docs 00–22 drafted; ADR framework created",
             "Freeze scope before code; make decisions auditable"],
            ["P1 — Baselines (historical)",
             "Pre-sprint (archived)",
             "FER2013 baselines: VGG16 official baseline 67.84% / 66.58% F1 "
             "(EXP-004); ResNet-50 / EfficientNet-B2 runs superseded",
             "Mandatory baseline requirement; informed accuracy-first pivot"],
            ["P2 — RAF-DB pivot & data preparation",
             "2026-09-22",
             "Repo cleanup of legacy artifacts; scripts/prepare_rafdb.py builds "
             "train/val/test; Kaggle GPU notebook prepared (optional)",
             "RAF-DB is the standard realistic FER benchmark; official split for "
             "comparability"],
            ["P3 — In-house model training",
             "2026-09-23",
             "Staged fine-tunes on Apple M4 MPS: EfficientNet-B3 (EXP-007, "
             "60 epochs staged) and ViT-Small (EXP-008, 3+7+10+70 staged) + "
             "10-view TTA",
             "Own trainable baselines with full hyper-parameter transparency"],
            ["P4 — Released-checkpoint evaluation, ensemble, calibration",
             "2026-09-23",
             "DAN official ckpt reproduced exactly (EXP-011: 89.70%); DDAMFN++ "
             "released ckpt evaluated (EXP-009: 83.83%); ensemble leaderboard + "
             "temperature scaling (EXP-010); ADR-001 accepted",
             "Establish strong references and a calibrated champion without "
             "claiming their weights as our own"],
            ["P5 — Product engineering",
             "2026-09-22 – 2026-09-23",
             "FastAPI + WebSocket + SQLite backend; Next.js five-area front-end; "
             "standalone webcam module; ranking engine wired with region evidence; "
             "event engine + narrator; PDF/CSV/JSON reports; auth & settings",
             "Turn the research core into a usable, private, local product"],
            ["P6 — Single-person product semantics",
             "2026-09-23",
             "Face tracker pins one subject (id 0) with IoU continuity; 1.0 s "
             "transition debounce; face re-entry events; narrator rewritten to a "
             "single-subject voice; EMA smoothing configurable",
             "Kill multi-face flicker and chatty narration; make outputs stable "
             "and honest"],
            ["P7 — UX professionalisation batches B1–B5",
             "2026-09-23 (current)",
             "Trust surfaces (model inspector fix, health chip, privacy line, page "
             "titles, favicon), first-run dashboard + PD sample faces, live "
             "pause/mirror/reconnect/shortcuts, analytics heat-maps/valence/"
             "filters/comparison, toasts/skeletons/two-click delete/a11y",
             "Make the system presentable and trustworthy for evaluation"],
        ],
        widths=[3.4, 2.6, 6.6, 4.0],
        font_size=8.5,
    )
    h(doc, 2, "3.3 Development environment & scale")
    table(
        doc,
        ["Item", "Value"],
        [
            ["Hardware", "Apple MacBook M4, 16 GB unified memory (training & "
             "inference on MPS; CUDA unavailable)"],
            ["Language / ML stack", "Python 3.12.14, PyTorch 2.14.0 (MPS), "
             "torchvision, scikit-learn, OpenCV (<5, Haar cascades), albumentations"],
            ["Backend stack", "FastAPI, Uvicorn, SQLAlchemy 2, SQLite, "
             "python-multipart, e-mail-validator; auth = stdlib PBKDF2 + HMAC tokens "
             "(no external JWT lib)"],
            ["Frontend stack", "Next.js 14.2 (App Router), React 18, TypeScript, "
             "Tailwind CSS, Zustand, Recharts, Framer Motion"],
            ["Code volume (this repository)", "src/ ≈ 4,300 lines Python (44 "
             "files), ml/ + scripts/ complete training/XAI tooling (74 Python files "
             "across src/ml/scripts/tests), frontend/src ≈ 4,200 lines TS/TSX, "
             "tests/ ≈ 1,240 lines"],
            ["Automated tests", "93 (all passing; 5 test files)"],
            ["Design documents", "30 (22 specs + 7 ADRs + evidence log)"],
            ["Experiment folders on disk", "7 (EXP-004/005/007/008/009/010/011)"],
            ["Checkpoint storage", "≈ 1.2 GB experiments/ + 151 MB released DAN "
             "weights + 97 MB prepared RAF-DB images"],
            ["Version control", "Git repository; 20 commits (2026-09-22 → 2026-09-23); "
             "front-end layer completed and pending initial commit"],
        ],
        widths=[4.5, 12.0],
        font_size=9,
    )

    page_break(doc)

    # ---------------- 4 architecture ----------------
    h(doc, 1, "4. System Architecture")
    h(doc, 2, "4.1 Layered view")
    table(
        doc,
        ["Layer", "Components", "Responsibility"],
        [
            ["Presentation", "Next.js app: /dashboard /live /lab /analytics "
             "/settings + login/register; Header (health chip, About, privacy line)",
             "User interaction, charts (Recharts), canvas mirror/overlay, speech "
             "(browser speechSynthesis), toasts/skeletons"],
            ["API gateway", "FastAPI routers: auth, analyze, sessions, analytics, "
             "settings, models, health; envelope responses",
             "REST JSON with status/data/meta/error envelope; auth dependency; "
             "ownership checks"],
            ["Streaming", "WebSocket /ws/emotion-stream (binary JPEG ≈ 8 fps; "
             "text actions xai/ping)",
             "Per-user live analysis loop; query params token / title / "
             "xai_interval; max 30 connections per user"],
            ["Analysis engine", "SessionAnalysis: face tracker → Grad-CAM on "
             "demand → expression model → EMA → ranking engine → event engine → "
             "narrator",
             "Single-person policy, debounce, certainty flags, timeline samples "
             "(1/s), narration lines"],
            ["Model zoo", "ml/models production loader: DAN (deployed), DDAMFN++, "
             "ViT-Small, EfficientNet-B3, ensemble eval script",
             "Unified load_production / build_transform; class-order remapping to "
             "canonical 7-emotion space"],
            ["Persistence", "SQLite via SQLAlchemy: users, sessions, events, XAI "
             "records, settings, model_registry (seeded + idempotent upsert)",
             "Local-only storage; session meta JSON (titles); ON DELETE cascades"],
            ["Reporting", "src/api/services/report.py — compute_summary + CSV / "
             "JSON / dependency-free PDF with honesty notes section",
             "Exports with named files affecta_<label>_<date>.<fmt>"],
        ],
        widths=[2.8, 7.0, 6.7],
        font_size=8.5,
    )
    h(doc, 2, "4.2 Runtime topology")
    bullets(doc, [
        "API server: uvicorn src.api.main:app on 127.0.0.1:8765 with "
        "AFFECTA_WARM_MODEL=1 AFFECTA_MODEL=dan (champion model pre-loaded; health "
        "endpoint reports model_loaded=true).",
        "Web server: next start on 127.0.0.1:3000; frontend env "
        "NEXT_PUBLIC_API_URL / NEXT_PUBLIC_WS_URL point at the local API.",
        "Browsers connect only to localhost; no external services are called at "
        "runtime. Speech synthesis and all inference run on-device.",
        "A standalone offline module (python -m src.pipeline.webcam) provides the "
        "same detection loop without the web stack (headless --image mode used by "
        "tests).",
    ])
    h(doc, 2, "4.3 Repository structure (deliverable map)")
    table(
        doc,
        ["Directory", "Contents"],
        [
            ["src/core, src/models", "Config (EMOTIONS order), classifier wrappers, "
             "ranking engine, intensity, temporal/EMA, AU detector"],
            ["src/api", "FastAPI app, routers (auth/analyze/sessions/analytics/"
             "settings/ws), services (inference, tracker, report), core (config, "
             "database seeds, security), schemas"],
            ["src/pipeline", "Standalone webcam application (Haar → model → EMA → "
             "HUD, Grad-CAM toggle, recording)"],
            ["ml/", "Training (staged trainer), evaluation, data + picklable TTA, "
             "models (DAN/DDAMFN vendored, backbones, production zoo), XAI "
             "(Grad-CAM, LRP-epsilon research engine, region mapping), benchmarks "
             "(calibration, latency, XAI probe)"],
            ["scripts/", "prepare_rafdb.py, ensemble_production.py, "
             "ensemble_eval.py, supervised_train.py, chain.py, report generator"],
            ["tests/", "104 tests across test_api, test_event_narrator, test_models, "
             "test_production_webcam, test_tools"],
            ["experiments/", "EXP-004…011: config.json, environment.json, "
             "train_history.json, test_metrics.json, confusion matrices, checkpoints"],
            ["docs/", "00–22 specifications + ADRs + evidence log"],
            ["frontend/", "Next.js application (src/app routes, components, stores, "
             "lib), public/samples (PD demo faces + credits)"],
        ],
        widths=[3.6, 12.9],
        font_size=9,
    )

    # ---------------- 5 dataset ----------------
    h(doc, 1, "5. Dataset & Emotion Taxonomy")
    h(doc, 2, "5.1 Primary dataset — RAF-DB")
    bullets(doc, [
        ("Source & preparation — ", "raw parquets under data/raw/rafdb; "
         "scripts/prepare_rafdb.py materialises image folders "
         "data/processed/rafdb/{train,val,test} (≈ 97 MB prepared on disk)."),
        ("Official test split — ", "n = 3,068 images; every headline metric in this "
         "report is measured on this split (or the RetinaFace-realigned variant "
         "test_rf for EXP-009, declared separately)."),
        ("Why RAF-DB — ", "real-world label noise and pose variation make it the "
         "standard reliability benchmark for FER; the official split allows "
         "meaningful comparison with published checkpoints."),
        ("Class imbalance — ", "test supports: Happy 1,185 · Neutral 680 · Sad 478 · "
         "Surprise 329 · Angry 162 · Disgust 160 · Fear 74. Therefore accuracy alone "
         "is reported together with balanced accuracy / macro-F1."),
    ])
    h(doc, 2, "5.2 Historical dataset — FER2013")
    p(doc,
      "FER2013 was used only for the mandatory VGG16 baseline (EXP-004: 67.84% acc / "
      "66.58% macro-F1) before the accuracy-first pivot to RAF-DB. Those runs are "
      "retained as historical context and are not used for any product claim.")
    h(doc, 2, "5.3 Canonical emotion space (LOCKED)")
    table(
        doc,
        ["Index", "Emotion", "Index", "Emotion"],
        [
            ["0", "Angry", "4", "Neutral"],
            ["1", "Disgust", "5", "Sad"],
            ["2", "Fear", "6", "Surprise"],
            ["3", "Happy", "", ""],
        ],
        widths=[2.0, 5.5, 2.0, 5.5],
    )
    p(doc,
      "All production models internally re-map their native label order to this "
      "alphabetical canonical order (verified by unit tests: bijective, complete "
      "mapping; DDAMFN gather order test).")

    page_break(doc)

    # ---------------- 6 model development ----------------
    h(doc, 1, "6. Model Development, Experiments & Results")
    h(doc, 2, "6.1 Training protocol (in-house runs)")
    p(doc,
      "Both in-house RAF-DB runs use the same staged fine-tuning discipline "
      "(progressive unfreezing) with ImageNet-pretrained backbones, seed 42, mixed "
      "precision, and strong augmentation — chosen to avoid catastrophic forgetting "
      "while eventually training deep blocks on a small dataset (12,271 train images).")
    table(
        doc,
        ["Hyper-parameter", "EfficientNet-B3 (EXP-007)", "ViT-Small (EXP-008)"],
        [
            ["Stages (epochs)", "head 3 → blk5 7 → blk4-5 10 → blk3-5 40 (= 60)",
             "head 3 → blk5 7 → blk4-5 10 → blk3-5 70 (= 90)"],
            ["Learning rates", "head 3e-4 · backbone 1e-5 · wd 1e-4",
             "head 3e-4 · backbone 1e-5 · wd 1e-4"],
            ["Batch / input", "24 · 300×300", "32 · 224×224"],
            ["Loss / regularisation", "Cross-entropy, label smoothing 0.05, class "
             "weights, strong augmentation", "identical"],
            ["Early stop", "monitor val macro-F1, patience 7, min Δ 0.001",
             "identical"],
            ["Extras", "AMP on MPS; atomic checkpointing (flaky MPS writes)",
             "+ 10-view TTA at evaluation; extended-tail option "
             "(--extend-last-stage)"],
            ["Final test result", "66.49% acc · 59.13 macro-F1 "
             "(test_metrics.json)", "76.63% acc · 70.05 macro-F1 raw; "
             "78.88% / 73.23 bacc with 10-view TTA"],
        ],
        widths=[3.6, 6.5, 6.4],
        font_size=8.5,
    )
    h(doc, 2, "6.2 Experiment register (all runs on disk)")
    table(
        doc,
        ["ID", "Dataset / backbone", "Protocol", "Result (official test)",
         "Provenance"],
        [
            ["EXP-004", "FER2013 / VGG16", "Official baseline staged training",
             "67.84% acc · 66.58 F1", "In-house (ImageNet-pretrained)"],
            ["EXP-005 / 006", "FER2013 / ResNet-50, EffNet-B2",
             "Superseded by RAF-DB pivot", "Not reported", "In-house (historical)"],
            ["EXP-007", "RAF-DB / EfficientNet-B3", "60-epoch staged fine-tune",
             "66.49% acc · 59.13 macro-F1", "In-house (ImageNet-pretrained)"],
            ["EXP-008", "RAF-DB / ViT-Small", "90-epoch staged fine-tune + 10-view TTA",
             "76.63 raw → 78.88 / 73.23 bacc with TTA", "In-house (ImageNet-pretrained)"],
            ["EXP-009", "RAF-DB / DDAMFN++", "Released checkpoint evaluated on our "
             "split (RetinaFace-realigned; MTCNN variant too)",
             "83.83% acc · 73.40 bacc (82.40 MTCNN)", "Released ckpt — evaluated only"],
            ["EXP-010", "RAF-DB / ensemble", "Uniform avg-probs of 3 members + "
             "10-view TTA variants + temperature scaling",
             "ens_dan+ddamfn+vit 89.99% · 82.90 bacc", "Our recipe (no training)"],
            ["EXP-011", "RAF-DB / DAN", "Official released checkpoint reproduced on "
             "our 3,068-image split",
             "89.70% acc · 82.75 bacc (= published paper figure)",
             "Released ckpt — evaluated only"],
        ],
        widths=[2.0, 3.4, 4.6, 4.2, 2.5],
        font_size=8.5,
    )
    h(doc, 2, "6.3 Ensemble & calibration methodology")
    bullets(doc, [
        ("Ensemble — ", "uniform average of the three members’ probability "
         "vectors (equal weights). Weighted variant (w = 5:3:2) tried "
         "(86.11%) and rejected."),
        ("TTA — ", "10-view test-time augmentation; helps ViT (+2.25) but hurts DAN "
         "(89.70 → 83.83); champion ensemble therefore uses no TTA."),
        ("Calibration — ", "temperature scaling fitted on the validation split only, "
         "applied to test: T = 0.2569, ECE 0.1125 → 0.0865 "
         "(no test-set leakage)."),
        ("Authoritative leaderboard — ", "experiments/EXP-010-ENSEMBLE/"
         "production_ensemble_results.json (13 leaderboard entries)."),
    ])
    h(doc, 2, "6.4 Model provenance & research-integrity statement")
    p(doc,
      "This section exists because provenance must be auditable. Of the five models "
      "registered in the product’s Model Inspector, two were trained by this "
      "project, two were released third-party checkpoints that this project only "
      "evaluated, and one is this project’s ensemble recipe (no training).")
    table(
        doc,
        ["Registry entry", "Trained by us?", "Evidence", "What we did"],
        [
            ["dan v1.0 (89.70 / 82.75 bacc)", "NO — released checkpoint",
             "EXP-011 test_metrics.json: “model: DAN official RAF-DB ckpt "
             "(yaoing/DAN)”; weights file dated 2021; README: “released ckpt, "
             "paper-exact reproduction”; code+weights © yaoing/DAN (MIT), "
             "vendored",
             "Reproduced the published 89.70% exactly on our official split; "
             "deployed as the live inference model"],
            ["ddamfn v2.0 (83.83 / 73.40 bacc)", "NO — released checkpoint",
             "EXP-009 test_metrics_retinaface.json: “model: DDAMFN++ released "
             "ckpt”; architecture vendored from SainingZhang/DDAMFN",
             "Evaluated under our protocols (RetinaFace & MTCNN); provides "
             "architecture diversity for the ensemble"],
            ["vit_small v1.0 (78.88 / 73.23 bacc, +TTA)", "YES — in-house fine-tune",
             "EXP-008 config.json + train_history.json + environment.json (our "
             "seed 42, MPS, staged epochs, full loss/accuracy curves)",
             "Staged 90-epoch fine-tune from ImageNet-pretrained ViT-Small; "
             "TTA evaluation variant shown in registry"],
            ["efficientnet_b3 v1.0 (66.49 / 59.13 F1)", "YES — in-house fine-tune",
             "EXP-007 config.json + train_history.json (60 staged epochs, our "
             "environment); test_metrics.json final numbers",
             "Staged fine-tune from ImageNet-pretrained EfficientNet-B3; deployable "
             "lightweight member"],
            ["ensemble v1.0 (89.99 / 82.90 bacc)", "NO TRAINING — our recipe",
             "EXP-010 production_ensemble_results.json; scripts/"
             "ensemble_production.py",
             "Uniform avg-probs over the three members + val-fitted temperature; "
             "the only learned parameter is one temperature scalar"],
        ],
        widths=[3.0, 2.6, 6.0, 4.6],
        font_size=8,
    )
    callout(
        doc, "Reconciliation note (honesty):",
        "README and docs/22 still list EfficientNet-B3 at 57.30% / 50.21 F1 — that "
        "was an earlier mid-run snapshot. The finished EXP-007 artifact shows "
        "66.49% / 59.13 macro-F1, which is what the product’s Model Inspector "
        "now serves (seed upserts on every API start). Documentation tables will be "
        "refreshed to the final artifact numbers in the next documentation pass.",
    )
    h(doc, 2, "6.5 Headline results (RAF-DB official test, n = 3,068)")
    table(
        doc,
        ["Rank", "System", "Accuracy", "Balanced acc / F1", "Calibration (ECE)",
         "Notes"],
        [
            ["1", "ens_dan+ddamfn+vit (EXP-010)", "89.99%", "82.90 bacc",
             "0.113 raw → 0.087 calibrated (T=0.257)", "Champion recipe; no TTA"],
            ["2", "DAN released ckpt (EXP-011)", "89.70%", "82.75 bacc", "0.033",
             "= published paper number; deployed live model"],
            ["3", "ens_dan+vit (EXP-010)", "89.31%", "83.12 bacc (best bacc)", "0.129",
             "Best balanced accuracy"],
            ["4", "ens_dan+ddamfn (EXP-010)", "89.24%", "81.69 bacc", "0.018",
             "Best-calibrated system"],
            ["5", "DDAMFN++ released ckpt (EXP-009)", "83.83%", "73.40 bacc", "0.080",
             "RetinaFace-realigned protocol"],
            ["6", "ViT-Small + TTA (EXP-008)", "78.88%", "73.23 bacc / 70.05 F1 raw",
             "0.118", "In-house; +2.25 from TTA"],
            ["7", "EfficientNet-B3 (EXP-007)", "66.49%", "59.13 macro-F1", "—",
             "In-house final artifact (see reconciliation note)"],
        ],
        widths=[1.2, 4.6, 2.0, 3.4, 3.4, 3.0],
        font_size=8,
    )

    page_break(doc)

    # ---------------- 7 explainability ----------------
    h(doc, 1, "7. Explainability & Ranking Core (How & Why)")
    h(doc, 2, "7.1 Evidence-ranking formula (core contribution)")
    p(doc, "For each emotion e:   S_e = w_p · P_e + w_a · A_e + w_r · R_e + w_t · T_e "
      "(scores renormalised to a ranking).")
    table(
        doc,
        ["Term", "Meaning", "Production weight", "Status / rationale"],
        [
            ["P_e", "Classifier probability", "w_p = 0.6",
             "Primary signal from the deployed model"],
            ["A_e", "Action-Unit evidence", "w_a = 0.0",
             "AU branch exists but is UNTRAINED → excluded honestly "
             "(action_units omitted from outputs, au_model_status=untrained)"],
            ["R_e", "Regional attribution evidence", "w_r = 0.2",
             "Grad-CAM mass over geometric face regions, mapped through "
             "region→emotion priors"],
            ["T_e", "Temporal evidence", "w_t = 0.2",
             "Recent-emotion history (EMA-stabilised state)"],
        ],
        widths=[1.6, 4.4, 3.2, 7.5],
        font_size=9,
    )
    p(doc,
      "Unit-tested: ranking order sanity, softmax validity, preliminary ranking "
      "behaviour. ADR-002 (full ranking ablation, experiment EXP-002) remains open "
      "by design — the pending research question is whether adding regional evidence "
      "significantly beats probability+AU alone (Section 14).")
    h(doc, 2, "7.2 Visual explanations (production path — Grad-CAM)")
    bullets(doc, [
        ("Method — ", "Grad-CAM on the deployed classifier’s final conv/feature "
         "stage; computed on demand (button, on transition, or every N frames — "
         "user-selectable), never blindly every frame (CPU budget)."),
        ("Region shares — ", "CAM mass is aggregated over five geometric boxes "
         "(mouth, eyes, eyebrows, nose, cheeks) → percentages that sum to 1; shown "
         "as region evidence and fed into R_e."),
        ("Honest limits (printed in UI and PDF) — ", "regions are geometric, not "
         "landmark-based; this is attribution, not causal proof."),
        ("LRP toolkit (research path) — ", "ml/xai/lrp_epsilon.py implements true "
         "LRP-ε for faithfulness studies (spec names LRP as primary XAI; Grad-CAM "
         "serves production). Faithfulness evaluation against LRP is future work."),
    ])
    h(doc, 2, "7.3 Uncertainty & certainty states")
    p(doc,
      "The model surfaces its own doubt from the top-1 vs top-2 probability margin: "
      "certainty ∈ {certain, uncertain, ambiguous, insufficient}. Ambiguity and "
      "confidence drops become timeline events (ambiguity_enter/exit, "
      "confidence_drop), counted in the Analytics “Uncertainty” card, and "
      "spoken only at higher narrator detail levels. This is model-derived doubt — "
      "not a human judgement.")
    h(doc, 2, "7.4 Intensity (explicit proxy)")
    p(doc,
      "Intensity (0–100, banded low/medium/high) is the extremity of the emotion "
      "distribution — a transparent proxy, not a trained regressor. It is labelled "
      "as a proxy in the UI, the PDF footer and this report (ADR-005 remains the "
      "path to a learned intensity head).")
    h(doc, 2, "7.5 Action Units — current honest status")
    p(doc,
      "The AU module (detector forward + sigmoid outputs, unit-tested) is wired for "
      "future integration, but no AU head has been trained on this project’s data. "
      "Consequently w_a = 0, AU values are omitted from API responses, and every "
      "surface states au_model_status = untrained. This prevents any appearance of "
      "AU evidence the system cannot actually provide.")
    h(doc, 2, "7.6 Narrator (deterministic, offline)")
    bullets(doc, [
        "Templates only — no LLM, no cloud TTS; speech uses the browser’s local "
        "speechSynthesis with user-selected voice and speed.",
        "Single-subject voice: always “Person 01”; chatty events suppressed by "
        "detail level (calm = enter/exit/changes; normal adds intensity/ambiguity; "
        "detailed adds confidence drops).",
        "Evidence-grounded lines, e.g. “Person 01 changed from Neutral to Happy. "
        "Confidence 91%. Strongest evidence: mouth region.” (unit-tested).",
    ])

    # ---------------- 8 real-time engine ----------------
    h(doc, 1, "8. Real-time Engine & Product Behaviour (How)")
    h(doc, 2, "8.1 Frame loop")
    numbered(doc, [
        "Browser captures 640×480 video, draws to an off-screen canvas, encodes "
        "JPEG (quality 0.7) and streams to the WebSocket at ~8 fps (120 ms), with "
        "backpressure guard (skip when > 1 MB buffered).",
        "Server runs Haar-cascade face detection (OpenCV), applies the single-person "
        "tracking policy, crops and transforms the face, runs the deployed model "
        "(DAN by default) on MPS/CPU.",
        "Probability vectors are EMA-smoothed (α default 0.35, user-tunable "
        "0.10–1.00) before display and eventing.",
        "Ranking engine scores all seven emotions; certainty assigned from margins.",
        "Event engine diffs the state machine (debounced) and emits typed events; "
        "narrator renders lines; timeline samples appended once per second.",
        "On request / transition / interval: Grad-CAM computed, heatmap + regions "
        "returned; client draws overlay (mirrored to match display).",
        "On disconnect: aggregates flushed to SQLite (frames, events, duration, "
        "avg confidence/intensity, timeline).",
    ])
    h(doc, 2, "8.2 Single-person policy (stability engineering)")
    table(
        doc,
        ["Mechanism", "Rule", "Why"],
        [
            ["Pin", "Track exactly one face: IoU-continuous match preferred, else "
             "largest box; all frames labelled id 0 / “Person 01”",
             "Deterministic single-subject product semantics"],
            ["Transition debounce", "Emotion change must hold ≥ 1.0 s before a "
             "transition event fires (env-configurable; tests use 0)",
             "Kills A→B→A flicker (unit-tested with 7-step oscillation)"],
            ["Re-entry", "Face absent ≥ exit timeout → face_exit; return fires "
             "face_enter with reentry=true and a fresh state (no ghost transition)",
             "Prevents stale cross-gap transitions"],
            ["Max faces setting", "settings.max_faces = 1 for the product stream",
             "Matches product promise; multi-face capable engine retained for tests"],
            ["EMA smoothing", "Configurable α on the probability vector",
             "Calmer charts/narration without hiding raw model behaviour"],
        ],
        widths=[3.2, 7.5, 5.9],
        font_size=9,
    )
    h(doc, 2, "8.3 Event engine taxonomy (9 typed events)")
    table(
        doc,
        ["Event", "Fires when", "Narrator (normal detail)"],
        [
            ["face_enter", "First sight / re-entry", "“Person 01 entered the scene.”"],
            ["face_exit", "Absent beyond timeout", "“Person 01 left the scene.”"],
            ["transition", "Debounced emotion change",
             "“Person 01 changed from X to Y. Confidence …%. Strongest evidence: …”"],
            ["intensity_change", "|Δ intensity| ≥ threshold",
             "“… expression grew stronger/weaker.”"],
            ["stabilization", "Same emotion held ≥ N seconds",
             "“Person 01 has held X steady for N seconds.”"],
            ["ambiguity_enter", "Certainty drops to ambiguous/insufficient",
             "“… expression became ambiguous.”"],
            ["ambiguity_exit", "Certainty restored", "detailed level only"],
            ["confidence_drop", "Confidence falls ≥ threshold",
             "detailed level only: “… confidence dropped N points.”"],
            ["(analytics)", "All events stored regardless of narration level",
             "Feeds Analytics filters, donut, uncertainty card, exports"],
        ],
        widths=[3.0, 6.5, 7.1],
        font_size=8.5,
    )
    h(doc, 2, "8.4 Standalone webcam module")
    p(doc,
      "python -m src.pipeline.webcam provides the same loop without servers "
      "(--model, --cam Grad-CAM overlay, --image headless JSON mode, --save "
      "recording; keys q/ESC/c/s). Headless mode is exercised by the automated "
      "suite, proving offline operability.")

    page_break(doc)

    # ---------------- 9 backend ----------------
    h(doc, 1, "9. Backend Services — What Was Built")
    h(doc, 2, "9.1 REST API surface")
    table(
        doc,
        ["Method & path", "Auth", "Purpose"],
        [
            ["POST /api/auth/register", "—", "Create local account (e-mail "
             "validated; .local-style internal domains accepted)"],
            ["POST /api/auth/login", "—", "Issue bearer token (PBKDF2 password "
             "verification; login rate-limit covered by tests)"],
            ["POST /api/auth/logout", "token", "Revoke token server-side"],
            ["GET /api/auth/me", "token", "Current user profile"],
            ["DELETE /api/auth/me", "token", "Danger zone: delete account + all "
             "sessions/events/XAI/settings (ORM cascade)"],
            ["GET/PUT /api/settings", "token",
             "theme, narrator on/speed/voice/detail, EMA α, XAI frame interval, "
             "analysis frame rate (1-15), emotion hold (0-600 s), analysis "
             "mode live|periodic, verdict-window interval (30-600 s), motion "
             "trigger; SQLite migration for added columns"],
            ["GET /api/models", "token", "Model registry (idempotent seed upsert: "
             "dan, ddamfn, vit, b3, ensemble with normalized metric keys)"],
            ["POST /api/analyze", "token", "Multipart photo analysis; optional "
             "xai=true → Grad-CAM heatmap + regions + evidence ranking"],
            ["POST /api/analyze/video", "token", "Multipart clip analysis (≤60MB, "
             "≤120s): OpenCV decode, sampled through the same pipeline → emotion "
             "distribution, transitions, timeline, narration, peak frame + "
             "Grad-CAM; not persisted as a session"],
            ["GET /api/sessions", "token", "Session list (newest first, with meta "
             "titles + per-session statistics)"],
            ["DELETE /api/sessions/{id}", "token", "Delete own session (ownership "
             "checked; covered by tests)"],
            ["GET /api/analytics/summary?session_id=", "token",
             "Distribution, per-person, transition matrix, uncertainty counts"],
            ["GET /api/analytics/timeline?session_id=", "token",
             "Timeline samples (1/s) + full event list"],
            ["GET /api/analytics/export/{id}?format=", "token",
             "PDF (stdlib generator incl. honesty notes) / CSV / JSON"],
            ["GET /api/health", "—", "status, app, version, model, model_loaded "
             "(public; powers the header health chip)"],
        ],
        widths=[5.6, 1.6, 9.4],
        font_size=8.5,
    )
    h(doc, 2, "9.2 WebSocket protocol /ws/emotion-stream")
    table(
        doc,
        ["Direction", "Payload", "Notes"],
        [
            ["Connect query", "?token= [ &title= ] [ &xai_interval= ]",
             "4001 invalid token · 1013 >30 connections/user; title stored in "
             "session meta; xai_interval override enables transition-only mode"],
            ["Client → server", "binary JPEG frames", "≈ 8 fps, backpressure-aware"],
            ["Client → server", '{"action":"xai","face_id":0} / {"action":"ping"}',
             "On-demand explanation; keepalive"],
            ["Server → client", '{"type":"session","session_id"}',
             "Handshake with created session id"],
            ["Server → client", '{"type":"analysis", faces, events, narration, xai, '
             'timeline, fps, frame_id, model}',
             "Full per-frame packet; narration lines include event_type"],
            ["Server → client", '{"type":"xai"} / {"type":"error"} / {"type":"pong"}',
             "Async heatmap; structured errors (bad frame etc.); keepalive reply"],
        ],
        widths=[2.8, 7.2, 6.6],
        font_size=8.5,
    )
    h(doc, 2, "9.3 Persistence model (SQLite)")
    table(
        doc,
        ["Table", "Key columns", "Notes"],
        [
            ["users", "id, email (unique), password_hash (PBKDF2), created_at",
             "Cascades delete to sessions/settings"],
            ["sessions", "id, user_id, start/end, duration, frames, events, "
             "avg_conf/instensity, meta JSON",
             "meta.title = human session name; timeline aggregates flushed on "
             "close"],
            ["events", "session_id, face_id, ts, type, from/to emotion, confidence, "
             "certainty, metadata", "Feeds summary/matrix/filters/exports"],
            ["xai records", "session/frame refs, method, regions JSON, heatmap ref",
             "Region evidence retained; raw frames never stored"],
            ["settings", "theme, narrator_*, ema_alpha, xai_frame_interval, "
             "analysis_fps (1-15), emotion_hold_seconds (0-10)",
             "ALTER TABLE migration for columns added post-deploy"],
            ["model_registry", "name+version unique, metrics JSON, checkpoint, status",
             "Seeded idempotently: trained/validated/deployed statuses; only "
             "deployed serves inference"],
        ],
        widths=[3.0, 7.0, 6.6],
        font_size=8.5,
    )
    h(doc, 2, "9.4 Security posture (local-single-machine threat model)")
    bullets(doc, [
        ("Passwords — ", "PBKDF2-HMAC (stdlib, salted iterations) — no plaintext, no "
         "external auth library."),
        ("Tokens — ", "HMAC-signed bearer tokens with server-side revocation on "
         "logout/account deletion."),
        ("Authorization — ", "every session/analytics query is scoped to the "
         "authenticated user (ownership tests)."),
        ("Input validation — ", "Pydantic schemas; e-mail validation; multipart size "
         "type checks (garbage upload rejected in tests); WS bad-token/bad-frame "
         "error paths tested."),
        ("Privacy — ", "no third-party endpoints; frames in RAM only; local SQLite "
         "file; report PDF states processing is local and frames are not stored."),
        ("Explicit non-goal — ", "no identity recognition, no cloud, no telemetry."),
    ])

    # ---------------- 10 frontend ----------------
    h(doc, 1, "10. Frontend Application — What Was Built")
    h(doc, 2, "10.1 Route map")
    table(
        doc,
        ["Route", "Role", "Key features"],
        [
            ["/login, /register", "Local account entry", "Validated forms; clear "
             "errors (FastAPI 422 array parsing); redirect to /dashboard"],
            ["/dashboard", "First-run home",
             "Quick-start checklist, recent sessions with auto-names, empty-state "
             "guidance"],
            ["/live", "Live Observer (primary mode)",
             "Camera canvas with mirror toggle, overlay boxes, pause/resume (WS "
             "stays up), reconnect pill with auto-backoff, session title input, "
             "explain mode (interval vs transitions-only), voice on/off, snapshot "
             "PNG, event counter, shortcuts (Space/M/H/S/?/Esc) + help sheet, "
             "aria-live narration, ranking & region cards, heatmap card"],
            ["/lab", "Research Lab",
             "Upload or one-click PD samples (Einstein/Obama/Astronaut + credits), "
             "ranking breakdown, Grad-CAM view, Model Inspector (metric labels, "
             "status colours, accuracy bars, In-use pill, n=3,068 footnote, loading "
             "skeletons)"],
            ["/analytics", "Session Analytics",
             "Session list w/ date-range filter + human labels, stats grid, "
             "expression pie with slice drill-down, emotion-band timeline + "
             "metrics/valence toggle (proxy caption), 7×7 transition heat-map, "
             "uncertainty card, event-type donut + filter chips, confidence "
             "histogram, compare-with-previous, named exports "
             "(affecta_<label>_<date>.*), two-click delete"],
            ["/settings", "Configuration & account",
             "Theme, narrator enable/detail/speed, voice dropdown (system voices) + "
             "Test, EMA α slider, XAI interval, two-click Delete-account danger "
             "zone"],
            ["Header (global)", "Trust & navigation",
             "Health chip (polls /api/health 30 s), privacy line (“frames never "
             "leave this machine”), About modal with honesty bullets, nav incl. "
             "Dashboard"],
        ],
        widths=[2.6, 3.4, 10.6],
        font_size=8.5,
    )
    h(doc, 2, "10.2 Capture pacing & display dwell (new)")
    bullets(doc, [
        "Analysis frame rate (Settings → Capture & display, 1–15 fps, default 8): "
        "replaces the previously hardcoded 120 ms capture interval; the live "
        "status bar shows the configured rate, the value travels as the WS "
        "“fps” parameter and is persisted in session metadata as applied_fps."
        "Standalone webcam module gained the equivalent --fps flag."
        "Emotion hold (0–10 s, default 0 = off): presentation-only dwell — a "
        "detected emotion stays painted with a countdown bar for N seconds "
        "after a change. Narration, events and the Person panel remain live, so "
        "no stale value is ever presented as current."
        "Bounds are enforced server-side (422 outside 1–15 fps / 0–10 s) and "
        "covered by tests, including a WebSocket metadata assertion.",
    ])
    h(doc, 2, "10.3 Periodic (verdict-window) analysis mode")
    bullets(doc, [
        "Requested as “slower the frame rate to 1.5–2 minutes” and “one detected "
        "emotion to stay on screen for 1–2 minutes”. A frame rate cannot be "
        "expressed in minutes, so this was implemented as an explicit verdict "
        "window rather than a rate: analysis_mode = periodic with a configurable "
        "interval (30 s – 10 min, default 2:00) plus a minute-scale display hold "
        "(0 – 10 min).",
        "The existing continuous mode is untouched — this is an additional mode, "
        "so live demos, research and the test suite behave exactly as before.",
        "Frames keep streaming at the configured analysis rate; every window the "
        "readings are mean-pooled into one pinned verdict on the face (dominant "
        "emotion + pooled confidence + mean box + mean region evidence). A pooled "
        "top-1 below 40% pins UNCERTAIN instead of a bare label. Aggregation is "
        "entirely client-side (lib/verdict.ts); the server is unchanged.",
        "Motion trigger: a 64×48 grayscale frame difference (no model inference) "
        "flushes the current window into an instant early verdict, with a 5 s "
        "cooldown. Motion accelerates the schedule, never replaces it.",
        "Honesty: the face is labelled “VERDICT · JOY 74% · window n=…”. When the "
        "face leaves the frame the verdict is frozen at reduced opacity with a "
        "STALE marker — never presented as a live reading — and re-acquiring the "
        "face flushes a fresh verdict within a few frames. Readings that arrive "
        "during a hold are never replayed late — only the newest is shown when "
        "it expires.",
        "Cadence is recorded per session (analysis_mode, applied_interval_seconds, "
        "motion_trigger) so exports state what produced the run; values are clamped "
        "server-side and covered by tests.",
    ])
    h(doc, 2, "10.4 Research Lab: photo + video")
    bullets(doc, [
        "One drop zone accepts both a photo and a video (MP4/MOV/AVI/MKV/WEBM); "
        "the button label, preview and result panel follow the input type.",
        "Video runs through the identical live pipeline — tracker, EMA, evidence "
        "ranking, event engine, narrator — decoded locally with OpenCV, sampled "
        "at the user’s configured rate (capped at 10 fps) and never stored.",
        "The response aggregates the whole clip: dominant emotion, distribution "
        "with shares, transition list, peak frame with its Grad-CAM, timeline and "
        "narration lines.",
        "Transition semantics match the live stream (1.0 s debounce by default) so "
        "per-frame flicker is not reported as real change; a 0 value can be passed "
        "to study raw flips.",
        "Frames without a detected face yield no classification and the response "
        "reports face coverage, so a poor crop is visible rather than papered over.",
        "Video runs are ad-hoc research artifacts and are not recorded as Observer "
        "sessions — the session list stays an accurate record of live sessions.",
    ])
    h(doc, 2, "10.5 UX professionalisation batches (B1–B5, completed)")

    
    table(
        doc,
        ["Batch", "Goal", "Delivered"],
        [
            ["B1", "Trust & correctness",
             "Model Inspector rebuilt (normalized metric labels test_acc/test_bacc, "
             "correct deployed/validated/trained colours — fixing a dead "
             "“active” status bug — accuracy bars, In-use pill, honest "
             "footnote; registry seed upserts corrected stale B3 numbers and added "
             "the ensemble row); header health chip; privacy line; per-page document "
             "titles; favicon + Open Graph metadata"],
            ["B2", "First-run experience",
             "Dashboard page (checklist + recent sessions + auto session names from "
             "titles/dominants); nav & login redirects; public-domain sample faces "
             "with credits README; loading skeletons"],
            ["B3", "Live-session polish",
             "Pause/Resume without dropping the session; WS status pill + auto "
             "reconnect (5-attempt backoff); mirror-aware display with unmirrored "
             "network frames; explain-on-transitions mode; session title persisted "
             "server-side; snapshot download; keyboard shortcuts + help; narrator "
             "detail levels & voice test; EMA slider wired to stream; event counter; "
             "aria-live"],
            ["B4", "Analytics depth",
             "Transition heat-map matrix; emotion band strip; valence trend with "
             "proxy disclaimer; uncertainty breakdown; event-type donut + chips; pie "
             "drill-down linking to events; confidence histogram; previous-session "
             "comparison; date-range filter; export file naming; two-click delete"],
            ["B5", "System polish",
             "Toast notifications; skeletons; InfoTip affordances; About modal; "
             "CVD-safe disgust colour (pink) + reduced-motion CSS + global "
             "focus-visible rings; delete-account danger zone wired to "
             "DELETE /api/auth/me; PDF cover & honesty footer"],
        ],
        widths=[1.4, 3.4, 11.8],
        font_size=8.5,
    )

    page_break(doc)

    # ---------------- 11 QA ----------------
    h(doc, 1, "11. Quality Assurance & Verification")
    h(doc, 2, "11.1 Automated test suite — 104/104 passing")
    table(
        doc,
        ["Test file", "Count", "Covers"],
        [
            ["tests/test_api.py", "24",
             "Register/login/me/logout, duplicate & invalid e-mails, rate-limit, "
             "unauthorized access, settings roundtrip (incl. narrator_detail/EMA, "
             "analysis_fps range + emotion_hold_seconds + WS applied_fps), "
             "model registry seeds (ensemble present, B3 self-heal 66.49, keys "
             "normalized), analyze upload + XAI + garbage rejection, session "
             "ownership/delete, summary/timeline/PDF-CSV-JSON exports, WebSocket "
             "stream (analysis, narration, bad token/frame, session title) , "
             "account deletion"],
            ["tests/test_event_narrator.py", "24",
             "Event engine: enter/exit/transition/debounce (flicker blocked; fires "
             "once after stable hold), re-entry, intensity/confidence thresholds, "
             "stabilization, multi-face independence; Narrator: templates for every "
             "spoken type, single-subject voice, evidence grounding, detail levels "
             "(calm/normal/detailed), budget, disabled mode"],
            ["tests/test_models.py", "18",
             "VGG16/ResNet50/EfficientNet forward shapes; DAN & DDAMFN forward + "
             "label gather permutation; AU detector sigmoid outputs; intensity "
             "range; Grad-CAM shape/finiteness; region aggregation sums to 1; "
             "ranking order; production loaders for DAN & DDAMFN; label mapping "
             "bijectivity"],
            ["tests/test_production_webcam.py", "16",
             "Face detector load + synthetic detection, IoU, tracker stability "
             "single-face pin via IoU, ghost-then-new handling, EMA smoothing, "
             "annotate rendering, headless --image mode, argparser defaults, "
             "predictor EMA"],
            ["tests/test_tools.py", "11",
             "Dataset parquet parsing (2/3-column, bijective labels, rename map), "
             "evaluation metric shapes, perfect-calibration ECE=0, temperature "
             "scaling validity, three backbones’ forward paths"],
            ["Total", "93", "Run: source .venv/bin/activate && pytest tests/ -q"],
        ],
        widths=[4.2, 1.4, 11.0],
        font_size=8.5,
    )
    h(doc, 2, "11.2 Static checks & build")
    bullets(doc, [
        "Frontend: next lint — 0 errors / 0 warnings; next build — compiled "
        "successfully, type-checked, 13 static routes generated "
        "(shared First Load JS ≈ 87 kB).",
        "Backend: full pytest green (only 4 deprecation-style warnings from "
        "sklearn); manual flake8/black configured in Makefile.",
    ])
    h(doc, 2, "11.3 End-to-end smoke validation (against running servers)")
    p(doc, "A 38-assertion scripted smoke run was executed against the live stack:")
    table(
        doc,
        ["Area", "Checks (all passed)"],
        [
            ["Frontend", "11 routes/assets HTTP 200 (/, dashboard, live, lab, "
             "analytics, settings, login, register, icon.png, sample image, credits)"],
            ["Auth & danger zone", "register → token; DELETE account → me/login 401 "
             "afterwards"],
            ["Model registry", "ensemble present; B3 metrics = 66.49 upsert; dan "
             "deployed; normalized keys; ensemble bacc 82.90"],
            ["Settings", "defaults (narrator_detail=normal, ema=0.35) + roundtrip "
             "(calm/0.5)"],
            ["Live WebSocket", "session handshake; analysis frames; narration "
             "single-subject only; ?title= persisted into session meta"],
            ["Analysis", "Einstein sample upload → face detected, Grad-CAM heatmap "
             "and regions returned"],
            ["Exports", "PDF magic + honesty terms present; CSV/JSON valid; "
             "summary/timeline endpoints"],
            ["Health", "public /api/health, model_loaded=true (DAN warm)"],
        ],
        widths=[3.4, 13.2],
        font_size=8.5,
    )
    h(doc, 2, "11.4 Runtime status at report time")
    bullets(doc, [
        "API: http://127.0.0.1:8765 — uvicorn, warm model dan, health "
        '{"status":"ok","model_loaded":true}.',
        "Web: http://127.0.0.1:3000 — next start (production build).",
        "Logs clean (no tracebacks/errors during smoke).",
    ])

    # ---------------- 12 status ----------------
    h(doc, 1, "12. Current Status & Progress Assessment")
    h(doc, 2, "12.1 Workstream status")
    table(
        doc,
        ["Workstream", "Status", "Evidence / completion notes"],
        [
            ["Specification & design docs (00–22, ADRs)", "COMPLETE (living docs)",
             "Master spec APPROVED; 30 documents; ADR-001 accepted"],
            ["Dataset pipeline (RAF-DB)", "COMPLETE",
             "prepare_rafdb.py; processed splits on disk (97 MB)"],
            ["In-house training (B3, ViT)", "COMPLETE for current budget",
             "Full staged runs + artifacts; longer tails listed as future work"],
            ["Released-ckpt evaluation (DAN, DDAMFN++)", "COMPLETE",
             "EXP-009/011 artifacts; DAN paper-exact"],
            ["Ensemble + calibration", "COMPLETE",
             "EXP-010 leaderboard JSON; T-scaling val-fitted"],
            ["Explainability (Grad-CAM, regions, uncertainty, intensity)", "COMPLETE",
             "Unit-tested; wired to API/WS; honest captions everywhere"],
            ["LRP faithfulness path", "PARTIAL",
             "LRP-ε research engine implemented (ml/xai); production path uses "
             "Grad-CAM; faithfulness study pending"],
            ["Action Units", "STRUCTURED, NOT TRAINED",
             "Detector code + tests exist; w_a=0; disclosed on all surfaces"],
            ["Ranking-engine ablation (ADR-002 / EXP-002)", "PENDING (planned)",
             "Formula implemented with honest weights; formal ablation not run"],
            ["Backend (API/WS/DB/auth/reports)", "COMPLETE",
             "14 endpoints + WS; 104-test suite green; live"],
            ["Frontend (5 areas + auth)", "COMPLETE (pending initial git commit)",
             "Lint/build clean; live; B1–B5 polish applied"],
            ["Single-person real-time semantics", "COMPLETE",
             "Tracker pin + debounce + re-entry + narrator rewrite; tests added"],
            ["QA (unit/integration/E2E)", "COMPLETE for current scope",
             "104 tests + lint + build + 38-check smoke"],
            ["Documentation refresh (stale README numbers)", "OPEN (small)",
             "B3 57.30→66.49 reconciliation noted; inspector provenance labels "
             "queued"],
        ],
        widths=[5.0, 3.4, 8.2],
        font_size=8.5,
    )
    h(doc, 2, "12.2 Deliverables inventory")
    table(
        doc,
        ["Deliverable", "Quantity"],
        [
            ["Runnable product (API + Web + standalone webcam)", "3 entry points"],
            ["Automated tests (passing)", "93"],
            ["REST endpoints + WebSocket protocol", "14 + 1 stream protocol"],
            ["Registered models (registry)", "5 (1 deployed, 3 validated, 1 trained)"],
            ["Experiment folders with artifacts", "7"],
            ["Checkpoints & weights stored", "≈ 1.4 GB total"],
            ["Design/spec documents", "30"],
            ["Frontend routes built", "8 pages (+icon/samples assets)"],
            ["Exports supported", "PDF, CSV, JSON (named files)"],
            ["Public-domain demo samples", "3 images + credits README"],
        ],
        widths=[9.5, 7.1],
        font_size=9,
    )
    h(doc, 2, "12.3 Known gaps (declared, not hidden)")
    bullets(doc, [
        "README/docs leaderboard still shows the mid-run B3 figure (57.30); final "
        "artifact 66.49 is what the product serves — docs refresh pending.",
        "Model Inspector does not yet print per-row provenance labels "
        "(released vs in-house vs ensemble-recipe) — wording agreed, UI change queued.",
        "vit_small registry row shows the +TTA metric (78.88) without a visible "
        "“+TTA” suffix — raw is 76.63; label fix queued.",
        "Frontend layer and today’s backend refinements are not yet committed to "
        "git (20 existing commits cover through the 2026-09-23 research results).",
        "Architecture docs mention PostgreSQL/WebRTC as production targets; the "
        "deliberate local build ships SQLite/WebSocket — swap-in path documented in "
        "deployment docs.",
        "AU branch untrained; LRP not the production explanation yet; ADR-002 "
        "ranking experiment open.",
    ])

    # ---------------- 13 ethics ----------------
    h(doc, 1, "13. Ethical Considerations, Privacy & Limitations")
    h(doc, 2, "13.1 Privacy-by-design measures implemented")
    bullets(doc, [
        "Camera frames processed in memory only; never written to disk or sent "
        "off-machine (only labels/statistics stored).",
        "Single local database file; no cloud services, no analytics beacons, no "
        "third-party fonts/CDNs at runtime.",
        "No identity recognition — expressions only; single-person labelling is "
        "positional (“Person 01”), not identity.",
        "Demo images are public-domain portraits with credits; no subject data "
        "collected from users of the report.",
        "Account deletion removes all personal rows (sessions, events, XAI, "
        "settings) in one action.",
    ])
    h(doc, 2, "13.2 Disclosed limitations")
    bullets(doc, [
        "Not a diagnostic/clinical tool; intensity and valence are proxies; "
        "narration is templated commentary, not interpretation of inner states.",
        "Class imbalance: fear (74 test images) remains weakest — per-class recall "
        "disclosed in artifacts.",
        "In-house models trail the released DAN reference; honest leaderboard keeps "
        "them visible rather than hiding them.",
        "AU evidence excluded (w_a = 0) until an AU head is trained on approved "
        "data.",
        "Geometric region boxes approximate facial areas without landmarks.",
        "Single-person policy means a second face in frame is ignored by design.",
        "Browser speech quality varies by OS; server never sends audio anywhere.",
    ])
    h(doc, 2, "13.3 Research integrity summary")
    table(
        doc,
        ["Principle", "How it is enforced in this project"],
        [
            ["No fabricated metrics", "Every reported number maps to a stored JSON "
             "artifact path; stale doc numbers are reconciled, not copied"],
            ["Provenance honesty", "Section 6.4 table; docs wording “released "
             "ckpt, paper-exact reproduction”; UI provenance labels queued"],
            ["No test leakage", "Temperature fitted on validation only; single "
             "official test evaluation discipline"],
            ["Attribution", "DAN (MIT) and DDAMFN architectures vendored with "
             "credits in README + source headers"],
            ["Ethical boundary", "No identity recognition; no clinical claims; "
             "frames never stored"],
        ],
        widths=[4.2, 12.4],
        font_size=9,
    )

    # ---------------- 14 future ----------------
    h(doc, 1, "14. Future Work & Roadmap")
    table(
        doc,
        ["Priority", "Item", "Rationale / hook"],
        [
            ["High", "Refresh README/docs leaderboard to final artifacts (B3 66.49) "
             "and add provenance labels + “+TTA” suffix in Model Inspector",
             "Closes declared gaps in §12.3"],
            ["High", "Commit front-end + backend refinements; tag v1.0-report",
             "Reproducibility of the submitted state"],
            ["High", "ADR-002 ranking ablation (EXP-002): softmax vs +regional vs "
             "+temporal on Top-1 / NDCG@3", "Completes the core research claim"],
            ["Medium", "Train AU branch (approved AU labels) → re-enable w_a > 0",
             "Restores the designed ranking term; removes the untrained caveat"],
            ["Medium", "LRP-ε faithfulness evaluation vs Grad-CAM (insertion/"
             "deletion curves)", "Spec names LRP as primary XAI; kit already exists"],
            ["Medium", "Longer in-house fine-tunes (100–150 epochs, per ADR-001 "
             "note) ± ensemble weight learning", "Close the gap to released "
             "references"],
            ["Medium", "Learned intensity head (ADR-005) replacing distribution "
             "proxy", "Removes a proxy limitation"],
            ["Low", "PostgreSQL + Dockerised deployment profile; optional WebRTC",
             "Matches architecture docs for non-local deployments"],
            ["Low", "Serve ensemble directly as a production loader entry (status "
             "deployed) once latency-verified", "Champion currently research-"
             "validated only"],
        ],
        widths=[1.8, 8.0, 6.8],
        font_size=8.5,
    )

    # ---------------- 15 conclusion ----------------
    h(doc, 1, "15. Conclusion")
    p(doc,
      "From an approved specification (2026-09-10) to a verified local product "
      "(2026-09-23), this project delivered a complete ranking-based explainable "
      "facial emotion system: seven experiment-tracked model runs with on-disk "
      "artifacts, a calibrated 89.99%-accuracy ensemble recipe and a paper-exact "
      "89.70% DAN reference on RAF-DB’s official 3,068-image test split, an "
      "explainability stack (ranking engine, Grad-CAM regions, certainty states, "
      "intensity proxy, event engine, deterministic narrator), a production-grade "
      "local backend and five-area web frontend, and a 104-test quality gate that is "
      "currently green alongside lint, production build and a 38-check end-to-end "
      "smoke run.")
    p(doc,
      "Equally important is how the work was done: experiment or artifact, honest "
      "provenance for third-party weights, explicit disclosure of untrained branches "
      "and proxies, privacy-first local processing, and no identity recognition. "
      "The remaining items (Section 14) are research completions and documentation "
      "polish — not hidden deficiencies.")
    p(doc, "")

    # sign-off
    table(
        doc,
        ["Role", "Name", "Signature", "Date"],
        [
            ["Prepared by (student / developer)", "", "", ""],
            ["Project guide / supervisor", "", "", ""],
            ["Head of Department / Reviewing authority", "", "", ""],
        ],
        widths=[5.6, 4.4, 4.0, 2.6],
        font_size=9.5,
    )

    page_break(doc)

    # ---------------- appendices ----------------
    h(doc, 1, "Appendix A — Reproduction Commands")
    table(
        doc,
        ["Step", "Command"],
        [
            ["Install Python deps", "pip install -r requirements.txt  (venv: "
             "source .venv/bin/activate)"],
            ["Install frontend deps", "cd frontend && npm install"],
            ["Prepare RAF-DB", "python scripts/prepare_rafdb.py --raw "
             "data/raw/rafdb --out data/processed/rafdb"],
            ["Train in-house model", "make train-expression   (staged fine-tune, "
             "ml/training)"],
            ["Evaluate a run", "make evaluate   (or ml/evaluate.py --experiment-dir "
             "… --split test)"],
            ["Ensemble + calibration", "python scripts/ensemble_production.py"],
            ["Run tests", "pytest tests/ -q   (expect 93 passed)"],
            ["Start API (warm champion)",
             "AFFECTA_WARM_MODEL=1 AFFECTA_MODEL=dan uvicorn src.api.main:app "
             "--host 127.0.0.1 --port 8765"],
            ["Start web", "cd frontend && npm run build && npm run start   "
             "(http://127.0.0.1:3000)"],
            ["Standalone webcam", "python -m src.pipeline.webcam [--cam|--image f.jpg]"],
            ["Regenerate this report",
             "python scripts/generate_project_report.py"],
        ],
        widths=[4.6, 12.0],
        font_size=8.5,
    )

    h(doc, 1, "Appendix B — Experiment Artifacts Inventory")
    table(
        doc,
        ["Path", "Contains"],
        [
            ["experiments/EXP-004-VGG16-CLEAN/", "FER2013 baseline results (historical)"],
            ["experiments/EXP-007-EFFICIENTNET-B3/",
             "config.json, environment.json, train_history.json (60 epochs), "
             "test_metrics.json (66.49 / 59.13), confusion matrix PNG/JSON, "
             "checkpoints (best/last + epochs 53–57)"],
            ["experiments/EXP-008-VIT-SMALL/",
             "config/history, test_metrics.json (76.63 raw), TTA scan "
             "(scan_vit_tta_final.json), ensemble_b3_vit_test.json, checkpoints"],
            ["experiments/EXP-009-DDAMFN-PP/",
             "Released ckpt + test_metrics_mtcnn.json (82.40) & "
             "test_metrics_retinaface.json (83.83 / 73.40)"],
            ["experiments/EXP-010-ENSEMBLE/",
             "production_ensemble_results.json — 13-entry leaderboard, TTA "
             "variants, temperature scaling (T=0.2569)"],
            ["experiments/EXP-011-DAN-REPRO/",
             "test_metrics.json — official released ckpt reproduction "
             "(89.70 / 82.75, n=3068)"],
            ["data/raw/dan_weights/RAF-DB.pth", "Released DAN weights (151 MB, "
             "file dated 2021 — not produced by this project)"],
            ["data/processed/rafdb/", "Prepared image splits (≈ 97 MB)"],
            ["docs/evidence/", "Kaggle GPU probe log (optional cloud path; recorded "
             "runs are local MPS per environment.json)"],
        ],
        widths=[5.4, 11.2],
        font_size=8.5,
    )

    h(doc, 1, "Appendix C — Design Document Set")
    p(doc,
      "docs/: 00 Master Specification · 01 Requirements · 02 System Architecture · "
      "03 ML Model Architecture · 04 Ranking Engine · 05 Action Unit Module · "
      "06 XAI (LRP/Grad-CAM) · 07 Intensity & Uncertainty · 08 Dataset & Training · "
      "09 Evaluation & Ablation · 10 Real-time Video Pipeline · 11 Database Schema · "
      "12 FastAPI Backend · 13 Frontend UX · 14 Security & Privacy · 15 Reporting & "
      "Analytics · 16 Deployment · 17 Testing & QA · 18 Development Roadmap · "
      "19 Research Novelty · 20 Event Engine · 21 Model Registry · 22 Experiment "
      "Tracking.")
    p(doc,
      "ADRS/: ADR-001 Backbone Selection (ACCEPTED) · ADR-002 Ranking Formula "
      "(PENDING experiment) · ADR-003 AU Subset · ADR-004 LRP Frequency · "
      "ADR-005 Intensity Architecture · ADR-006 Uncertainty Thresholds · "
      "ADR-007 Temporal Model.")

    h(doc, 1, "Appendix D — References & Third-Party Attributions")
    numbered(doc, [
        "Li, S., Deng, W., & Du, J. (2017). Reliable Crowdsourcing and Deep "
        "Locality-Preserving Learning for Expression Recognition in the Wild. "
        "CVPR 2017. (RAF-DB dataset.)",
        "DAN — Dual Attention Network for Facial Expression Recognition, "
        "arXiv:2304.03108; code & released RAF-DB checkpoint © yaoing/DAN (MIT), "
        "vendored at ml/models/dan.py. Evaluated, not retrained, by this project.",
        "DDAMFN / DDAMFN++ — architecture © SainingZhang/DDAMFN (research use), "
        "vendored at ml/models/ddam.py; released checkpoint evaluated only by this "
        "project.",
        "Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers "
        "for Image Recognition at Scale. ICLR 2021. (ViT backbone.)",
        "Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for "
        "Convolutional Neural Networks. ICML 2019.",
        "Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep "
        "Networks via Gradient-based Localization. ICCV 2017.",
        "Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML 2017. "
        "(Temperature scaling.)",
        "Goodfellow, I. et al. (2013). Challenges in Representation Learning "
        "(FER2013 / Kaggle FER challenge) — historical baseline dataset only.",
        "Viola, P., & Jones, M. (2004). Robust Real-time Face Detection. IJCV — "
        "via OpenCV Haar CascadeClassifier (opencv-python <5).",
        "Internal: Human Facial Emotion Detection specs 00–22, ADRs 001–007, and all "
        "experiments/*/test_metrics.json artifacts (paths cited inline).",
    ])

    p(doc, "")
    p(doc, "— End of report —", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output))
    print(f"Report written: {output}  ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "Human_Facial_Emotion_Detection_Project_Report.docx")
    build(out)
