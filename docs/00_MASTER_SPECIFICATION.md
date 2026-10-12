# Human Facial Emotion Detection — Master Technical Specification

```
Version: V1.0
Architecture Status: APPROVED
Last Updated: 2026-09-10
```

---

## 1. Project Identity

### Academic Title
**A Ranking-Based Explainable AI Model for Human Facial Emotion Detection**

### Product Name
**Human Facial Emotion Detection — Evidence-Ranked Facial Emotion Intelligence**

### Core Concept
A real-time ranking-based explainable facial-expression analysis system with Action Unit evidence, region-level attribution, intensity estimation, uncertainty analysis and temporal multi-face tracking.

---

## 2. Core Research Contribution

An integrated evidence-ranking framework that combines:
1. Facial-expression probabilities
2. Action Unit evidence
3. Region-level LRP relevance
4. Temporal information

With:
- Separate expression-intensity estimation
- Uncertainty estimation
- Faithfulness evaluation of explanations

---

## 3. Seven Questions the System Answers

| Question | System Output |
|----------|---------------|
| **WHAT?** | Emotion ranking |
| **HOW SURE?** | Calibrated confidence |
| **HOW STRONG?** | Expression intensity |
| **WHY?** | LRP explanation |
| **WHICH FACIAL CUES?** | Action Units |
| **WHICH REGIONS?** | Region contribution ranking |
| **WHAT CHANGED?** | Temporal expression analysis |

---

## 4. Decision Status Legend

| Status | Meaning |
|--------|---------|
| LOCKED | Frozen decision, will not change |
| EXPERIMENTAL | To be tested, selection based on evidence |
| IMPL-DEPENDENT | Decided during implementation |

---

## 5. LOCKED Decisions

### Product
- 7 emotion classes: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- Three product modes: Live Observer, Research Lab, Session Analytics
- No identity recognition
- Privacy-first approach
- Evidence-grounded narrator

### ML Architecture
- VGG16 = mandatory baseline
- LRP = primary XAI method
- Grad-CAM = secondary comparison
- AU branch = yes (minimum 6 AUs)
- Ranking engine = core research contribution
- Intensity = separate from confidence
- Uncertainty detection = yes
- Temporal analysis = yes
- Multi-face tracking = yes

### Execution
- V0 Research Core before full product
- ML-first development order
- ADRs for all experimental decisions

---

## 6. EXPERIMENTAL Decisions

| Decision | Candidates | Experiment ID |
|----------|------------|---------------|
| Final backbone | EfficientNet-B2, ResNet-50 | ADR-001 |
| Ranking formula | Softmax-only, Weighted, Learned | ADR-002 |
| AU subset | 6, 12, 20 AUs | ADR-003 |
| LRP frequency | Every frame, Every N frames, Event-triggered | ADR-004 |
| Intensity architecture | MLP, CNN head, Regression | ADR-005 |
| Uncertainty thresholds | Learned from calibration | ADR-006 |
| Temporal model | EMA, LSTM, Transformer | ADR-007 |

---

## 7. IMPL-DEPENDENT Decisions

- WebSocket vs alternative streaming
- Browser vs server face detection
- GPU/CPU deployment strategy
- Database storage optimization
- Authentication mechanism (JWT vs session cookies)

---

## 8. Terminology

| Term | Definition |
|------|------------|
| Prediction | Observed facial-expression category |
| Intensity | Observed expression strength (0-100) |
| Confidence | Model confidence in its classification |
| Evidence | Model-attributed facial regions and AU activations |
| Narration | Evidence-grounded interpretation of model output |
| Ambiguity | Insufficient evidence for single-class prediction |

**Important:** The system estimates observable facial expressions, not private emotional states.

---

## 9. System Architecture (Summary)

```
Camera → Face Detection → Tracking → Alignment
    → Expression Model + AU Model → Preliminary Ranking
    → [LRP on-demand] → Final Ranking
    → Intensity + Uncertainty → Event Engine
    → Explanation → Output
```

See: `02_SYSTEM_ARCHITECTURE.md` for full details.

---

## 10. Module Relationships

| Module | Depends On | Feeds Into |
|--------|------------|------------|
| Face Detection | Camera | Tracking |
| Tracking | Face Detection | Alignment |
| Expression Model | Alignment | Preliminary Ranking |
| AU Model | Alignment | Preliminary Ranking |
| Preliminary Ranking | Expression + AU | Temporal, XAI Trigger |
| LRP Engine | Preliminary Ranking | Region Evidence |
| Final Ranking | All evidence sources | Intensity, Uncertainty |
| Event Engine | Final Ranking | Narrator, Analytics |
| Explanation Engine | All evidence | Frontend |

---

## 11. Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React/Next.js, Tailwind CSS, WebRTC |
| Backend | FastAPI, Python |
| ML | PyTorch |
| Database | PostgreSQL |
| Real-time | WebSocket |
| Deployment | Docker, Vercel/Railway |

---

## 12. Development Phases

| Phase | Weeks | Focus | Gate |
|-------|-------|-------|------|
| V0 | 1-4 | Research Core | Pass criteria |
| V1 | 5-8 | ML Validation | Experiments complete |
| V2 | 9-12 | Real-time + Backend | Latency met |
| V3 | 13-16 | Frontend + Product | UI functional |
| V4 | 17-20 | Polish + Deploy | Final validation |

See: `18_DEVELOPMENT_ROADMAP.md` for full details.

---

## 13. Definition of DONE

Status as of V1.0 (2026-10-11) — checked items are shipped in this repository;
unchecked items are honest gaps, annotated.

### AI
- [x] 7-class expression recognition (DAN 89.70% / ensemble 89.99% on RAF-DB test)
- [ ] AU detection (6+ AUs) — module shipped **untrained** (w_AU = 0, status reported everywhere; per ADR-003)
- [x] Evidence-based ranking (src/models/ranking_engine.py, ADR-002)
- [x] Intensity estimation (distribution-extremity proxy, ADR-005)
- [x] Confidence calibration (ECE + temperature scaling, EXP-010)
- [x] Ambiguity detection (margin + confidence events, ADR-006)

### XAI
- [x] LRP explanation (offline only — ml/xai, ADR-004)
- [x] Region ranking (Grad-CAM mass over geometric regions)
- [x] Grad-CAM comparison (served via Lab XAI inspector)
- [x] Explanation faithfulness validated (ml/benchmarks/xai_probe.py)

### Live System
- [x] Camera integration (getUserMedia → WebSocket stream)
- [ ] Multiple face tracking — **single-person mode by design** (max_faces = 1; multi-face tracking not shipped)
- [x] Temporal smoothing (EMA, ADR-007)
- [x] Expression transitions (debounced event engine)
- [x] Event timeline (sessions + analytics)

### Product
- [x] Authentication (register/login/logout, HMAC bearer tokens)
- [x] Live Observer mode (verdict mode + continuous)
- [x] Research Lab mode (photo XAI + video)
- [x] Session Analytics mode (summary, timeline, exports)
- [x] Report generation (PDF/JSON/CSV + docx project report)
- [x] Narrator (event-driven TTS, detail levels)

### Research
- [x] VGG16 baseline (EXP-004-VGG16-CLEAN)
- [x] Improved model (EfficientNet-B3 → ViT → DDAMFN++ → DAN → 3-model ensemble)
- [x] Ablation study (TTA on/off, ensemble member subsets, EXP-010)
- [ ] Cross-dataset testing — **not shipped** (single dataset: RAF-DB; generalization unmeasured)
- [x] Documented limitations (report §limitations, ADRs, honesty footers)

---

## 14. What We Will NOT Build

| Feature | Reason |
|---------|--------|
| Lie detection | Scientifically invalid |
| Personality assessment | Not supported by evidence |
| Mental health diagnosis | Dangerous, requires certification |
| Identity recognition | Out of scope |
| Demographic inference | Ethical concerns |
| Mind reading | Impossible |

---

## 15. Document Index

| Doc | Title | Status |
|-----|-------|--------|
| 00 | Master Specification | This file |
| 01 | Product Requirements | See file |
| 02 | System Architecture | See file |
| 03 | ML Model Architecture | See file |
| 04 | Ranking Engine | See file |
| 05 | Action Unit Module | See file |
| 06 | XAI: LRP + Grad-CAM | See file |
| 07 | Intensity & Uncertainty | See file |
| 08 | Dataset & Training | See file |
| 09 | Evaluation & Ablation | See file |
| 10 | Real-Time Video Pipeline | See file |
| 11 | Database Schema | See file |
| 12 | FastAPI Backend | See file |
| 13 | Frontend UX | See file |
| 14 | Security & Privacy | See file |
| 15 | Reporting & Analytics | See file |
| 16 | Deployment Infrastructure | See file |
| 17 | Testing & QA | See file |
| 18 | Development Roadmap | See file |
| 19 | Research Novelty | See file |
| 20 | Event Engine | See file |
| 21 | Model Registry | See file |
| 22 | Experiment Tracking | See file |
| ADR-001 | Backbone Selection | See file |
| ADR-002 | Ranking Formula | See file |
| ADR-003 | AU Subset | See file |
| ADR-004 | LRP Frequency | See file |
| ADR-005 | Intensity Architecture | See file |
| ADR-006 | Uncertainty Thresholds | See file |
| ADR-007 | Temporal Model | See file |
