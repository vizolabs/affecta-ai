# 02 — System Architecture

```
Status: LOCKED
Version: V1.0
```

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Auth    │  │  Camera  │  │  Canvas  │  │  Narrator│       │
│  │  Module   │  │  (WebRTC)│  │ Overlay  │  │  (TTS)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│         │            │             │              │               │
│         └────────────┴──────┬──────┴──────────────┘               │
│                             │ WebSocket + REST                    │
└─────────────────────────────┼───────────────────────────────────┘
                              │
┌─────────────────────────────┼───────────────────────────────────┐
│                      SERVER (FastAPI)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Auth   │  │ Session  │  │   ML     │  │  Event   │       │
│  │ Service  │  │ Manager  │  │ Inference │  │  Engine  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│         │            │             │              │               │
│         └────────────┴──────┬──────┴──────────────┘               │
│                             │                                     │
│  ┌──────────────────────────┴──────────────────────────┐        │
│  │              PostgreSQL Database                     │        │
│  │    (Users, Sessions, Events, XAI Snapshots)         │        │
│  └─────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

### Fast Path (Every Frame)
```
Camera Frame
    → Face Detection (<30ms)
    → Alignment (<5ms)
    → Expression Model (<50ms)
    → AU Model (<40ms)
    → Temporal Smoothing (<5ms)
    → Preliminary Ranking (<5ms)
    → Send to Frontend
```

**Total: <140ms per frame**

### Slow Path (On-Demand)
```
Trigger Event
    → LRP (<80ms)
    → Region Aggregation (<10ms)
    → Grad-CAM Comparison (<60ms)
    → Final Ranking (<10ms)
    → Store XAI Snapshot
    → Send to Frontend
```

**Triggered: ~160ms per computation**

---

## 3. Trigger Conditions for XAI

LRP and detailed XAI compute when:
1. Expression transition detected
2. Every N frames (configurable, default: 10)
3. User clicks "Why?" button
4. Significant intensity change
5. Ambiguity state entered

---

## 4. Module Responsibilities

| Module | Input | Output | Latency |
|--------|-------|--------|---------|
| Face Detection | RGB frame | Bounding boxes | < 30ms |
| Tracking | Detections + history | Tracked faces | < 5ms |
| Alignment | Bounding box + frame | Aligned face crop | < 5ms |
| Expression Model | Aligned face | 7-class probabilities | < 50ms |
| AU Model | Aligned face | 6 AU activations | < 40ms |
| Preliminary Ranking | Probs + AUs | Ranked emotions | < 5ms |
| LRP Engine | Model + input | Relevance map | < 80ms |
| Region Aggregation | Relevance + landmarks | Region percentages | < 10ms |
| Final Ranking | All evidence | Final ranked emotions | < 10ms |
| Intensity Estimator | Features | 0-100 score | < 20ms |
| Uncertainty Engine | Probabilities | Certainty state | < 5ms |
| Temporal Engine | History | Smoothed output | < 5ms |
| Event Engine | State changes | Events | < 5ms |
| Explanation Engine | All evidence | Natural language | < 10ms |

---

## 5. Communication Patterns

### Browser ↔ Backend
- **WebSocket**: Real-time emotion streaming
- **REST**: Auth, sessions, settings, reports

### Backend ↔ ML
- **In-process**: PyTorch inference (no IPC overhead)

### Database
- **PostgreSQL**: Persistent storage
- **In-memory**: Real-time buffers, temporal history

---

## 6. Deployment Architecture

### Development
```
Frontend: localhost:3000 (Next.js dev)
Backend: localhost:8000 (Uvicorn)
Database: localhost:5432 (PostgreSQL)
```

### Production
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Vercel    │────▶│   Railway   │────▶│  PostgreSQL │
│  (Frontend) │     │  (Backend)  │     │  (Database) │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 7. Three Product Modes

### Live Observer
- Real-time camera feed
- Multi-face tracking
- Emotion + intensity + ranking
- AU evidence bars
- Live timeline
- Rule-based voice commentary

### Research Lab
- Upload image/video
- Model prediction
- LRP + Grad-CAM comparison
- AU results
- Evidence ranking
- Model inspector

### Session Analytics
- Expression timeline
- Expression distribution
- Transitions
- Per-person summary
- Export report

---

## 8. Security Boundaries

```
┌─────────────────────────────────────┐
│           TRUST BOUNDARY            │
│                                     │
│  Camera frames → Transient only     │
│  Face data → Never stored           │
│  Emotion readings → Stored          │
│  XAI snapshots → Stored (on-demand) │
│  User accounts → Stored (hashed)    │
│                                     │
└─────────────────────────────────────┘
```
