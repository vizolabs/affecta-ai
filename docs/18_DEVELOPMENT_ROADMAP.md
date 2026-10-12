# 18 — Development Roadmap

```
Status: LOCKED
Version: V1.0
```

---

## 1. Execution Philosophy

**ML first → Validation → Real-time → Backend → Frontend → Polish → Research**

Build the product around a validated AI core, not the other way around.

---

## 2. Phase Overview

| Phase | Weeks | Focus | Gate |
|-------|-------|-------|------|
| V0 | 1-4 | Research Core | Pass criteria |
| V1 | 5-8 | ML Validation | Experiments complete |
| V2 | 9-12 | Real-time + Backend | Latency met |
| V3 | 13-16 | Frontend + Product | UI functional |
| V4 | 17-20 | Polish + Deploy | Final validation |

---

## 3. V0: Research Core (Weeks 1-4)

### Week 1: Dataset & Baseline
- [x] Set up project structure (2026-10)
- [x] Download/verify datasets (FER2013, RAF-DB) (2026-10)
- [x] Implement data loading pipeline (2026-10)
- [x] Train VGG16 baseline (2026-10)
- [x] Establish baseline metrics (2026-10)

### Week 2: XAI Proof-of-Concept
- [x] Implement LRP-ε (2026-10)
- [x] Test on sample images (2026-10)
- [x] Implement region aggregation (2026-10)
- [x] Visualize LRP heatmaps (2026-10)

### Week 3: AU Prototype
- [ ] Not shipped (scope decision): trained AU detection — AU training pipeline (BP4D/DISFA)
- [ ] Not shipped (scope decision): trained AU detection — train AU detector on BP4D/DISFA
- [ ] Not shipped (scope decision): trained AU detection — evaluate AU F1 scores
- [ ] Integrate AU evidence with ranking — engine-capable, inactive (w_AU = 0)

### Week 4: Ranking & Evaluation
- [x] Implement ranking engine (R0, R1) (2026-10)
- [x] Implement intensity prototype (2026-10)
- [x] Implement uncertainty engine (2026-10)
- [x] Build minimal Research Lab UI (2026-10)
- [x] Run evaluation suite (2026-10)

### V0 Gate Criteria
```
✅ VGG16 baseline established
✅ LRP produces meaningful heatmaps
✅ AU detection F1 ≥ 0.6
✅ Ranking produces correct ordering
✅ Intensity correlates with human ratings
✅ Can inspect predictions in Research Lab
```

---

## 4. V1: ML Validation (Weeks 5-8)

### Week 5: Backbone Comparison
- [ ] Train EfficientNet-B2
- [x] Train ResNet-50 (2026-10)
- [ ] Benchmark all three
- [x] Select production backbone (2026-10)
- [x] Document ADR-001 (2026-10)

### Week 6: Ranking Experiments
- [x] Implement R2 (+ LRP) (2026-10)
- [x] Implement R3 (+ Temporal) (2026-10)
- [ ] Run ranking comparison
- [x] Optimize weights (2026-10)
- [x] Document ADR-002 (2026-10)

### Week 7: Advanced Features
- [x] Refine intensity model (2026-10)
- [x] Calibrate uncertainty thresholds (2026-10)
- [x] Implement temporal smoothing (2026-10)
- [x] Run ablation study (2026-10)

### Week 8: Validation
- [ ] Not shipped (scope decision): cross-dataset testing (CK+)
- [ ] XAI faithfulness experiments — probe tooling shipped; formal study pending
- [ ] Not shipped (scope decision): failure-mode testing
- [x] Final model selection (2026-10)

### V1 Gate Criteria
```
✅ Production backbone selected (ADR-001)
✅ Ranking formula validated (ADR-002)
✅ Cross-dataset generalization tested
✅ Ablation study complete
✅ All experimental decisions documented
```

---

## 5. V2: Real-Time + Backend (Weeks 9-12)

### Week 9: Real-Time Pipeline
- [ ] Implement face detection (MediaPipe) — shipped implementation uses Haar cascade
- [ ] Not shipped (scope decision): multi-face tracking
- [ ] Implement face alignment
- [x] Optimize inference pipeline (2026-10)

### Week 10: Backend Core
- [x] Set up FastAPI project (2026-10)
- [x] Implement authentication (2026-10)
- [x] Implement session management (2026-10)
- [ ] Set up PostgreSQL — shipped: SQLite at data/app.db

### Week 11: WebSocket Streaming
- [x] Implement WebSocket handler (2026-10)
- [x] Stream real-time results (2026-10)
- [x] Implement fast path (2026-10)
- [x] Implement slow path triggers (2026-10)

### Week 12: Integration
- [x] Connect ML pipeline to backend (2026-10)
- [x] Test real-time performance (2026-10)
- [x] Optimize latency (2026-10)
- [ ] Load testing

### V2 Gate Criteria
```
✅ ≥ 15 FPS for 1-3 faces
✅ Backend API functional
✅ WebSocket streaming working
✅ Authentication secure
✅ Latency targets met
```

---

## 6. V3: Frontend + Product (Weeks 13-16)

### Week 13: Frontend Foundation
- [x] Set up Next.js project (2026-10)
- [x] Implement authentication UI (2026-10)
- [x] Implement camera access (src/pipeline/webcam.py, Haar cascade) (2026-10)
- [x] Implement basic layout (2026-10)

### Week 14: Live Observer
- [x] Implement camera feed (live HUD, EMA, Grad-CAM overlay, --image headless) (2026-10)
- [x] Implement face overlays (2026-10)
- [x] Implement person cards (2026-10)
- [x] Implement emotion spectrum (2026-10)

### Week 15: Research Lab
- [x] Implement upload interface (2026-10)
- [x] Implement analysis panel (2026-10)
- [x] Implement XAI inspector (2026-10)
- [x] Implement model info (2026-10)

### Week 16: Analytics + Reports
- [x] Implement session list (2026-10)
- [x] Implement statistics (2026-10)
- [x] Implement timeline (2026-10)
- [x] Implement PDF export (2026-10)

### V3 Gate Criteria
```
✅ Live Observer functional
✅ Research Lab functional
✅ Session Analytics functional
✅ Reports generating correctly
✅ UI responsive
```

---

## 7. V4: Polish + Deploy (Weeks 17-20)

### Week 17: Narrator + Events
- [x] Implement event engine (2026-10)
- [x] Implement rule-based narrator (2026-10)
- [x] Implement voice synthesis (2026-10)
- [x] Narrator controls (2026-10)

### Week 18: Security + Privacy
- [ ] Security audit
- [ ] Privacy dashboard
- [x] User data deletion (2026-10)
- [x] Rate limiting (2026-10)

### Week 19: Deployment
- [ ] Set up production environment
- [ ] Deploy frontend (Vercel)
- [ ] Deploy backend (Railway)
- [ ] Database setup

### Week 20: Documentation + Final
- [x] API documentation (2026-10)
- [ ] User documentation
- [ ] Final testing
- [ ] Presentation preparation

### V4 Gate Criteria
```
✅ All features complete
✅ Security verified
✅ Deployed and accessible
✅ Documentation complete
✅ Presentation ready
```

---

## 8. Risk Management

| Risk | Mitigation |
|------|------------|
| Dataset access denied | Use publicly available subsets |
| Model accuracy too low | Adjust augmentation, try different architectures |
| Latency too high | Optimize model, reduce XAI frequency |
| Backend issues | Start with simple REST, add WebSocket later |
| Frontend complexity | Build minimal viable UI first |
