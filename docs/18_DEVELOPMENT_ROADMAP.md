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
- [ ] Set up project structure
- [ ] Download/verify datasets (FER2013, RAF-DB)
- [ ] Implement data loading pipeline
- [ ] Train VGG16 baseline
- [ ] Establish baseline metrics

### Week 2: XAI Proof-of-Concept
- [ ] Implement LRP-ε
- [ ] Test on sample images
- [ ] Implement region aggregation
- [ ] Visualize LRP heatmaps

### Week 3: AU Prototype
- [ ] Set up AU training pipeline
- [ ] Train AU detector on BP4D/DISFA
- [ ] Evaluate AU F1 scores
- [ ] Integrate AU evidence with ranking

### Week 4: Ranking & Evaluation
- [ ] Implement ranking engine (R0, R1)
- [ ] Implement intensity prototype
- [ ] Implement uncertainty engine
- [ ] Build minimal Research Lab UI
- [ ] Run evaluation suite

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
- [ ] Train ResNet-50
- [ ] Benchmark all three
- [ ] Select production backbone
- [ ] Document ADR-001

### Week 6: Ranking Experiments
- [ ] Implement R2 (+ LRP)
- [ ] Implement R3 (+ Temporal)
- [ ] Run ranking comparison
- [ ] Optimize weights
- [ ] Document ADR-002

### Week 7: Advanced Features
- [ ] Refine intensity model
- [ ] Calibrate uncertainty thresholds
- [ ] Implement temporal smoothing
- [ ] Run ablation study

### Week 8: Validation
- [ ] Cross-dataset testing (CK+)
- [ ] XAI faithfulness experiments
- [ ] Failure mode testing
- [ ] Final model selection

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
- [ ] Implement face detection (MediaPipe)
- [ ] Implement multi-face tracking
- [ ] Implement face alignment
- [ ] Optimize inference pipeline

### Week 10: Backend Core
- [ ] Set up FastAPI project
- [ ] Implement authentication
- [ ] Implement session management
- [ ] Set up PostgreSQL

### Week 11: WebSocket Streaming
- [ ] Implement WebSocket handler
- [ ] Stream real-time results
- [ ] Implement fast path
- [ ] Implement slow path triggers

### Week 12: Integration
- [ ] Connect ML pipeline to backend
- [ ] Test real-time performance
- [ ] Optimize latency
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
- [ ] Set up Next.js project
- [ ] Implement authentication UI
- [ ] Implement camera access
- [ ] Implement basic layout

### Week 14: Live Observer
- [ ] Implement camera feed
- [ ] Implement face overlays
- [ ] Implement person cards
- [ ] Implement emotion spectrum

### Week 15: Research Lab
- [ ] Implement upload interface
- [ ] Implement analysis panel
- [ ] Implement XAI inspector
- [ ] Implement model info

### Week 16: Analytics + Reports
- [ ] Implement session list
- [ ] Implement statistics
- [ ] Implement timeline
- [ ] Implement PDF export

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
- [ ] Implement event engine
- [ ] Implement rule-based narrator
- [ ] Implement voice synthesis
- [ ] Narrator controls

### Week 18: Security + Privacy
- [ ] Security audit
- [ ] Privacy dashboard
- [ ] User data deletion
- [ ] Rate limiting

### Week 19: Deployment
- [ ] Set up production environment
- [ ] Deploy frontend (Vercel)
- [ ] Deploy backend (Railway)
- [ ] Database setup

### Week 20: Documentation + Final
- [ ] API documentation
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
