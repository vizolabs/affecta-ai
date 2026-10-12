# 01 — Product Requirements

```
Status: LOCKED
Version: V1.0
```

---

## 1. User Roles

| Role | Description |
|------|-------------|
| Guest | Not authenticated, cannot access system |
| Authenticated User | Full access to all modes |
| Admin (optional) | User management, system config |

---

## 2. Core User Flows

### Flow 1: Live Analysis
```
Login → Camera Permission → System Check → Live Observer → Analysis
```

### Flow 2: Image/Video Upload
```
Login → Research Lab → Upload → Analysis → Explanation → Download
```

### Flow 3: Session Review
```
Login → Session Analytics → Select Session → View Report → Export
```

---

## 3. Feature Priorities

Status as of V1.0 — see docs/00 §13 for the annotated Definition of Done.

### Must Have (MVP — V0)
- [x] Authentication (register/login/logout)
- [x] Camera access + face detection
- [x] 7-class emotion detection
- [x] Emotion ranking
- [x] LRP explanation (offline; Grad-CAM served)
- [ ] Multi-face tracking — not shipped (single-person mode, max_faces = 1, by design)
- [x] Basic narrator

### Should Have (V1)
- [ ] AU detection (6 AUs) — module shipped untrained (w_AU = 0, per ADR-003)
- [x] Intensity estimation (proxy, ADR-005)
- [x] Uncertainty detection (ADR-006)
- [x] Temporal timeline
- [x] Session analytics
- [x] PDF report export
- [x] Event engine (transitions)

### Nice to Have (V2)
- [x] Research Lab mode (photo XAI + video analysis)
- [ ] Human vs AI comparison — not shipped
- [ ] Prediction replay — not shipped
- [x] Model card (Lab Model inspector + registry)
- [ ] Failure mode testing — not shipped (see docs/17 honest gaps)

---

## 4. UI/UX Requirements

### Responsive Design
- Desktop: ≥ 1024px (full features)
- Tablet: 768px - 1023px (limited features)
- Mobile: < 768px (view only)

### Accessibility
- WCAG 2.1 AA compliance
- ARIA labels
- Keyboard navigation
- High contrast mode
- Screen reader support

### Theming
- Dark mode (default)
- Light mode
- System preference detection

---

## 5. Performance Requirements

| Metric | Target | Stretch |
|--------|--------|---------|
| Face detection latency | < 50ms | < 30ms |
| Emotion classification | < 100ms | < 60ms |
| End-to-end latency | < 250ms | < 150ms |
| Frame rate (1-3 faces) | ≥ 15 FPS | ≥ 20 FPS |
| Frame rate (4-6 faces) | ≥ 10 FPS | ≥ 15 FPS |
| Multi-face capacity | Up to 10 faces | Stress test |

---

## 6. Security Requirements

- Passwords hashed (bcrypt) — shipped: PBKDF2-HMAC (210k iters); min 8 chars, no complexity rule
- Secure session management — shipped: custom HMAC bearer token; 7-day TTL; no rotation (logout revokes)
- HTTPS in production
- CSRF protection
- Rate limiting — shipped: API-wide limiter not shipped (local-first); login 10/min + register 5/min per IP
- No raw video storage by default
- User data deletion capability

---

## 7. Privacy Requirements

- Camera frames processed transiently
- No facial recognition/identity templates stored
- Session data stored (emotion readings only)
- User can delete all their data
- Clear privacy dashboard
- Consent required before camera access

---

## 8. Reporting Requirements

### Session Report
- Session overview (duration, frames, faces)
- Expression distribution (pie chart)
- Per-person analysis
- Timeline graph
- AU evidence summary
- LRP heatmaps
- Uncertainty events
- Export as PDF/CSV

---

## 9. Browser Support

| Browser | Version |
|---------|---------|
| Chrome | ≥ 90 |
| Firefox | ≥ 88 |
| Safari | ≥ 14 |
| Edge | ≥ 90 |

---

## 10. Constraints

- No identity recognition
- No "mind reading" claims
- No mental health diagnosis
- No lie detection
- Observable expressions only
- Evidence-grounded explanations only
