# 13 — Frontend UX Design

```
Status: LOCKED
Version: V1.0
```

---

## 1. Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Animation | Framer Motion |
| Charts | Recharts |
| Camera | WebRTC |
| State | Zustand / React Context |

---

## 2. Three Product Modes

### Live Observer

```
┌────────────────────────────────────────────────────┐
│  Header: Human Facial Emotion Detection    [Live Observer]  [Settings] │
├──────────────────────────────────┬─────────────────┤
│                                  │  Person 01      │
│                                  │  😊 HAPPY       │
│       LIVE CAMERA FEED           │  Confidence 92% │
│       (with face overlays)       │  Intensity 84   │
│                                  │  Certainty HIGH │
│                                  │                 │
│                                  │  [Why?] [Heatmap]│
├──────────────────────────────────┴─────────────────┤
│  Timeline: Neutral ─── Happy ─── Strong Happy       │
├────────────────────────────────────────────────────┤
│  Controls: [Start] [Stop] [Narrator] [Export]      │
└────────────────────────────────────────────────────┘
```

### Research Lab

```
┌────────────────────────────────────────────────────┐
│  Header: Human Facial Emotion Detection    [Research Lab]  [Settings]  │
├──────────────────┬─────────────────────────────────┤
│                  │                                 │
│   UPLOAD AREA    │   ANALYSIS PANEL                │
│   (photo|video)  │   photo → ranking, confidence,   │
│   JPG/PNG ·      │     intensity, AUs               │
│   MP4/MOV/AVI/   │   video → dominant emotion,      │
│   MKV/WEBM       │     distribution, transitions,   │
│                  │     peak frame, narration        │
├──────────────────┼─────────────────────────────────┤
│  XAI INSPECTOR   │  MODEL INFO                     │
│  - LRP heatmap   │  - Model version                │
│  - Grad-CAM      │  - Training data                │
│  - Regions       │  - Metrics                      │
└──────────────────┴─────────────────────────────────┘
```

The same drop zone accepts a photo or a video; the button and the analysis
panel switch with the input type. Video runs are ad-hoc research artifacts —
they are **not** recorded as Observer sessions and never touch the session list.

### Session Analytics

```
┌────────────────────────────────────────────────────┐
│  Header: Human Facial Emotion Detection    [Analytics]  [Settings]     │
├──────────────────┬─────────────────────────────────┤
│                  │                                 │
│  SESSION LIST    │  STATISTICS                     │
│  - Session 1     │  - Expression distribution      │
│  - Session 2     │  - Per-person analysis          │
│  - Session 3     │  - Timeline                     │
│                  │  - Transitions                   │
├──────────────────┼─────────────────────────────────┤
│                  │  [Export PDF] [Export CSV]       │
└──────────────────┴─────────────────────────────────┘
```

---

## 3. Component Structure

```
components/
├── layout/
│   ├── Header.tsx
│   ├── Sidebar.tsx
│   └── Layout.tsx
├── auth/
│   ├── LoginForm.tsx
│   ├── RegisterForm.tsx
│   └── ProtectedRoute.tsx
├── camera/
│   ├── CameraFeed.tsx
│   ├── FaceOverlay.tsx
│   └── MultiFaceTracker.tsx
├── emotion/
│   ├── PersonCard.tsx
│   ├── EmotionSpectrum.tsx
│   ├── AUBars.tsx
│   └── CertaintyBadge.tsx
├── xai/
│   ├── LRPHeatmap.tsx
│   ├── GradCAMOverlay.tsx
│   └── RegionChart.tsx
├── timeline/
│   ├── EmotionTimeline.tsx
│   └── TransitionMarker.tsx
├── analytics/
│   ├── SessionStats.tsx
│   ├── ExpressionPie.tsx
│   └── SessionList.tsx
├── narrator/
│   ├── NarratorControls.tsx
│   └── CommentaryPanel.tsx
└── common/
    ├── Button.tsx
    ├── Card.tsx
    └── ThemeToggle.tsx
```

---

## 4. Responsive Breakpoints

| Breakpoint | Width | Features |
|------------|-------|----------|
| Desktop | ≥ 1024px | Full features |
| Tablet | 768px - 1023px | Limited sidebar |
| Mobile | < 768px | View only |

---

## 5. Accessibility

- ARIA labels on all interactive elements
- Keyboard navigation (Tab, Enter, Escape)
- High contrast mode
- Screen reader support
- Focus indicators

---

## 6. Theming

Design tokens live in `src/app/globals.css`. The dark **HUD** theme is the
primary experience and matches the login/register screens.

```css
:root, .dark {
  --bg: #04070c;      /* deep space navy */
  --surface: #060b14;
  --text: #e8f1f5;
  --accent: #22d3ee;  /* cyan */
  --muted: #7c93a3;
  --border: rgba(148, 197, 214, 0.14);
}

.light {
  --bg: #f4f7f9;
  --surface: #ffffff;
  --text: #0b1620;
  --accent: #0e7490;
  --muted: #5a6b78;
  --border: rgba(15, 40, 55, 0.12);
}
```

Auth screens additionally layer a self-contained HUD frame (grid field, radial
glows, scan sweep, corner brackets, telemetry rail) via
`components/auth/AuthShell.tsx`; shared field/button primitives live in
`components/auth/AuthControls.tsx`.

---

## 7. Settings Screen

Cards: Account · Appearance · Narrator · Analysis smoothing · **Capture &
display** · XAI · Danger zone.

**Capture & display** exposes the runtime controls for capture cadence and
presentation:

| Control | Range | Default | Effect |
|---------|-------|---------|--------|
| Analysis mode | Live / Periodic | live | Continuous per-frame labels, or a pinned verdict every N seconds |
| Analysis frame rate (live mode) | 1–15 fps | 8 | Capture/analysis interval (`1000 / fps` ms); status bar shows `set N fps` |
| Verdict every (periodic mode) | 30 s – 10 min | 2:00 | Verdict-window flush cadence; presets 0:30 / 1:00 / 1:30 / 2:00 / 3:00 / 5:00 |
| Re-analyse on movement (periodic) | on / off | on | 64×48 frame-difference flushes an instant verdict (5 s cooldown) |
| Emotion hold | 0 – 10 min | off | Display dwell: a detected emotion stays painted for N after a change, with a countdown bar; shown in minutes above 60 s |

Periodic mode ("verdict mode") keeps frames streaming at the configured rate —
only the *display* is aggregated: every window the readings are mean-pooled
into one verdict pinned to the face (`VERDICT · JOY 74% · window n=…`), and the
Live bar reports the countdown to the next flush. When the face leaves the
frame the verdict dims with a `STALE` marker instead of vanishing or being
presented as live; re-acquiring the face flushes a fresh verdict within a few
frames. Frames that arrive during a hold are never replayed late; only the
newest reading is shown once the hold expires.

The hold is presentation-only: narration, events and the Person panel always
show live values, so the UI never displays a stale number as if it were current.
Logic lives in `lib/verdict.ts` (`VerdictBuffer`), `lib/emotionHold.ts`
(`useEmotionHold`), the cadence timer in `lib/cadence.ts` (`startCadence`) and
the motion check in `lib/motion.ts` (`computeMotionScore` / `shouldTriggerMotion`).

---

## 8. Browser Support

| Browser | Version |
|---------|---------|
| Chrome | ≥ 90 |
| Firefox | ≥ 88 |
| Safari | ≥ 14 |
| Edge | ≥ 90 |
