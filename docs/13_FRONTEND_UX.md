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
│  Header: AFFECTA AI    [Live Observer]  [Settings] │
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
│  Header: AFFECTA AI    [Research Lab]  [Settings]  │
├──────────────────┬─────────────────────────────────┤
│                  │                                 │
│   UPLOAD AREA    │   ANALYSIS PANEL                │
│   (image/video)  │   - Emotion ranking             │
│                  │   - Confidence                   │
│                  │   - Intensity                    │
│                  │   - AUs                          │
├──────────────────┼─────────────────────────────────┤
│  XAI INSPECTOR   │  MODEL INFO                     │
│  - LRP heatmap   │  - Model version                │
│  - Grad-CAM      │  - Training data                │
│  - Regions       │  - Metrics                      │
└──────────────────┴─────────────────────────────────┘
```

### Session Analytics

```
┌────────────────────────────────────────────────────┐
│  Header: AFFECTA AI    [Analytics]  [Settings]     │
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

```typescript
// Tailwind config
const theme = {
  dark: {
    bg: '#0a0a0a',
    surface: '#1a1a1a',
    text: '#ffffff',
    accent: '#3b82f6'
  },
  light: {
    bg: '#ffffff',
    surface: '#f5f5f5',
    text: '#0a0a0a',
    accent: '#2563eb'
  }
}
```

---

## 7. Browser Support

| Browser | Version |
|---------|---------|
| Chrome | ≥ 90 |
| Firefox | ≥ 88 |
| Safari | ≥ 14 |
| Edge | ≥ 90 |
