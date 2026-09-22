# 15 — Reporting & Analytics

```
Status: LOCKED
Version: V1.0
```

---

## 1. Session Report Structure

### Section 1: Overview

```
SESSION OVERVIEW

Session ID:     abc-123-def
Duration:       03:42
Total Frames:   3,420
Faces Detected: 4
Start Time:     2026-09-10 14:32:00
End Time:       2026-09-10 14:35:42
```

### Section 2: Expression Distribution

```
EXPRESSION DISTRIBUTION

Happy          42%  ████████████████
Neutral        31%  ████████████
Surprise       13%  █████
Sad             9%  ███
Angry           5%  ██
Fear            0%  
Disgust         0%  
```

### Section 3: Per-Person Analysis

```
PERSON 01

Dominant Expression:    Happy
Peak Intensity:         88/100
Average Confidence:     89%
Major Transition:       Neutral → Happy
Duration of Happiness:  2.1 seconds (longest)

Top Evidence:
- Mouth region:        41%
- Cheek region:        28%
- AU12:                Strong
- AU6:                 Strong
```

### Section 4: Timeline

```
EXPRESSION TIMELINE

00:00  Neutral ──────────┐
00:15                    │
00:30                    ▼
00:45              Slight Happy
01:00                    │
01:15                    ▼
01:30                 Happy
01:45                    │
02:00                    ▼
02:15             Strong Happy
02:30                    │
02:45                    ▼
03:00              Neutral
03:15  ─────────────────┘
```

### Section 5: Event Log

```
EVENTS

14:32:15  Face Enter     Person 01
14:32:18  Transition     Neutral → Happy
14:32:25  Ambiguity      Person 02 (low confidence)
14:32:30  Face Enter     Person 02
14:32:45  Transition     Happy → Strong Happy
14:33:00  Face Exit      Person 02
14:33:15  Intensity ↑    Person 01 (78 → 88)
14:35:42  Session End
```

---

## 2. Export Formats

### PDF Report

```python
def generate_pdf_report(session_id):
    """Generate formatted PDF report."""
    session = get_session(session_id)
    events = get_events(session_id)
    xai_snapshots = get_xai_snapshots(session_id)
    
    pdf = PDFReport()
    pdf.add_section("Overview", session)
    pdf.add_section("Expression Distribution", session)
    pdf.add_section("Per-Person Analysis", session)
    pdf.add_section("Timeline", events)
    pdf.add_section("XAI Evidence", xai_snapshots)
    pdf.add_section("Uncertainty Events", events)
    
    return pdf.export()
```

### CSV Export

```python
def generate_csv_export(session_id):
    """Export raw data as CSV."""
    events = get_events(session_id)
    
    csv_data = []
    for event in events:
        csv_data.append({
            'timestamp': event.timestamp,
            'face_id': event.face_id,
            'event_type': event.event_type,
            'from_emotion': event.from_emotion,
            'to_emotion': event.to_emotion,
            'confidence': event.confidence,
            'intensity': event.intensity
        })
    
    return csv_data
```

---

## 3. Analytics Queries

### Expression Distribution

```sql
SELECT 
    to_emotion as emotion,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM events
WHERE session_id = ? AND event_type = 'transition'
GROUP BY to_emotion
ORDER BY count DESC;
```

### Per-Person Dominant Emotion

```sql
SELECT 
    face_id,
    to_emotion as dominant_emotion,
    COUNT(*) as occurrences
FROM events
WHERE session_id = ? AND event_type = 'transition'
GROUP BY face_id, to_emotion
ORDER BY face_id, occurrences DESC;
```

### Transition Matrix

```sql
SELECT 
    from_emotion,
    to_emotion,
    COUNT(*) as transitions
FROM events
WHERE session_id = ? AND event_type = 'transition'
GROUP BY from_emotion, to_emotion;
```

---

## 4. Real-Time Analytics (In-Memory)

```python
class SessionAnalytics:
    """Real-time analytics during session."""
    
    def __init__(self):
        self.emotion_counts = Counter()
        self.transitions = []
        self.intensity_history = []
        self.events = []
    
    def update(self, face_id, emotion, confidence, intensity):
        """Update analytics with new observation."""
        # Track emotion distribution
        self.emotion_counts[emotion] += 1
        
        # Track transitions
        if self.last_emotion.get(face_id) != emotion:
            transition = {
                'from': self.last_emotion.get(face_id, 'neutral'),
                'to': emotion,
                'timestamp': time.time()
            }
            self.transitions.append(transition)
        
        self.last_emotion[face_id] = emotion
```
