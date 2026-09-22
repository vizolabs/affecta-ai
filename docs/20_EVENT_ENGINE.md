# 20 — Event Engine

```
Status: LOCKED
Version: V1.0
```

---

## 1. Purpose

Detect significant state changes to:
- Trigger XAI computation
- Generate meaningful narration
- Create useful analytics
- Avoid repetitive output

---

## 2. Event Types

| Event | Trigger | Description |
|-------|---------|-------------|
| `transition` | Emotion changes | Neutral → Happy |
| `ambiguity_enter` | State becomes ambiguous | Low confidence |
| `ambiguity_exit` | State becomes certain | Confidence restored |
| `intensity_change` | Intensity changes > 20 | Significant shift |
| `face_enter` | New face detected | Person appears |
| `face_exit` | Face lost for > 2s | Person leaves |
| `confidence_drop` | Confidence drops > 20 | Uncertainty increase |
| `stabilization` | Expression stable > 3s | Expression settled |

---

## 3. Event Detection Algorithm

```python
class EventEngine:
    def __init__(self, config):
        self.config = config
        self.state = {}  # track_id -> last state
    
    def detect_events(self, track_id, current_state):
        """Detect events based on state change."""
        events = []
        prev = self.state.get(track_id, {})
        
        # Transition detection
        if prev.get('emotion') != current_state['emotion']:
            events.append(Event(
                type='transition',
                face_id=track_id,
                from_emotion=prev.get('emotion'),
                to_emotion=current_state['emotion'],
                confidence=current_state['confidence']
            ))
        
        # Ambiguity detection
        if prev.get('certainty') != current_state['certainty']:
            if current_state['certainty'] == 'ambiguous':
                events.append(Event(
                    type='ambiguity_enter',
                    face_id=track_id,
                    confidence=current_state['confidence']
                ))
            elif prev.get('certainty') == 'ambiguous':
                events.append(Event(
                    type='ambiguity_exit',
                    face_id=track_id,
                    confidence=current_state['confidence']
                ))
        
        # Intensity change
        if prev.get('intensity') is not None:
            intensity_diff = abs(
                current_state['intensity'] - prev['intensity']
            )
            if intensity_diff > self.config.intensity_threshold:
                events.append(Event(
                    type='intensity_change',
                    face_id=track_id,
                    from_intensity=prev['intensity'],
                    to_intensity=current_state['intensity']
                ))
        
        # Update state
        self.state[track_id] = current_state
        
        return events
```

---

## 4. Event Schema

```python
class Event:
    id: UUID
    session_id: UUID
    face_id: int
    timestamp: datetime
    event_type: str
    from_emotion: Optional[str]
    to_emotion: Optional[str]
    confidence: Optional[float]
    intensity: Optional[float]
    certainty: Optional[str]
    metadata: Optional[dict]
```

---

## 5. Integration Points

### Trigger XAI

```python
def on_event(event):
    if event.type in ['transition', 'ambiguity_enter']:
        trigger_lrp(event.face_id)
```

### Generate Narration

```python
def generate_narration(event):
    if event.type == 'transition':
        return f"Person {event.face_id} transitioned from {event.from_emotion} to {event.to_emotion}."
    elif event.type == 'ambiguity_enter':
        return f"Expression became ambiguous for Person {event.face_id}."
```

### Update Analytics

```python
def on_event(event):
    analytics.record_event(event)
```

---

## 6. Narration Rules

### Transition

```
Template: "Person {id} transitioned from {from} to {to}."
Example: "Person 01 transitioned from Neutral to Happy."
```

### Intensity Change

```
Template: "Expression intensity {direction} for Person {id}."
Example: "Expression intensity increased for Person 01."
```

### Ambiguity

```
Template: "Expression became ambiguous for Person {id}."
Example: "Expression became ambiguous for Person 02."
```

### Face Enter/Exit

```
Template: "Person {id} {entered/left} the scene."
Example: "Person 03 entered the scene."
```

---

## 7. Event Storage

```sql
-- Events are stored in the events table
-- See: 11_DATABASE_SCHEMA.md

-- Example:
INSERT INTO events (session_id, face_id, event_type, 
                    from_emotion, to_emotion, confidence)
VALUES (?, ?, 'transition', 'neutral', 'happy', 0.91);
```
