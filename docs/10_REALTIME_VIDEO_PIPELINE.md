# 10 — Real-Time Video Pipeline

```
Status: LOCKED (structure) / IMPL-DEPENDENT (specifics)
Version: V1.0
```

**Implementation status:** webcam stage implemented in `src/pipeline/webcam.py`
(Haar face detection, production-model inference via `ml/models/production.py`,
EMA smoothing, HUD, Grad-CAM toggle, headless `--image` mode, `make webcam`).
Covered by `tests/test_production_webcam.py` (16 tests).

---

## 1. Pipeline Overview

```
Camera (WebRTC)
    → Frame Capture
    → Face Detection
    → Multi-Face Tracking
    → Face Alignment
    → [Fast Path] Expression + AU + Temporal → Preliminary Ranking
    → [Slow Path] LRP + Region Evidence → Final Ranking
    → Intensity + Uncertainty
    → Event Detection
    → Output to Frontend
```

---

## 2. Face Detection

### Browser-Side (Primary)

```javascript
// MediaPipe Face Detection
const detector = await faceDetection.createDetector({
    model: 'short',
    maxFaces: 10,
    minDetectionConfidence: 0.5
});
```

- Model: MediaPipe Face Detection
- Latency: < 30ms
- Max faces: 10
- **Status: IMPL-DEPENDENT**

### Server-Side (Fallback)

```python
# MTCNN or RetinaFace
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda')
```

---

## 3. Multi-Face Tracking

### Assignment Strategy

```python
class FaceTracker:
    def __init__(self, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
    
    def update(self, detections):
        """Assign tracks to new detections."""
        # Compute IoU between existing tracks and new detections
        # Match using Hungarian algorithm
        # Create new tracks for unmatched detections
        # Remove stale tracks
        pass
```

### Identity

- Temporary session IDs (Person 01, Person 02, ...)
- **No facial recognition** — just tracking within session
- Reset when face leaves frame for > 2 seconds

---

## 4. Face Alignment

```python
def align_face(image, landmarks, target_size=(224, 224)):
    """Align face using affine transformation."""
    # Use eye landmarks for alignment
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    
    # Compute rotation angle
    angle = np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    ))
    
    # Compute center
    center = (left_eye + right_eye) / 2
    
    # Apply affine transformation
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    aligned = cv2.warpAffine(image, M, target_size)
    
    return aligned
```

---

## 5. Fast Path (Every Frame)

```python
def fast_path(frame, tracker):
    """Process every frame — must be fast."""
    # 1. Detect faces
    detections = detect_faces(frame)  # < 30ms
    
    # 2. Track
    tracks = tracker.update(detections)  # < 5ms
    
    results = []
    for track in tracks:
        # 3. Align
        aligned = align_face(frame, track.landmarks)  # < 5ms
        
        # 4. Expression model
        probs = expression_model(aligned)  # < 50ms
        
        # 5. AU model
        au_activations = au_model(aligned)  # < 40ms
        
        # 6. Temporal smoothing
        smoothed_probs = temporal_smooth(track.history, probs)  # < 5ms
        
        # 7. Preliminary ranking
        ranking = ranking_engine.preliminary_ranking(
            smoothed_probs, au_activations
        )  # < 5ms
        
        results.append({
            'track_id': track.id,
            'ranking': ranking,
            'probs': smoothed_probs,
            'aus': au_activations,
            'bbox': track.bbox
        })
    
    return results  # Total: < 140ms
```

---

## 6. Slow Path (On-Demand)

```python
def slow_path(frame, track, target_class):
    """Compute detailed XAI — triggered, not every frame."""
    aligned = align_face(frame, track.landmarks)
    
    # 1. LRP
    lrp_heatmap = lrp_engine(expression_model, aligned, target_class)  # < 80ms
    
    # 2. Region aggregation
    region_evidence = aggregate_regions(lrp_heatmap, track.landmarks)  # < 10ms
    
    # 3. Grad-CAM comparison
    gradcam_heatmap = grad_cam(expression_model, aligned, target_class)  # < 60ms
    agreement = compute_agreement(lrp_heatmap, gradcam_heatmap)  # < 5ms
    
    # 4. Final ranking
    final_ranking = ranking_engine.final_ranking(
        track.current_probs,
        track.current_aus,
        region_evidence,
        track.temporal_evidence
    )  # < 10ms
    
    return {
        'lrp_heatmap': lrp_heatmap,
        'region_evidence': region_evidence,
        'gradcam_heatmap': gradcam_heatmap,
        'agreement': agreement,
        'final_ranking': final_ranking
    }
```

---

## 7. Performance Targets

### Latency Budget

| Step | Fast Path | Slow Path |
|------|-----------|-----------|
| Face detection | 30ms | — |
| Tracking | 5ms | — |
| Alignment | 5ms | — |
| Expression model | 50ms | — |
| AU model | 40ms | — |
| Temporal smoothing | 5ms | — |
| Preliminary ranking | 5ms | — |
| LRP | — | 80ms |
| Region aggregation | — | 10ms |
| Grad-CAM | — | 60ms |
| Final ranking | — | 10ms |
| **Total** | **< 140ms** | **< 160ms** |

### Frame Rate Targets

| Scenario | Target | Notes |
|----------|--------|-------|
| 1-3 faces | ≥ 15 FPS | Primary target |
| 4-6 faces | ≥ 10 FPS | Stretch target |
| 7-10 faces | Stress test | May degrade |

---

## 8. XAI Trigger Conditions

```python
def should_compute_xai(track, frame_count, config):
    """Determine if slow path should run."""
    # 1. Transition detected
    if track.transition_detected:
        return True
    
    # 2. Every N frames
    if frame_count % config.xai_frame_interval == 0:
        return True
    
    # 3. User request
    if track.user_requested_xai:
        return True
    
    # 4. Significant intensity change
    if abs(track.intensity - track.prev_intensity) > 20:
        return True
    
    # 5. Ambiguity state
    if track.certainty == 'ambiguous':
        return True
    
    return False
```

---

## 9. Browser-Side Processing

### Option A: Server-Side Inference (Recommended for V1)

```javascript
// Send frame to backend via WebSocket
ws.send(frame.toBlob());

// Receive results
ws.onmessage = (event) => {
    const results = JSON.parse(event.data);
    updateUI(results);
};
```

### Option B: Browser-Side Inference (Future)

```javascript
// TensorFlow.js / ONNX Runtime Web
const model = await tf.loadGraphModel('expression_model.json');
const prediction = model.predict(tf.browser.fromPixels(video));
```

**Status: IMPL-DEPENDENT — Test both, select based on performance.**
