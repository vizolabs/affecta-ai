# 17 — Testing & QA

```
Status: LOCKED
Version: V1.0
```

---

## 1. Test Types

### Unit Tests

```python
# Backend
pytest tests/unit/
    test_auth.py          # Authentication logic
    test_session.py       # Session management
    test_ranking.py       # Ranking engine
    test_intensity.py     # Intensity estimation
    test_uncertainty.py   # Uncertainty engine

# Frontend
npm run test
    components.test.tsx   # Component tests
    hooks.test.ts         # Hook tests
```

### Integration Tests

```python
pytest tests/integration/
    test_api_endpoints.py    # API endpoint tests
    test_websocket.py        # WebSocket tests
    test_database.py         # Database operations
    test_auth_flow.py        # Auth flow tests
```

### ML Tests

```python
pytest tests/ml/
    test_inference.py        # Model inference
    test_lrp.py              # LRP output validation
    test_ranking.py          # Ranking consistency
    test_au_detection.py     # AU detection
```

### E2E Tests

```javascript
// Playwright
tests/e2e/
    login.spec.ts           # Login flow
    camera.spec.ts          # Camera access
    analysis.spec.ts        # Analysis flow
    export.spec.ts          # Report export
```

---

## 2. Test Coverage Targets

| Component | Target |
|-----------|--------|
| Backend API | 80% |
| ML inference | 90% |
| Frontend components | 70% |
| Critical paths | 95% |

---

## 3. ML Model Tests

### Inference Consistency

```python
def test_inference_consistency():
    """Same input should produce same output."""
    model = load_model()
    input_tensor = create_test_input()
    
    output1 = model(input_tensor)
    output2 = model(input_tensor)
    
    assert torch.allclose(output1, output2)
```

### LRP Validation

```python
def test_lrp_relevance_sum():
    """LRP relevance should sum to prediction score."""
    model = load_model()
    input_tensor = create_test_input()
    
    prediction = model(input_tensor).max().item()
    relevance = lrp_engine(model, input_tensor)
    
    assert abs(relevance.sum() - prediction) < 1e-5
```

### Ranking Consistency

```python
def test_ranking_ordering():
    """Higher probability should rank higher."""
    engine = RankingEngine()
    
    probs = torch.tensor([0.8, 0.1, 0.05, 0.03, 0.01, 0.01, 0.0])
    au_activations = torch.tensor([0.9, 0.1, 0.1, 0.1, 0.1, 0.9])
    
    ranking = engine.preliminary_ranking(probs, au_activations)
    
    assert list(ranking.keys())[0] == 'happy'
```

---

## 4. Performance Tests

### Latency Benchmark

```python
def test_inference_latency():
    """Inference should meet latency requirements."""
    model = load_model()
    input_tensor = create_test_input()
    
    times = []
    for _ in range(100):
        start = time.time()
        model(input_tensor)
        times.append(time.time() - start)
    
    avg_latency = np.mean(times) * 1000  # ms
    assert avg_latency < 50  # Should be under 50ms
```

### Memory Profiling

```python
def test_memory_usage():
    """Model should not leak memory."""
    import tracemalloc
    
    tracemalloc.start()
    
    model = load_model()
    for _ in range(100):
        model(create_test_input())
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 1024 * 1024 * 500  # < 500MB
```

---

## 5. Failure Mode Tests

### Lighting Conditions

```python
@pytest.mark.parametrize("brightness", [0.3, 0.5, 0.7, 1.0])
def test_lighting_variation(brightness):
    """Model should handle lighting variations."""
    image = load_test_image()
    adjusted = adjust_brightness(image, brightness)
    
    prediction = model(adjusted)
    assert prediction is not None
```

### Pose Variation

```python
@pytest.mark.parametrize("angle", [-15, -5, 0, 5, 15])
def test_pose_variation(angle):
    """Model should handle slight pose variations."""
    image = load_test_image()
    rotated = rotate_image(image, angle)
    
    prediction = model(rotated)
    assert prediction is not None
```

### Occlusion

```python
def test_partial_occlusion():
    """Model should handle partial face occlusion."""
    image = load_test_image()
    occluded = add_occlusion(image, region='mouth')
    
    prediction = model(occluded)
    assert prediction is not None
```

---

## 6. Bug Tracking

### Labels

- `bug`: Something broken
- `feature`: New feature
- `documentation`: Docs update
- `performance`: Performance issue
- `security`: Security concern

### Priority

- `P0`: Critical, blocks development
- `P1`: High, should fix soon
- `P2`: Medium, normal priority
- `P3`: Low, nice to have
