# 06 — Explainable AI: LRP + Grad-CAM

```
Status: LOCKED (LRP primary, Grad-CAM comparison)
Version: V1.0
```

---

## 1. Primary XAI: Layer-wise Relevance Propagation (LRP)

### Algorithm

1. **Forward pass**: Compute prediction
2. **Backward pass**: Distribute relevance from output to input
3. **Conservation**: Σ relevance = prediction score

### LRP-ε Rule

```
R_j = Σ (a_j * z_j / (Σ a_j + ε)) * R_k
```

Where:
- a_j = activation of neuron j
- z_j = weight × activation
- ε = stabilization term (1e-6)
- R_k = relevance from layer above

### Implementation

```python
def lrp_epsilon(model, input_tensor, target_class, epsilon=1e-6):
    """Compute LRP relevance map for target class."""
    # Forward pass
    output = model(input_tensor)
    
    # Create relevance seed
    relevance = torch.zeros_like(output)
    relevance[0, target_class] = output[0, target_class]
    
    # Backward pass through layers
    for layer in reversed(model.features):
        relevance = lrp_layer(layer, relevance, epsilon)
    
    # Reshape to spatial dimensions
    relevance = relevance.view(input_tensor.shape)
    
    return relevance
```

---

## 2. Region Aggregation

### Step 1: Pixel-Level Relevance
```
LRP heatmap: [H, W] (one relevance value per pixel)
```

### Step 2: Facial Region Mapping
Using 68-point facial landmarks to define regions:

| Region | Landmarks |
|--------|-----------|
| Mouth | 48-67 |
| Cheeks | 1-17 (left/right) |
| Eyes | 36-47 |
| Eyebrows | 17-27 |
| Nose | 28-35 |

### Step 3: Aggregate

```python
def aggregate_regions(lrp_heatmap, landmarks):
    regions = {
        'mouth': extract_region(landmarks, 'mouth'),
        'cheeks': extract_region(landmarks, 'cheeks'),
        'eyes': extract_region(landmarks, 'eyes'),
        'eyebrows': extract_region(landmarks, 'eyebrows'),
        'nose': extract_region(landmarks, 'nose')
    }
    
    region_relevance = {}
    for name, mask in regions.items():
        region_relevance[name] = np.sum(np.abs(lrp_heatmap * mask))
    
    # Normalize to percentages
    total = sum(region_relevance.values())
    for k in region_relevance:
        region_relevance[k] /= total
    
    return region_relevance
```

### Step 4: Region Ranking

```
1. Mouth       41%
2. Cheeks      28%
3. Eyes        19%
4. Eyebrows     8%
5. Other        4%
```

---

## 3. Secondary XAI: Grad-CAM (Comparison)

### Algorithm

```python
def grad_cam(model, input_tensor, target_class, layer_name):
    """Compute Grad-CAM heatmap for target class."""
    # Get feature maps
    feature_maps = None
    def hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(hook)
    
    # Forward pass
    output = model(input_tensor)
    
    # Compute gradients
    model.zero_grad()
    output[0, target_class].backward()
    
    # Weight feature maps by gradient averages
    weights = torch.mean(layer.weight.grad, dim=[2, 3])
    
    # Weighted combination + ReLU
    cam = torch.relu(torch.sum(weights * feature_maps, dim=1))
    
    # Upsample to input size
    cam = F.interpolate(cam.unsqueeze(1), 
                        size=input_tensor.shape[2:],
                        mode='bilinear')
    
    return cam.squeeze()
```

---

## 4. Comparison Protocol

| Aspect | LRP | Grad-CAM |
|--------|-----|----------|
| Resolution | Pixel-level | Coarse heatmap |
| Faithfulness | High | Medium |
| Speed | Slower | Faster |
| Use case | Primary explanation | Validation |

### Agreement Measurement
```python
def compute_agreement(lrp_heatmap, gradcam_heatmap):
    # Normalize both to [0, 1]
    lrp_norm = (lrp_heatmap - lrp_heatmap.min()) / (lrp_heatmap.max() - lrp_heatmap.min())
    gradcam_norm = (gradcam_heatmap - gradcam_heatmap.min()) / (gradcam_heatmap.max() - gradcam_heatmap.min())
    
    # Compute correlation
    agreement = np.corrcoef(lrp_norm.flatten(), gradcam_norm.flatten())[0, 1]
    return agreement
```

---

## 5. XAI Trigger Conditions

LRP computes when:
1. Expression transition detected
2. Every N frames (configurable, default: 10)
3. User clicks "Why?" button
4. Significant intensity change (> 20 points)
5. Ambiguity state entered

---

## 6. Evaluation Metrics

### Deletion Test
```
1. Start with original image
2. Remove top-ranked region
3. Measure prediction drop
4. Remove next region
5. Repeat

Metric: AUC of prediction vs. regions removed
```

### Insertion Test
```
1. Start with blank image
2. Add top-ranked region
3. Measure prediction rise
4. Add next region
5. Repeat

Metric: AUC of prediction vs. regions added
```

### Stability Test
```
1. Add small noise to input
2. Compute LRP
3. Measure heatmap change

Metric: Heatmap consistency under perturbation
```

---

## 7. Implementation Checklist

- [ ] LRP-ε implementation
- [ ] Region aggregation
- [ ] Grad-CAM implementation
- [ ] Comparison protocol
- [ ] Trigger conditions
- [ ] Deletion/insertion tests
- [ ] Stability tests
