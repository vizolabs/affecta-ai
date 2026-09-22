# 03 — ML Model Architecture

```
Status: LOCKED (structure) / EXPERIMENTAL (specific choices)
Version: V1.0
```

---

## 1. Three-Model Strategy

| Model | Purpose | Status |
|-------|---------|--------|
| Model A: Expression Classifier | 7-class emotion | VGG16 baseline locked; final backbone experimental |
| Model B: AU Detector | Action Unit detection | 6 AUs locked |
| Model C: Ranking Engine | Evidence-scored ranking | Formula experimental |

---

## 2. Model A: Expression Classifier

### Baseline: VGG16

```python
VGG16(pretrained=True)
    → Freeze features
    → Replace classifier:
        Linear(25088, 4096) → ReLU → Dropout
        Linear(4096, 4096) → ReLU → Dropout
        Linear(4096, 7)
```

- Input: 224×224×3 aligned face
- Output: 7-class softmax
- Training: Fine-tune last 3 layers
- **Status: LOCKED**

### Candidate: EfficientNet-B2

```python
EfficientNet-B2(pretrained=True)
    → Freeze early layers
    → Replace classifier head:
        Linear(1408, 7)
```

- Input: 260×260×3 aligned face
- Output: 7-class softmax
- Training: Progressive unfreezing
- **Status: EXPERIMENTAL — Select after benchmarking**

### Candidate: ResNet-50

```python
ResNet50(pretrained=True)
    → Freeze early layers
    → Replace fc layer:
        Linear(2048, 7)
```

- Input: 224×224×3 aligned face
- Output: 7-class softmax
- Training: Fine-tune last block + head
- **Status: EXPERIMENTAL — Select after benchmarking**

---

## 3. Model B: AU Detector

### Architecture

```python
ResNet18(pretrained=True)
    → Replace classifier:
        Linear(512, 6)
        → Sigmoid (per-AU)
```

- Input: 224×224×3 aligned face
- Output: 6 sigmoid activations
- Loss: Binary Cross-Entropy (per-AU)
- **Status: LOCKED**

### AU Subset (Initial)

| AU | Name | Primary Emotion Evidence |
|----|------|--------------------------|
| AU4 | Brow Lowerer | Angry, Sad |
| AU6 | Cheek Raiser | Happy |
| AU7 | Lid Tightener | Angry |
| AU9 | Nose Wrinkler | Disgust |
| AU10 | Upper Lip Raiser | Disgust |
| AU12 | Lip Corner Puller | Happy |

---

## 4. Model C: Ranking Engine

### Architecture

Not a neural network — MLP or weighted combination.

```python
class RankingEngine:
    def __init__(self, weights=None):
        self.weights = weights or [0.4, 0.25, 0.25, 0.1]
    
    def forward(self, P_e, A_e, R_e=None, T_e=0):
        # P_e: classifier probability
        # A_e: AU evidence score
        # R_e: LRP regional evidence (optional)
        # T_e: temporal evidence
        
        if R_e is None:
            # Preliminary ranking (fast path)
            S = self.weights[0] * P_e + self.weights[1] * A_e
        else:
            # Final ranking (slow path)
            S = (self.weights[0] * P_e + 
                 self.weights[1] * A_e + 
                 self.weights[2] * R_e + 
                 self.weights[3] * T_e)
        return S
```

### Three Ranking Variants (EXPERIMENTAL)

| Variant | Formula | Status |
|---------|---------|--------|
| R0 | S = P_e (softmax only) | Baseline |
| R1 | S = P_e + A_e | + AU evidence |
| R2 | S = P_e + A_e + R_e | + LRP evidence |
| R3 | S = P_e + A_e + R_e + T_e | + Temporal |

**Research Question:** Does adding LRP-derived evidence improve ranking over probability + AU alone?

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate (backbone) | 1e-4 |
| Learning Rate (heads) | 1e-3 |
| Batch Size | 32 |
| Max Epochs | 50 |
| LR Scheduler | CosineAnnealing |
| Early Stopping | Patience=10 |
| Weight Decay | 1e-4 |

### Data Augmentation

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 6. Inference Pipeline

```python
def inference(face_image):
    # Stage 1: Expression
    probs = expression_model(face_image)  # [7]
    
    # Stage 2: AU
    au_activations = au_model(face_image)  # [6]
    A_e = compute_au_evidence(au_activations)
    
    # Stage 3: Preliminary ranking
    S_prelim = ranking_engine(probs, A_e)
    
    # Stage 4: LRP (on-demand)
    if trigger_lrp:
        lrp_map = lrp_engine(expression_model, face_image)
        R_e = aggregate_regions(lrp_map)
        S_final = ranking_engine(probs, A_e, R_e, T_e)
    else:
        S_final = S_prelim
    
    # Stage 5: Intensity
    intensity = intensity_estimator(face_image, au_activations)
    
    # Stage 6: Uncertainty
    certainty = compute_uncertainty(probs)
    
    return {
        'ranking': S_final,
        'intensity': intensity,
        'certainty': certainty,
        'aus': au_activations
    }
```

---

## 7. Model Inputs/Outputs

| Model | Input Shape | Output Shape |
|-------|-------------|--------------|
| Expression (VGG16) | [B, 3, 224, 224] | [B, 7] softmax |
| Expression (EfficientNet) | [B, 3, 260, 260] | [B, 7] softmax |
| AU Detector | [B, 3, 224, 224] | [B, 6] sigmoid |
| Intensity | [B, 512+6] | [B, 1] scalar |

---

## 8. Loss Functions

| Model | Loss |
|-------|------|
| Expression | CrossEntropyLoss |
| AU | BCELoss (per-AU) |
| Ranking | PairwiseRankingLoss (experimental) |
| Intensity | MSELoss or OrdinalLoss |

---

## 9. Hardware Requirements

### Training
- GPU: NVIDIA GPU with ≥ 8GB VRAM
- RAM: ≥ 16GB
- Storage: ≥ 50GB (datasets + checkpoints)

### Inference
- CPU-only: Possible (slower)
- GPU: Recommended for real-time
- Browser: TensorFlow.js / ONNX Runtime Web
