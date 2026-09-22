# 21 — Model Registry

```
Status: LOCKED
Version: V1.0
```

---

## 1. Purpose

Track model versions, configurations, and deployment status.

---

## 2. Registry Schema

```python
class ModelRecord:
    id: UUID
    model_name: str        # e.g., "affecta-expression"
    version: str           # e.g., "v1.0.0"
    model_type: str        # "expression", "au", "intensity", "ranking"
    dataset_used: str      # e.g., "FER2013+RAF-DB"
    training_config: dict  # Hyperparameters
    metrics: dict          # Accuracy, F1, etc.
    checkpoint_path: str   # Path to model file
    checksum: str          # SHA256 of checkpoint
    status: str            # "training", "trained", "validated", "deployed"
    created_at: datetime
    deployed_at: Optional[datetime]
```

---

## 3. Model Naming Convention

```
{project}-{type}-{version}

Examples:
affecta-expression-v1.0.0
affecta-au-v1.0.0
affecta-intensity-v1.0.0
affecta-ranking-v1.0.0
```

---

## 4. Status Workflow

```
training → trained → validated → deployed
                          ↓
                      deprecated
```

---

## 5. Registry Operations

### Register New Model

```python
def register_model(name, version, type, config, metrics, checkpoint):
    """Register a new model version."""
    record = ModelRecord(
        model_name=name,
        version=version,
        model_type=type,
        training_config=config,
        metrics=metrics,
        checkpoint_path=checkpoint,
        checksum=compute_checksum(checkpoint),
        status='trained'
    )
    save_to_registry(record)
```

### Deploy Model

```python
def deploy_model(model_id):
    """Mark model as deployed."""
    record = get_model(model_id)
    record.status = 'deployed'
    record.deployed_at = datetime.now()
    
    # Mark previous version as deprecated
    deprecated_previous_version(record.model_name)
    
    update_registry(record)
```

### Get Active Model

```python
def get_active_model(model_type):
    """Get currently deployed model."""
    return query_registry(
        model_type=model_type,
        status='deployed'
    )
```

---

## 6. Database Table

```sql
CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    dataset_used VARCHAR(255),
    training_config JSONB,
    metrics JSONB,
    checkpoint_path VARCHAR(500),
    checksum VARCHAR(64),
    status VARCHAR(20) DEFAULT 'trained',
    created_at TIMESTAMP DEFAULT NOW(),
    deployed_at TIMESTAMP,
    UNIQUE(model_name, version)
);
```

---

## 7. Example Registry Entry

```json
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "model_name": "affecta-expression",
    "version": "v1.0.0",
    "model_type": "expression",
    "dataset_used": "FER2013+RAF-DB",
    "training_config": {
        "backbone": "efficientnet-b2",
        "epochs": 50,
        "learning_rate": 0.0001,
        "batch_size": 32
    },
    "metrics": {
        "accuracy": 0.742,
        "macro_f1": 0.731,
        "cross_dataset_accuracy": 0.68
    },
    "checkpoint_path": "/models/affecta-expression-v1.0.0.pt",
    "checksum": "sha256:abc123...",
    "status": "deployed",
    "created_at": "2026-09-15T10:00:00Z",
    "deployed_at": "2026-09-20T14:00:00Z"
}
```
