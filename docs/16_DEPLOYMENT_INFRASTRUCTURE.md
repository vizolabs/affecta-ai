# 16 — Deployment Infrastructure

```
Status: IMPL-DEPENDENT
Version: V1.0
```

---

## 1. Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/affecta
      - JWT_SECRET=dev-secret
    
  db:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=affecta
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## 2. Production Options

### Option A: Vercel + Railway (Recommended)

```
Frontend → Vercel (Next.js)
Backend → Railway (FastAPI)
Database → Railway PostgreSQL
```

### Option B: Docker Compose

```
Frontend → Nginx container
Backend → Uvicorn container
Database → PostgreSQL container
```

### Option C: Cloud (AWS/GCP)

```
Frontend → S3 + CloudFront
Backend → EC2/Compute Engine
Database → RDS/Cloud SQL
```

---

## 3. Environment Variables

### Backend

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/affecta

# Authentication
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# ML Models
MODEL_PATH=/models
EXPRESSION_MODEL=affecta-expression-v1.pt
AU_MODEL=affecta-au-v1.pt

# CORS
CORS_ORIGINS=http://localhost:3000,https://your-domain.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
```

### Frontend

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## 4. CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          cd backend && pip install -r requirements.txt
          pytest tests/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        run: railway up
```

---

## 5. Monitoring

### Health Check Endpoint

```python
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": check_db_connection(),
        "ml_models": check_models_loaded(),
        "version": "1.0.0"
    }
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 6. Model Deployment

### Model Files

```
models/
├── affecta-expression-v1.pt      # Expression classifier
├── affecta-au-v1.pt              # AU detector
├── lrp_config.json               # LRP configuration
└── model_registry.json           # Model metadata
```

### Loading

```python
class ModelManager:
    def __init__(self, model_path):
        self.expression_model = self.load_model(f"{model_path}/expression.pt")
        self.au_model = self.load_model(f"{model_path}/au.pt")
    
    def load_model(self, path):
        model = torch.load(path)
        model.eval()
        return model
```
