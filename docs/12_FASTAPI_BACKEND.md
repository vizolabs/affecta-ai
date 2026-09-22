# 12 — FastAPI Backend

```
Status: LOCKED
Version: V1.0
```

---

## 1. Project Structure

```
backend/
├── main.py                 # FastAPI app
├── core/
│   ├── config.py           # Settings
│   ├── security.py         # Auth utilities
│   └── database.py         # DB connection
├── api/
│   ├── auth.py             # Authentication endpoints
│   ├── sessions.py         # Session management
│   ├── analytics.py        # Analytics endpoints
│   ├── settings.py         # User settings
│   └── websocket.py        # WebSocket handler
├── models/
│   ├── user.py             # User model
│   ├── session.py          # Session model
│   └── event.py            # Event model
├── schemas/
│   ├── auth.py             # Auth schemas
│   ├── session.py          # Session schemas
│   └── emotion.py          # Emotion schemas
├── services/
│   ├── auth_service.py     # Auth logic
│   ├── session_service.py  # Session logic
│   └── emotion_service.py  # Emotion processing
└── ml/
    ├── inference.py        # Model inference
    ├── lrp.py              # LRP engine
    └── tracker.py          # Face tracker
```

---

## 2. API Endpoints

### Authentication

```python
@router.post("/api/auth/register")
async def register(email: str, password: str, full_name: str = None):
    """Register new user."""
    pass

@router.post("/api/auth/login")
async def login(email: str, password: str):
    """Login and return token."""
    pass

@router.post("/api/auth/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Invalidate session."""
    pass

@router.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user."""
    pass
```

### Sessions

```python
@router.get("/api/sessions")
async def list_sessions(current_user: User = Depends(get_current_user)):
    """List user's sessions."""
    pass

@router.get("/api/sessions/{session_id}")
async def get_session(session_id: UUID):
    """Get session details."""
    pass

@router.get("/api/sessions/{session_id}/events")
async def get_session_events(session_id: UUID):
    """Get session events."""
    pass

@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: UUID):
    """Delete session and all data."""
    pass
```

### Analytics

```python
@router.get("/api/analytics/summary")
async def get_summary(session_id: UUID = None):
    """Get emotion summary."""
    pass

@router.get("/api/analytics/timeline")
async def get_timeline(session_id: UUID):
    """Get emotion timeline."""
    pass

@router.get("/api/analytics/export/{session_id}")
async def export_report(session_id: UUID, format: str = "pdf"):
    """Export session report."""
    pass
```

### Settings

```python
@router.get("/api/settings")
async def get_settings(current_user: User = Depends(get_current_user)):
    """Get user settings."""
    pass

@router.put("/api/settings")
async def update_settings(settings: SettingsUpdate,
                         current_user: User = Depends(get_current_user)):
    """Update user settings."""
    pass
```

### WebSocket

```python
@router.websocket("/ws/emotion-stream")
async def emotion_stream(websocket: WebSocket):
    """Real-time emotion analysis stream."""
    await websocket.accept()
    
    # Authenticate
    token = await websocket.receive_text()
    user = await verify_token(token)
    if not user:
        await websocket.close(code=4001)
        return
    
    # Process frames
    while True:
        frame_data = await websocket.receive_bytes()
        results = await process_frame(frame_data)
        await websocket.send_json(results)
```

---

## 3. Response Format

```python
class ApiResponse(BaseModel):
    status: str  # 'success' | 'error'
    data: Any = None
    meta: dict = None
    error: str = None
```

---

## 4. Error Handling

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limited |
| 500 | Internal Server Error |

---

## 5. Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@router.post("/api/auth/login")
@limiter.limit("10/minute")
async def login(...):
    pass
```

---

## 6. CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
