# 11 — Database Schema (PostgreSQL)

```
Status: LOCKED
Version: V1.0
```

---

## 1. Design Principles

- Store **events**, not every frame
- Store **XAI snapshots** on-demand, not per-frame
- Minimal storage for privacy
- User can delete all data

---

## 2. Tables

### users

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_email ON users(email);
```

### sessions

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    total_frames INTEGER DEFAULT 0,
    total_events INTEGER DEFAULT 0,
    avg_confidence FLOAT,
    avg_intensity FLOAT,
    metadata JSONB
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_start ON sessions(start_time);
```

### events

```sql
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    face_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    -- Event types: 'transition', 'ambiguity', 'intensity_change',
    --             'face_enter', 'face_exit', 'confidence_drop'
    from_emotion VARCHAR(50),
    to_emotion VARCHAR(50),
    confidence FLOAT,
    intensity FLOAT,
    certainty VARCHAR(20),
    action_units JSONB,
    metadata JSONB
);

CREATE INDEX idx_events_session ON events(session_id);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp);
```

### xai_snapshots

```sql
CREATE TABLE xai_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    face_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    emotion_label VARCHAR(50),
    confidence FLOAT,
    lrp_heatmap BYTEA,  -- Compressed image
    region_relevance JSONB,
    au_snapshot JSONB,
    gradcam_heatmap BYTEA,  -- Optional
    metadata JSONB
);

CREATE INDEX idx_xai_session ON xai_snapshots(session_id);
```

### settings

```sql
CREATE TABLE settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    theme VARCHAR(20) DEFAULT 'dark',
    narrator_enabled BOOLEAN DEFAULT TRUE,
    narrator_speed FLOAT DEFAULT 1.0,
    narrator_voice VARCHAR(50) DEFAULT 'default',
    xai_frame_interval INTEGER DEFAULT 10,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### model_registry

```sql
CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    -- Types: 'expression', 'au', 'intensity', 'ranking'
    dataset_used VARCHAR(255),
    training_config JSONB,
    metrics JSONB,
    checkpoint_path VARCHAR(500),
    checksum VARCHAR(64),
    status VARCHAR(20) DEFAULT 'trained',
    -- Status: 'training', 'trained', 'validated', 'deployed', 'deprecated'
    created_at TIMESTAMP DEFAULT NOW(),
    deployed_at TIMESTAMP,
    UNIQUE(model_name, version)
);
```

---

## 3. Storage Estimates

### Per Session (1 hour, moderate activity)

| Table | Records | Size |
|-------|---------|------|
| sessions | 1 | ~1 KB |
| events | ~500-1000 | ~500 KB |
| xai_snapshots | ~50-100 | ~5 MB |
| **Total** | — | **~5.5 MB** |

### Per User (10 sessions)

| Table | Size |
|-------|------|
| User data | ~1 KB |
| Session data | ~55 MB |
| **Total** | **~55 MB** |

**Much smaller than frame-level storage (which would be ~500 MB/session)**

---

## 4. Data Retention

```sql
-- Optional: Auto-delete old sessions
DELETE FROM sessions 
WHERE start_time < NOW() - INTERVAL '90 days';

-- User-initiated: Delete all user data
DELETE FROM users WHERE id = ?;
-- (Cascades to all related tables)
```

---

## 5. Privacy Compliance

- No facial embeddings stored
- No identity templates stored
- Camera frames: transient only
- Stored data: emotion readings + XAI snapshots only
- User can delete all data via API
