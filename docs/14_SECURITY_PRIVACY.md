# 14 — Security & Privacy

```
Status: LOCKED (requirements) / IMPL-DEPENDENT (mechanisms)
Version: V1.0
```

---

## 1. Authentication Requirements

### Password Security

| Requirement | Specification |
|-------------|---------------|
| Hashing | bcrypt (12 rounds) — shipped: PBKDF2-HMAC (210k iters) |
| Minimum length | 8 characters |
| Complexity | Upper, lower, number, special — shipped: min 8 chars, no complexity rule |
| Storage | Hashed only, never plaintext |

### Session Management

| Requirement | Specification |
|-------------|---------------|
| Token type | JWT or secure session cookie — shipped: custom HMAC bearer token |
| Access token lifetime | 15 minutes — shipped: 7-day TTL |
| Refresh token lifetime | 7 days |
| Rotation | On each use — shipped: no rotation (logout revokes) |
| Invalidation | On logout |

### Transport Security

| Requirement | Specification |
|-------------|---------------|
| Protocol | HTTPS (production) |
| WSS | WebSocket Secure |
| HSTS | Enabled |
| Secrets | Environment variables only |

---

## 2. Privacy Architecture

### Data Classification

| Data Type | Stored | Duration | User Control |
|-----------|--------|----------|--------------|
| Camera frames | No | Transient | N/A |
| Face embeddings | No | Never | N/A |
| Identity templates | No | Never | N/A |
| Emotion readings | Yes | Session lifetime | Can delete |
| XAI snapshots | Yes | Session lifetime | Can delete |
| User account | Yes | Until deleted | Can delete all |

### Privacy Principles

1. **No facial recognition** — System does not identify people
2. **No identity templates** — No faceprints stored
3. **Transient processing** — Camera frames not saved
4. **Minimal storage** — Only analysis metadata stored
5. **User control** — Can delete all data anytime
6. **Consent required** — Camera access requires explicit consent

---

## 3. Security Measures

### Rate Limiting

```python
# Login: 10 attempts per minute
# API-wide limiter not shipped (local-first); login 10/min + register 5/min per IP
# WebSocket: 30 connections per user
```

### CSRF Protection

```python
# SameSite cookies
# CSRF token for state-changing operations
```

### Input Validation

```python
# Validate all API inputs
# Sanitize file uploads
# Limit file sizes
# Check content types
```

### Error Handling

```python
# Never expose internal errors
# Log errors server-side
# Return generic messages to client
```

---

## 4. Privacy Dashboard

User can view:
- What data is stored
- When it was created
- Option to delete individual sessions
- Option to delete all data
- Camera usage indicator

---

## 5. Compliance Notes

### GDPR Considerations

- Right to access: User can download their data
- Right to deletion: User can delete all data
- Data minimization: Store only what's necessary
- Consent: Required before camera access

### Important Disclaimer

```
The system does not intentionally create or store face 
embeddings or identity templates. Facial imagery is 
processed transiently by default, and stored records 
are limited to analysis metadata.

Legal compliance should be verified with appropriate 
counsel for specific jurisdictions.
```

---

## 6. What We Do NOT Store

| Data Type | Status |
|-----------|--------|
| Facial images | Never stored |
| Face embeddings | Never created |
| Identity templates | Never created |
| Bounding boxes | Transient only |
| Raw video | Never stored |
| Location data | Not collected |
| Biometric identifiers | Not created |

---

## 7. Security Checklist

- [x] Passwords hashed with PBKDF2-HMAC-SHA256 (210k iterations, per-user salt) — stdlib-only by design; bcrypt/jose deliberately not vendored
- [x] Bearer-token auth implemented correctly (HMAC-SHA256 signed, exp-checked, denylist on logout) — covered by tests/test_api.py
- [x] HTTPS N/A — local-first single-machine app (localhost only); if exposed beyond localhost, terminate TLS in front of uvicorn
- [x] CORS configured properly (localhost dev origins only)
- [x] Rate limiting active (login: 10 attempts / minute / client)
- [x] Input validation complete (pydantic schemas; settings clamped e.g. ema_alpha ∈ [0.05, 1.0]; oversized passwords rejected pre-hash)
- [x] Error handling secure (no stack traces leaked; 401/404/422/503 envelopes)
- [x] Secrets in environment variables / local file (token secret in data/.token_secret, gitignored; never committed)
- [x] No facial data stored (frames processed in memory only; never persisted)
- [x] User deletion working (DELETE /api/auth/me cascades sessions/events/settings — tested)
- [x] Privacy posture surfaced in-product (ethics copy on exports, "frames never leave this device", untrained-AU notices)
