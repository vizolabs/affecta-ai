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
| Hashing | bcrypt (12 rounds) |
| Minimum length | 8 characters |
| Complexity | Upper, lower, number, special |
| Storage | Hashed only, never plaintext |

### Session Management

| Requirement | Specification |
|-------------|---------------|
| Token type | JWT or secure session cookie |
| Access token lifetime | 15 minutes |
| Refresh token lifetime | 7 days |
| Rotation | On each use |
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
# API: 100 requests per minute
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

- [ ] Passwords hashed with bcrypt
- [ ] JWT/session implemented correctly
- [ ] HTTPS enforced in production
- [ ] CORS configured properly
- [ ] Rate limiting active
- [ ] Input validation complete
- [ ] Error handling secure
- [ ] Secrets in environment variables
- [ ] No facial data stored
- [ ] User deletion working
- [ ] Privacy dashboard functional
