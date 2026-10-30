# Production Deployment & Containerization Guide

This document specifies the deployment architecture, Docker multi-stage containerization, Docker Compose service orchestration, environment variables, and health check monitoring.

---

## 1. Containerization Architecture (Multi-Stage Dockerfile)

The application uses an optimized multi-stage build that compiles the React 18 frontend and bundles it directly into the production FastAPI ASGI container, eliminating separate web-server proxy overhead for monolithic deployments.

```
+-------------------------------------------------------------+
| Stage 1: Frontend Build (node:18-alpine)                   |
| 1. Install dependencies via npm ci                          |
| 2. Compile React/Vite assets: npm run build                 |
| 3. Output: /frontend/dist                                   |
+-------------------------------------------------------------+
                               | COPY --from=frontend-builder
+-------------------------------------------------------------+
| Stage 2: Runtime Production (python:3.11-slim)              |
| 1. System packages: libgl1, libglib2.0, curl                |
| 2. Python virtualenv & dependencies from requirements.txt   |
| 3. Non-root user: appuser (UID 10001)                       |
| 4. Static asset mount: /app/frontend/dist                   |
| 5. Health probe: curl -f http://localhost:8000/health/live  |
| 6. Entrypoint: uvicorn backend.main:app --host 0.0.0.0      |
+-------------------------------------------------------------+
```

---

## 2. Docker Compose Orchestration

The `docker-compose.yml` orchestrates the backend web service and an optional production PostgreSQL database:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: dr_screening_app
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite+aiosqlite:///./data/dr_screening.db
      - SECRET_KEY=production-secure-random-32-byte-hex
      - ACCESS_TOKEN_EXPIRE_MINUTES=1440
      - ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
      - NO_ALBUMENTATIONS_UPDATE=1
    volumes:
      - uploads_data:/app/uploads
      - reports_data:/app/reports
      - db_data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  uploads_data:
  reports_data:
  db_data:
```

---

## 3. Environment Variable Configuration

| Variable | Required | Default Value | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | `sqlite+aiosqlite:///./dr_screening.db` | SQLAlchemy async connection string (PostgreSQL or SQLite) |
| `SECRET_KEY` | Yes | `super-secret-key-dr-screening-2026` | 256-bit symmetric encryption key for signing JWT tokens |
| `ALLOWED_ORIGINS` | No | `*` | Comma-separated list of allowed CORS origins |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `1440` (24 hours) | Token validity lifespan |
| `NO_ALBUMENTATIONS_UPDATE` | No | `1` | Suppresses Albumentations outbound update checks |
| `PORT` | No | `8000` | Port on which Uvicorn binds |

---

## 4. Operational Health Checks & Liveness Probes

Kubernetes and container runtimes utilize two standardized health endpoints:
- **Liveness Probe (`GET /health/live`):** Returns HTTP 200 `{"status": "alive"}` to verify process availability without querying external dependencies.
- **Readiness Probe (`GET /health/ready`):** Returns HTTP 200 only when the database is reachable and the CNN model weights are loaded into memory (`{"status": "ready", "model_loaded": true, "database": "connected"}`).
