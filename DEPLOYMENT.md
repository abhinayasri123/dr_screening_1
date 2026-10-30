# Deployment & Cloud Hosting Guide

**Diabetic Retinopathy Clinical Decision Support System**

---

## 1. Local One-Click Startup (Already Running)

The application is running as a single unified server serving both the React frontend and FastAPI backend:
- **Link:** **[http://localhost:8000](http://localhost:8000)**
- **Credentials:** `admin@dr-screening.com` / `admin123`

To restart the server at any time:
```bash
cd dr_screening
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 2. Docker & Containerized Deployment

A production container definition is ready. Build and run with:

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run Container
```bash
docker build -t dr-screening-system .
docker run -p 8000:8000 dr-screening-system
```

---

## 3. Free Cloud Hosting Deployment Options

### Option A: Render.com (Recommended - 1 Click)
1. Fork or push this repository to GitHub.
2. Sign in to [Render.com](https://render.com) -> New **Web Service**.
3. Select your GitHub repository.
4. Settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Render generates a public HTTPS link: `https://dr-screening.onrender.com`

---

### Option B: Railway.app
1. Push code to GitHub.
2. Link your repo on [Railway.app](https://railway.app).
3. Railway auto-detects `requirements.txt` and uses:
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}`
4. Generates a live URL: `https://dr-screening.up.railway.app`

---

### Option C: Hugging Face Spaces (Ideal for ML / Research Demo)
1. Create a Space on [huggingface.co/spaces](https://huggingface.co/spaces) with SDK: **Docker**.
2. Push this project folder.
3. Hugging Face deploys the app with a public research link: `https://huggingface.co/spaces/<your-username>/dr-screening`
