# Deployment and Hosting Guide

### Official Project Title
**Early Disease Risk Prediction from Electronic Health Records via Convolutional Neural Network-based Radiographic Classification, Applied to Diabetic Retinopathy Screening in Ophthalmology Clinics — Enhanced via Federated Learning for Privacy-preserving Multi-hospital Training, Validated on Clinician-in-the-loop Web App for Diagnosis Review**

---

## 1. Live Public Hosting Link (Active Now)

The project is currently hosted and live on Cloudflare's global edge network:

- **Primary Cloudflare Live Link:** [https://groove-members-auckland-baghdad.trycloudflare.com](https://groove-members-auckland-baghdad.trycloudflare.com)
- **Interactive Swagger API Docs:** [https://groove-members-auckland-baghdad.trycloudflare.com/docs](https://groove-members-auckland-baghdad.trycloudflare.com/docs)
- **Local Application Link:** [http://localhost:8000](http://localhost:8000)

### Default Authentication Credentials:
- **Email:** `admin@dr-screening.com`
- **Password:** `admin123`
- **Role:** `Clinician / Administrator`

---

## 2. Deploying to Cloud Platforms (Permanent Free Hosting)

### Option A: Render (Free Web Service)
1. Push this repository to GitHub.
2. Log in to [render.com](https://render.com) and click **New +** &rarr; **Web Service**.
3. Select your GitHub repository.
4. Render will automatically detect `render.yaml` or `Dockerfile`.
   - **Environment:** Docker
   - **Health Check Path:** `/api/v1/health`
5. Click **Deploy Web Service**.
6. Render will build both the React frontend and FastAPI backend into a single container and assign a permanent URL (e.g., `https://dr-screening.onrender.com`).

---

### Option B: Hugging Face Spaces (Free Docker Space)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a **New Space**.
2. Select **Docker** as the Space SDK.
3. Push this project code to the Hugging Face repository.
4. The space will automatically build the `Dockerfile` and serve the full application at:
   `https://huggingface.co/spaces/<your-username>/<your-space-name>`

---

### Option C: Docker Local / VPS Deployment
To run anywhere using Docker:
```bash
# Build and run with docker-compose
docker compose up -d --build

# Open the browser at:
# http://localhost:8000
```

---

## 3. Running Locally / Starting New Public Tunnels

### Step 1: Start Backend & Frontend Unified Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Step 2: Open an Instant Public Tunnel (No Signup Required)
```bash
ssh -o StrictHostKeyChecking=no -R 80:127.0.0.1:8000 nokey@localhost.run
```
Copy the generated `https://<subdomain>.lhr.life` link from the console.
