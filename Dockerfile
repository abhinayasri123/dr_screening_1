# Multi-stage Dockerfile for DR Screening System
# 1. Frontend Build Stage
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy package manifests first for optimal Docker layer caching
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source code and compile production assets
COPY frontend/ ./
RUN npm run build

# 2. Python Backend & Production Runtime Stage
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure all application and storage directories exist prior to copying
RUN mkdir -p /app/checkpoints /app/models /app/outputs /app/uploads /app/reports /app/data

# Copy backend application code and configurations
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/
COPY outputs/ ./outputs/

# Copy compiled frontend from builder stage into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (Render sets $PORT dynamically)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Start unified FastAPI server with shell expansion for PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
