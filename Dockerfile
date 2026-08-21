# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_URL=sqlite:////app/data/claims.db \
    FRAUD_API_URL=http://127.0.0.1:8000 \
    GRADIO_SHARE=true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application files (includes fraud_detection_model.pkl and
# synthetic_health_claims.csv, both required at runtime)
COPY . .

# Create directory for database persistence, and make the startup script executable
RUN mkdir -p /app/data && chmod +x start.sh

# Expose ports
# 8000 for FastAPI
# 7860 for Gradio UI
EXPOSE 8000 7860

# Health check -- verifies both services are actually responding, not just
# that the container process is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:7860/ || exit 1

# Runs both FastAPI and Gradio in this one container -- see start.sh
CMD ["./start.sh"]