# Docker Setup Guide

## Overview
This application can be containerized using Docker. Two files are provided:
- **Dockerfile**: Build a Docker image for the application
- **docker-compose.yml**: Run the entire stack with multiple services

## Prerequisites
- Docker installed (https://www.docker.com/products/docker-desktop)
- Docker Compose installed (comes with Docker Desktop)

## Quick Start

### Run Both Backend & Frontend Together (Recommended):
```bash
docker-compose up
```

Access:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:7860
- **API Docs**: http://localhost:8000/docs

### Run Backend Only:
```bash
docker-compose up api
```

### Docker Compose Commands

```bash
# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build

# Remove all volumes (reset database)
docker-compose down -v
```

## Docker Commands (Raw Docker)

### Build Image
```bash
docker build -t fraud-detection-api .
```

### Run Container
```bash
# Basic run
docker run -p 8000:8000 fraud-detection-api

# Run with volume for persistent database
docker run -p 8000:8000 -v $(pwd)/claims.db:/app/claims.db fraud-detection-api

# Run with environment variables
docker run -p 8000:8000 -e DATABASE_URL=sqlite:///./claims.db fraud-detection-api
```

### Docker Compose

```bash
# Start services in background
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Remove all volumes
docker-compose down -v
```

## Environment Variables

Both backend and frontend services use these environment variables:

```env
DATABASE_URL=sqlite:///./claims.db
FRAUD_API_URL=http://api:8000  # URL for Gradio to connect to backend API
```

**Note**: Inside Docker network, services use `http://api:8000`. From your local machine, use `http://localhost:8000`.

## Ports

- **8000**: FastAPI backend (REST API)
- **7860**: Gradio frontend (Web UI)

## Persistence

Database file (`claims.db`) is mounted as a volume to persist data between container runs:

```bash
docker run -p 8000:8000 -v $(pwd)/claims.db:/app/claims.db fraud-detection-api
```

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Create a Claim
```bash
curl -X POST http://localhost:8000/claim \
  -H "Content-Type: application/json" \
  -d '{
    "claim_date": "2024-08-15",
    "service_date": "2024-08-10",
    "policy_expiration_date": "2025-12-31",
    "claim_amount": 1500.50,
    "patient_age": 45,
    "patient_gender": "M",
    "patient_city": "New York",
    "patient_state": "NY",
    "provider_type": "Hospital",
    "provider_specialty": "Cardiology",
    "provider_city": "New York",
    "provider_state": "NY",
    "diagnosis_code": "I10",
    "procedure_code": "99213",
    "number_of_procedures": 2,
    "admission_type": "Emergency",
    "discharge_type": "Home",
    "length_of_stay_days": 3,
    "service_type": "Inpatient",
    "deductible_amount": 250.00,
    "copay_amount": 30.00,
    "num_previous_claims_patient": 5,
    "num_previous_claims_provider": 45,
    "provider_patient_distance_miles": 2.5,
    "claim_submitted_late": false
  }'
```

### View API Documentation
Open http://localhost:8000/docs in your browser

## Troubleshooting

### Container won't start
```bash
docker logs <container_id>
```

### Port already in use
```bash
# Change port in docker run or docker-compose.yml
docker run -p 8001:8000 fraud-detection-api
```

### Database connection issues
Ensure the database volume is correctly mounted and has write permissions.

## Deployment

For production deployment, consider:
1. Using a production ASGI server (already using uvicorn)
2. Adding SSL/TLS certificates
3. Using environment-specific configurations
4. Setting up logging and monitoring
5. Using a managed database service instead of SQLite
