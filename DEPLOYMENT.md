# 🚀 Deployment Guide

Complete guide for deploying the Rubik's Cube Solver application in production.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Environment Configuration](#environment-configuration)
- [Production Deployment](#production-deployment)
- [Cloud Providers](#cloud-providers)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Git** for cloning the repository
- At least 2GB RAM and 5GB disk space

### For Development

- **Python** 3.11+
- **Node.js** 20+
- **npm** or **yarn**

## Docker Deployment

The easiest way to deploy is using Docker. Everything is containerized and ready to go.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jitesh523/rubiks.git
cd rubiks

# Start with Docker Compose
./deploy/start-docker.sh
```

That's it! The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Manual Docker Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Rebuild from Scratch

```bash
./deploy/rebuild.sh
```

## Environment Configuration

### Creating Your `.env` File

Copy the example file and customize:

```bash
cp .env.example .env
```

### Environment Variables

#### API Configuration

```env
API_HOST=0.0.0.0          # Host to bind to
API_PORT=8000             # Port for the API
API_RELOAD=false          # Auto-reload on code changes (dev only)
```

#### CORS Configuration

```env
# Comma-separated list of allowed origins
CORS_ORIGINS=http://localhost:3000,http://yourdomain.com
```

#### ML Model Configuration

```env
ML_MODEL_PATH=./ml_color_model.pkl
ML_SCALER_PATH=./ml_color_scaler.pkl
ML_METADATA_PATH=./ml_color_metadata.json
ML_CONFIDENCE_THRESHOLD=0.7
```

#### Logging

```env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Production Deployment

### Security Checklist

- [ ] Set specific CORS origins (not `*`)
- [ ] Use strong secrets and API keys
- [ ] Enable HTTPS/TLS
- [ ] Configure firewalls properly
- [ ] Keep dependencies updated
- [ ] Review and limit exposed ports

### Production Environment Setup

1. **Update Environment Variables**:
   ```env
   API_RELOAD=false
   LOG_LEVEL=WARNING
   CORS_ORIGINS=https://yourdomain.com
   ```

2. **Build Production Images**:
   ```bash
   docker-compose build --no-cache
   ```

3. **Start Services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify Health**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:3000/health
   ```

### Using a Reverse Proxy (Recommended)

For production, use Nginx or Traefik as a reverse proxy with SSL/TLS.

Example Nginx configuration:

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Cloud Providers

### Deploy to DigitalOcean

1. **Create a Droplet** (Ubuntu 22.04, 2GB RAM minimum)

2. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

3. **Clone and Deploy**:
   ```bash
   git clone https://github.com/jitesh523/rubiks.git
   cd rubiks
   cp .env.example .env
   # Edit .env with your domain
   ./deploy/start-docker.sh
   ```

4. **Configure Firewall**:
   ```bash
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw allow 22
   sudo ufw enable
   ```

### Deploy to AWS EC2

1. **Launch EC2 Instance** (t2.small or larger, Ubuntu 22.04)

2. **SSH into instance** and install Docker:
   ```bash
   sudo yum update -y
   sudo yum install docker -y
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```

3. **Install Docker Compose**:
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

4. **Deploy Application**:
   ```bash
   git clone https://github.com/jitesh523/rubiks.git
   cd rubiks
   ./deploy/start-docker.sh
   ```

### Deploy to Heroku

Heroku supports Docker deployments via `heroku.yml`:

```yaml
build:
  docker:
    web: frontend/Dockerfile
    api: Dockerfile
run:
  web: /usr/sbin/nginx -g "daemon off;"
  api: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

```bash
heroku create rubiks-solver
heroku stack:set container
git push heroku main
```

### Deploy to Google Cloud Run

```bash
# Build and push images
gcloud builds submit --tag gcr.io/PROJECT_ID/rubiks-backend
gcloud builds submit --tag gcr.io/PROJECT_ID/rubiks-frontend frontend/

# Deploy services
gcloud run deploy rubiks-backend \
  --image gcr.io/PROJECT_ID/rubiks-backend \
  --platform managed \
  --allow-unauthenticated

gcloud run deploy rubiks-frontend \
  --image gcr.io/PROJECT_ID/rubiks-frontend \
  --platform managed \
  --allow-unauthenticated
```

## Troubleshooting

### Port Already in Use

If ports 3000 or 8000 are already in use:

```bash
# Stop conflicting services
docker-compose down

# Or change ports in docker-compose.yml
ports:
  - "8080:8000"  # Change 8080 to any available port
```

### Container Not Starting

Check logs:
```bash
docker-compose logs backend
docker-compose logs frontend
```

Common issues:
- Missing ML model files (train the model first)
- Port conflicts
- Insufficient memory

### API Not Accessible

1. Check if backend is running:
   ```bash
   docker ps
   curl http://localhost:8000/health
   ```

2. Check CORS settings in `.env`:
   ```env
   CORS_ORIGINS=http://localhost:3000
   ```

3. Restart services:
   ```bash
   docker-compose restart
   ```

### Frontend Not Loading

1. Check if Nginx is serving files:
   ```bash
   docker exec -it rubiks-frontend ls /usr/share/nginx/html
   ```

2. Check build logs:
   ```bash
   docker-compose logs frontend
   ```

### ML Model Not Found

Train the model before deployment:

```bash
# On the host machine
python auto_calibrator.py

# Models will be created:
# - ml_color_model.pkl
# - ml_color_scaler.pkl
# - ml_color_metadata.json
```

## Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000/health

# ML model status
curl http://localhost:8000/api/v1/ml/info
```

### Resource Usage

```bash
# Container stats
docker stats

# Disk usage
docker system df
```

## Scaling

### Horizontal Scaling

Use Docker Swarm or Kubernetes for horizontal scaling:

```bash
# Docker Swarm example
docker swarm init
docker stack deploy -c docker-compose.yml rubiks
docker service scale rubiks_backend=3
```

### Vertical Scaling

Allocate more resources in `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Backup and Restore

### Backup ML Models

```bash
# Create backup directory
mkdir -p backups

# Backup models
cp ml_color_model.pkl backups/
cp ml_color_scaler.pkl backups/
cp ml_color_metadata.json backups/
```

### Restore ML Models

```bash
# Restore from backup
cp backups/ml_color_model.pkl .
cp backups/ml_color_scaler.pkl .
cp backups/ml_color_metadata.json .

# Restart backend
docker-compose restart backend
```

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/jitesh523/rubiks/issues
- **Documentation**: See README.md
- **API Docs**: http://localhost:8000/docs (when running)

---

**Happy Deploying! 🚀**
