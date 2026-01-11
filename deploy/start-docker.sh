#!/bin/bash
# Start all services with Docker Compose

echo "🚀 Starting Rubik's Cube Solver with Docker..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Build and start services
echo "📦 Building Docker images..."
docker-compose build

echo ""
echo "🎬 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 5

# Check service health
echo ""
echo "🏥 Health Check:"
echo "  Backend:  $(curl -s http://localhost:8000/health || echo '❌ Not responding')"
echo "  Frontend: $(curl -s http://localhost:3000/health || echo '❌ Not responding')"

echo ""
echo "✅ Services started successfully!"
echo ""
echo "📍 Access the application:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "📋 View logs with: docker-compose logs -f"
echo "🛑 Stop services with: docker-compose down"
