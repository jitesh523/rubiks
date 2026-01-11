#!/bin/bash
# Rebuild Docker images and restart services

echo "🔄 Rebuilding Rubik's Cube Solver..."
echo ""

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose down

echo ""
echo "🗑️  Removing old images..."
docker-compose rm -f

echo ""
echo "📦 Building fresh images (no cache)..."
docker-compose build --no-cache

echo ""
echo "🎬 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 5

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "📍 Access the application:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
