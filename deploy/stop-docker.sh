#!/bin/bash
# Stop all Docker services

echo "🛑 Stopping Rubik's Cube Solver services..."

docker-compose down

echo ""
echo "✅ All services stopped."
echo ""
echo "💡 To remove volumes as well, run: docker-compose down -v"
