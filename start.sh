#!/bin/bash

# Rubik's Cube Solver Launcher
# This script handles environment setup and launching the application

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "🎲 Initializing Rubik's Cube Solver..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in your PATH."
    exit 1
fi

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if ! pip freeze | grep -q "kociemba"; then
    echo "⬇️  Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "🚀 Starting application..."
python main.py
