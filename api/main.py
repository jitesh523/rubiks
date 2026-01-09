"""
FastAPI application for Rubik's Cube Solver

REST API providing cube solving, ML color detection, and camera streaming.
"""

import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import ml_model, solver

# Create FastAPI application
app = FastAPI(
    title="Rubik's Cube Solver API",
    description="AI-powered Rubik's Cube solver with ML color detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(solver.router, prefix="/api/v1/solver", tags=["Solver"])
app.include_router(ml_model.router, prefix="/api/v1/ml", tags=["ML Model"])


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "name": "Rubik's Cube Solver API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    from api.services.ml_service import ml_service

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_model_loaded": ml_service.is_model_available(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
