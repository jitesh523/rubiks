"""
ML Model service

Business logic for ML color detection operations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_color_detector import MLColorDetector


class MLService:
    """Service for ML model operations"""

    def __init__(self):
        self.detector = MLColorDetector()
        self.load_model_if_available()

    def load_model_if_available(self):
        """Load ML model if it exists"""
        try:
            self.detector.load_model()
        except:
            pass  # Model not available, that's okay

    def is_model_available(self) -> bool:
        """Check if ML model is trained and ready"""
        return self.detector.is_trained

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.detector.is_trained:
            metadata = self.detector.model_metadata
            return {
                "is_trained": True,
                "accuracy": metadata.get("accuracy"),
                "last_trained": metadata.get("trained_at"),
                "confidence_threshold": self.detector.confidence_threshold,
                "metadata": metadata,
            }
        else:
            return {
                "is_trained": False,
                "accuracy": None,
                "last_trained": None,
                "confidence_threshold": self.detector.confidence_threshold,
                "metadata": {},
            }

    async def predict_color(self, rgb: Tuple[int, int, int]) -> Dict[str, Any]:
        """Predict color with ML"""
        if not self.detector.is_trained:
            return {
                "color": "unknown",
                "confidence": 0.0,
                "used_ml": False,
                "error": "Model not trained",
            }

        try:
            color, confidence = self.detector.predict_color(rgb)
            return {
                "color": color,
                "confidence": confidence,
                "used_ml": True,
                "error": None,
            }
        except Exception as e:
            return {
                "color": "unknown",
                "confidence": 0.0,
                "used_ml": False,
                "error": str(e),
            }

    async def train_model(
        self, training_data: List[dict], confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Train ML model"""
        try:
            self.detector.confidence_threshold = confidence_threshold
            metrics = self.detector.train(training_data)
            self.detector.save_model()

            return {
                "accuracy": metrics["accuracy"],
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
            }
        except Exception as e:
            raise ValueError(f"Training failed: {str(e)}")


# Singleton instance
ml_service = MLService()
