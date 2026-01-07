"""
ML-based Color Detection for Rubik's Cube Scanner

Uses scikit-learn KNN classifier with LAB+HSV+RGB ratio features
for robust color detection across different lighting conditions.
"""

import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
from datetime import datetime


class MLColorDetector:
    """Machine learning based color detector using KNN classification."""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize ML color detector.

        Args:
            confidence_threshold: Minimum confidence for ML prediction (0-1)
        """
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.scaler = StandardScaler()
        self.confidence_threshold = confidence_threshold
        self.is_trained = False
        self.color_names = ["white", "red", "green", "yellow", "orange", "blue"]
        self.model_metadata: Dict = {}

    def extract_features(self, rgb_color: Tuple[int, int, int]) -> np.ndarray:
        """
        Extract color features for classification.

        Combines LAB, HSV, and RGB ratio features for robust classification.

        Args:
            rgb_color: RGB color tuple (0-255 range)

        Returns:
            Feature vector (9 features: LAB + HSV + RGB ratios)
        """
        r, g, b = rgb_color

        # Convert to numpy array for OpenCV
        rgb_array = np.uint8([[[b, g, r]]])  # OpenCV uses BGR

        # LAB color space (perceptually uniform)
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2LAB)[0][0]
        l, a, b_lab = lab

        # HSV color space
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv

        # RGB ratios (normalized)
        total = r + g + b + 1  # Avoid division by zero
        r_ratio = r / total
        g_ratio = g / total
        b_ratio = b / total

        # Combine all features (9 features total)
        features = np.array(
            [
                l,
                a,
                b_lab,  # LAB (3)
                h,
                s,
                v,  # HSV (3)
                r_ratio,
                g_ratio,
                b_ratio,  # RGB ratios (3)
            ]
        )

        return features

    def train(
        self, training_data: List[Dict], test_size: float = 0.2, random_state: int = 42
    ) -> Dict:
        """
        Train the classifier with labeled color samples.

        Args:
            training_data: List of dicts with 'rgb' and 'color' keys
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training metrics
        """
        if len(training_data) == 0:
            raise ValueError("Training data is empty")

        # Extract features and labels
        X = []
        y = []

        for sample in training_data:
            rgb = tuple(sample["rgb"])
            color = sample["color"]

            features = self.extract_features(rgb)
            X.append(features)
            y.append(color)

        X = np.array(X)
        y = np.array(y)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train classifier
        self.knn_classifier.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.knn_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Store metadata
        self.model_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(training_data),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": float(accuracy),
            "color_names": list(self.color_names),
        }

        # Return detailed metrics
        metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=self.color_names
            ).tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        return metrics

    def predict_color(
        self, rgb_color: Tuple[int, int, int]
    ) -> Tuple[str, float]:
        """
        Predict color with confidence score.

        Args:
            rgb_color: RGB color tuple

        Returns:
            Tuple of (predicted_color, confidence_score)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Extract and scale features
        features = self.extract_features(rgb_color).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Get prediction and probabilities
        prediction = self.knn_classifier.predict(features_scaled)[0]
        probabilities = self.knn_classifier.predict_proba(features_scaled)[0]

        # Confidence is the max probability
        confidence = float(np.max(probabilities))

        return prediction, confidence

    def predict_with_fallback(
        self,
        rgb_color: Tuple[int, int, int],
        hsv_detector_func=None,
    ) -> Tuple[str, float, bool]:
        """
        Predict color with optional HSV fallback.

        Args:
            rgb_color: RGB color tuple
            hsv_detector_func: Optional HSV detection function for fallback

        Returns:
            Tuple of (color, confidence, used_ml)
        """
        if not self.is_trained:
            if hsv_detector_func:
                return hsv_detector_func(rgb_color), 0.5, False
            raise RuntimeError("Model not trained and no fallback provided")

        color, confidence = self.predict_color(rgb_color)

        # Use ML if confidence is high enough
        if confidence >= self.confidence_threshold:
            return color, confidence, True

        # Fall back to HSV if available
        if hsv_detector_func:
            hsv_color = hsv_detector_func(rgb_color)
            return hsv_color, confidence, False

        # Return ML prediction even with low confidence
        return color, confidence, True

    def save_model(self, model_dir: str = ".") -> None:
        """
        Save trained model to disk.

        Args:
            model_dir: Directory to save model files
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save classifier
        with open(model_path / "ml_color_model.pkl", "wb") as f:
            pickle.dump(self.knn_classifier, f)

        # Save scaler
        with open(model_path / "ml_color_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            **self.model_metadata,
            "confidence_threshold": self.confidence_threshold,
            "color_names": self.color_names,
        }

        with open(model_path / "ml_color_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model saved to {model_path}")

    def load_model(self, model_dir: str = ".") -> bool:
        """
        Load trained model from disk.

        Args:
            model_dir: Directory containing model files

        Returns:
            True if successful, False otherwise
        """
        model_path = Path(model_dir)

        try:
            # Load classifier
            with open(model_path / "ml_color_model.pkl", "rb") as f:
                self.knn_classifier = pickle.load(f)

            # Load scaler
            with open(model_path / "ml_color_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            # Load metadata
            with open(model_path / "ml_color_metadata.json", "r") as f:
                self.model_metadata = json.load(f)

            self.is_trained = True
            print(f"✅ Model loaded from {model_path}")
            print(f"   Accuracy: {self.model_metadata.get('accuracy', 'N/A'):.2%}")
            return True

        except FileNotFoundError as e:
            print(f"⚠️  Model files not found: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        return {
            "is_trained": self.is_trained,
            "confidence_threshold": self.confidence_threshold,
            "metadata": self.model_metadata,
        }


def demo_ml_detector():
    """Demo the ML color detector with sample data."""
    print("🎨 ML Color Detector Demo")
    print("=" * 50)

    # Create sample training data
    training_data = []

    # White samples
    for _ in range(10):
        training_data.append(
            {"rgb": [245 + np.random.randint(-10, 10)] * 3, "color": "white"}
        )

    # Red samples
    for _ in range(10):
        training_data.append(
            {
                "rgb": [200 + np.random.randint(-20, 20), 30, 30],
                "color": "red",
            }
        )

    # Green samples
    for _ in range(10):
        training_data.append(
            {
                "rgb": [30, 180 + np.random.randint(-20, 20), 30],
                "color": "green",
            }
        )

    # Blue samples
    for _ in range(10):
        training_data.append(
            {
                "rgb": [30, 30, 200 + np.random.randint(-20, 20)],
                "color": "blue",
            }
        )

    # Yellow samples
    for _ in range(10):
        training_data.append(
            {
                "rgb": [220 + np.random.randint(-20, 20), 220, 30],
                "color": "yellow",
            }
        )

    # Orange samples
    for _ in range(10):
        training_data.append(
            {
                "rgb": [220 + np.random.randint(-20, 20), 100, 30],
                "color": "orange",
            }
        )

    # Train detector
    detector = MLColorDetector(confidence_threshold=0.7)
    print(f"\n📊 Training with {len(training_data)} samples...")

    metrics = detector.train(training_data)
    print(f"✅ Training complete!")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Train samples: {metrics['n_train']}")
    print(f"   Test samples: {metrics['n_test']}")

    # Test predictions
    print("\n🧪 Testing predictions:")
    test_colors = {
        "white": (245, 245, 245),
        "red": (200, 30, 30),
        "green": (30, 180, 30),
        "blue": (30, 30, 200),
        "yellow": (220, 220, 30),
        "orange": (220, 100, 30),
    }

    for expected, rgb in test_colors.items():
        predicted, confidence = detector.predict_color(rgb)
        check = "✓" if predicted == expected else "✗"
        print(
            f"  {check} {expected:6s} -> {predicted:6s} (confidence: {confidence:.2f})"
        )

    # Save model
    print("\n💾 Saving model...")
    detector.save_model()
    print("   Files: ml_color_model.pkl, ml_color_scaler.pkl, ml_color_metadata.json")


if __name__ == "__main__":
    demo_ml_detector()
