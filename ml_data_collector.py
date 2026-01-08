"""
ML-Based Training Data Collector for Rubik's Cube

Enhanced training data collection tool with:
- Real-time camera interface
- Live ML predictions (if model exists)
- Confidence score display
- JSON export format
- Integration with MLColorDetector
"""

import json
from datetime import datetime

import cv2
import numpy as np

from ml_color_detector import MLColorDetector


class MLDataCollector:
    """Interactive tool for collecting ML training data with live predictions."""

    def __init__(self):
        """Initialize the data collector."""
        self.detector = MLColorDetector()
        self.samples: list[dict] = []
        self.color_names = ["white", "red", "green", "blue", "yellow", "orange"]

        # Try to load existing model for live predictions
        self.model_loaded = self.detector.load_model()

        # Keyboard mappings
        self.label_map = {
            ord("1"): "white",
            ord("2"): "red",
            ord("3"): "green",
            ord("4"): "blue",
            ord("5"): "yellow",
            ord("6"): "orange",
        }

    def get_center_rgb(self, frame: np.ndarray) -> tuple[int, int, int]:
        """Extract RGB color from center of frame."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Sample 5x5 region around center for stability
        region = frame[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3]

        # Convert BGR to RGB and get average
        mean_color = region.mean(axis=(0, 1))
        b, g, r = mean_color
        return int(r), int(g), int(b)

    def display_status(self) -> str:
        """Generate status text showing sample counts."""
        counts = {}
        for color in self.color_names:
            counts[color] = sum(1 for s in self.samples if s["color"] == color)

        status_parts = [f"{color[:3].upper()}:{counts[color]}" for color in self.color_names]
        return " | ".join(status_parts)

    def add_sample(self, rgb: tuple[int, int, int], label: str) -> None:
        """Add a labeled sample to the training data."""
        self.samples.append({"rgb": list(rgb), "color": label})
        print(f"✅ Added {label}: RGB{rgb} (Total samples: {len(self.samples)})")

    def collect_samples(self) -> None:
        """Interactive camera-based sample collection."""
        print("🎨 ML Training Data Collector")
        print("=" * 60)
        print("Instructions:")
        print("  • Point center crosshair at a cube sticker")
        print("  • Press 1-6 to label the current color")
        print("    1:White  2:Red  3:Green  4:Blue  5:Yellow  6:Orange")
        print("  • Press 's' to save samples")
        print("  • Press 't' to train model with collected data")
        print("  • Press 'c' to clear all samples")
        print("  • Press 'q' to quit")
        print("=" * 60)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break

            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2

            # Get current RGB at center
            rgb = self.get_center_rgb(frame)

            # Draw crosshair
            cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), 3)
            cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 0), 2)
            cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 0), 2)

            # Display RGB values
            cv2.putText(
                frame, f"RGB: {rgb}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            # Show ML prediction if model loaded
            y_offset = 70
            if self.model_loaded and self.detector.is_trained:
                try:
                    predicted_color, confidence = self.detector.predict_color(rgb)

                    # Color code by confidence
                    color = (0, 255, 0) if confidence >= 0.7 else (0, 165, 255)

                    cv2.putText(
                        frame,
                        f"ML Prediction: {predicted_color.upper()} ({confidence:.1%})",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                    y_offset += 40
                except Exception:
                    cv2.putText(
                        frame,
                        "ML: Not ready",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (128, 128, 128),
                        1,
                    )
                    y_offset += 40
            else:
                cv2.putText(
                    frame,
                    "ML: No model loaded",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (128, 128, 128),
                    1,
                )
                y_offset += 40

            # Display instructions
            cv2.putText(
                frame,
                "Press 1-6 to label color",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            y_offset += 30

            cv2.putText(
                frame,
                "1:W 2:R 3:G 4:B 5:Y 6:O | s:Save t:Train q:Quit",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            # Display sample counts
            status_text = self.display_status()
            cv2.putText(
                frame,
                f"Samples: {status_text}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow("ML Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in self.label_map:
                # Add labeled sample
                label = self.label_map[key]
                self.add_sample(rgb, label)

            elif key == ord("s"):
                # Save samples
                self.export_data()

            elif key == ord("t"):
                # Train model with collected data
                if len(self.samples) >= 30:  # Minimum samples
                    self.train_and_test()
                else:
                    print(f"⚠️  Need at least 30 samples (current: {len(self.samples)})")

            elif key == ord("c"):
                # Clear samples
                self.samples = []
                print("🗑️  Cleared all samples")

            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Data collection finished!")

    def export_data(self, filename: str | None = None) -> None:
        """Export samples in JSON format compatible with MLColorDetector."""
        if not self.samples:
            print("⚠️  No samples to export")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_training_data_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(self.samples),
            "samples": self.samples,
            "color_counts": {
                color: sum(1 for s in self.samples if s["color"] == color)
                for color in self.color_names
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Exported {len(self.samples)} samples to {filename}")

        # Show summary
        print("\n📊 Training Data Summary:")
        for color in self.color_names:
            count = data["color_counts"][color]
            status = "✅ Good" if count >= 10 else "⚠️  Few" if count >= 5 else "❌ Too few"
            print(f"  {color.capitalize():8}: {count:3d} samples {status}")

    def load_data(self, filename: str) -> bool:
        """Load training data from JSON file."""
        try:
            with open(filename) as f:
                data = json.load(f)

            self.samples = data.get("samples", [])
            print(f"✅ Loaded {len(self.samples)} samples from {filename}")
            return True

        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON file: {filename}")
            return False

    def train_and_test(self) -> dict | None:
        """Train model with collected data and show results."""
        if len(self.samples) == 0:
            print("❌ No training data available")
            return None

        print(f"\n🧠 Training ML model with {len(self.samples)} samples...")

        try:
            metrics = self.detector.train(self.samples)

            print("✅ Training complete!")
            print(f"   Accuracy: {metrics['accuracy']:.2%}")
            print(f"   Training samples: {metrics['n_train']}")
            print(f"   Test samples: {metrics['n_test']}")

            # Save the trained model
            self.detector.save_model()
            self.model_loaded = True

            print("\n📊 Classification Report:")
            print(metrics["classification_report"])

            return metrics

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None


def main():
    """Main entry point for the data collector."""
    collector = MLDataCollector()

    if collector.model_loaded:
        print("✅ Existing ML model loaded - Live predictions enabled!")
    else:
        print("ℹ️  No existing model - Collect data and press 't' to train")

    print()
    collector.collect_samples()

    # Offer to save if there's unsaved data
    if collector.samples:
        response = input("\n💾 Save collected samples? (y/n): ")
        if response.lower() == "y":
            collector.export_data()


if __name__ == "__main__":
    main()
