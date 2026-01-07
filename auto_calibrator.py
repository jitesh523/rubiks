"""
Auto-Calibration Tool for ML Color Detection

Automatically calibrates the ML color detector by scanning a solved cube.
No manual data collection needed!
"""

import cv2
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from ml_color_detector import MLColorDetector


class AutoCalibrator:
    """Automatic calibration using a solved Rubik's cube."""

    def __init__(self, samples_per_face: int = 30):
        """
        Initialize auto-calibrator.

        Args:
            samples_per_face: Number of samples to collect per face
        """
        self.samples_per_face = samples_per_face
        self.face_order = ["U", "R", "F", "D", "L", "B"]
        self.face_colors = {
            "U": "white",
            "R": "red",
            "F": "green",
            "D": "yellow",
            "L": "orange",
            "B": "blue",
        }
        self.training_data: List[Dict] = []

    def get_center_color(self, frame: np.ndarray) -> Tuple[int, int, int]:
        """
        Extract color from the center of the frame.

        Args:
            frame: Camera frame

        Returns:
            RGB tuple of center color
        """
        h, w = frame.shape[:2]
        center_size = 50  # 50x50 pixel region

        # Extract center region
        y1 = h // 2 - center_size // 2
        y2 = h // 2 + center_size // 2
        x1 = w // 2 - center_size // 2
        x2 = w // 2 + center_size // 2

        center_region = frame[y1:y2, x1:x2]

        # Calculate average color (BGR to RGB)
        avg_bgr = center_region.mean(axis=0).mean(axis=0)
        r, g, b = int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])

        return (r, g, b)

    def draw_ui(
        self,
        frame: np.ndarray,
        current_face: str,
        samples_collected: int,
        instruction: str,
    ) -> np.ndarray:
        """Draw calibration UI on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw center square
        center_size = 100
        y1 = h // 2 - center_size // 2
        y2 = h // 2 + center_size // 2
        x1 = w // 2 - center_size // 2
        x2 = w // 2 + center_size // 2

        # Green square
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw center point
        cv2.circle(display, (w // 2, h // 2), 5, (0, 0, 255), -1)

        # Progress bar
        progress = samples_collected / self.samples_per_face
        bar_width = 300
        bar_height = 20
        bar_x = w // 2 - bar_width // 2
        bar_y = h - 80

        # Background
        cv2.rectangle(
            display,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100),
            -1,
        )

        # Progress
        filled_width = int(bar_width * progress)
        if filled_width > 0:
            cv2.rectangle(
                display,
                (bar_x, bar_y),
                (bar_x + filled_width, bar_y + bar_height),
                (0, 255, 0),
                -1,
            )

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        title = f"Calibrating: {self.face_colors[current_face].upper()} face ({current_face})"
        cv2.putText(display, title, (20, 30), font, 0.8, (255, 255, 255), 2)

        # Instruction
        cv2.putText(display, instruction, (20, h - 120), font, 0.6, (255, 255, 255), 1)

        # Progress text
        progress_text = f"{samples_collected}/{self.samples_per_face} samples"
        cv2.putText(
            display, progress_text, (bar_x, bar_y - 10), font, 0.5, (255, 255, 255), 1
        )

        return display

    def calibrate_face(
        self, cap: cv2.VideoCapture, face_name: str
    ) -> List[Tuple[int, int, int]]:
        """
        Calibrate one face of the cube.

        Args:
            cap: OpenCV video capture
            face_name: Face to calibrate (U/R/F/D/L/B)

        Returns:
            List of RGB samples
        """
        color_name = self.face_colors[face_name]
        samples = []

        print(f"\n📸 Show {color_name.upper()} ({face_name}) face to camera")
        print(f"   Collecting {self.samples_per_face} samples...")

        frame_count = 0
        sample_interval = 5  # Sample every 5 frames

        while len(samples) < self.samples_per_face:
            ret, frame = cap.read()
            if not ret:
                continue

            # Collect sample every N frames
            if frame_count % sample_interval == 0:
                rgb = self.get_center_color(frame)
                samples.append(rgb)

            # Draw UI
            instruction = "Hold cube steady - capturing automatically..."
            display = self.draw_ui(frame, face_name, len(samples), instruction)

            cv2.imshow("Auto Calibration", display)

            # Allow early exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("   ⚠️  Calibration cancelled")
                return []

            frame_count += 1

        print(f"   ✅ Collected {len(samples)} samples for {color_name}")
        return samples

    def run_calibration(self) -> bool:
        """
        Run full calibration process.

        Returns:
            True if successful
        """
        print("=" * 60)
        print("🎨 AUTO-CALIBRATION FOR ML COLOR DETECTION")
        print("=" * 60)
        print("\nYou will need a SOLVED Rubik's cube.")
        print("The system will scan each face and learn the colors.\n")

        input("Press ENTER to start calibration...")

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return False

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            # Calibrate each face
            for face_name in self.face_order:
                color_name = self.face_colors[face_name]

                # Show instructions between faces
                if len(self.training_data) > 0:
                    print(f"\n▶️  Next: {color_name.upper()} face")
                    print("   Rotate cube and press ENTER when ready...")
                    input()

                # Collect samples
                samples = self.calibrate_face(cap, face_name)

                if len(samples) == 0:
                    # User cancelled
                    cap.release()
                    cv2.destroyAllWindows()
                    return False

                # Add to training data
                for rgb in samples:
                    self.training_data.append({"rgb": list(rgb), "color": color_name})

            # Success message
            print("\n" + "=" * 60)
            print(f"✅ Calibration complete!")
            print(f"   Total samples collected: {len(self.training_data)}")
            print(f"   Samples per color: {self.samples_per_face}")
            print("=" * 60)

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return True

    def save_training_data(self, filepath: str = "ml_training_data.json") -> None:
        """Save collected training data to file."""
        data = {
            "samples": self.training_data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "calibration_type": "auto",
                "samples_per_face": self.samples_per_face,
                "total_samples": len(self.training_data),
                "camera": "default",
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Training data saved to: {filepath}")

    def train_and_save_model(self, model_dir: str = ".") -> Optional[Dict]:
        """
        Train ML model with collected data and save it.

        Args:
            model_dir: Directory to save model

        Returns:
            Training metrics or None if failed
        """
        if len(self.training_data) == 0:
            print("❌ No training data collected!")
            return None

        print("\n🤖 Training ML model...")

        detector = MLColorDetector(confidence_threshold=0.7)

        try:
            metrics = detector.train(self.training_data)

            print(f"✅ Training complete!")
            print(f"   Accuracy: {metrics['accuracy']:.2%}")
            print(f"   Train samples: {metrics['n_train']}")
            print(f"   Test samples: {metrics['n_test']}")

            # Save model
            detector.save_model(model_dir)

            return metrics

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None


def main():
    """Main auto-calibration program."""
    calibrator = AutoCalibrator(samples_per_face=30)

    # Run calibration
    success = calibrator.run_calibration()

    if success:
        # Save training data
        calibrator.save_training_data()

        # Train and save model
        metrics = calibrator.train_and_save_model()

        if metrics:
            print("\n🎉 Auto-calibration successful!")
            print("   ML model is ready to use.")
            print("\n📝 Next steps:")
            print("   • Run the scanner with ML detection enabled")
            print("   • Enjoy improved color accuracy!")
        else:
            print("\n⚠️  Model training failed - please try again")
    else:
        print("\n⚠️  Calibration cancelled or failed")


if __name__ == "__main__":
    main()
