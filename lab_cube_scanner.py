import os

import cv2
import joblib
import numpy as np


class LABCubeScanner:
    def __init__(self):
        self.face_names = [
            "White (U)",
            "Red (F)",
            "Green (R)",
            "Blue (B)",
            "Yellow (D)",
            "Orange (L)",
        ]
        self.current_face = 0
        self.cube_state = {}
        self.knn_model = None

        # Color mapping for visualization
        self.color_map = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
        }

        # Load the trained model
        self.load_trained_model()

    def load_trained_model(self):
        """Load the trained KNN model"""
        if os.path.exists("lab_knn.pkl"):
            try:
                self.knn_model = joblib.load("lab_knn.pkl")
                print("✅ Loaded trained LAB KNN model")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
        else:
            print("❌ No trained model found!")
            print("🔧 Please run: python train_lab_knn.py")
            return False

    def detect_face_lab(self, roi):
        """Detect colors in a 3x3 grid using LAB KNN classifier"""
        if self.knn_model is None:
            return [["unknown" for _ in range(3)] for _ in range(3)], [
                [0.0 for _ in range(3)] for _ in range(3)
            ]

        # Convert to LAB color space
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]

        cube_colors = []
        confidences = []

        for i in range(3):
            row = []
            conf_row = []
            for j in range(3):
                # Get center pixel of each sticker
                y = int((i + 0.5) * h / 3)
                x = int((j + 0.5) * w / 3)
                pixel = lab[y, x].reshape(1, -1)

                # Predict color using KNN model
                prediction = self.knn_model.predict(pixel)[0]
                confidence = self.knn_model.predict_proba(pixel).max()

                row.append(prediction)
                conf_row.append(confidence)

            cube_colors.append(row)
            confidences.append(conf_row)

        return cube_colors, confidences

    def draw_detection_overlay(self, frame, roi, colors, confidences):
        """Draw detected colors and confidence scores on the ROI"""
        h, w = roi.shape[:2]

        # Create overlay
        overlay = roi.copy()

        for i in range(3):
            for j in range(3):
                # Calculate sticker position
                y = int((i + 0.5) * h / 3)
                x = int((j + 0.5) * w / 3)

                color = colors[i][j]
                conf = confidences[i][j]

                # Draw circle with detected color
                draw_color = self.color_map.get(color, (128, 128, 128))
                cv2.circle(overlay, (x, y), 20, draw_color, -1)
                cv2.circle(overlay, (x, y), 20, (0, 0, 0), 2)

                # Draw confidence as text
                conf_text = f"{conf:.2f}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2

                cv2.putText(
                    overlay,
                    conf_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

                # Draw grid lines
                if j < 2:  # Vertical lines
                    line_x = int((j + 1) * w / 3)
                    cv2.line(overlay, (line_x, 0), (line_x, h), (0, 0, 0), 2)
                if i < 2:  # Horizontal lines
                    line_y = int((i + 1) * h / 3)
                    cv2.line(overlay, (0, line_y), (w, line_y), (0, 0, 0), 2)

        return overlay

    def scan_face(self, frame):
        """Scan current face and return detected colors"""
        h, w, _ = frame.shape

        # Define ROI for cube face (center square)
        roi_size = min(h, w) // 2
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        roi = frame[y1:y2, x1:x2]

        # Detect colors using LAB KNN
        colors, confidences = self.detect_face_lab(roi)

        # Draw overlay
        overlay_roi = self.draw_detection_overlay(frame, roi, colors, confidences)
        frame[y1:y2, x1:x2] = overlay_roi

        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return colors, confidences

    def display_info(self, frame, colors, confidences):
        """Display scanning information on frame"""
        face_name = self.face_names[self.current_face]

        # Main title
        cv2.putText(
            frame, f"Scanning: {face_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
        )

        # Face counter
        cv2.putText(
            frame,
            f"Face {self.current_face + 1}/6",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Average confidence
        avg_conf = np.mean([conf for row in confidences for conf in row])
        conf_color = (
            (0, 255, 0) if avg_conf > 0.8 else (0, 255, 255) if avg_conf > 0.6 else (0, 0, 255)
        )
        cv2.putText(
            frame,
            f"Confidence: {avg_conf:.1%}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            conf_color,
            2,
        )

        # Quality indicator
        if avg_conf > 0.8:
            quality = "Excellent"
            quality_color = (0, 255, 0)
        elif avg_conf > 0.6:
            quality = "Good"
            quality_color = (0, 255, 255)
        else:
            quality = "Poor - Check lighting"
            quality_color = (0, 0, 255)

        cv2.putText(
            frame, f"Quality: {quality}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2
        )

        # Controls
        cv2.putText(
            frame,
            "SPACE: Capture | n: Next | r: Reset | q: Quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def run_scanner(self):
        """Main scanning loop"""
        if self.knn_model is None:
            print("❌ Cannot run scanner without trained model!")
            return

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return

        print("🚀 LAB-Powered Cube Scanner Started!")
        print("=" * 50)
        print("🎯 Using LAB color space + KNN classification")
        print("📋 Controls:")
        print("  SPACE: Capture current face")
        print("  n: Next face")
        print("  r: Reset/restart")
        print("  q: Quit")
        print("=" * 50)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Scan current face
            colors, confidences = self.scan_face(frame)

            # Display information
            self.display_info(frame, colors, confidences)

            cv2.imshow("LAB Cube Scanner", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # Space to capture
                face_name = self.face_names[self.current_face]
                self.cube_state[face_name] = colors

                avg_conf = np.mean([conf for row in confidences for conf in row])
                print(f"\\n✅ Captured {face_name} (Confidence: {avg_conf:.1%})")
                print("Detected colors:")
                for row in colors:
                    print("  " + " ".join(f"{color:>8}" for color in row))

                # Auto-advance to next face
                if self.current_face < 5:
                    self.current_face += 1
                    print(f"➡️ Next: {self.face_names[self.current_face]}")
                else:
                    print("🎉 All faces captured! Press 'q' to finish or 'r' to restart.")

            elif key == ord("n"):  # Next face
                if self.current_face < 5:
                    self.current_face += 1
                    print(f"➡️ Next: {self.face_names[self.current_face]}")

            elif key == ord("r"):  # Reset
                self.current_face = 0
                self.cube_state = {}
                print("🔄 Reset to first face")

            elif key == ord("q"):  # Quit
                break

        cap.release()
        cv2.destroyAllWindows()

        # Show final results
        if self.cube_state:
            self.display_final_results()

    def display_final_results(self):
        """Display the final cube state"""
        print("\\n🎲 Final Cube State (LAB Detection):")
        print("=" * 50)

        for face, colors in self.cube_state.items():
            print(f"\\n{face}:")
            for row in colors:
                print("  " + " ".join(f"{color:>8}" for color in row))

        # Save results
        if len(self.cube_state) == 6:
            print("\\n💾 Saving cube state...")
            import json

            with open("cube_state_lab.json", "w") as f:
                json.dump(self.cube_state, f, indent=2)
            print("✅ Saved to cube_state_lab.json")

            print("\\n🧩 Ready for solving!")
            print("🔄 Next: Integrate with your cube solver")


def main():
    print("🧠 LAB-Based Rubik's Cube Scanner")
    print("Using optimized LAB color space + KNN classification")
    print("=" * 60)

    scanner = LABCubeScanner()
    if scanner.knn_model is not None:
        scanner.run_scanner()
    else:
        print("\\n📋 Setup Instructions:")
        print("1. Run: python color_trainer_lab.py")
        print("2. Run: python train_lab_knn.py")
        print("3. Run: python lab_cube_scanner.py")


if __name__ == "__main__":
    main()
