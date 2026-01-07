#!/usr/bin/env python3
"""
HSV Color Calibration Tool for Rubik's Cube Scanner
==================================================

Features:
✅ Real-time HSV value display
✅ Live color detection preview
✅ Save/load calibration presets
✅ Debug mode for fine-tuning
"""

import json
import os

import cv2
import numpy as np


class ColorCalibrator:
    def __init__(self):
        # Default safe HSV ranges - can be adjusted
        self.color_ranges = {
            "white": ([0, 0, 200], [180, 40, 255]),
            "red": ([0, 80, 80], [15, 255, 255]),
            "red2": ([170, 80, 80], [180, 255, 255]),  # Red wraparound
            "orange": ([11, 100, 100], [25, 255, 255]),
            "yellow": ([26, 100, 100], [35, 255, 255]),
            "green": ([36, 100, 100], [85, 255, 255]),
            "blue": ([90, 100, 100], [130, 255, 255]),
        }

        self.color_samples = {}  # Store HSV samples for each color
        self.calibration_mode = False

    def detect_color_with_debug(self, roi, debug=True):
        """Detect color with debug information"""
        if roi.size == 0:
            return "?", None

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Get center pixel for debugging
        h, w = hsv.shape[:2]
        center_hsv = hsv[h // 2, w // 2]

        # Test each color range
        best_match = "?"
        max_pixels = 0
        match_confidence = 0

        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name == "red2":  # Skip the secondary red range in main loop
                continue

            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Special case for red (check both ranges)
            if color_name == "red":
                lower2, upper2 = self.color_ranges["red2"]
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask, mask2)

            pixel_count = cv2.countNonZero(mask)
            confidence = pixel_count / roi.size

            if pixel_count > max_pixels and confidence > 0.3:
                max_pixels = pixel_count
                best_match = color_name
                match_confidence = confidence

        debug_info = (
            {"center_hsv": center_hsv, "confidence": match_confidence, "pixel_count": max_pixels}
            if debug
            else None
        )

        return best_match, debug_info

    def run_calibration_mode(self):
        """Interactive calibration mode"""
        cap = cv2.VideoCapture(0)

        print("🎨 HSV Color Calibration Mode")
        print("=" * 40)
        print("Instructions:")
        print("• Hold each cube color at the center circle")
        print("• Press keys 1-6 to save HSV values:")
        print("  1: White  2: Red    3: Orange")
        print("  4: Yellow 5: Green  6: Blue")
        print("• Press 'S' to save calibration")
        print("• Press 'Q' to quit")

        color_keys = {
            ord("1"): "white",
            ord("2"): "red",
            ord("3"): "orange",
            ord("4"): "yellow",
            ord("5"): "green",
            ord("6"): "blue",
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Center point
            cx, cy = w // 2, h // 2

            # Draw center circle
            cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            # Get HSV at center
            center_pixel = frame[cy - 10 : cy + 10, cx - 10 : cx + 10]
            if center_pixel.size > 0:
                hsv_pixel = cv2.cvtColor(center_pixel, cv2.COLOR_BGR2HSV)
                avg_hsv = np.mean(hsv_pixel.reshape(-1, 3), axis=0).astype(int)

                # Detect current color
                color, debug_info = self.detect_color_with_debug(center_pixel)

                # Display HSV values
                cv2.putText(
                    frame,
                    f"HSV: {avg_hsv}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Detected: {color}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                if debug_info:
                    cv2.putText(
                        frame,
                        f"Confidence: {debug_info['confidence']:.2f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # Instructions
            cv2.putText(
                frame,
                "Hold cube color at center circle",
                (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press 1-6 to save, S to save config, Q to quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

            # Show saved samples
            y_offset = 160
            for color_name, hsv_val in self.color_samples.items():
                cv2.putText(
                    frame,
                    f"{color_name}: {hsv_val}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                y_offset += 25

            cv2.imshow("HSV Color Calibrator", frame)

            key = cv2.waitKey(1) & 0xFF

            # Save color sample
            if key in color_keys and center_pixel.size > 0:
                color_name = color_keys[key]
                hsv_pixel = cv2.cvtColor(center_pixel, cv2.COLOR_BGR2HSV)
                avg_hsv = np.mean(hsv_pixel.reshape(-1, 3), axis=0).astype(int)
                self.color_samples[color_name] = list(avg_hsv)
                print(f"✅ Saved {color_name}: HSV {avg_hsv}")

            # Save calibration
            elif key == ord("s"):
                self.save_calibration()
                print("💾 Calibration saved!")

            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def generate_ranges_from_samples(self):
        """Generate HSV ranges from collected samples"""
        if len(self.color_samples) < 6:
            print("⚠️ Need all 6 colors to generate ranges")
            return

        new_ranges = {}

        for color_name, hsv in self.color_samples.items():
            h, s, v = hsv

            if color_name == "white":
                # White: low saturation, high value
                new_ranges["white"] = ([0, 0, max(180, v - 20)], [180, 50, 255])

            elif color_name == "red":
                # Red: handle hue wraparound
                if h <= 10:
                    new_ranges["red"] = (
                        [max(0, h - 10), max(50, s - 30), max(50, v - 30)],
                        [min(15, h + 10), 255, 255],
                    )
                    new_ranges["red2"] = ([170, max(50, s - 30), max(50, v - 30)], [180, 255, 255])
                else:
                    new_ranges["red"] = (
                        [max(170, h - 10), max(50, s - 30), max(50, v - 30)],
                        [180, 255, 255],
                    )
                    new_ranges["red2"] = ([0, max(50, s - 30), max(50, v - 30)], [10, 255, 255])

            else:
                # Other colors: create range around detected values
                h_range = 8 if color_name in ["orange", "yellow"] else 15
                new_ranges[color_name] = (
                    [max(0, h - h_range), max(80, s - 40), max(80, v - 40)],
                    [min(179, h + h_range), 255, 255],
                )

        self.color_ranges.update(new_ranges)
        print("✅ Generated new HSV ranges from samples")

    def save_calibration(self, filename="cube_calibration.json"):
        """Save calibration to file"""
        if self.color_samples:
            self.generate_ranges_from_samples()

        calibration_data = {"color_ranges": self.color_ranges, "color_samples": self.color_samples}

        with open(filename, "w") as f:
            json.dump(calibration_data, f, indent=2)

        print(f"💾 Calibration saved to {filename}")

    def load_calibration(self, filename="cube_calibration.json"):
        """Load calibration from file"""
        if os.path.exists(filename):
            with open(filename) as f:
                calibration_data = json.load(f)

            self.color_ranges = calibration_data.get("color_ranges", self.color_ranges)
            self.color_samples = calibration_data.get("color_samples", {})

            print(f"📂 Calibration loaded from {filename}")
            return True
        else:
            print(f"⚠️ Calibration file {filename} not found")
            return False

    def test_detection(self):
        """Test color detection with current calibration"""
        cap = cv2.VideoCapture(0)

        print("🧪 Testing Color Detection")
        print("Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Test area
            test_size = 100
            tx = w // 2 - test_size // 2
            ty = h // 2 - test_size // 2
            test_roi = frame[ty : ty + test_size, tx : tx + test_size]

            # Draw test area
            cv2.rectangle(frame, (tx, ty), (tx + test_size, ty + test_size), (0, 255, 0), 2)

            # Detect color
            if test_roi.size > 0:
                color, debug_info = self.detect_color_with_debug(test_roi)

                # Display results
                cv2.putText(
                    frame,
                    f"Detected: {color}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                if debug_info:
                    cv2.putText(
                        frame,
                        f"HSV: {debug_info['center_hsv']}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Confidence: {debug_info['confidence']:.2f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            cv2.imshow("Color Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main calibration interface"""
    calibrator = ColorCalibrator()

    # Try to load existing calibration
    calibrator.load_calibration()

    while True:
        print("\n🎨 HSV Color Calibration Tool")
        print("=" * 35)
        print("1. 🎯 Run Calibration Mode")
        print("2. 🧪 Test Current Detection")
        print("3. 💾 Save Current Settings")
        print("4. 📂 Load Settings")
        print("5. 🚪 Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            calibrator.run_calibration_mode()
        elif choice == "2":
            calibrator.test_detection()
        elif choice == "3":
            calibrator.save_calibration()
        elif choice == "4":
            filename = input("Enter filename (or press Enter for default): ").strip()
            if not filename:
                filename = "cube_calibration.json"
            calibrator.load_calibration(filename)
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
