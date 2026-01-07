#!/usr/bin/env python3
"""
Red vs Orange Training Data Collector
=====================================

Specialized trainer for collecting high-quality red and orange samples
to fix the notorious red/orange confusion in Rubik's cube scanners.

Features:
✅ Focuses specifically on red/orange samples
✅ Shows live HSV analysis for optimal sample collection
✅ Collects both BGR and LAB+HSV features
✅ Visual feedback for sample quality
✅ Guided collection process
"""

import os
import pickle

import cv2
import numpy as np


class RedOrangeTrainer:
    def __init__(self):
        self.samples_file = "enhanced_lab_samples.pkl"
        self.samples = self.load_existing_samples()

        # Target sample counts
        self.target_samples = {"red": 20, "orange": 20}

        # HSV guidance ranges for better collection
        self.ideal_ranges = {
            "red": {"h": (0, 10, 170, 180), "s": (100, 255), "v": (50, 255)},
            "orange": {"h": (10, 25), "s": (100, 255), "v": (50, 255)},
        }

    def load_existing_samples(self) -> dict:
        """Load existing LAB samples or create new structure"""
        if os.path.exists(self.samples_file):
            try:
                with open(self.samples_file, "rb") as f:
                    return pickle.load(f)
            except:
                print("⚠️ Could not load existing samples, starting fresh")

        # Create default structure with all 6 colors
        return {"white": [], "red": [], "orange": [], "yellow": [], "green": [], "blue": []}

    def analyze_pixel_hsv(self, bgr_pixel: np.ndarray) -> tuple[int, int, int]:
        """Analyze HSV values of a pixel"""
        bgr_reshaped = bgr_pixel.reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr_reshaped, cv2.COLOR_BGR2HSV)[0, 0]
        return int(hsv[0]), int(hsv[1]), int(hsv[2])

    def get_sample_quality(self, hsv: tuple[int, int, int], color: str) -> str:
        """Assess the quality of a sample based on HSV values"""
        h, s, v = hsv

        if color not in self.ideal_ranges:
            return "❓ Unknown"

        ranges = self.ideal_ranges[color]

        # Check hue
        if color == "red":
            hue_good = (0 <= h <= 10) or (170 <= h <= 180)
        else:  # orange
            hue_good = ranges["h"][0] <= h <= ranges["h"][1]

        # Check saturation and value
        sat_good = ranges["s"][0] <= s <= ranges["s"][1]
        val_good = ranges["v"][0] <= v <= ranges["v"][1]

        if hue_good and sat_good and val_good:
            return "🟢 Excellent"
        elif hue_good and sat_good:
            return "🟡 Good"
        elif hue_good:
            return "🟠 OK"
        else:
            return "🔴 Poor"

    def create_lab_hue_features(self, bgr_pixel: np.ndarray) -> list[float]:
        """Create LAB + Hue feature vector from BGR pixel"""
        # Convert to LAB
        bgr_reshaped = bgr_pixel.reshape(1, 1, 3)
        lab = cv2.cvtColor(bgr_reshaped, cv2.COLOR_BGR2LAB)[0, 0]

        # Convert to HSV for hue
        hsv = cv2.cvtColor(bgr_reshaped, cv2.COLOR_BGR2HSV)[0, 0]
        hue = hsv[0]

        # Return [L, A, B, H] feature vector
        return [float(lab[0]), float(lab[1]), float(lab[2]), float(hue)]

    def display_status(self, frame: np.ndarray) -> None:
        """Display current collection status on frame"""
        y_offset = 30

        # Title
        cv2.putText(
            frame,
            "Red vs Orange Trainer",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        y_offset += 40

        # Current counts and targets
        for color in ["red", "orange"]:
            current = len(self.samples.get(color, []))
            target = self.target_samples[color]

            if current >= target:
                status_color = (0, 255, 0)  # Green
                status_text = "✅ Complete"
            elif current >= target * 0.7:
                status_color = (0, 255, 255)  # Yellow
                status_text = "🟡 Almost"
            else:
                status_color = (0, 0, 255)  # Red
                status_text = "🔴 Need more"

            text = f"{color.capitalize()}: {current}/{target} {status_text}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            y_offset += 25

        # Instructions
        y_offset += 10
        instructions = [
            "Instructions:",
            "R = Add Red sample",
            "O = Add Orange sample",
            "S = Save samples",
            "Q = Quit",
        ]

        for instruction in instructions:
            cv2.putText(
                frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

    def display_crosshair_analysis(self, frame: np.ndarray, center_pixel: np.ndarray) -> None:
        """Display HSV analysis at crosshair location"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Draw crosshair
        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), 2)
        cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (0, 255, 0), 2)

        # Analyze pixel
        h_val, s_val, v_val = self.analyze_pixel_hsv(center_pixel)

        # Display HSV analysis
        analysis_x = w - 250
        analysis_y = 30

        cv2.putText(
            frame,
            "HSV Analysis:",
            (analysis_x, analysis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        analysis_y += 25

        cv2.putText(
            frame,
            f"Hue: {h_val}°",
            (analysis_x, analysis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        analysis_y += 20

        cv2.putText(
            frame,
            f"Sat: {s_val}",
            (analysis_x, analysis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        analysis_y += 20

        cv2.putText(
            frame,
            f"Val: {v_val}",
            (analysis_x, analysis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        analysis_y += 30

        # Show quality assessment for both colors
        for color in ["red", "orange"]:
            quality = self.get_sample_quality((h_val, s_val, v_val), color)
            cv2.putText(
                frame,
                f"{color.capitalize()}: {quality}",
                (analysis_x, analysis_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            analysis_y += 20

        # Show ideal ranges
        analysis_y += 10
        cv2.putText(
            frame,
            "Ideal Hue Ranges:",
            (analysis_x, analysis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        analysis_y += 20

        cv2.putText(
            frame,
            "Red: 0-10°, 170-180°",
            (analysis_x, analysis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
        analysis_y += 15

        cv2.putText(
            frame,
            "Orange: 10-25°",
            (analysis_x, analysis_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )

    def add_sample(self, color: str, bgr_pixel: np.ndarray) -> bool:
        """Add a sample for the specified color"""
        if color not in self.samples:
            self.samples[color] = []

        # Create LAB+Hue feature vector
        features = self.create_lab_hue_features(bgr_pixel)

        # Add to samples
        self.samples[color].append(features)

        # Show feedback
        h, s, v = self.analyze_pixel_hsv(bgr_pixel)
        quality = self.get_sample_quality((h, s, v), color)
        count = len(self.samples[color])
        target = self.target_samples.get(color, 15)

        print(f"✅ Added {color} sample #{count}: HSV({h}°, {s}, {v}) - {quality}")

        if count >= target:
            print(f"🎉 {color.capitalize()} collection complete! ({count}/{target})")
            return True
        else:
            print(f"   Need {target - count} more {color} samples")
            return False

    def save_samples(self) -> None:
        """Save samples to file"""
        with open(self.samples_file, "wb") as f:
            pickle.dump(self.samples, f)

        print(f"\n💾 Saved samples to {self.samples_file}")

        # Show summary
        print("📊 Sample Summary:")
        total = 0
        for color, samples in self.samples.items():
            count = len(samples)
            total += count
            target = self.target_samples.get(color, "N/A")
            status = "✅" if isinstance(target, int) and count >= target else "⚠️"
            print(f"  {color.capitalize():8}: {count:2d} samples {status}")

        print(f"  Total: {total} samples")

        # Check if ready for training
        red_ready = len(self.samples.get("red", [])) >= self.target_samples["red"]
        orange_ready = len(self.samples.get("orange", [])) >= self.target_samples["orange"]

        if red_ready and orange_ready:
            print("\n🚀 Ready to train enhanced model!")
            print("   Run: python enhanced_red_orange_detector.py")
        else:
            print("\n📝 Collect more samples:")
            if not red_ready:
                needed = self.target_samples["red"] - len(self.samples.get("red", []))
                print(f"   Red: need {needed} more")
            if not orange_ready:
                needed = self.target_samples["orange"] - len(self.samples.get("orange", []))
                print(f"   Orange: need {needed} more")

    def run(self) -> None:
        """Run the training interface"""
        print("🎯 Red vs Orange Training Data Collector")
        print("=" * 50)
        print("📋 Goal: Collect high-quality red and orange samples")
        print("💡 Tips:")
        print("  • Use actual cube stickers for best results")
        print("  • Vary lighting conditions between samples")
        print("  • Try different angles and distances")
        print("  • Look for 'Excellent' quality ratings")
        print("=" * 50)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            center_pixel = frame[h // 2, w // 2]

            # Display UI elements
            self.display_status(frame)
            self.display_crosshair_analysis(frame, center_pixel)

            cv2.imshow("Red vs Orange Trainer", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r") or key == ord("R"):
                self.add_sample("red", center_pixel)

            elif key == ord("o") or key == ord("O"):
                self.add_sample("orange", center_pixel)

            elif key == ord("s") or key == ord("S"):
                self.save_samples()

            elif key == ord("q") or key == ord("Q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n👋 Training session complete!")

        # Final save prompt
        red_count = len(self.samples.get("red", []))
        orange_count = len(self.samples.get("orange", []))

        if red_count > 0 or orange_count > 0:
            print(f"💾 You collected {red_count} red and {orange_count} orange samples")
            save_prompt = input("Save samples before exit? (y/n): ").lower()
            if save_prompt == "y":
                self.save_samples()


if __name__ == "__main__":
    trainer = RedOrangeTrainer()
    trainer.run()
