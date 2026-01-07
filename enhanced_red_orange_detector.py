#!/usr/bin/env python3
"""
Enhanced Red vs Orange Color Detector
=====================================

Fixes the notorious red vs orange confusion using:
✅ Hybrid LAB + HSV classification
✅ Hue-aware KNN training (LAB + Hue)
✅ Manual override when uncertain
✅ Enhanced training data collection
✅ Visual confidence indicators

Based on the curated guidance for fixing red/orange confusion.
"""

import os
import pickle

import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class EnhancedRedOrangeDetector:
    def __init__(self, debug=False):
        self.debug = debug
        self.samples_file = "enhanced_lab_samples.pkl"
        self.model_file = "hybrid_knn.pkl"

        # Load existing model if available
        self.knn = None
        self.scaler = None
        self.load_model()

        # Color display mapping
        self.display_colors = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "?": (128, 128, 128),
        }

        # Confidence thresholds for manual override
        self.uncertain_threshold = 0.6  # Below this, ask user
        self.min_confidence = 0.3  # Minimum to accept

    def load_model(self):
        """Load trained enhanced KNN model"""
        if os.path.exists(self.model_file):
            try:
                model_data = joblib.load(self.model_file)
                if isinstance(model_data, dict):
                    # Handle hybrid model format
                    self.knn = model_data.get("model")
                    self.scaler = model_data.get("scaler")
                    print("✅ Loaded hybrid KNN model with LAB+Hue features and scaler")
                else:
                    # Handle simple model format
                    self.knn = model_data
                    self.scaler = None
                    print("✅ Loaded enhanced KNN model with LAB+Hue features")
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                self.knn = None
                self.scaler = None
        else:
            print("⚠️ No enhanced model found. Please train first!")
            self.knn = None
            self.scaler = None

    def extract_features(self, bgr_pixel):
        """Extract LAB + Hue features from BGR pixel"""
        # Convert single pixel to LAB
        bgr_reshaped = bgr_pixel.reshape(1, 1, 3)
        lab = cv2.cvtColor(bgr_reshaped, cv2.COLOR_BGR2LAB)[0, 0]

        # Convert to HSV for hue
        hsv = cv2.cvtColor(bgr_reshaped, cv2.COLOR_BGR2HSV)[0, 0]
        hue = hsv[0]

        # Combine LAB + Hue as feature vector
        features = np.array([lab[0], lab[1], lab[2], hue])
        return features

    def predict_with_hue_rule(self, bgr_pixel):
        """
        Predict color using hybrid LAB+Hue approach

        Returns:
            (color_name, confidence, method_used)
        """
        if self.knn is None:
            return "?", 0.0, "no_model"

        # Extract features (LAB + Hue)
        features = self.extract_features(bgr_pixel)

        # Apply scaling if scaler is available
        if self.scaler is not None:
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = [features]

        # Get base prediction from KNN
        base_prediction = self.knn.predict(features_scaled)[0]
        base_confidence = self.knn.predict_proba(features_scaled).max()

        # Extract hue for rule-based refinement
        hsv = cv2.cvtColor(bgr_pixel.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        hue = hsv[0]

        # Apply hue-based rules for red/orange disambiguation
        final_color = base_prediction
        method = "base_knn"

        if base_prediction in ["red", "orange"]:
            if 8 < hue < 20:  # Orange hue range
                if base_prediction == "red":
                    final_color = "orange"
                    method = "hue_corrected_to_orange"
            elif hue <= 8 or hue >= 170:  # Red hue range (wraps around)
                if base_prediction == "orange":
                    final_color = "red"
                    method = "hue_corrected_to_red"
            else:
                method = "hue_confirmed"

        if self.debug:
            print(
                f"  Hue: {hue:3.0f}° | Base: {base_prediction} | Final: {final_color} | Method: {method}"
            )

        return final_color, base_confidence, method

    def detect_single_cubelet(self, roi):
        """Detect color of single cubelet with enhanced red/orange handling"""
        if roi.size == 0:
            return "?", 0.0, {}

        # Get center pixel (more stable than averaging)
        h, w = roi.shape[:2]
        center_y, center_x = h // 2, w // 2
        center_pixel = roi[center_y, center_x]

        # Predict using hybrid approach
        color, confidence, method = self.predict_with_hue_rule(center_pixel)

        debug_info = {
            "roi_size": roi.shape[:2],
            "center_pixel_bgr": center_pixel.tolist(),
            "method": method,
            "base_confidence": confidence,
        }

        return color, confidence, debug_info

    def detect_face_colors(self, roi, cell_padding=8):
        """
        Detect 3x3 grid colors with enhanced red/orange detection

        Returns:
            (colors_grid, confidence_grid, uncertain_cells, debug_info)
        """
        if roi is None or roi.size == 0:
            empty_grid = [["?" for _ in range(3)] for _ in range(3)]
            zero_conf = [[0.0 for _ in range(3)] for _ in range(3)]
            return empty_grid, zero_conf, [], {}

        h, w = roi.shape[:2]
        colors_grid = []
        confidence_grid = []
        uncertain_cells = []
        all_debug_info = {}

        if self.debug:
            print(f"\n🔍 Enhanced Detection (ROI: {w}x{h})")

        for i in range(3):
            color_row = []
            conf_row = []

            for j in range(3):
                # Calculate cell boundaries
                cell_h = h // 3
                cell_w = w // 3

                y1 = i * cell_h + cell_padding
                y2 = (i + 1) * cell_h - cell_padding
                x1 = j * cell_w + cell_padding
                x2 = (j + 1) * cell_w - cell_padding

                # Extract cell ROI
                cell_roi = roi[y1:y2, x1:x2]

                # Detect color
                color, confidence, debug_info = self.detect_single_cubelet(cell_roi)

                # Check if we need manual override
                if color in ["red", "orange"] and confidence < self.uncertain_threshold:
                    uncertain_cells.append((i, j, color, confidence))

                color_row.append(color)
                conf_row.append(confidence)
                all_debug_info[f"cell_{i}_{j}"] = debug_info

                if self.debug:
                    print(f"  Cell [{i},{j}]: {color} ({confidence:.1%})")

            colors_grid.append(color_row)
            confidence_grid.append(conf_row)

        return colors_grid, confidence_grid, uncertain_cells, all_debug_info

    def manual_override_prompt(self, uncertain_cells, frame=None):
        """
        Show manual override for uncertain red/orange cells

        Returns updated colors_grid or None if cancelled
        """
        if not uncertain_cells:
            return None

        print(f"\n🤔 Found {len(uncertain_cells)} uncertain red/orange detections")

        overrides = {}
        for i, j, detected_color, confidence in uncertain_cells:
            print(f"\nCell [{i},{j}]: Detected {detected_color} ({confidence:.1%} confidence)")

            while True:
                response = input("Is this (R)ed, (O)range, or (S)kip? ").upper()
                if response == "R":
                    overrides[(i, j)] = "red"
                    break
                elif response == "O":
                    overrides[(i, j)] = "orange"
                    break
                elif response == "S":
                    break
                else:
                    print("Please enter R, O, or S")

        return overrides

    def apply_manual_overrides(self, colors_grid, overrides):
        """Apply manual overrides to colors grid"""
        if not overrides:
            return colors_grid

        updated_grid = [row[:] for row in colors_grid]  # Deep copy

        for (i, j), new_color in overrides.items():
            if 0 <= i < 3 and 0 <= j < 3:
                updated_grid[i][j] = new_color
                print(f"✅ Override: Cell [{i},{j}] → {new_color}")

        return updated_grid

    def get_display_color(self, color_name):
        """Get BGR color for display overlay"""
        return self.display_colors.get(color_name, (128, 128, 128))

    def draw_enhanced_overlay(self, frame, colors, confidences, start_x, start_y, grid_size):
        """Draw enhanced overlay with red/orange highlighting"""
        cell_size = grid_size // 3

        for i in range(3):
            for j in range(3):
                if i < len(colors) and j < len(colors[i]):
                    color = colors[i][j]
                    conf = (
                        confidences[i][j]
                        if i < len(confidences) and j < len(confidences[i])
                        else 0.0
                    )

                    # Cell center
                    cell_x = start_x + j * cell_size + cell_size // 2
                    cell_y = start_y + i * cell_size + cell_size // 2

                    # Enhanced visualization for red/orange
                    if color in ["red", "orange"]:
                        # Special highlighting for red/orange
                        circle_color = (
                            (0, 0, 255) if color == "red" else (0, 165, 255)
                        )  # Red or Orange
                        thickness = 3 if conf < self.uncertain_threshold else 2

                        # Draw confidence circle
                        radius = int(15 + conf * 15)
                        cv2.circle(frame, (cell_x, cell_y), radius, circle_color, thickness)

                        # Add uncertainty indicator
                        if conf < self.uncertain_threshold:
                            cv2.circle(
                                frame, (cell_x, cell_y), radius + 5, (0, 255, 255), 1
                            )  # Yellow warning
                    else:
                        # Standard visualization for other colors
                        display_color = self.get_display_color(color)
                        radius = int(10 + conf * 15)
                        cv2.circle(frame, (cell_x, cell_y), radius, display_color, 2)

                    # Confidence text
                    conf_text = f"{conf:.0%}"
                    text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    text_x = cell_x - text_size[0] // 2
                    text_y = cell_y + text_size[1] // 2

                    cv2.putText(
                        frame,
                        conf_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

                    # Color label
                    label = (
                        "🔴"
                        if color == "red"
                        else "🟠" if color == "orange" else color[:1].upper()
                    )
                    label_y = cell_y - 20
                    cv2.putText(
                        frame,
                        label,
                        (cell_x - 10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )


def train_enhanced_model():
    """Train enhanced KNN model with LAB + Hue features"""
    print("🧠 Training Enhanced KNN with LAB + Hue Features")
    print("=" * 60)

    # Load existing LAB samples
    if not os.path.exists("lab_samples.pkl"):
        print("❌ No LAB training data found!")
        print("🔧 Please run: python color_trainer_lab.py")
        return None

    with open("lab_samples.pkl", "rb") as f:
        lab_samples = pickle.load(f)

    print("📊 Converting LAB samples to LAB+Hue features...")

    X = []
    y = []

    # For each LAB sample, we need to estimate HSV
    # Since we don't have the original BGR, we'll convert LAB back to BGR, then to HSV
    for color, lab_pixels in lab_samples.items():
        print(f"  Processing {color}: {len(lab_pixels)} samples")

        for lab_pixel in lab_pixels:
            # Convert LAB back to BGR (approximate)
            lab_array = np.array([[[lab_pixel[0], lab_pixel[1], lab_pixel[2]]]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(lab_array, cv2.COLOR_LAB2BGR)[0, 0]

            # Now get HSV hue
            hsv_array = cv2.cvtColor(bgr_pixel.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            hue = hsv_array[0]

            # Create feature vector: [L, A, B, H]
            features = [lab_pixel[0], lab_pixel[1], lab_pixel[2], hue]

            X.append(features)
            y.append(color)

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Created {len(X)} feature vectors with LAB+Hue")

    # Train enhanced KNN
    print("🔄 Training enhanced KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance", metric="euclidean")
    knn.fit(X, y)

    # Save enhanced model
    joblib.dump(knn, "enhanced_knn_model.pkl")
    print("✅ Enhanced KNN model saved!")

    # Test on training data
    print("\n🧪 Testing enhanced model:")
    correct = 0
    red_orange_tests = 0
    red_orange_correct = 0

    for i, (features, true_color) in enumerate(zip(X, y, strict=False)):
        pred_color = knn.predict([features])[0]
        confidence = knn.predict_proba([features]).max()

        if pred_color == true_color:
            correct += 1

        if true_color in ["red", "orange"]:
            red_orange_tests += 1
            if pred_color == true_color:
                red_orange_correct += 1

        if i < 12:  # Show first few predictions
            status = "✅" if pred_color == true_color else "❌"
            print(f"  {true_color:7} → {pred_color:7} ({confidence:.1%}) {status}")

    overall_accuracy = correct / len(X)
    red_orange_accuracy = red_orange_correct / red_orange_tests if red_orange_tests > 0 else 0

    print(f"\n📈 Overall accuracy: {overall_accuracy:.1%}")
    print(
        f"🎯 Red/Orange accuracy: {red_orange_accuracy:.1%} ({red_orange_correct}/{red_orange_tests})"
    )

    print("\n🚀 Enhanced model ready!")
    print("💡 Usage: detector = EnhancedRedOrangeDetector()")

    return knn


if __name__ == "__main__":
    print("🎯 Enhanced Red vs Orange Detector")
    print("=" * 50)

    # Check if we need to train
    if not os.path.exists("enhanced_knn_model.pkl"):
        print("📝 Training enhanced model first...")
        train_enhanced_model()

    # Test with camera
    detector = EnhancedRedOrangeDetector(debug=True)

    if detector.knn is None:
        print("❌ No model loaded. Please train first.")
        exit(1)

    print("\n📷 Testing enhanced detector (press 'q' to quit)")
    print("🎯 Focus on red and orange stickers to test the fix!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Test region in center
        size = 200
        x1 = w // 2 - size // 2
        y1 = h // 2 - size // 2
        x2 = x1 + size
        y2 = y1 + size

        roi = frame[y1:y2, x1:x2]

        # Detect colors
        colors, confidences, uncertain_cells, debug_info = detector.detect_face_colors(roi)

        # Draw detection results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        detector.draw_enhanced_overlay(frame, colors, confidences, x1, y1, size)

        # Status
        red_count = sum(row.count("red") for row in colors)
        orange_count = sum(row.count("orange") for row in colors)
        uncertain_count = len(uncertain_cells)

        status = f"Red: {red_count} | Orange: {orange_count} | Uncertain: {uncertain_count}"
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(
            frame,
            "Enhanced Red/Orange Detection",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )

        cv2.imshow("Enhanced Red vs Orange Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
