import json
import os
from collections import Counter

import cv2
import numpy as np


# Load calibrated HSV color ranges from file
def load_color_ranges():
    """Load calibrated color ranges from cube_calibration.json"""
    default_ranges = {
        "white": ([0, 0, 200], [180, 50, 255]),
        "red": ([0, 100, 100], [10, 255, 255]),
        "red2": ([170, 100, 100], [180, 255, 255]),
        "orange": ([11, 100, 100], [25, 255, 255]),
        "yellow": ([26, 100, 100], [34, 255, 255]),
        "green": ([35, 100, 100], [85, 255, 255]),
        "blue": ([86, 100, 100], [130, 255, 255]),
    }

    calibration_file = "cube_calibration.json"
    if os.path.exists(calibration_file):
        try:
            with open(calibration_file) as f:
                calibration_data = json.load(f)

            # Use calibrated ranges if available
            if "color_ranges" in calibration_data:
                print("📂 Using calibrated HSV ranges from cube_calibration.json")
                return calibration_data["color_ranges"]
        except Exception as e:
            print(f"⚠️ Error loading calibration: {e}")

    print("📝 Using default HSV ranges")
    return default_ranges


# Load color ranges at module import
color_ranges = load_color_ranges()

# Color priority for better detection
color_priority = ["white", "yellow", "red", "orange", "green", "blue"]

# BGR colors for display
display_colors = {
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "?": (128, 128, 128),
}


def detect_single_cubelet_color(roi):
    """Detect color of a single cubelet region"""
    if roi.size == 0:
        return "?"

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Test each color range
    best_match = "?"
    max_pixels = 0

    for color_name in color_priority:
        if color_name not in color_ranges:
            continue

        lower, upper = color_ranges[color_name]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Special case for red (check both ranges)
        if color_name == "red":
            lower2, upper2 = color_ranges["red2"]
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)

        # Count pixels
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > max_pixels and pixel_count > roi.size * 0.3:  # At least 30% confidence
            max_pixels = pixel_count
            best_match = color_name

    return best_match


def detect_cube_face_colors(roi):
    """Detect colors in 3x3 grid from ROI with real-time feedback"""
    if roi is None or roi.size == 0:
        return [["?" for _ in range(3)] for _ in range(3)], True

    h, w = roi.shape[:2]
    cube_colors = []
    unknown_flag = False

    for i in range(3):
        row = []
        for j in range(3):
            # Calculate precise cell boundaries
            cell_h = h // 3
            cell_w = w // 3

            # Add padding to avoid edges
            padding = 8
            y1 = i * cell_h + padding
            y2 = (i + 1) * cell_h - padding
            x1 = j * cell_w + padding
            x2 = (j + 1) * cell_w - padding

            # Extract cell ROI
            cell_roi = roi[y1:y2, x1:x2]

            # Detect color
            detected_color = detect_single_cubelet_color(cell_roi)

            if detected_color == "?":
                unknown_flag = True

            row.append(detected_color)
        cube_colors.append(row)

    return cube_colors, unknown_flag


def get_display_color(color_name):
    """Get BGR color for display overlay"""
    return display_colors.get(color_name, (128, 128, 128))


def validate_face_colors(colors):
    """Validate face has reasonable color distribution"""
    if not colors:
        return False, "Empty face data"

    # Flatten colors
    flat_colors = [color for row in colors for color in row]

    # Check for unknowns
    unknown_count = flat_colors.count("?")
    if unknown_count > 2:  # Allow up to 2 unknowns for flexibility
        return False, f"Too many unknown colors: {unknown_count}"

    # Check color distribution
    color_counts = Counter(flat_colors)
    most_common = color_counts.most_common(1)[0]

    if most_common[1] > 7:  # No single color should dominate completely
        return False, f"Face seems monochrome: {most_common[0]} x {most_common[1]}"

    return True, "Face looks good"


def stabilize_colors(recent_detections, stability_threshold=8):
    """Stabilize color detection by requiring consistency"""
    if len(recent_detections) < stability_threshold:
        return None, False

    # Get recent detections
    recent = list(recent_detections)[-stability_threshold:]

    # Find most common color for each cell
    stable_colors = []
    for i in range(3):
        stable_row = []
        for j in range(3):
            cell_colors = [detection[i][j] for detection in recent]
            most_common = Counter(cell_colors).most_common(1)[0]

            # Require 60% consistency for stability
            if most_common[1] >= stability_threshold * 0.6:
                stable_row.append(most_common[0])
            else:
                stable_row.append("?")
        stable_colors.append(stable_row)

    # Check if stable detection is valid
    unknown_count = sum(row.count("?") for row in stable_colors)
    is_stable = unknown_count <= 1  # Allow 1 unknown max

    return stable_colors, is_stable
