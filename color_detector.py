import cv2
import numpy as np
from collections import Counter


class ColorDetector:
    def __init__(self):
        # Improved HSV color ranges for better detection
        self.color_ranges = {
            "white": ([0, 0, 180], [180, 30, 255]),
            "red": ([0, 120, 70], [10, 255, 255]),
            "red2": ([170, 120, 70], [180, 255, 255]),  # Red wraps around hue
            "orange": ([10, 120, 120], [25, 255, 255]),
            "yellow": ([20, 120, 120], [30, 255, 255]),
            "green": ([40, 70, 70], [80, 255, 255]),
            "blue": ([100, 120, 70], [130, 255, 255]),
        }

        # Color priority for better detection
        self.color_priority = ["white", "yellow", "red", "orange", "green", "blue"]

        # BGR colors for display
        self.display_colors = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "?": (128, 128, 128),
        }

    def detect_single_color(self, roi):
        """Detect the dominant color in a region of interest"""
        if roi.size == 0:
            return "?"

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Test each color range
        best_match = "?"
        max_pixels = 0

        for color_name in self.color_priority:
            if color_name not in self.color_ranges:
                continue

            lower, upper = self.color_ranges[color_name]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Special case for red (check both ranges)
            if color_name == "red":
                lower2, upper2 = self.color_ranges["red2"]
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask, mask2)

            # Count pixels
            pixel_count = cv2.countNonZero(mask)

            if pixel_count > max_pixels and pixel_count > roi.size * 0.3:  # At least 30% of pixels
                max_pixels = pixel_count
                best_match = color_name

        return best_match

    def detect_cube_face_colors(self, roi, grid_size=3):
        """Detect colors in a 3x3 grid from the ROI"""
        if roi is None or roi.size == 0:
            return [["?" for _ in range(grid_size)] for _ in range(grid_size)], True

        h, w = roi.shape[:2]
        cube_colors = []
        unknown_count = 0

        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                # Calculate cell boundaries with some padding
                cell_h = h // grid_size
                cell_w = w // grid_size

                y1 = i * cell_h + cell_h // 4
                y2 = (i + 1) * cell_h - cell_h // 4
                x1 = j * cell_w + cell_w // 4
                x2 = (j + 1) * cell_w - cell_w // 4

                # Extract cell ROI
                cell_roi = roi[y1:y2, x1:x2]

                # Detect color
                detected_color = self.detect_single_color(cell_roi)

                if detected_color == "?":
                    unknown_count += 1

                row.append(detected_color)
            cube_colors.append(row)

        # Face is invalid if more than 1 cell is unknown
        unknown_flag = unknown_count > 1

        return cube_colors, unknown_flag

    def get_display_color(self, color_name):
        """Get BGR color for display"""
        return self.display_colors.get(color_name, (128, 128, 128))

    def validate_face_colors(self, colors):
        """Validate that a face has reasonable color distribution"""
        if not colors:
            return False, "Empty face data"

        # Flatten the colors
        flat_colors = [color for row in colors for color in row]

        # Check for unknowns
        unknown_count = flat_colors.count("?")
        if unknown_count > 1:
            return False, f"Too many unknown colors: {unknown_count}"

        # Check color distribution (no single color should dominate too much)
        color_counts = Counter(flat_colors)
        most_common = color_counts.most_common(1)[0]

        if most_common[1] > 6:  # More than 6 of the same color is suspicious
            return False, f"Too many {most_common[0]} colors: {most_common[1]}"

        return True, "Face looks good"

    def stabilize_colors(self, recent_detections, stability_threshold=5):
        """Stabilize color detection by requiring consistency"""
        if len(recent_detections) < stability_threshold:
            return None, False

        # Check if the last few detections are consistent
        # Convert deque to list for slicing
        recent_list = list(recent_detections)
        recent = recent_list[-stability_threshold:]

        # Compare all recent detections
        stable_colors = []
        for i in range(3):
            stable_row = []
            for j in range(3):
                # Get the most common color for this cell
                cell_colors = [detection[i][j] for detection in recent]
                most_common = Counter(cell_colors).most_common(1)[0]

                # Require at least 60% consistency
                if most_common[1] >= stability_threshold * 0.6:
                    stable_row.append(most_common[0])
                else:
                    stable_row.append("?")
            stable_colors.append(stable_row)

        # Check if stable detection is valid
        unknown_count = sum(row.count("?") for row in stable_colors)
        is_stable = unknown_count <= 1

        return stable_colors, is_stable
