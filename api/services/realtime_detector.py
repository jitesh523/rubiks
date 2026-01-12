"""
Real-time Color Detection Service

Optimized for frame-by-frame color detection from camera feed.
Uses existing enhanced color detector with performance optimizations.
"""

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from enhanced_color_detector import (
    color_ranges,
)


class RealtimeDetector:
    """Real-time color detection with confidence scoring"""

    def __init__(self):
        self.color_ranges = color_ranges
        self.detection_history: dict[str, list] = {}
        self.stabilization_window = 5  # frames to average

    def decode_image_from_base64(self, image_data: str) -> np.ndarray:
        """Decode base64 image to OpenCV format"""
        # Remove data URL prefix if present
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # Convert to OpenCV format (BGR)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return frame

    def detect_color_with_confidence(self, roi: np.ndarray) -> tuple[str, float]:
        """
        Detect color from ROI with confidence score

        Args:
            roi: Region of interest (BGR image)

        Returns:
            Tuple of (color_name, confidence_score)
        """
        if roi is None or roi.size == 0:
            return "unknown", 0.0

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Sample multiple points (center and corners)
        h, w = hsv.shape[:2]
        sample_points = [
            (h // 2, w // 2),  # center
            (h // 4, w // 4),  # top-left
            (3 * h // 4, w // 4),  # bottom-left
            (h // 4, 3 * w // 4),  # top-right
            (3 * h // 4, 3 * w // 4),  # bottom-right
        ]

        color_votes: dict[str, int] = {}
        confidence_scores: dict[str, list[float]] = {}

        for y, x in sample_points:
            pixel_hsv = hsv[y, x]

            # Check each color range
            best_color = None
            best_score = 0.0

            for color_name, (lower, upper) in self.color_ranges.items():
                # Check if pixel is in range
                if np.all(pixel_hsv >= lower) and np.all(pixel_hsv <= upper):
                    # Calculate confidence based on how centered the value is
                    center = (np.array(lower, dtype=float) + np.array(upper, dtype=float)) / 2
                    distance = np.linalg.norm(pixel_hsv.astype(float) - center)
                    max_distance = np.linalg.norm(np.array(upper, dtype=float) - center)

                    score = 1.0 - distance / max_distance if max_distance > 0 else 1.0

                    if score > best_score:
                        best_score = score
                        best_color = color_name

            if best_color:
                color_votes[best_color] = color_votes.get(best_color, 0) + 1
                if best_color not in confidence_scores:
                    confidence_scores[best_color] = []
                confidence_scores[best_color].append(best_score)

        # Determine final color by vote
        if not color_votes:
            return "unknown", 0.0

        final_color = max(color_votes, key=color_votes.get)
        # Average confidence
        avg_confidence = np.mean(confidence_scores[final_color])

        return final_color, float(avg_confidence)

    def detect_face_grid(
        self, frame: np.ndarray, grid_region: dict[str, int]
    ) -> dict[str, any]:
        """
        Detect 3x3 grid of colors from frame region

        Args:
            frame: Input camera frame
            grid_region: Dict with x, y, width, height of detection region

        Returns:
            Dict with colors, confidences, and validation status
        """
        x = grid_region.get("x", 0)
        y = grid_region.get("y", 0)
        width = grid_region.get("width", 300)
        height = grid_region.get("height", 300)

        # Extract region
        roi = frame[y : y + height, x : x + width]

        if roi.size == 0:
            return {
                "colors": [["unknown"] * 3 for _ in range(3)],
                "confidences": [[0.0] * 3 for _ in range(3)],
                "avg_confidence": 0.0,
                "is_valid": False,
            }

        # Divide into 3x3 grid
        cell_height = height // 3
        cell_width = width // 3

        colors = []
        confidences = []

        for row in range(3):
            color_row = []
            conf_row = []

            for col in range(3):
                cell_y = row * cell_height
                cell_x = col * cell_width

                # Extract cell with some margin
                margin = 5
                cell_roi = roi[
                    cell_y + margin : cell_y + cell_height - margin,
                    cell_x + margin : cell_x + cell_width - margin,
                ]

                color, confidence = self.detect_color_with_confidence(cell_roi)
                color_row.append(color)
                conf_row.append(confidence)

            colors.append(color_row)
            confidences.append(conf_row)

        # Calculate average confidence
        flat_conf = [c for row in confidences for c in row]
        avg_confidence = float(np.mean(flat_conf))

        # Validate face (check if we have valid colors)
        flat_colors = [c for row in colors for c in row]
        is_valid = all(c != "unknown" for c in flat_colors) and avg_confidence > 0.6

        return {
            "colors": colors,
            "confidences": confidences,
            "avg_confidence": avg_confidence,
            "is_valid": is_valid,
        }

    def validate_move_execution(
        self, before_state: list[list[str]], after_state: list[list[str]], expected_move: str
    ) -> dict[str, any]:
        """
        Validate if a move was executed correctly

        Args:
            before_state: 3x3 color grid before move
            after_state: 3x3 color grid after move
            expected_move: Move notation (e.g., "R", "U'", "F2")

        Returns:
            Dict with validation result and feedback
        """
        # For now, we'll do a simple check:
        # 1. States should be different
        # 2. Center color should remain the same (center never moves)

        states_different = before_state != after_state

        # Check center remains same
        center_before = before_state[1][1] if len(before_state) > 1 else None
        center_after = after_state[1][1] if len(after_state) > 1 else None
        center_same = center_before == center_after

        # Basic validation
        is_valid = states_different and center_same

        return {
            "is_valid": is_valid,
            "states_different": states_different,
            "center_preserved": center_same,
            "message": (
                "Move appears correct!"
                if is_valid
                else "Move may not be correct. Please try again."
            ),
        }


# Singleton instance
realtime_detector = RealtimeDetector()
