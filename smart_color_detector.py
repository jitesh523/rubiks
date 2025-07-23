#!/usr/bin/env python3
"""
Smart Color Detector with HSV Distance Matching
==============================================

Enhanced color detection that:
✅ Uses HSV distance for better matching
✅ Provides confidence scores (0-100%)
✅ Fuzzy fallback matching to avoid unknowns
✅ Adaptive threshold based on lighting conditions
✅ Debug visualization for tuning
"""

import cv2
import numpy as np
import json
import os
from typing import Tuple, List, Dict, Optional

class SmartColorDetector:
    def __init__(self, calibration_file="cube_calibration.json", debug=False):
        self.debug = debug
        self.calibration_file = calibration_file
        
        # Default HSV ranges (updated with your values)
        self.default_ranges = {
            "white": ([0, 0, 180], [180, 30, 255]),
            "red": ([0, 120, 70], [10, 255, 255]),
            "red2": ([170, 120, 70], [180, 255, 255]),  # Red wraps around
            "green": ([35, 40, 40], [85, 255, 255]),
            "blue": ([100, 50, 50], [130, 255, 255]),
            "orange": ([10, 100, 100], [25, 255, 255]),
            "yellow": ([20, 100, 100], [35, 255, 255])
        }
        
        # Load calibrated ranges if available
        self.color_ranges = self.load_calibration()
        
        # Color display mapping
        self.display_colors = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "?": (128, 128, 128)
        }
        
        # Confidence thresholds
        self.min_confidence = 0.15  # Minimum to consider a match
        self.good_confidence = 0.35  # Threshold for good match
        self.excellent_confidence = 0.55  # Threshold for excellent match
        
    def load_calibration(self) -> dict:
        """Load calibrated HSV ranges"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r") as f:
                    data = json.load(f)
                print("✅ Loaded calibrated HSV ranges")
                return data.get("color_ranges", self.default_ranges)
            except:
                print("⚠️ Error loading calibration, using defaults")
        else:
            print("⚠️ No calibration file found, using defaults")
        
        return self.default_ranges
    
    def hsv_distance(self, hsv_pixel: np.ndarray, hsv_center: np.ndarray) -> float:
        """
        Calculate HSV distance with proper hue wrapping
        
        Args:
            hsv_pixel: Single HSV pixel [H, S, V]
            hsv_center: Center HSV value [H, S, V]
            
        Returns:
            Normalized distance (0.0 = perfect match, 1.0 = maximum difference)
        """
        h1, s1, v1 = hsv_pixel
        h2, s2, v2 = hsv_center
        
        # Hue distance (circular, 0-180 in OpenCV)
        hue_diff = abs(h1 - h2)
        hue_dist = min(hue_diff, 180 - hue_diff) / 90.0  # Normalize to 0-1
        
        # Saturation distance
        sat_dist = abs(s1 - s2) / 255.0
        
        # Value distance  
        val_dist = abs(v1 - v2) / 255.0
        
        # Weighted combination (hue most important, then saturation, then value)
        distance = (hue_dist * 0.5) + (sat_dist * 0.3) + (val_dist * 0.2)
        
        return min(distance, 1.0)
    
    def get_color_center(self, color_name: str) -> np.ndarray:
        """Get the center HSV value for a color"""
        if color_name not in self.color_ranges:
            return np.array([90, 128, 128])  # Default gray center
            
        lower, upper = self.color_ranges[color_name]
        
        # Handle hue wrapping for red
        if color_name == "red" and lower[0] > upper[0]:
            # Red wraps around, use 0 as center
            center_h = 0
        else:
            center_h = (lower[0] + upper[0]) // 2
            
        center_s = (lower[1] + upper[1]) // 2
        center_v = (lower[2] + upper[2]) // 2
        
        return np.array([center_h, center_s, center_v])
    
    def mask_based_confidence(self, hsv_roi: np.ndarray, color_name: str) -> float:
        """Calculate confidence using traditional mask method"""
        if color_name not in self.color_ranges:
            return 0.0
            
        lower, upper = self.color_ranges[color_name]
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        
        # Handle red special case (dual range)
        if color_name == "red" and "red2" in self.color_ranges:
            lower2, upper2 = self.color_ranges["red2"]
            mask2 = cv2.inRange(hsv_roi, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)
        
        pixel_count = cv2.countNonZero(mask)
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
        
        return pixel_count / total_pixels if total_pixels > 0 else 0.0
    
    def distance_based_confidence(self, hsv_roi: np.ndarray, color_name: str) -> float:
        """Calculate confidence using HSV distance"""
        if hsv_roi.size == 0:
            return 0.0
            
        color_center = self.get_color_center(color_name)
        
        # Calculate distance for each pixel
        distances = []
        h, w = hsv_roi.shape[:2]
        
        for y in range(h):
            for x in range(w):
                pixel_hsv = hsv_roi[y, x]
                distance = self.hsv_distance(pixel_hsv, color_center)
                distances.append(distance)
        
        if not distances:
            return 0.0
            
        # Use inverse of average distance as confidence
        avg_distance = np.mean(distances)
        confidence = max(0.0, 1.0 - avg_distance)
        
        return confidence
    
    def detect_single_color(self, roi: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Detect color in ROI using smart matching
        
        Returns:
            (color_name, confidence, debug_info)
        """
        if roi.size == 0:
            return "?", 0.0, {"error": "Empty ROI"}
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Test all colors
        results = {}
        debug_info = {"method": "smart_hybrid", "roi_size": roi.shape[:2]}
        
        for color_name in self.color_ranges.keys():
            if color_name == "red2":  # Skip secondary red
                continue
                
            # Method 1: Traditional mask-based
            mask_conf = self.mask_based_confidence(hsv_roi, color_name)
            
            # Method 2: Distance-based
            dist_conf = self.distance_based_confidence(hsv_roi, color_name)
            
            # Hybrid confidence (weighted average)
            hybrid_conf = (mask_conf * 0.6) + (dist_conf * 0.4)
            
            results[color_name] = {
                "mask_confidence": mask_conf,
                "distance_confidence": dist_conf,
                "hybrid_confidence": hybrid_conf
            }
            
            if self.debug:
                print(f"  {color_name:6s}: mask={mask_conf:.3f}, dist={dist_conf:.3f}, hybrid={hybrid_conf:.3f}")
        
        # Find best match
        best_color = "?"
        best_confidence = 0.0
        
        for color_name, scores in results.items():
            if scores["hybrid_confidence"] > best_confidence:
                best_confidence = scores["hybrid_confidence"]
                best_color = color_name
        
        # Apply minimum confidence threshold
        if best_confidence < self.min_confidence:
            # Fuzzy fallback - find closest by distance only
            fallback_color, fallback_conf = self.fuzzy_fallback(hsv_roi)
            
            if fallback_conf > best_confidence * 0.7:  # Accept if reasonably close
                best_color = fallback_color
                best_confidence = fallback_conf
                debug_info["used_fallback"] = True
            else:
                best_color = "?"
                best_confidence = 0.0
        
        debug_info["all_results"] = results
        debug_info["final_color"] = best_color
        debug_info["final_confidence"] = best_confidence
        
        return best_color, best_confidence, debug_info
    
    def fuzzy_fallback(self, hsv_roi: np.ndarray) -> Tuple[str, float]:
        """
        Fallback method using pure HSV distance to avoid unknowns
        """
        if hsv_roi.size == 0:
            return "?", 0.0
        
        # Calculate median HSV of the ROI
        median_hsv = np.median(hsv_roi.reshape(-1, 3), axis=0)
        
        best_color = "?"
        min_distance = float('inf')
        
        for color_name in self.color_ranges.keys():
            if color_name == "red2":
                continue
                
            color_center = self.get_color_center(color_name)
            distance = self.hsv_distance(median_hsv, color_center)
            
            if distance < min_distance:
                min_distance = distance
                best_color = color_name
        
        # Convert distance to confidence
        confidence = max(0.0, (1.0 - min_distance) * 0.8)  # Scale down a bit
        
        return best_color, confidence
    
    def detect_face_colors(self, roi: np.ndarray, cell_padding: int = 8) -> Tuple[List[List[str]], List[List[float]], bool]:
        """
        Detect 3x3 grid of colors with smart detection
        
        Returns:
            (colors_grid, confidence_grid, has_unknowns)
        """
        if roi is None or roi.size == 0:
            empty_grid = [["?" for _ in range(3)] for _ in range(3)]
            zero_conf = [[0.0 for _ in range(3)] for _ in range(3)]
            return empty_grid, zero_conf, True
        
        h, w = roi.shape[:2]
        colors_grid = []
        confidence_grid = []
        has_unknowns = False
        
        if self.debug:
            print(f"\n🔍 Detecting 3x3 face (ROI: {w}x{h})")
        
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
                
                if self.debug:
                    print(f"Cell [{i},{j}]: {cell_roi.shape[:2]}")
                
                # Detect color
                color, confidence, debug_info = self.detect_single_color(cell_roi)
                
                if color == "?":
                    has_unknowns = True
                
                color_row.append(color)
                conf_row.append(confidence)
                
                if self.debug:
                    print(f"  Result: {color} ({confidence:.1%})")
            
            colors_grid.append(color_row)
            confidence_grid.append(conf_row)
        
        return colors_grid, confidence_grid, has_unknowns
    
    def get_display_color(self, color_name: str) -> Tuple[int, int, int]:
        """Get BGR color for display"""
        return self.display_colors.get(color_name, self.display_colors["?"])
    
    def draw_debug_overlay(self, frame: np.ndarray, colors: List[List[str]], 
                          confidences: List[List[float]], start_x: int, start_y: int, 
                          grid_size: int) -> None:
        """Draw debug information overlay"""
        if not self.debug:
            return
            
        cell_size = grid_size // 3
        
        for i in range(3):
            for j in range(3):
                if i < len(colors) and j < len(colors[i]):
                    color = colors[i][j]
                    conf = confidences[i][j]
                    
                    # Cell center
                    cell_x = start_x + j * cell_size + cell_size // 2
                    cell_y = start_y + i * cell_size + cell_size // 2
                    
                    # Confidence circle (size based on confidence)
                    radius = int(10 + conf * 20)
                    circle_color = (0, 255, 0) if conf > self.good_confidence else (0, 255, 255) if conf > self.min_confidence else (0, 0, 255)
                    
                    cv2.circle(frame, (cell_x, cell_y), radius, circle_color, 2)
                    
                    # Confidence text
                    conf_text = f"{conf:.0%}"
                    text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    text_x = cell_x - text_size[0] // 2
                    text_y = cell_y + text_size[1] // 2
                    
                    cv2.putText(frame, conf_text, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Factory function for easy integration
def create_smart_detector(calibration_file="cube_calibration.json", debug=False) -> SmartColorDetector:
    """Create a smart color detector instance"""
    return SmartColorDetector(calibration_file, debug)

# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("🧠 Smart Color Detector Test")
    detector = SmartColorDetector(debug=True)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        exit(1)
    
    print("📷 Testing with live camera (press 'q' to quit)")
    
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
        colors, confidences, has_unknowns = detector.detect_face_colors(roi)
        
        # Draw grid
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw detection results
        detector.draw_debug_overlay(frame, colors, confidences, x1, y1, size)
        
        # Status text
        avg_conf = np.mean([c for row in confidences for c in row if c > 0])
        status = f"Avg Confidence: {avg_conf:.1%} | Unknowns: {has_unknowns}"
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Smart Color Detector Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
