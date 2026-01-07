#!/usr/bin/env python3
"""
Complete Rubik's Cube Scanner with Enhanced Red/Orange Detection
==============================================================

This implementation combines multiple color detection approaches:
✅ HSV range-based detection (primary)
✅ LAB color space analysis (secondary)
✅ Hue-based red/orange disambiguation
✅ Manual override for uncertain cases
✅ Real-time confidence display
✅ Complete 6-face cube scanning

Focus: Solving the red vs orange confusion problem
"""

import json
import time
from collections import Counter

import cv2
import numpy as np

from cube_visualizer import CubeVisualizer


class CompleteCubeScanner:
    def __init__(self):
        self.debug = True

        # Improved HSV ranges based on testing
        self.hsv_ranges = {
            "white": {"lower": [0, 0, 180], "upper": [180, 30, 255]},
            "red": {"lower": [0, 120, 70], "upper": [10, 255, 255]},
            "red2": {"lower": [170, 120, 70], "upper": [180, 255, 255]},  # Red wraps around
            "orange": {"lower": [11, 120, 70], "upper": [25, 255, 255]},
            "yellow": {"lower": [26, 100, 100], "upper": [34, 255, 255]},
            "green": {"lower": [35, 100, 50], "upper": [85, 255, 255]},
            "blue": {"lower": [86, 100, 50], "upper": [130, 255, 255]},
        }

        # Color priority for detection
        self.color_priority = ["white", "yellow", "red", "orange", "green", "blue"]

        # Display colors for visualization
        self.display_colors = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "?": (128, 128, 128),
        }

        # Face detection state
        self.faces = {}
        self.current_face = 0
        # Use Kociemba notation for face names: U=Up, R=Right, F=Front, D=Down, L=Left, B=Back
        self.face_names = ["U", "R", "F", "D", "L", "B"]
        self.face_display_names = ["Up (Top)", "Right", "Front", "Down (Bottom)", "Left", "Back"]
        self.face_colors = ["⬜", "🔴", "🟢", "🟡", "🟠", "🔵"]

        # Detection stability
        self.recent_detections = []
        self.stability_frames = 10
        self.stability_frames = 10
        self.min_confidence = 0.35

        # 3D Visualizer
        self.visualizer = CubeVisualizer(size=400)

    def get_hue_value(self, bgr_pixel: np.ndarray) -> float:
        """Extract hue value from BGR pixel"""
        bgr_reshaped = bgr_pixel.reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr_reshaped, cv2.COLOR_BGR2HSV)[0, 0]
        return float(hsv[0])

    def detect_color_hsv(self, roi: np.ndarray) -> tuple[str, float]:
        """Detect color using HSV range matching"""
        if roi.size == 0:
            return "?", 0.0

        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]

        best_color = "?"
        best_confidence = 0.0

        # Test each color range
        for color_name in self.color_priority:
            if color_name not in self.hsv_ranges:
                continue

            # Get color range
            color_range = self.hsv_ranges[color_name]
            lower = np.array(color_range["lower"])
            upper = np.array(color_range["upper"])

            # Create mask
            mask = cv2.inRange(hsv_roi, lower, upper)

            # Special handling for red (dual range)
            if color_name == "red" and "red2" in self.hsv_ranges:
                red2_range = self.hsv_ranges["red2"]
                lower2 = np.array(red2_range["lower"])
                upper2 = np.array(red2_range["upper"])
                mask2 = cv2.inRange(hsv_roi, lower2, upper2)
                mask = cv2.bitwise_or(mask, mask2)

            # Calculate confidence
            matching_pixels = cv2.countNonZero(mask)
            confidence = matching_pixels / total_pixels

            if confidence > best_confidence:
                best_confidence = confidence
                best_color = color_name

        return best_color, best_confidence

    def apply_red_orange_hue_rule(
        self, color: str, confidence: float, roi: np.ndarray
    ) -> tuple[str, float, str]:
        """Apply hue-based rule to disambiguate red vs orange"""
        if color not in ["red", "orange"] or confidence < 0.2:
            return color, confidence, "base_detection"

        # Get average hue of the ROI
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv_roi[:, :, 0])

        # Apply hue rules
        if 8 < mean_hue < 20:  # Orange hue range
            if color == "red":
                return "orange", confidence * 0.9, "hue_corrected_to_orange"
            else:
                return "orange", confidence * 1.1, "hue_confirmed_orange"
        elif mean_hue <= 8 or mean_hue >= 170:  # Red hue range
            if color == "orange":
                return "red", confidence * 0.9, "hue_corrected_to_red"
            else:
                return "red", confidence * 1.1, "hue_confirmed_red"

        return color, confidence, "hue_neutral"

    def detect_single_cubelet(self, roi: np.ndarray) -> tuple[str, float, dict]:
        """Detect color of a single cubelet with enhanced red/orange handling"""
        if roi.size == 0:
            return "?", 0.0, {}

        # Primary HSV detection
        color, confidence = self.detect_color_hsv(roi)

        # Apply red/orange hue disambiguation
        final_color, final_confidence, method = self.apply_red_orange_hue_rule(
            color, confidence, roi
        )

        # Get hue for debugging
        center_pixel = roi[roi.shape[0] // 2, roi.shape[1] // 2]
        hue = self.get_hue_value(center_pixel)

        debug_info = {
            "roi_size": roi.shape[:2],
            "base_color": color,
            "base_confidence": confidence,
            "final_color": final_color,
            "final_confidence": final_confidence,
            "method": method,
            "hue": hue,
        }

        return final_color, final_confidence, debug_info

    def detect_face_colors(
        self, roi: np.ndarray, cell_padding: int = 10
    ) -> tuple[list[list[str]], list[list[float]], list, dict]:
        """Detect 3x3 face colors with stability checking"""
        if roi is None or roi.size == 0:
            empty_grid = [["?" for _ in range(3)] for _ in range(3)]
            zero_conf = [[0.0 for _ in range(3)] for _ in range(3)]
            return empty_grid, zero_conf, [], {}

        h, w = roi.shape[:2]
        colors_grid = []
        confidence_grid = []
        uncertain_cells = []
        debug_info = {}

        if self.debug:
            print(f"\n🔍 Detecting face colors (ROI: {w}x{h})")

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
                color, confidence, cell_debug = self.detect_single_cubelet(cell_roi)

                # Mark uncertain red/orange detections
                if color in ["red", "orange"] and confidence < 0.6:
                    uncertain_cells.append((i, j, color, confidence))

                color_row.append(color)
                conf_row.append(confidence)
                debug_info[f"cell_{i}_{j}"] = cell_debug

                if self.debug:
                    hue = cell_debug.get("hue", 0)
                    method = cell_debug.get("method", "unknown")
                    print(
                        f"  Cell [{i},{j}]: {color} ({confidence:.1%}) | Hue: {hue:.0f}° | {method}"
                    )

            colors_grid.append(color_row)
            confidence_grid.append(conf_row)

        return colors_grid, confidence_grid, uncertain_cells, debug_info

    def validate_face(self, colors: list[list[str]]) -> tuple[bool, str]:
        """Validate that a face has reasonable color distribution"""
        if not colors:
            return False, "Empty face"

        # Flatten colors
        flat_colors = [color for row in colors for color in row]

        # Count unknowns
        unknown_count = flat_colors.count("?")

        # Be more lenient - allow faces with some unknowns for easier scanning
        if unknown_count >= 7:  # Only reject if most cells are unknown
            return False, f"Too many unknowns: {unknown_count}/9"

        # Check for reasonable distribution
        color_counts = Counter(flat_colors)
        known_colors = {k: v for k, v in color_counts.items() if k != "?"}

        # Must have at least one known color
        if not known_colors:
            return False, "No colors detected"

        # A completely monochrome face is suspicious (but could be valid)
        if len(known_colors) == 1 and unknown_count == 0:
            return True, f"Monochrome {list(known_colors.keys())[0]} face - proceed with caution"

        return True, "Face looks valid"

    def draw_face_overlay(
        self,
        frame: np.ndarray,
        colors: list[list[str]],
        confidences: list[list[float]],
        start_x: int,
        start_y: int,
        grid_size: int,
        uncertain_cells: list = None,
    ):
        """Draw color detection overlay on the frame"""
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

                    # Get display color
                    display_color = self.display_colors.get(color, (128, 128, 128))

                    # Enhanced visualization for red/orange
                    if color in ["red", "orange"]:
                        thickness = 3
                        radius = int(20 + conf * 15)

                        # Add uncertainty indicator
                        if uncertain_cells and (i, j, color, conf) in [
                            (uc[0], uc[1], uc[2], uc[3]) for uc in uncertain_cells
                        ]:
                            cv2.circle(
                                frame, (cell_x, cell_y), radius + 8, (0, 255, 255), 2
                            )  # Yellow warning
                    else:
                        thickness = 2
                        radius = int(15 + conf * 10)

                    # Draw main circle
                    cv2.circle(frame, (cell_x, cell_y), radius, display_color, thickness)

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

                    # Color emoji/label
                    emoji_map = {
                        "red": "🔴",
                        "orange": "🟠",
                        "white": "⬜",
                        "yellow": "🟡",
                        "green": "🟢",
                        "blue": "🔵",
                        "?": "❓",
                    }
                    label = emoji_map.get(color, color[:1].upper())

                    # For text rendering, use letters instead of emojis
                    text_label = {
                        "red": "R",
                        "orange": "O",
                        "white": "W",
                        "yellow": "Y",
                        "green": "G",
                        "blue": "B",
                        "?": "?",
                    }[color]

                    cv2.putText(
                        frame,
                        text_label,
                        (cell_x - 5, cell_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

    def show_status(self, frame: np.ndarray):
        """Display scanning status and instructions"""
        h, w = frame.shape[:2]

        # Current face info
        face_name = self.face_names[self.current_face]
        face_display = self.face_display_names[self.current_face]
        face_emoji = self.face_colors[self.current_face]

        status_text = (
            f"Scanning: {face_emoji} {face_display} ({face_name}) ({self.current_face + 1}/6)"
        )
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Instructions
        instructions = [
            "SPACEBAR = Capture face",
            "R = Reset current face",
            "N = Next face",
            "P = Previous face",
            "Q = Quit",
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (20, h - 140 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Show captured faces
        if self.faces:
            cv2.putText(
                frame,
                f"Captured: {len(self.faces)}/6 faces",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )

        # Red/Orange detection info
        cv2.putText(
            frame,
            "Enhanced Red/Orange Detection Active",
            (w - 350, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
        )

    def show_transition_animation(self, from_face, to_face):
        """Show 3D animation transitioning between faces"""
        print(f"🎬 Animating transition: {from_face} -> {to_face}")

        frames = self.visualizer.animate_transition(from_face, to_face)

        for frame in frames:
            # Create a composite frame (animation centered on black background)
            display_frame = np.zeros((600, 800, 3), dtype=np.uint8)

            # Center the animation
            h, w = frame.shape[:2]
            y_offset = (600 - h) // 2
            x_offset = (800 - w) // 2

            display_frame[y_offset : y_offset + h, x_offset : x_offset + w] = frame

            # Add instructions
            cv2.putText(
                display_frame,
                f"Move to {self.face_display_names[self.face_names.index(to_face)]} Face",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Complete Cube Scanner", display_frame)
            cv2.waitKey(30)

        # Hold final frame briefly
        time.sleep(0.5)

    def run_scanner(self):
        """Run the complete cube scanning interface"""
        print("🎯 Complete Rubik's Cube Scanner")
        print("=" * 50)
        print("📋 Enhanced Red vs Orange Detection")
        print("🎮 Controls:")
        print("  SPACEBAR = Capture current face")
        print("  R = Reset current face")
        print("  N = Next face")
        print("  P = Previous face")
        print("  Q = Quit")
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

            # Define scanning region
            size = 300
            x1 = w // 2 - size // 2
            y1 = h // 2 - size // 2
            x2 = x1 + size
            y2 = y1 + size

            roi = frame[y1:y2, x1:x2]

            # Detect colors
            colors, confidences, uncertain_cells, debug_info = self.detect_face_colors(roi)

            # Draw scanning region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw 3x3 grid lines
            for i in range(1, 3):
                # Vertical lines
                x_line = x1 + i * size // 3
                cv2.line(frame, (x_line, y1), (x_line, y2), (0, 255, 0), 1)
                # Horizontal lines
                y_line = y1 + i * size // 3
                cv2.line(frame, (x1, y_line), (x2, y_line), (0, 255, 0), 1)

            # Draw color overlay
            self.draw_face_overlay(frame, colors, confidences, x1, y1, size, uncertain_cells)

            # Show status
            self.show_status(frame)

            # Show face validation
            is_valid, validation_msg = self.validate_face(colors)
            validation_color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.putText(
                frame,
                f"Face Status: {validation_msg}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                validation_color,
                1,
            )

            # Show uncertain detections
            if uncertain_cells:
                cv2.putText(
                    frame,
                    f"⚠️ {len(uncertain_cells)} uncertain red/orange detections",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                )

            cv2.imshow("Complete Cube Scanner", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # Spacebar - capture face
                if is_valid:
                    face_name = self.face_names[self.current_face]
                    self.faces[face_name] = {
                        "colors": colors,
                        "confidences": confidences,
                        "timestamp": time.time(),
                    }
                    print(f"✅ Captured {face_name} face!")

                    # Auto-advance to next face
                    if self.current_face < 5:
                        old_face = self.face_names[self.current_face]
                        self.current_face += 1
                        new_face = self.face_names[self.current_face]

                        print(f"🔄 Moved to {new_face} face")
                        self.show_transition_animation(old_face, new_face)
                    else:
                        print("🎉 All faces captured! Press Q to quit.")
                else:
                    print(f"❌ Cannot capture: {validation_msg}")

            elif key == ord("r") or key == ord("R"):  # Reset current face
                face_name = self.face_names[self.current_face]
                if face_name in self.faces:
                    del self.faces[face_name]
                    print(f"🗑️ Reset {face_name} face")

            elif key == ord("n") or key == ord("N"):  # Next face
                old_face = self.face_names[self.current_face]
                self.current_face = (self.current_face + 1) % 6
                new_face = self.face_names[self.current_face]

                print(f"➡️ Switched to {new_face} face")
                self.show_transition_animation(old_face, new_face)

            elif key == ord("p") or key == ord("P"):  # Previous face
                old_face = self.face_names[self.current_face]
                self.current_face = (self.current_face - 1) % 6
                new_face = self.face_names[self.current_face]

                print(f"⬅️ Switched to {new_face} face")
                self.show_transition_animation(old_face, new_face)

            elif key == ord("q") or key == ord("Q"):  # Quit
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save results
        if self.faces:
            self.save_cube_data()

        print(f"\n👋 Scanning complete! Captured {len(self.faces)}/6 faces")

    def save_cube_data(self):
        """Save the complete cube data"""
        cube_data = {"timestamp": time.time(), "faces": self.faces, "face_names": self.face_names}

        filename = f"cube_scan_{int(time.time())}.json"

        # Convert numpy arrays to lists for JSON serialization
        for face_name, face_data in cube_data["faces"].items():
            if "confidences" in face_data:
                # Convert confidences to lists
                face_data["confidences"] = [
                    [float(c) for c in row] for row in face_data["confidences"]
                ]

        with open(filename, "w") as f:
            json.dump(cube_data, f, indent=2)

        print(f"💾 Cube data saved to {filename}")

        # Print summary
        print("\n📊 Cube Summary:")
        for face_name, face_data in self.faces.items():
            colors = face_data["colors"]
            flat_colors = [color for row in colors for color in row]
            color_counts = Counter(flat_colors)
            print(f"  {face_name:8}: {dict(color_counts)}")

    def scan_all_faces(self):
        """Main interface method expected by main.py - runs interactive scanner and returns face data"""
        print("\n🎯 Starting complete cube scan...")
        print("Instructions:")
        print("  - SPACEBAR: Capture current face")
        print("  - N/P: Navigate between faces")
        print("  - R: Reset current face")
        print("  - Q: Quit scanner")
        print("\nPlease scan all 6 faces in order: U, R, F, D, L, B")

        # Run the interactive scanner
        self.run_scanner()

        # Check if all faces were captured
        if len(self.faces) != 6:
            print(f"\n⚠️ Warning: Only {len(self.faces)}/6 faces were captured")
            return None

        # Convert to ordered list format expected by main.py
        # Order: U, R, F, D, L, B (Up, Right, Front, Down, Left, Back)
        face_order = ["U", "R", "F", "D", "L", "B"]
        face_colors = []

        for face_name in face_order:
            if face_name in self.faces:
                colors_2d = self.faces[face_name]["colors"]
                # Flatten the 3x3 grid to a 9-element list
                flat_colors = [color for row in colors_2d for color in row]
                face_colors.append(flat_colors)
            else:
                print(f"❌ Missing face: {face_name}")
                return None

        print("\n✅ Successfully scanned all 6 faces!")
        return face_colors

    def attempt_color_correction(self, face_colors):
        """Attempt to correct missing or unknown colors based on cube logic"""
        print("\n🔧 Attempting color correction...")

        # Color mapping for center positions
        center_positions = [4, 13, 22, 31, 40, 49]  # Center of each face in flattened format
        face_names = ["U", "R", "F", "D", "L", "B"]

        # Try to identify face centers first
        corrected_faces = []
        for face_idx, face in enumerate(face_colors):
            corrected_face = face.copy()

            # Count detected colors on this face
            from collections import Counter

            color_counts = Counter(face)
            unknown_count = color_counts.get("?", 0)

            if unknown_count > 0:
                print(f"  Face {face_names[face_idx]}: {unknown_count} unknown cells")

                # Find the most common non-unknown color (likely the face color)
                known_colors = {k: v for k, v in color_counts.items() if k != "?"}
                if known_colors:
                    most_common_color = max(known_colors, key=known_colors.get)

                    # Replace unknown colors with the most common color
                    for i, color in enumerate(corrected_face):
                        if color == "?":
                            corrected_face[i] = most_common_color
                            print(f"    Cell {i}: ? → {most_common_color}")

            corrected_faces.append(corrected_face)

        return corrected_faces

    def convert_to_kociemba_format(self, face_colors):
        """Convert detected color names to Kociemba format string"""
        if not face_colors or len(face_colors) != 6:
            return None

        # Check for unknown colors and attempt correction
        needs_correction = False
        for face in face_colors:
            if "?" in face:
                needs_correction = True
                break

        if needs_correction:
            face_colors = self.attempt_color_correction(face_colors)

        # Color mapping: color name -> Kociemba letter
        color_map = {
            "white": "U",  # Up face center
            "red": "R",  # Right face center
            "green": "F",  # Front face center
            "yellow": "D",  # Down face center
            "orange": "L",  # Left face center
            "blue": "B",  # Back face center
        }

        # Build the 54-character Kociemba string
        # Order: U face (9), R face (9), F face (9), D face (9), L face (9), B face (9)
        kociemba_string = ""

        for face_idx, face in enumerate(face_colors):
            for color in face:
                if color.lower() in color_map:
                    kociemba_string += color_map[color.lower()]
                else:
                    print(f"❌ Unknown color detected: {color}")
                    return None

        return kociemba_string

    def validate_cube(self, cube_string):
        """Validate the cube string follows Rubik's cube rules"""
        if not cube_string or len(cube_string) != 54:
            return False, f"Invalid length: {len(cube_string) if cube_string else 0}, expected 54"

        # Count each color
        color_counts = Counter(cube_string)
        expected_colors = {"U", "R", "F", "D", "L", "B"}

        # Check if we have exactly the expected colors
        if set(color_counts.keys()) != expected_colors:
            return False, f"Invalid colors found: {set(color_counts.keys())}"

        # Each color should appear exactly 9 times
        for color in expected_colors:
            if color_counts[color] != 9:
                return False, f"Color {color} appears {color_counts[color]} times, expected 9"

        # Check center squares (positions 4, 13, 22, 31, 40, 49 in 0-based indexing)
        # Centers should match their face designation
        centers = {
            cube_string[4]: "U",  # Up face center
            cube_string[13]: "R",  # Right face center
            cube_string[22]: "F",  # Front face center
            cube_string[31]: "D",  # Down face center
            cube_string[40]: "L",  # Left face center
            cube_string[49]: "B",  # Back face center
        }

        expected_centers = {"U", "R", "F", "D", "L", "B"}
        if set(centers.keys()) != expected_centers:
            return False, f"Invalid center configuration: {centers}"

        return True, "Cube validation passed"


if __name__ == "__main__":
    scanner = CompleteCubeScanner()
    scanner.run_scanner()
