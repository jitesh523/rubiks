import threading
import time
from collections import deque

import cv2

from color_detector import ColorDetector


class SimpleAdvancedScanner:
    def __init__(self, use_voice=True):
        self.color_detector = ColorDetector()
        self.cube_faces = {}
        self.face_names = ["U", "R", "F", "D", "L", "B"]
        self.face_colors = ["White", "Red", "Green", "Yellow", "Orange", "Blue"]
        self.current_face_index = 0

        # Live detection state
        self.recent_detections = deque(maxlen=10)
        self.stability_counter = 0
        self.stable_colors = None
        self.is_stable = False
        self.auto_capture_enabled = True
        self.stability_threshold = 8

        # TTS setup
        self.use_voice = use_voice
        self.tts_engine = None
        if use_voice:
            try:
                import pyttsx3

                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", 150)
                self.tts_engine.setProperty("volume", 0.8)
            except:
                self.use_voice = False

    def speak_async(self, text):
        """Speak text in a separate thread"""
        if self.use_voice and self.tts_engine:

            def speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass

            thread = threading.Thread(target=speak, daemon=True)
            thread.start()

    def draw_enhanced_grid(self, frame, roi_coords, colors, is_stable, stability_progress):
        """Draw an enhanced 3x3 grid with real-time color feedback"""
        x, y, w, h = roi_coords
        cell_w = w // 3
        cell_h = h // 3

        # Grid colors based on stability
        if is_stable:
            grid_color = (0, 255, 0)  # Green when stable
            thickness = 3
        elif stability_progress > 0.5:
            grid_color = (0, 255, 255)  # Yellow when getting stable
            thickness = 2
        else:
            grid_color = (255, 255, 255)  # White when unstable
            thickness = 2

        # Draw grid lines
        for i in range(4):
            # Vertical lines
            line_x = x + i * cell_w
            cv2.line(frame, (line_x, y), (line_x, y + h), grid_color, thickness)
            # Horizontal lines
            line_y = y + i * cell_h
            cv2.line(frame, (x, line_y), (x + w, line_y), grid_color, thickness)

        # Fill cells with detected colors
        for i in range(3):
            for j in range(3):
                cell_x = x + j * cell_w
                cell_y = y + i * cell_h

                if colors and i < len(colors) and j < len(colors[i]):
                    color = colors[i][j]
                    display_color = self.color_detector.get_display_color(color)

                    # Fill cell with semi-transparent color
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (cell_x + 5, cell_y + 5),
                        (cell_x + cell_w - 5, cell_y + cell_h - 5),
                        display_color,
                        -1,
                    )
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                    # Add color text
                    text_color = (0, 0, 0) if color == "white" else (255, 255, 255)
                    text = color.upper()[:3] if color != "?" else "???"

                    # Calculate text position
                    text_x = cell_x + cell_w // 2 - 15
                    text_y = cell_y + cell_h // 2 + 5

                    cv2.putText(
                        frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
                    )

    def draw_ui_elements(
        self, frame, face_index, face_name, is_stable, stability_progress, message=""
    ):
        """Draw UI elements on the frame"""
        h, w = frame.shape[:2]

        # Background for text
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 120), (50, 50, 50), 2)

        # Title
        title = f"Advanced Cube Scanner - Face {face_index + 1}/6"
        cv2.putText(frame, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Current face
        face_text = f"Scanning: {face_name}"
        cv2.putText(frame, face_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Stability indicator
        stability_color = (
            (0, 255, 0)
            if is_stable
            else (0, 165, 255) if stability_progress > 0.5 else (255, 255, 255)
        )
        stability_text = f"Stability: {stability_progress:.0%}"
        cv2.putText(
            frame, stability_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1
        )

        # Progress bar
        bar_width = 200
        bar_x = 10
        bar_y = 85
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (100, 100, 100), -1)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + int(bar_width * stability_progress), bar_y + 10),
            stability_color,
            -1,
        )

        # Status message
        if message:
            cv2.putText(
                frame, message, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Instructions in bottom
        instructions = [
            "Hold cube steady in the grid",
            "SPACE: Manual capture | A: Toggle auto-capture | Q: Quit",
            f"Auto-capture: {'ON' if self.auto_capture_enabled else 'OFF'}",
        ]

        for i, instruction in enumerate(instructions):
            y_pos = h - 60 + i * 20
            cv2.putText(
                frame, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )

    def scan_face_advanced(self):
        """Advanced face scanning with live preview and auto-capture"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access camera")
            return None

        # Reset state
        self.recent_detections.clear()
        self.stability_counter = 0
        self.is_stable = False
        face_captured = False
        captured_colors = None

        face_name = f"{self.face_colors[self.current_face_index]} ({self.face_names[self.current_face_index]})"
        print(f"Scanning face {self.current_face_index + 1}/6: {face_name}")

        # Announce face
        self.speak_async(f"Show me the {self.face_colors[self.current_face_index]} face")

        while not face_captured:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from camera")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Define ROI
            roi_size = 240
            roi_x = w // 2 - roi_size // 2
            roi_y = h // 2 - roi_size // 2
            roi = frame[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]

            # Detect colors
            current_colors, has_unknowns = self.color_detector.detect_cube_face_colors(roi)

            # Add to recent detections
            if not has_unknowns:
                self.recent_detections.append(current_colors)

                # Check for stability
                if len(self.recent_detections) >= self.stability_threshold:
                    stable_colors, is_stable = self.color_detector.stabilize_colors(
                        self.recent_detections, self.stability_threshold
                    )

                    if is_stable and not self.is_stable:
                        # Just became stable
                        self.is_stable = True
                        self.stable_colors = stable_colors
                        self.stability_counter = self.stability_threshold

                        # Validate the face
                        is_valid, msg = self.color_detector.validate_face_colors(stable_colors)

                        if is_valid and self.auto_capture_enabled:
                            # Auto-capture after brief delay
                            print("Face stable - Auto-capturing in 1 second...")
                            self.speak_async("Face detected")
                            time.sleep(1)
                            captured_colors = stable_colors
                            face_captured = True
                            break
                    elif is_stable:
                        self.stability_counter = min(
                            self.stability_counter + 1, self.stability_threshold * 2
                        )
                    else:
                        self.is_stable = False
                        self.stability_counter = max(self.stability_counter - 1, 0)
                else:
                    self.stability_counter = len(self.recent_detections)
            else:
                # Reset if we detect unknowns
                if len(self.recent_detections) > 0:
                    self.recent_detections.clear()
                self.stability_counter = 0
                self.is_stable = False

            # Calculate stability progress
            stability_progress = min(self.stability_counter / self.stability_threshold, 1.0)

            # Choose colors to display
            display_colors = self.stable_colors if self.is_stable else current_colors

            # Draw enhanced grid
            self.draw_enhanced_grid(
                frame,
                (roi_x, roi_y, roi_size, roi_size),
                display_colors,
                self.is_stable,
                stability_progress,
            )

            # Determine status message
            message = ""
            if has_unknowns:
                message = "Improve lighting - unknown colors detected"
            elif self.is_stable:
                if self.auto_capture_enabled:
                    message = "Ready! Auto-capturing..."
                else:
                    message = "Stable - Press SPACE to capture"
            elif stability_progress > 0.5:
                message = "Stabilizing... hold steady"
            else:
                message = "Position cube face in center grid"

            # Draw UI
            self.draw_ui_elements(
                frame,
                self.current_face_index,
                face_name,
                self.is_stable,
                stability_progress,
                message,
            )

            # Show frame
            cv2.imshow("Advanced Cube Scanner", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):  # Manual capture
                if self.is_stable and self.stable_colors:
                    captured_colors = self.stable_colors
                    face_captured = True
                    print("Manual capture!")
                    self.speak_async("Captured")
                else:
                    print("Face not stable enough for capture")

            elif key == ord("a"):  # Toggle auto-capture
                self.auto_capture_enabled = not self.auto_capture_enabled
                status = "enabled" if self.auto_capture_enabled else "disabled"
                print(f"Auto-capture {status}")
                self.speak_async(f"Auto capture {status}")

            elif key == ord("q"):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return None

        cap.release()
        cv2.destroyAllWindows()

        if captured_colors:
            print(f"Captured {face_name}: {captured_colors}")
            return captured_colors
        else:
            return None
