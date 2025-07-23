#!/usr/bin/env python3
"""
Calibrated Live Cube Scanner with Confidence Overlay
===================================================

Enhanced version with:
✅ Uses calibrated HSV ranges from cube_calibration.json
✅ Real-time confidence display for each cubelet
✅ HSV distance visualization
✅ Improved accuracy with your specific cube colors
"""

import cv2
import numpy as np
import time
import threading
import json
import os
from collections import deque
from enhanced_color_detector import detect_cube_face_colors, get_display_color, validate_face_colors, stabilize_colors, color_ranges
from enhanced_solver import solve_cube_string, get_move_explanation

class CalibratedLiveScanner:
    def __init__(self, use_voice=True):
        self.use_voice = use_voice
        self.tts_engine = None
        
        # Face information
        self.face_labels = ['White (U)', 'Red (R)', 'Green (F)', 'Yellow (D)', 'Orange (L)', 'Blue (B)']
        self.captured_faces = []
        
        # Live detection state
        self.recent_detections = deque(maxlen=15)
        self.auto_capture_delay = 1.5
        self.last_stable_time = None
        self.show_confidence = True  # New feature
        
        # Load calibration info
        self.load_calibration_info()
        
        # Initialize TTS
        if use_voice:
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.8)
            except:
                print("TTS not available, continuing without voice")
                self.use_voice = False
    
    def load_calibration_info(self):
        """Load calibration information"""
        if os.path.exists("cube_calibration.json"):
            with open("cube_calibration.json", "r") as f:
                self.calibration = json.load(f)
            print("✅ Loaded calibration data")
        else:
            print("⚠️ No calibration found - using defaults")
            self.calibration = {}
    
    def speak_async(self, text):
        """Speak text in background thread"""
        if self.use_voice and self.tts_engine:
            def speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            
            thread = threading.Thread(target=speak, daemon=True)
            thread.start()
    
    def detect_color_with_confidence(self, roi):
        """Detect color with confidence score"""
        if roi.size == 0:
            return "?", 0.0
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        best_match = "?"
        max_confidence = 0.0
        
        for color_name, (lower, upper) in color_ranges.items():
            if color_name == "red2":  # Skip secondary red range
                continue
                
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Special handling for red
            if color_name == "red" and "red2" in color_ranges:
                lower2, upper2 = color_ranges["red2"]
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask, mask2)
            
            # Calculate confidence as percentage of matching pixels
            pixel_count = cv2.countNonZero(mask)
            confidence = pixel_count / roi.size
            
            if confidence > max_confidence and confidence > 0.3:
                max_confidence = confidence
                best_match = color_name
        
        return best_match, max_confidence
    
    def detect_face_with_confidence(self, roi):
        """Detect 3x3 face colors with confidence scores"""
        if roi is None or roi.size == 0:
            return [["?" for _ in range(3)] for _ in range(3)], [[0.0 for _ in range(3)] for _ in range(3)], True
        
        h, w = roi.shape[:2]
        cube_colors = []
        confidence_matrix = []
        unknown_flag = False
        
        for i in range(3):
            color_row = []
            confidence_row = []
            for j in range(3):
                # Calculate cell boundaries
                cell_h = h // 3
                cell_w = w // 3
                
                padding = 8
                y1 = i * cell_h + padding
                y2 = (i + 1) * cell_h - padding
                x1 = j * cell_w + padding
                x2 = (j + 1) * cell_w - padding
                
                cell_roi = roi[y1:y2, x1:x2]
                
                # Detect with confidence
                color, confidence = self.detect_color_with_confidence(cell_roi)
                
                if color == "?":
                    unknown_flag = True
                
                color_row.append(color)
                confidence_row.append(confidence)
            
            cube_colors.append(color_row)
            confidence_matrix.append(confidence_row)
        
        return cube_colors, confidence_matrix, unknown_flag
    
    def draw_enhanced_grid(self, frame, start_x, start_y, grid_size, colors, confidence_matrix, is_stable, stability_progress):
        """Draw grid with confidence overlay"""
        cell_size = grid_size // 3
        
        # Grid color based on stability
        if is_stable:
            grid_color = (0, 255, 0)  # Green when stable
            thickness = 4
        elif stability_progress > 0.5:
            grid_color = (0, 255, 255)  # Yellow when getting stable
            thickness = 3
        else:
            grid_color = (255, 255, 255)  # White when unstable
            thickness = 2
        
        # Draw grid lines
        for i in range(4):
            # Vertical lines
            x = start_x + i * cell_size
            cv2.line(frame, (x, start_y), (x, start_y + grid_size), grid_color, thickness)
            # Horizontal lines
            y = start_y + i * cell_size
            cv2.line(frame, (start_x, y), (start_x + grid_size, y), grid_color, thickness)
        
        # Fill cells with detected colors and confidence
        for i in range(3):
            for j in range(3):
                if colors and i < len(colors) and j < len(colors[i]):
                    color = colors[i][j]
                    confidence = confidence_matrix[i][j] if confidence_matrix else 0.0
                    
                    display_color = get_display_color(color)
                    
                    # Cell coordinates
                    x1 = start_x + j * cell_size + 4
                    y1 = start_y + i * cell_size + 4
                    x2 = start_x + (j + 1) * cell_size - 4
                    y2 = start_y + (i + 1) * cell_size - 4
                    
                    # Color overlay with confidence-based transparency
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), display_color, -1)
                    alpha = 0.2 + (confidence * 0.3)  # 0.2 to 0.5 based on confidence
                    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
                    
                    # Color text
                    text_color = (0, 0, 0) if color == "white" else (255, 255, 255)
                    text = color.upper()[:3] if color != "?" else "???"
                    
                    # Center text in cell
                    text_x = x1 + (cell_size - 60) // 2
                    text_y = y1 + (cell_size - 10) // 2
                    
                    cv2.putText(frame, text, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    # Confidence display (optional)
                    if self.show_confidence and confidence > 0:
                        conf_text = f"{confidence:.0%}"
                        conf_y = text_y + 20
                        cv2.putText(frame, conf_text, (text_x + 5, conf_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    def draw_ui_overlay(self, frame, face_idx, is_stable, stability_progress, message, avg_confidence=0.0):
        """Draw comprehensive UI with confidence info"""
        h, w = frame.shape[:2]
        
        # Dark overlay at top
        cv2.rectangle(frame, (0, 0), (w, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 160), (50, 50, 50), 3)
        
        # Title
        title = f"🎲 Calibrated Live Scanner - Face {face_idx + 1}/6"
        cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Current face
        face_text = f"Show: {self.face_labels[face_idx]}"
        cv2.putText(frame, face_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Stability and confidence
        stability_color = (0, 255, 0) if is_stable else (0, 255, 255) if stability_progress > 0.5 else (255, 255, 255)
        stability_text = f"Stability: {stability_progress:.0%} | Avg Confidence: {avg_confidence:.0%}"
        cv2.putText(frame, stability_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1)
        
        # Progress bar
        bar_width = 300
        bar_x = 350
        bar_y = 90
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * stability_progress), bar_y + 15), stability_color, -1)
        
        # Status message
        if message:
            cv2.putText(frame, message, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calibration status
        calib_status = "✅ Using calibrated HSV" if self.calibration else "⚠️ Using defaults"
        cv2.putText(frame, calib_status, (w - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Instructions at bottom
        instructions = [
            "Hold cube face steady in center grid",
            "SPACE: Manual | A: Auto-capture | C: Toggle confidence | Q: Quit",
            f"Progress: {len(self.captured_faces)}/6 faces captured"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 80 + i * 25
            cv2.putText(frame, instruction, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def scan_face(self, face_idx):
        """Scan face with calibrated detection and confidence display"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None
        
        print(f"\\n📷 Scanning {self.face_labels[face_idx]} with calibrated detection")
        self.speak_async(f"Show me the {self.face_labels[face_idx].split()[0]} face")
        
        # Reset state
        self.recent_detections.clear()
        self.last_stable_time = None
        face_captured = False
        captured_colors = None
        auto_capture = True
        
        while not face_captured:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Larger ROI
            grid_size = 300  # Even bigger for better accuracy
            start_x = w // 2 - grid_size // 2
            start_y = h // 2 - grid_size // 2
            roi = frame[start_y:start_y + grid_size, start_x:start_x + grid_size]
            
            # Detect with confidence
            colors, confidence_matrix, has_unknowns = self.detect_face_with_confidence(roi)
            
            # Calculate average confidence
            if confidence_matrix:
                flat_conf = [conf for row in confidence_matrix for conf in row if conf > 0]
                avg_confidence = np.mean(flat_conf) if flat_conf else 0.0
            else:
                avg_confidence = 0.0
            
            # Add to recent detections if valid
            if not has_unknowns and avg_confidence > 0.5:
                self.recent_detections.append(colors)
                
                # Check stability
                stable_colors, is_stable = stabilize_colors(self.recent_detections)
                
                if is_stable:
                    if self.last_stable_time is None:
                        self.last_stable_time = time.time()
                    
                    # Auto-capture logic
                    stable_duration = time.time() - self.last_stable_time
                    if auto_capture and stable_duration >= self.auto_capture_delay:
                        is_valid, validation_msg = validate_face_colors(stable_colors)
                        
                        if is_valid:
                            captured_colors = stable_colors
                            face_captured = True
                            print(f"✅ {self.face_labels[face_idx]} auto-captured! (Confidence: {avg_confidence:.0%})")
                            self.speak_async("Face captured")
                            break
                        else:
                            self.last_stable_time = time.time()
                else:
                    self.last_stable_time = None
                
                stability_progress = min(len(self.recent_detections) / 10, 1.0)
            else:
                self.recent_detections.clear()
                self.last_stable_time = None
                stability_progress = 0
                stable_colors = colors
                is_stable = False
            
            # Status message
            if has_unknowns:
                message = "⚠️ Unknown colors - adjust lighting/position"
            elif avg_confidence < 0.5:
                message = f"⚠️ Low confidence ({avg_confidence:.0%}) - move closer/improve lighting"
            elif is_stable:
                if auto_capture:
                    remaining = max(0, self.auto_capture_delay - (time.time() - (self.last_stable_time or time.time())))
                    message = f"✅ Stable! Auto-capturing in {remaining:.1f}s"
                else:
                    message = "✅ Stable - Press SPACE to capture"
            elif stability_progress > 0.5:
                message = "🔄 Stabilizing... hold steady"
            else:
                message = "📱 Position cube face in center grid"
            
            # Draw interface
            self.draw_enhanced_grid(frame, start_x, start_y, grid_size, stable_colors, confidence_matrix, is_stable, stability_progress)
            self.draw_ui_overlay(frame, face_idx, is_stable, stability_progress, message, avg_confidence)
            
            cv2.imshow("Calibrated Live Scanner", frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Manual capture
                if not has_unknowns and avg_confidence > 0.5:
                    is_valid, validation_msg = validate_face_colors(colors)
                    if is_valid:
                        captured_colors = colors
                        face_captured = True
                        print(f"✅ {self.face_labels[face_idx]} manually captured!")
                        self.speak_async("Captured")
                    else:
                        print(f"⚠️ {validation_msg}")
                else:
                    print("⚠️ Cannot capture - low quality detection")
                    
            elif key == ord('a'):  # Toggle auto-capture
                auto_capture = not auto_capture
                status = "enabled" if auto_capture else "disabled"
                print(f"🔄 Auto-capture {status}")
                
            elif key == ord('c'):  # Toggle confidence display
                self.show_confidence = not self.show_confidence
                print(f"🔄 Confidence display {'enabled' if self.show_confidence else 'disabled'}")
                
            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        return captured_colors
    
    def run_scanner(self):
        """Run the complete calibrated scanning process"""
        print("🎲 === Calibrated Live Cube Scanner ===")
        print("🎯 Using your calibrated HSV color ranges!")
        print("🚀 Features: Calibrated detection, Confidence display, Auto-capture")
        
        self.speak_async("Calibrated cube scanner ready. Starting with white face.")
        
        # Scan all faces
        for face_idx in range(6):
            colors = self.scan_face(face_idx)
            
            if colors is None:
                print("❌ Scanning cancelled")
                return
            
            self.captured_faces.append(colors)
            print(f"Face {face_idx + 1}/6 captured: {colors}")
            
            if face_idx < 5:
                next_face = self.face_labels[face_idx + 1].split()[0]
                print(f"✅ Ready for next face: {next_face}")
                self.speak_async(f"Great! Now show me the {next_face} face")
                time.sleep(1)
        
        print("\\n🎉 All faces scanned with calibrated detection!")
        self.speak_async("All faces captured. Solving cube.")
        
        # Solve the cube
        self.solve_and_display()
    
    def solve_and_display(self):
        """Solve and display solution"""
        print("\\n🧠 Solving cube...")
        solution, error = solve_cube_string(self.captured_faces)
        
        if error:
            print(f"❌ Error: {error}")
            self.speak_async("Could not solve cube. Try rescanning with better lighting.")
        else:
            print("\\n✅ Solution Steps:")
            for i, step in enumerate(solution, 1):
                explanation = get_move_explanation(step)
                print(f"{i:2d}. {step:3s} - {explanation}")
            
            print(f"\\n🎯 Total moves: {len(solution)}")
            self.speak_async(f"Perfect! Solution found with {len(solution)} moves.")

def main():
    """Main entry point"""
    try:
        scanner = CalibratedLiveScanner(use_voice=True)
        scanner.run_scanner()
    except KeyboardInterrupt:
        print("\\n\\n👋 Scanner interrupted. Goodbye!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
