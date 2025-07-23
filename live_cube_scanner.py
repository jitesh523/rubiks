#!/usr/bin/env python3
"""
Live Cube Scanner with Real-time AR-style Interface
==================================================

Dynamic real-time cube scanner with:
✅ Live Preview Cubelet Grid - Realtime detection of 9 cells
✅ Continuous Color Feedback - Colors update in realtime
✅ Auto Capture Mode - Captures when stable for 1-2 seconds
⚠️ Color Validation - No unknowns allowed before capturing
🎤 Voice Feedback - Announces face progress
📷 Bigger ROI - Large, clean scanning grid
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from enhanced_color_detector import detect_cube_face_colors, get_display_color, validate_face_colors, stabilize_colors
from enhanced_solver import solve_cube_string, get_move_explanation

class LiveCubeScanner:
    def __init__(self, use_voice=True):
        self.use_voice = use_voice
        self.tts_engine = None
        
        # Face information
        self.face_labels = ['White (U)', 'Red (R)', 'Green (F)', 'Yellow (D)', 'Orange (L)', 'Blue (B)']
        self.captured_faces = []
        
        # Live detection state
        self.recent_detections = deque(maxlen=15)
        self.stability_counter = 0
        self.auto_capture_delay = 1.5  # seconds
        self.last_stable_time = None
        
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
    
    def draw_live_grid(self, frame, start_x, start_y, grid_size, colors, is_stable, stability_progress):
        """Draw real-time 3x3 grid with color feedback"""
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
        
        # Fill cells with detected colors
        for i in range(3):
            for j in range(3):
                if colors and i < len(colors) and j < len(colors[i]):
                    color = colors[i][j]
                    display_color = get_display_color(color)
                    
                    # Cell coordinates
                    x1 = start_x + j * cell_size + 4
                    y1 = start_y + i * cell_size + 4
                    x2 = start_x + (j + 1) * cell_size - 4
                    y2 = start_y + (i + 1) * cell_size - 4
                    
                    # Semi-transparent color overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), display_color, -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    
                    # Color text
                    text_color = (0, 0, 0) if color == "white" else (255, 255, 255)
                    text = color.upper()[:3] if color != "?" else "???"
                    
                    # Center text in cell
                    text_x = x1 + (cell_size - 40) // 2
                    text_y = y1 + (cell_size + 10) // 2
                    
                    cv2.putText(frame, text, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    def draw_ui_overlay(self, frame, face_idx, is_stable, stability_progress, message, colors=None):
        """Draw comprehensive UI overlay"""
        h, w = frame.shape[:2]
        
        # Dark overlay at top
        cv2.rectangle(frame, (0, 0), (w, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 140), (50, 50, 50), 3)
        
        # Title
        title = f"🎲 Live Cube Scanner - Face {face_idx + 1}/6"
        cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Current face
        face_text = f"Show: {self.face_labels[face_idx]}"
        cv2.putText(frame, face_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Stability indicator
        stability_color = (0, 255, 0) if is_stable else (0, 255, 255) if stability_progress > 0.5 else (255, 255, 255)
        stability_text = f"Stability: {stability_progress:.0%}"
        cv2.putText(frame, stability_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
        
        # Progress bar
        bar_width = 300
        bar_x = 350
        bar_y = 90
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * stability_progress), bar_y + 15), stability_color, -1)
        
        # Status message
        if message:
            cv2.putText(frame, message, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions at bottom
        instructions = [
            "Hold cube face steady in center grid",
            "SPACE: Manual capture | A: Toggle auto-capture | Q: Quit",
            f"Auto-capture: {'ON' if hasattr(self, 'auto_capture') and self.auto_capture else 'ON'}"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 80 + i * 25
            cv2.putText(frame, instruction, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Face progress indicator
        progress_text = f"Progress: {len(self.captured_faces)}/6 faces captured"
        cv2.putText(frame, progress_text, (w - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def scan_face(self, face_idx):
        """Scan a single face with live preview and auto-capture"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None
        
        print(f"\\n📷 Show {self.face_labels[face_idx]} face... hold steady!")
        self.speak_async(f"Show me the {self.face_labels[face_idx].split()[0]} face")
        
        # Reset detection state
        self.recent_detections.clear()
        self.stability_counter = 0
        self.last_stable_time = None
        face_captured = False
        captured_colors = None
        auto_capture = True
        
        while not face_captured:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read from camera")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Larger ROI for better detection
            grid_size = 280  # Bigger grid as requested
            start_x = w // 2 - grid_size // 2
            start_y = h // 2 - grid_size // 2
            roi = frame[start_y:start_y + grid_size, start_x:start_x + grid_size]
            
            # Detect colors in real-time
            colors, has_unknowns = detect_cube_face_colors(roi)
            
            # Add to recent detections if valid
            if not has_unknowns:
                self.recent_detections.append(colors)
                
                # Check for stability
                stable_colors, is_stable = stabilize_colors(self.recent_detections)
                
                if is_stable:
                    if self.last_stable_time is None:
                        self.last_stable_time = time.time()
                    
                    # Auto-capture after delay
                    stable_duration = time.time() - self.last_stable_time
                    if auto_capture and stable_duration >= self.auto_capture_delay:
                        # Validate colors before capturing
                        is_valid, validation_msg = validate_face_colors(stable_colors)
                        
                        if is_valid:
                            captured_colors = stable_colors
                            face_captured = True
                            print(f"✅ {self.face_labels[face_idx]} auto-captured!")
                            self.speak_async("Face captured")
                            break
                        else:
                            print(f"⚠️ {validation_msg}")
                            self.last_stable_time = time.time()  # Reset timer
                else:
                    self.last_stable_time = None
                    
                # Calculate stability progress
                stability_progress = min(len(self.recent_detections) / 10, 1.0)
            else:
                # Reset if unknowns detected
                self.recent_detections.clear()
                self.last_stable_time = None
                stability_progress = 0
                stable_colors = colors
                is_stable = False
            
            # Determine status message
            if has_unknowns:
                message = "⚠️ Unknown colors - improve lighting"
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
            
            # Draw live grid and UI
            self.draw_live_grid(frame, start_x, start_y, grid_size, stable_colors, is_stable, stability_progress)
            self.draw_ui_overlay(frame, face_idx, is_stable, stability_progress, message, stable_colors)
            
            # Show frame
            cv2.imshow("Live Cube Scanner", frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Manual capture
                if not has_unknowns:
                    is_valid, validation_msg = validate_face_colors(colors)
                    if is_valid:
                        captured_colors = colors
                        face_captured = True
                        print(f"✅ {self.face_labels[face_idx]} manually captured!")
                        self.speak_async("Captured")
                    else:
                        print(f"⚠️ {validation_msg}")
                else:
                    print("⚠️ Cannot capture - unknown colors detected")
                    
            elif key == ord('a'):  # Toggle auto-capture
                auto_capture = not auto_capture
                status = "enabled" if auto_capture else "disabled"
                print(f"🔄 Auto-capture {status}")
                self.speak_async(f"Auto capture {status}")
                
            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        return captured_colors
    
    def run_scanner(self):
        """Run the complete 6-face scanning process"""
        print("🎲 === Live Cube Scanner ===")
        print("🚀 Features: Real-time detection, Auto-capture, Voice feedback")
        print("\\n🎯 You will scan all 6 faces in order:")
        for i, label in enumerate(self.face_labels):
            print(f"  {i+1}. {label}")
        
        print("\\n💡 Controls:")
        print("  • SPACE: Manual capture")
        print("  • A: Toggle auto-capture") 
        print("  • Q: Quit scanner")
        
        self.speak_async("Live cube scanner ready. Let's begin with the white face.")
        
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
        
        print("\\n🎉 All faces scanned successfully!")
        self.speak_async("All faces captured. Processing cube.")
        
        # Solve the cube
        self.solve_and_display()
    
    def solve_and_display(self):
        """Solve the cube and display solution"""
        print("\\n🧠 Solving cube...")
        solution, error = solve_cube_string(self.captured_faces)
        
        if error:
            print(f"❌ Error: {error}")
            self.speak_async("Sorry, I couldn't solve the cube. Please try scanning again.")
        else:
            print("\\n✅ Solution Steps:")
            for i, step in enumerate(solution, 1):
                explanation = get_move_explanation(step)
                print(f"{i:2d}. {step:3s} - {explanation}")
            
            print(f"\\n🎯 Total moves: {len(solution)}")
            self.speak_async(f"Solution found with {len(solution)} moves. Check the terminal for step by step instructions.")

def main():
    """Main entry point"""
    try:
        scanner = LiveCubeScanner(use_voice=True)
        scanner.run_scanner()
    except KeyboardInterrupt:
        print("\\n\\n👋 Scanner interrupted. Goodbye!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
