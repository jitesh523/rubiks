import cv2
import numpy as np
from collections import Counter
import time
import threading

class CubeScanner:
    def __init__(self):
        self.cube_faces = {}
        self.face_names = ['U', 'R', 'F', 'D', 'L', 'B']
        self.face_colors = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
        self.current_face_index = 0
        
        # HSV color ranges for cube colors
        self.color_ranges = {
            'White': ((0, 0, 200), (180, 30, 255)),
            'Red': ((0, 120, 70), (10, 255, 255)),
            'Green': ((40, 70, 70), (80, 255, 255)),
            'Yellow': ((20, 120, 120), (30, 255, 255)),
            'Orange': ((10, 120, 120), (25, 255, 255)),
            'Blue': ((100, 120, 70), (130, 255, 255))
        }
        
        # Alternative red range (wraps around hue)
        self.red_range2 = ((170, 120, 70), (180, 255, 255))
    
    def detect_grid_squares(self, frame):
        """Detect the 3x3 grid of squares on the cube face"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for squares
        squares = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly square
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if 1000 < area < 10000:  # Filter by area
                    squares.append(approx)
        
        return squares
    
    def get_dominant_color(self, roi):
        """Get the dominant color of a region of interest"""
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Test each color range
        best_match = None
        max_pixels = 0
        
        for color_name, (lower, upper) in self.color_ranges.items():
            # Create mask
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Special case for red (check both ranges)
            if color_name == 'Red':
                mask2 = cv2.inRange(hsv, np.array(self.red_range2[0]), np.array(self.red_range2[1]))
                mask = cv2.bitwise_or(mask, mask2)
            
            # Count pixels
            pixel_count = cv2.countNonZero(mask)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_match = color_name
        
        return best_match if best_match else 'Unknown'
    
    def scan_face(self):
        """Scan a single face of the cube"""
        cap = cv2.VideoCapture(0)
        face_data = []
        scanning = True
        capture_countdown = 0
        captured = False
        
        print(f"\n📸 Scanning face {self.current_face_index + 1}/6: {self.face_colors[self.current_face_index]} ({self.face_names[self.current_face_index]})")
        print("Position the cube face in the center of the frame")
        print("Press SPACE to capture when ready, 'q' to quit")
        
        while scanning:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Draw guide overlay (3x3 grid)
            center_x, center_y = w // 2, h // 2
            grid_size = 150
            cell_size = grid_size // 3
            
            # Draw grid with different colors based on state
            grid_color = (0, 255, 0) if not captured else (0, 255, 255)  # Green -> Cyan when captured
            line_thickness = 2 if not captured else 3
            
            for i in range(4):
                # Vertical lines
                x = center_x - grid_size // 2 + i * cell_size
                cv2.line(frame, (x, center_y - grid_size // 2), 
                        (x, center_y + grid_size // 2), grid_color, line_thickness)
                # Horizontal lines
                y = center_y - grid_size // 2 + i * cell_size
                cv2.line(frame, (center_x - grid_size // 2, y), 
                        (center_x + grid_size // 2, y), grid_color, line_thickness)
            
            # Add progress indicator
            progress_text = f"Face {self.current_face_index + 1}/6: {self.face_colors[self.current_face_index]} ({self.face_names[self.current_face_index]})"
            cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show capture status
            if captured:
                cv2.putText(frame, "✅ CAPTURED! Ready for next face", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to continue", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show captured colors in grid
                if face_data:
                    for row in range(3):
                        for col in range(3):
                            color = face_data[row][col]
                            x = center_x - grid_size // 2 + col * cell_size + 10
                            y = center_y - grid_size // 2 + row * cell_size + 20
                            cv2.putText(frame, color[:3], (x, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "Press SPACE to capture", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                           
                # Show countdown if capturing
                if capture_countdown > 0:
                    cv2.putText(frame, f"Capturing... {capture_countdown}", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    capture_countdown -= 1
            
            cv2.imshow('Cube Scanner', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture or continue
                if not captured:
                    # First SPACE press - capture the face
                    print("📸 Capturing...")
                    capture_countdown = 10  # Visual countdown
                    
                    # Extract colors from the 3x3 grid
                    colors = []
                    for row in range(3):
                        color_row = []
                        for col in range(3):
                            # Calculate ROI for each cell
                            x1 = center_x - grid_size // 2 + col * cell_size + 5
                            y1 = center_y - grid_size // 2 + row * cell_size + 5
                            x2 = x1 + cell_size - 10
                            y2 = y1 + cell_size - 10
                            
                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0:
                                color = self.get_dominant_color(roi)
                                color_row.append(color)
                            else:
                                color_row.append('Unknown')
                        colors.append(color_row)
                    
                    face_data = colors
                    captured = True
                    print(f"✅ Captured {self.face_names[self.current_face_index]} face: {colors}")
                    
                    # Add TTS feedback if available
                    try:
                        import pyttsx3
                        tts = pyttsx3.init()
                        tts.say(f"Face {self.current_face_index + 1} captured successfully")
                        tts.runAndWait()
                    except:
                        pass  # TTS optional
                        
                else:
                    # Second SPACE press - confirm and move to next face
                    scanning = False
                    print(f"➡️  Moving to next face...")
                    
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        return face_data
    
    def scan_all_faces(self):
        """Scan all 6 faces of the cube"""
        print("\n🎲 === Rubik's Cube Scanner === 🎲")
        print("You will scan all 6 faces of the cube.")
        print("Face order: U(White), R(Red), F(Green), D(Yellow), L(Orange), B(Blue)")
        print("\n💡 Instructions:")
        print("  1. Position face in center of green grid")
        print("  2. Press SPACE to capture")
        print("  3. Press SPACE again to continue to next face")
        print("  4. Repeat for all 6 faces")
        print("\n🚀 Let's begin!")
        
        for i in range(6):
            self.current_face_index = i
            face_data = self.scan_face()
            
            if face_data is None:
                print("❌ Scanning cancelled.")
                return None
            
            self.cube_faces[self.face_names[i]] = face_data
            
            # Show progress
            if i < 5:
                print(f"✅ Face {i+1}/6 complete! Get ready for the next face...")
                time.sleep(1)  # Brief pause between faces
            else:
                print("🎉 All faces scanned successfully!")
        
        return self.cube_faces
    
    def convert_to_kociemba_format(self, cube_faces):
        """Convert scanned colors to Kociemba solver format"""
        # Color mapping to single letters
        color_map = {
            'White': 'U', 'Red': 'R', 'Green': 'F',
            'Yellow': 'D', 'Orange': 'L', 'Blue': 'B'
        }
        
        cube_string = ""
        
        # Order for Kociemba: U, R, F, D, L, B
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            face_data = cube_faces[face]
            for row in face_data:
                for color in row:
                    if color in color_map:
                        cube_string += color_map[color]
                    else:
                        print(f"Warning: Unknown color '{color}' detected!")
                        cube_string += 'U'  # Default to U (white)
        
        return cube_string
    
    def validate_cube(self, cube_string):
        """Basic validation of the cube string"""
        if len(cube_string) != 54:
            return False, f"Invalid cube string length: {len(cube_string)} (should be 54)"
        
        # Check each color appears exactly 9 times
        color_counts = Counter(cube_string)
        for color in ['U', 'R', 'F', 'D', 'L', 'B']:
            if color_counts[color] != 9:
                return False, f"Color {color} appears {color_counts[color]} times (should be 9)"
        
        return True, "Cube validation passed!"
