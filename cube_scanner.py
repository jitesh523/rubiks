import cv2
import numpy as np
from collections import Counter
import time
import threading
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class CubeScanner:
    def __init__(self):
        self.cube_faces = {}
        self.face_names = ['U', 'R', 'F', 'D', 'L', 'B']
        self.face_colors = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
        self.current_face_index = 0
        
        # Load hybrid KNN model if available
        self.hybrid_model = None
        self.use_hybrid_model = False
        try:
            with open('hybrid_knn.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.hybrid_model = model_data['model']
                self.scaler = model_data['scaler']
                self.use_hybrid_model = True
                print("✅ Loaded hybrid KNN model for enhanced color detection")
        except FileNotFoundError:
            print("⚠️  Hybrid KNN model not found, falling back to HSV ranges")
        except Exception as e:
            print(f"⚠️  Error loading hybrid model: {e}, falling back to HSV ranges")
        
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
        
        # Color mapping for display (BGR format for OpenCV)
        self.color_map_bgr = {
            'White': (255, 255, 255),
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Yellow': (0, 255, 255),
            'Orange': (0, 128, 255),
            'Unknown': (50, 50, 50)
        }
    
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
    
    def extract_lab_hue_features(self, roi):
        """Extract LAB and Hue features from ROI for hybrid model"""
        try:
            # Convert to LAB
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_mean, a_mean, b_mean = np.mean(lab, axis=(0, 1))
            
            # Convert to HSV for hue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            
            return np.array([l_mean, a_mean, b_mean, h_mean])
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.array([50, 0, 0, 0])  # Default features
    
    def get_dominant_color(self, roi):
        """Get the dominant color of a region of interest using hybrid model or HSV fallback"""
        if self.use_hybrid_model and self.hybrid_model is not None:
            try:
                # Extract LAB+Hue features
                features = self.extract_lab_hue_features(roi)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Predict using hybrid model
                prediction = self.hybrid_model.predict(features_scaled)[0]
                
                # Get prediction confidence (distance to nearest neighbors)
                distances, indices = self.hybrid_model.kneighbors(features_scaled, n_neighbors=3)
                confidence = 1.0 / (1.0 + np.mean(distances))
                
                # Use hybrid prediction if confidence is reasonable
                if confidence > 0.1:  # Threshold for accepting hybrid prediction
                    return prediction
                else:
                    print(f"Low confidence ({confidence:.3f}), falling back to HSV")
                    
            except Exception as e:
                print(f"Error with hybrid model: {e}, falling back to HSV")
        
        # Fallback to original HSV-based method
        return self.get_dominant_color_hsv(roi)
    
    def get_dominant_color_hsv(self, roi):
        """Original HSV-based color detection method"""
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
    
    def display_scanned_face(self, face_colors, face_name="Scanned Face"):
        """Display scanned face with verification options"""
        cell_size = 100
        margin = 50
        grid_img = np.zeros((3*cell_size + 2*margin, 3*cell_size, 3), dtype=np.uint8)

        # Draw the 3x3 grid
        for i in range(3):
            for j in range(3):
                color = self.color_map_bgr.get(face_colors[i][j], (50, 50, 50))
                top_left = (j * cell_size, i * cell_size + margin)
                bottom_right = ((j+1) * cell_size, (i+1) * cell_size + margin)
                cv2.rectangle(grid_img, top_left, bottom_right, color, -1)
                cv2.rectangle(grid_img, top_left, bottom_right, (0, 0, 0), 2)
                
                # Add color name text
                text_pos = (j * cell_size + 10, i * cell_size + margin + 25)
                cv2.putText(grid_img, face_colors[i][j][:3].upper(), text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add title and instructions
        cv2.putText(grid_img, f"Face: {face_name}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(grid_img, "Press A to Accept, R to Retry, Any key to continue", 
                   (10, 3*cell_size + margin + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(f"Scanned Face: {face_name}", grid_img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(f"Scanned Face: {face_name}")
        
        if key == ord('r') or key == ord('R'):
            print("🔄 Retrying this face...")
            return 'retry'
        else:
            print("✅ Face accepted!")
            return 'accept'


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
        if self.use_hybrid_model:
            print("🤖 Using hybrid LAB+Hue KNN model for enhanced color detection")
        else:
            print("🔧 Using HSV range-based color detection")
        
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
        
        # Show face and allow retry
        if face_data:
            result = self.display_scanned_face(face_data, self.face_colors[self.current_face_index])
            if result == 'retry':
                return self.scan_face()  # Recursive retry
        
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
        
        # Display final cube net overview
        self.display_cube_net()
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
    
    def display_cube_net(self):
        """Display all 6 faces in a cube net layout"""
        cell_size = 60
        margin = 20
        
        # Net layout:    [U]
        #            [L][F][R][B]
        #                [D]
        net_width = 4 * cell_size * 3 + 5 * margin
        net_height = 3 * cell_size * 3 + 4 * margin
        net_img = np.zeros((net_height, net_width, 3), dtype=np.uint8)
        
        # Face positions in the net
        positions = {
            'U': (cell_size * 3 + margin, margin),
            'L': (margin, cell_size * 3 + 2 * margin),
            'F': (cell_size * 3 + 2 * margin, cell_size * 3 + 2 * margin),
            'R': (cell_size * 6 + 3 * margin, cell_size * 3 + 2 * margin),
            'B': (cell_size * 9 + 4 * margin, cell_size * 3 + 2 * margin),
            'D': (cell_size * 3 + margin, cell_size * 6 + 3 * margin)
        }
        
        for face_name, (start_x, start_y) in positions.items():
            if face_name in self.cube_faces:
                face_data = self.cube_faces[face_name]
                for i in range(3):
                    for j in range(3):
                        color = self.color_map_bgr.get(face_data[i][j], (50, 50, 50))
                        top_left = (start_x + j * cell_size, start_y + i * cell_size)
                        bottom_right = (start_x + (j+1) * cell_size, start_y + (i+1) * cell_size)
                        cv2.rectangle(net_img, top_left, bottom_right, color, -1)
                        cv2.rectangle(net_img, top_left, bottom_right, (0, 0, 0), 1)
                
                # Add face label
                label_pos = (start_x + cell_size, start_y - 10)
                cv2.putText(net_img, f"{face_name} ({self.face_colors[self.face_names.index(face_name)]})", 
                           label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(net_img, "Complete Cube Net - Press any key to continue", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Cube Net Overview", net_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Cube Net Overview")
