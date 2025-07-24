import cv2
import numpy as np
import joblib
import os
from lab_classifier import predict_color

class MLCubeScanner:
    def __init__(self):
        self.face_names = ['White (U)', 'Red (F)', 'Green (R)', 'Blue (B)', 'Yellow (D)', 'Orange (L)']
        self.current_face = 0
        self.cube_state = {}
        self.knn_model = None
        
        # Load the trained model
        if os.path.exists("knn_model.pkl"):
            self.knn_model = joblib.load("knn_model.pkl")
            print("✅ Loaded trained KNN model")
        else:
            print("❌ No trained model found! Run lab_classifier.py first!")
            
    def lab_pixel(self, frame, x, y):
        """Convert BGR pixel to LAB color space"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        return lab[y, x]
    
    def detect_face_colors_ml(self, roi):
        """Detect colors in a 3x3 grid using ML classifier"""
        if self.knn_model is None:
            return [["unknown" for _ in range(3)] for _ in range(3)]
        
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        h, w, _ = lab.shape
        cube_colors = []
        confidences = []

        for i in range(3):
            row = []
            conf_row = []
            for j in range(3):
                # Get center pixel of each sticker
                y = int((i + 0.5) * h / 3)
                x = int((j + 0.5) * w / 3)
                pixel = lab[y, x]
                
                # Predict color using ML model
                pred = self.knn_model.predict([pixel])[0]
                conf = self.knn_model.predict_proba([pixel]).max()
                
                row.append(pred)
                conf_row.append(conf)
            
            cube_colors.append(row)
            confidences.append(conf_row)
        
        return cube_colors, confidences
    
    def draw_detection_overlay(self, frame, roi, colors, confidences):
        """Draw detected colors and confidence scores"""
        h, w, _ = roi.shape
        
        for i in range(3):
            for j in range(3):
                # Calculate sticker position
                y = int((i + 0.5) * h / 3)
                x = int((j + 0.5) * w / 3)
                
                color = colors[i][j]
                conf = confidences[i][j]
                
                # Color mapping for visualization
                color_map = {
                    'white': (255, 255, 255),
                    'red': (0, 0, 255),
                    'green': (0, 255, 0),
                    'blue': (255, 0, 0),
                    'yellow': (0, 255, 255),
                    'orange': (0, 165, 255)
                }
                
                # Draw circle with detected color
                draw_color = color_map.get(color, (128, 128, 128))
                cv2.circle(roi, (x, y), 15, draw_color, -1)
                cv2.circle(roi, (x, y), 15, (0, 0, 0), 2)
                
                # Draw confidence text
                conf_text = f"{conf:.2f}"
                cv2.putText(roi, conf_text, (x-15, y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return roi
    
    def scan_face(self, frame):
        """Scan current face and return detected colors"""
        h, w, _ = frame.shape
        
        # Define ROI for cube face (center square)
        roi_size = min(h, w) // 2
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        roi = frame[y1:y2, x1:x2]
        
        # Detect colors using ML
        colors, confidences = self.detect_face_colors_ml(roi)
        
        # Draw overlay
        overlay_roi = self.draw_detection_overlay(frame.copy(), roi.copy(), colors, confidences)
        frame[y1:y2, x1:x2] = overlay_roi
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return colors, confidences
    
    def run_scanner(self):
        """Main scanning loop"""
        if self.knn_model is None:
            print("❌ Cannot run scanner without trained model!")
            print("🔧 Please run: python lab_classifier.py")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🚀 ML-Powered Cube Scanner Started!")
        print("=" * 50)
        print("Controls:")
        print("  SPACE: Capture current face")
        print("  n: Next face")
        print("  r: Reset/restart")
        print("  q: Quit")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Scan current face
            colors, confidences = self.scan_face(frame)
            
            # Display information
            face_name = self.face_names[self.current_face]
            cv2.putText(frame, f"Scanning: {face_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Face {self.current_face + 1}/6", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show average confidence
            avg_conf = np.mean([conf for row in confidences for conf in row])
            cv2.putText(frame, f"Confidence: {avg_conf:.1%}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display controls
            cv2.putText(frame, "SPACE: Capture | n: Next | r: Reset | q: Quit", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("ML Cube Scanner", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                self.cube_state[face_name] = colors
                print(f"✅ Captured {face_name}")
                print("Detected colors:")
                for row in colors:
                    print("  " + " ".join(f"{color:>6}" for color in row))
                print()
                
                # Auto-advance to next face
                if self.current_face < 5:
                    self.current_face += 1
                    print(f"➡️ Next: {self.face_names[self.current_face]}")
                else:
                    print("🎉 All faces captured! Press 'q' to finish.")
            
            elif key == ord('n'):  # Next face
                if self.current_face < 5:
                    self.current_face += 1
                    print(f"➡️ Next: {self.face_names[self.current_face]}")
            
            elif key == ord('r'):  # Reset
                self.current_face = 0
                self.cube_state = {}
                print("🔄 Reset to first face")
            
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final results
        if self.cube_state:
            print("\n🎲 Final Cube State:")
            print("=" * 30)
            for face, colors in self.cube_state.items():
                print(f"{face}:")
                for row in colors:
                    print("  " + " ".join(f"{color:>6}" for color in row))
                print()

def main():
    print("🧠 ML-Based Rubik's Cube Scanner")
    print("Using LAB color space + KNN classification")
    print("=" * 50)
    
    scanner = MLCubeScanner()
    scanner.run_scanner()

if __name__ == "__main__":
    main()
