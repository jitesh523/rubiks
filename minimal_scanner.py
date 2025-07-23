#!/usr/bin/env python3
"""
Ultra-minimal cube scanner to test basic functionality
"""

import cv2
import numpy as np

def detect_color_simple(roi):
    """Very simple color detection"""
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Get average HSV values
    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])
    
    # Simple color classification
    if v > 200 and s < 50:
        return "WHITE"
    elif 0 <= h <= 10 or 170 <= h <= 180:
        return "RED"
    elif 10 < h <= 25:
        return "ORANGE"  
    elif 25 < h <= 35:
        return "YELLOW"
    elif 35 < h <= 85:
        return "GREEN"
    elif 85 < h <= 130:
        return "BLUE"
    else:
        return "UNKNOWN"

def scan_minimal():
    """Minimal scanning function"""
    print("=== MINIMAL CUBE SCANNER ===")
    print("Controls:")
    print("- SPACE: Capture current view")
    print("- Q: Quit")
    print()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Define center region
        size = 200
        x = w // 2 - size // 2
        y = h // 2 - size // 2
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), 2)
        
        # Extract center region
        roi = frame[y:y+size, x:x+size]
        
        # Detect color
        color = detect_color_simple(roi)
        
        # Display info
        cv2.putText(frame, "Minimal Scanner", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Detected: {color}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | Q: Quit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Minimal Scanner', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            print(f"CAPTURED: {color}")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Scanner closed.")

if __name__ == "__main__":
    scan_minimal()
