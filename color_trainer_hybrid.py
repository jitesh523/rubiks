#!/usr/bin/env python3
"""
Hybrid Color Trainer for Rubik's Cube Scanner
Collects balanced LAB + HSV samples for each color with visual feedback
"""

import cv2
import numpy as np
import pickle
import os

class HybridColorTrainer:
    def __init__(self):
        self.samples = {
            'white': [],
            'red': [],
            'orange': [],
            'green': [],
            'blue': [],
            'yellow': []
        }
        self.color_map = {
            '1': 'white',
            '2': 'red', 
            '3': 'orange',
            '4': 'green',
            '5': 'blue',
            '6': 'yellow'
        }
        self.target_samples = 20  # Minimum samples per color
        
    def get_sample_counts(self):
        """Return current sample counts for each color"""
        return {color: len(samples) for color, samples in self.samples.items()}
        
    def draw_interface(self, frame):
        """Draw training interface with sample counts and instructions"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, "Hybrid Cube Color Trainer", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Keys: 1=White 2=Red 3=Orange 4=Green 5=Blue 6=Yellow",
            "Click on sticker center to collect sample",
            "Move cube for lighting variety! Target: 20 samples each",
            "Press 's' to save, 'q' to quit, 'c' to clear all"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (20, 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Sample counts with progress bars
        counts = self.get_sample_counts()
        y_start = 140
        
        for i, (color, count) in enumerate(counts.items()):
            x = 20 + (i % 3) * 200
            y = y_start + (i // 3) * 25
            
            # Color indicator
            color_bgr = self.get_color_bgr(color)
            cv2.rectangle(frame, (x, y-15), (x+15, y), color_bgr, -1)
            cv2.rectangle(frame, (x, y-15), (x+15, y), (255, 255, 255), 1)
            
            # Progress bar
            progress = min(count / self.target_samples, 1.0)
            bar_width = 100
            cv2.rectangle(frame, (x+20, y-12), (x+20+bar_width, y-3), (50, 50, 50), -1)
            cv2.rectangle(frame, (x+20, y-12), (x+20+int(bar_width*progress), y-3), (0, 255, 0) if progress >= 1.0 else (0, 255, 255), -1)
            
            # Count text
            status = "✓" if count >= self.target_samples else f"{count}/{self.target_samples}"
            cv2.putText(frame, f"{color}: {status}", (x+130, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def get_color_bgr(self, color_name):
        """Get BGR values for color visualization"""
        colors = {
            'white': (255, 255, 255),
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        return colors.get(color_name, (128, 128, 128))
    
    def extract_features(self, bgr_pixel):
        """Extract LAB + Hue features from BGR pixel"""
        bgr_array = np.uint8([[bgr_pixel]])
        
        # Convert to LAB
        lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)[0][0]
        
        # Convert to HSV for hue
        hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]
        
        return {
            'bgr': bgr_pixel,
            'lab': lab.tolist(),
            'hue': int(hsv[0]),
            'saturation': int(hsv[1]),
            'value': int(hsv[2])
        }
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for sample collection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param['frame']
            current_color = param['current_color']
            
            if current_color:
                # Extract pixel and features
                bgr_pixel = frame[y, x]
                features = self.extract_features(bgr_pixel)
                
                # Add to samples
                self.samples[current_color].append(features)
                
                print(f"✅ Collected {current_color} sample #{len(self.samples[current_color])}")
                print(f"   LAB: {features['lab']}, Hue: {features['hue']}°")
                
                # Visual feedback - draw circle at click point
                cv2.circle(frame, (x, y), 10, self.get_color_bgr(current_color), 2)
    
    def save_samples(self):
        """Save samples to pickle file"""
        # Convert to format compatible with existing code
        legacy_samples = {}
        for color, features_list in self.samples.items():
            legacy_samples[color] = [f['bgr'].tolist() for f in features_list]
        
        # Save both formats
        with open('lab_samples.pkl', 'wb') as f:
            pickle.dump(legacy_samples, f)
            
        with open('hybrid_samples.pkl', 'wb') as f:
            pickle.dump(self.samples, f)
            
        total_samples = sum(len(samples) for samples in self.samples.values())
        print(f"\n✅ Saved {total_samples} samples to lab_samples.pkl and hybrid_samples.pkl")
        
        # Show summary
        print("\n📊 Sample Summary:")
        for color, samples in self.samples.items():
            count = len(samples)
            status = "✅ Complete" if count >= self.target_samples else f"⚠️  Need {self.target_samples - count} more"
            print(f"  {color:>7}: {count:>2} samples {status}")
    
    def clear_samples(self):
        """Clear all collected samples"""
        for color in self.samples:
            self.samples[color] = []
        print("🗑️  Cleared all samples")
    
    def run(self):
        """Main training loop"""
        print("🎯 Hybrid Rubik's Cube Color Trainer")
        print("=" * 50)
        print("Instructions:")
        print("1. Position cube with good lighting (natural light preferred)")
        print("2. Press number keys (1-6) to select color mode")
        print("3. Click on sticker centers to collect samples")
        print("4. Move cube between samples for variety!")
        print("5. Collect at least 20 samples per color")
        print("6. Press 's' to save when done")
        print()
        
        # Load existing samples if available
        if os.path.exists('hybrid_samples.pkl'):
            try:
                with open('hybrid_samples.pkl', 'rb') as f:
                    self.samples = pickle.load(f)
                total = sum(len(samples) for samples in self.samples.values())
                print(f"📂 Loaded {total} existing samples")
            except:
                print("⚠️  Could not load existing samples, starting fresh")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        current_color = None
        param = {'frame': None, 'current_color': None}
        
        cv2.namedWindow('Hybrid Color Trainer')
        cv2.setMouseCallback('Hybrid Color Trainer', self.mouse_callback, param)
        
        print("🎥 Camera ready! Start collecting samples...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update callback parameters
            param['frame'] = frame.copy()
            param['current_color'] = current_color
            
            # Draw interface
            display_frame = self.draw_interface(frame)
            
            # Show current mode
            if current_color:
                mode_text = f"Mode: {current_color.upper()} (click on {current_color} stickers)"
                cv2.putText(display_frame, mode_text, (20, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.get_color_bgr(current_color), 2)
            
            cv2.imshow('Hybrid Color Trainer', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Color selection
            if chr(key) in self.color_map:
                current_color = self.color_map[chr(key)]
                print(f"🎯 Selected {current_color} mode - click on {current_color} stickers")
            
            # Commands
            elif key == ord('s'):
                self.save_samples()
            elif key == ord('c'):
                self.clear_samples()
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        total_samples = sum(len(samples) for samples in self.samples.values())
        if total_samples > 0:
            print(f"\n🎉 Training session complete! Collected {total_samples} total samples")
            print("Next step: Run 'python train_hybrid_knn.py' to train your model")
        else:
            print("\n⚠️  No samples collected")

if __name__ == "__main__":
    trainer = HybridColorTrainer()
    trainer.run()
