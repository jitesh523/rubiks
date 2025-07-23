#!/usr/bin/env python3
"""
Quick test script for the improved cube scanner
"""

from cube_scanner import CubeScanner

def test_single_face():
    """Test scanning just one face"""
    print("🧪 Testing single face scan...")
    scanner = CubeScanner()
    scanner.current_face_index = 0  # Test with White face
    
    face_data = scanner.scan_face()
    
    if face_data:
        print(f"✅ Successfully scanned face: {face_data}")
    else:
        print("❌ Scan cancelled or failed")

def test_improved_workflow():
    """Test the new improved workflow"""
    print("🧪 Testing improved SPACE-only workflow...")
    print("Instructions:")
    print("1. Position a cube face (any face) in the camera")
    print("2. Press SPACE to capture")
    print("3. You should see ✅ CAPTURED! message")
    print("4. Press SPACE again to finish")
    print("5. No need to press ENTER!")
    
    test_single_face()

if __name__ == "__main__":
    test_improved_workflow()
