#!/usr/bin/env python3
"""
Diagnostic tool to identify issues with the cube scanner
"""

import cv2
import sys


def test_camera():
    """Test basic camera functionality"""
    print("=== CAMERA TEST ===")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ FAILED: Cannot access camera")
            print("Solutions:")
            print("  • Check if camera is connected")
            print("  • Close other apps using camera")
            print("  • Check system permissions")
            return False

        ret, frame = cap.read()
        if not ret:
            print("❌ FAILED: Cannot read from camera")
            cap.release()
            return False

        print("✅ SUCCESS: Camera is working")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")

        # Quick test - show camera for 3 seconds
        print("  Showing camera feed for 3 seconds...")
        start_time = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                "Camera Test - Press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Camera Test", frame)

            # Auto-close after 3 seconds or if Q pressed
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > 3 or cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print(f"❌ FAILED: Camera test error: {e}")
        return False


def test_imports():
    """Test if all required modules can be imported"""
    print("\n=== MODULE IMPORT TEST ===")

    modules = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("kociemba", "Kociemba Solver"),
        ("pyttsx3", "Text-to-Speech (optional)"),
        ("collections", "Python Collections"),
    ]

    all_good = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: MISSING - {e}")
            if module != "pyttsx3":  # TTS is optional
                all_good = False

    return all_good


def test_color_detection():
    """Test basic color detection functionality"""
    print("\n=== COLOR DETECTION TEST ===")

    try:
        from color_detector import ColorDetector

        detector = ColorDetector()

        # Create a test image with known colors
        test_img = cv2.imread("/dev/null")  # This will fail, but that's expected

        if test_img is None:
            # Create a synthetic test image
            import numpy as np

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img[:, :] = [255, 255, 255]  # White image

        color = detector.detect_single_color(test_img)
        print(f"✅ Color detection works: detected '{color}'")

        return True

    except Exception as e:
        print(f"❌ Color detection failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_file_structure():
    """Check if all required files exist"""
    print("\n=== FILE STRUCTURE TEST ===")

    import os

    required_files = [
        "color_detector.py",
        "cube_scanner.py",
        "cube_solver.py",
        "main.py",
        "requirements.txt",
    ]

    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}: EXISTS")
        else:
            print(f"❌ {file}: MISSING")
            all_files_exist = False

    return all_files_exist


def main():
    """Run all diagnostic tests"""
    print("🔧 RUBIK'S CUBE SCANNER DIAGNOSTICS")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Camera Access", test_camera),
        ("Color Detection", test_color_detection),
    ]

    results = {}
    for test_name, test_func in tests:
        print()
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! Scanner should work.")
    else:
        print("⚠️  SOME TESTS FAILED. Please fix the issues above.")
        print("\nCommon solutions:")
        print("• Install missing modules: pip install opencv-python kociemba numpy pyttsx3")
        print("• Check camera permissions in System Preferences")
        print("• Close other applications using the camera")
        print("• Restart your terminal/IDE")


if __name__ == "__main__":
    main()
