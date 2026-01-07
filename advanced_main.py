#!/usr/bin/env python3
"""
Advanced Rubik's Cube Scanner with Real-time Color Detection
==========================================================

Features:
✅ Live preview with real-time color detection
✅ Auto-capture when face is stable
✅ Color validation and stability checking
✅ Professional AR-style interface
✅ Voice guidance and feedback
✅ Manual and automatic capture modes

Controls:
- SPACE: Manual capture
- A: Toggle auto-capture mode
- Q: Quit scanner

Author: AI Assistant
"""

from advanced_scanner import AdvancedCubeScanner
from cube_solver import CubeSolver
from move_tracker import MoveTracker


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🎲 ADVANCED RUBIK'S CUBE SCANNER WITH REAL-TIME DETECTION 🎲")
    print("=" * 70)
    print("🚀 NEW FEATURES:")
    print("✅ Live color preview in real-time")
    print("✅ Auto-capture when cube is stable")
    print("✅ Color validation and error detection")
    print("✅ Professional AR-style interface")
    print("✅ Voice guidance throughout process")
    print("✅ Manual and automatic capture modes")
    print("=" * 70)


def main():
    """Main application"""
    print_banner()

    print("\\n🎯 Choose scanning mode:")
    print("1. 🤖 Auto-capture mode (Recommended)")
    print("2. 📸 Manual capture mode")
    print("3. 🧪 Test single face")
    print("4. 🚪 Exit")

    choice = input("\\nSelect option (1-4): ").strip()

    if choice == "1":
        # Auto-capture mode
        print("\\n🤖 Starting Advanced Scanner in AUTO-CAPTURE mode...")
        scanner = AdvancedCubeScanner(use_voice=True)
        scanner.auto_capture_enabled = True

        # Scan all faces
        cube_faces = scanner.scan_all_faces_advanced()

        if cube_faces:
            print("\\n🔄 Converting to solver format...")
            cube_string = scanner.convert_to_kociemba_format(cube_faces)
            print(f"Cube string: {cube_string}")

            # Solve cube
            print("\\n🧠 Solving cube...")
            from solver import CubeSolver as SimpleSolver

            solver = SimpleSolver()
            moves, error = solver.solve(cube_string)

            if moves:
                success, result = True, moves
            else:
                success, result = False, error

            if success:
                if result:
                    solver.display_solution_summary()

                    # Offer guidance
                    print("\\n🎯 Would you like step-by-step guidance?")
                    guide_choice = input("(y/n): ").strip().lower()

                    if guide_choice == "y":
                        tracker = MoveTracker(use_voice=True)
                        tracker.guide_through_solution(solver)
                else:
                    print("🎉 Your cube is already solved!")
            else:
                print(f"❌ Solving failed: {result}")

    elif choice == "2":
        # Manual capture mode
        print("\\n📸 Starting Advanced Scanner in MANUAL CAPTURE mode...")
        scanner = AdvancedCubeScanner(use_voice=True)
        scanner.auto_capture_enabled = False

        cube_faces = scanner.scan_all_faces_advanced()

        if cube_faces:
            print("\\n🔄 Converting to solver format...")
            cube_string = scanner.convert_to_kociemba_format(cube_faces)
            print(f"Cube string: {cube_string}")

            # Solve cube
            print("\\n🧠 Solving cube...")
            solver = CubeSolver()
            success, result = solver.solve_cube(cube_string)

            if success and result:
                solver.display_solution_summary()
            elif success:
                print("🎉 Your cube is already solved!")
            else:
                print(f"❌ Solving failed: {result}")

    elif choice == "3":
        # Test single face
        print("\\n🧪 Testing single face detection...")
        scanner = AdvancedCubeScanner(use_voice=True)
        scanner.current_face_index = 0  # White face

        face_data = scanner.scan_face_advanced()

        if face_data:
            print(f"✅ Test successful! Detected: {face_data}")
        else:
            print("❌ Test cancelled or failed")

    elif choice == "4":
        print("\\n👋 Goodbye!")
        return

    else:
        print("❌ Invalid choice")
        return main()  # Show menu again


def demo_advanced_features():
    """Demo the advanced features"""
    print("\\n🎬 ADVANCED FEATURES DEMO")
    print("=" * 40)
    print("🔍 Live Color Detection:")
    print("  • Real-time HSV color analysis")
    print("  • 3x3 grid cell-by-cell detection")
    print("  • Semi-transparent color overlay")

    print("\\n🎯 Stability Tracking:")
    print("  • Tracks last 10 color detections")
    print("  • Requires 8 consistent frames")
    print("  • Visual stability progress bar")

    print("\\n🤖 Auto-Capture:")
    print("  • Automatically captures stable faces")
    print("  • 1-second delay before capture")
    print("  • Voice confirmation feedback")

    print("\\n🎨 Enhanced UI:")
    print("  • Color-coded grid (white → yellow → green)")
    print("  • Real-time status messages")
    print("  • Progress indicators")

    print("\\n🔊 Voice Guidance:")
    print("  • Announces each face to show")
    print("  • Confirms successful captures")
    print("  • Reports auto-capture status")


if __name__ == "__main__":
    try:
        # Check if camera is available
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  Warning: No camera detected!")
            print("Please ensure your camera is connected and not in use by another app.")
            exit(1)
        cap.release()

        main()

    except KeyboardInterrupt:
        print("\\n\\n👋 Scanner interrupted. Goodbye!")
    except Exception as e:
        print(f"\\n❌ An error occurred: {e}")
        print("Please ensure all dependencies are installed and camera is available.")
