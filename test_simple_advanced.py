#!/usr/bin/env python3
"""
Test the simplified advanced scanner
"""

from simple_advanced_scanner import SimpleAdvancedScanner


def test():
    print("Testing Advanced Cube Scanner")
    print("=" * 40)
    print("Features:")
    print("- Real-time color detection")
    print("- Stability tracking")
    print("- Auto-capture mode")
    print("- Semi-transparent color overlays")
    print("- Voice feedback")
    print("")
    print("Controls:")
    print("- SPACE: Manual capture")
    print("- A: Toggle auto-capture")
    print("- Q: Quit")
    print("")

    scanner = SimpleAdvancedScanner(use_voice=True)
    scanner.current_face_index = 0  # White face

    try:
        face_data = scanner.scan_face_advanced()

        if face_data:
            print("SUCCESS! Detected colors:")
            for i, row in enumerate(face_data):
                print(f"  Row {i+1}: {row}")
        else:
            print("Test cancelled or failed")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test()
