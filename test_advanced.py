#!/usr/bin/env python3
"""
Simple test for the advanced scanner features
"""

from advanced_scanner import AdvancedCubeScanner


def test_single_face():
    """Test the advanced single face scanning"""
    print("🧪 Testing Advanced Scanner - Single Face")
    print("=" * 50)
    print("✨ New Features:")
    print("  • Real-time color detection")
    print("  • Stability tracking with progress bar")
    print("  • Auto-capture when stable")
    print("  • Semi-transparent color overlays")
    print("  • Voice feedback")
    print("\n🎮 Controls:")
    print("  • SPACE: Manual capture")
    print("  • A: Toggle auto-capture")
    print("  • Q: Quit")
    print("\n🚀 Starting test...")

    scanner = AdvancedCubeScanner(use_voice=True)
    scanner.current_face_index = 0  # Test with white face

    try:
        face_data = scanner.scan_face_advanced()

        if face_data:
            print(f"✅ SUCCESS! Detected colors:")
            for i, row in enumerate(face_data):
                print(f"  Row {i+1}: {row}")
        else:
            print("❌ Test cancelled or failed")

    except Exception as e:
        print(f"❌ Error during test: {e}")


if __name__ == "__main__":
    test_single_face()
