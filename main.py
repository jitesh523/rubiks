#!/usr/bin/env python3
"""
Rubik's Cube Solver with Computer Vision
========================================

A complete system that can:
1. Scan a Rubik's cube using your camera
2. Solve it using the Kociemba algorithm  
3. Guide you through the solution step-by-step with visual and audio feedback

Author: AI Assistant
"""

import sys
import os
from cube_scanner import CubeScanner
from cube_solver import CubeSolver
from move_tracker import MoveTracker

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🎲 RUBIK'S CUBE SOLVER WITH COMPUTER VISION 🎲")
    print("=" * 60)
    print("Features:")
    print("✅ Camera-based cube scanning")
    print("✅ Kociemba algorithm solving")
    print("✅ Step-by-step visual guidance")
    print("✅ Text-to-speech instructions")
    print("✅ Progress tracking")
    print("=" * 60)

def print_menu():
    """Print main menu options"""
    print("\n📋 MAIN MENU:")
    print("1. 📸 Scan & Solve Cube (Full Experience)")
    print("2. 🔤 Enter Cube Manually")  
    print("3. 🧪 Test with Solved Cube")
    print("4. ℹ️  Show Instructions")
    print("5. 🚪 Exit")
    print("-" * 30)

def show_instructions():
    """Show detailed instructions for using the system"""
    print("\n📖 INSTRUCTIONS:")
    print("=" * 50)
    print("\n🎯 How to use the Cube Scanner:")
    print("1. Hold your scrambled Rubik's cube steady")
    print("2. Position one face in the center of the camera view")
    print("3. Align the cube with the green grid overlay")
    print("4. Press SPACE to capture when the colors are clear")
    print("5. Repeat for all 6 faces in order: U, R, F, D, L, B")
    
    print("\n🎨 Face Colors (Standard Cube):")
    print("• U (Up/Top): White")
    print("• R (Right): Red") 
    print("• F (Front): Green")
    print("• D (Down/Bottom): Yellow")
    print("• L (Left): Orange")
    print("• B (Back): Blue")
    
    print("\n🔄 Move Notation:")
    print("• R = Right face clockwise")
    print("• R' = Right face counter-clockwise")  
    print("• R2 = Right face 180 degrees")
    print("• Same pattern for L, U, D, F, B faces")
    
    print("\n💡 Tips for best results:")
    print("• Ensure good lighting (avoid shadows)")
    print("• Keep cube steady during scanning")
    print("• Clean cube faces for better color detection")
    print("• Follow the face order exactly as shown")
    
def manual_cube_entry():
    """Allow manual entry of cube state"""
    print("\n🔤 Manual Cube Entry")
    print("Enter the cube state as a 54-character string")
    print("Format: UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB")
    print("Order: U(9) R(9) F(9) D(9) L(9) B(9) faces")
    print("Colors: U=White, R=Red, F=Green, D=Yellow, L=Orange, B=Blue")
    print("\nExample (solved cube):")
    print("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB")
    
    while True:
        cube_string = input("\nEnter cube string (or 'back' to return): ").strip()
        
        if cube_string.lower() == 'back':
            return None
            
        if len(cube_string) != 54:
            print(f"❌ Invalid length: {len(cube_string)} (must be 54)")
            continue
            
        # Validate characters
        valid_chars = set('URFDLB')
        if not all(c in valid_chars for c in cube_string.upper()):
            print("❌ Invalid characters. Use only: U, R, F, D, L, B")
            continue
            
        return cube_string.upper()

def test_with_solved_cube():
    """Test the system with a solved cube (should return empty solution)"""
    print("\n🧪 Testing with solved cube...")
    solved_cube = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    
    solver = CubeSolver()
    success, result = solver.solve_cube(solved_cube)
    
    if success:
        if not solver.solution_moves:
            print("✅ Test passed! Solved cube requires no moves.")
        else:
            print(f"⚠️  Unexpected: Solved cube returned {len(solver.solution_moves)} moves")
            print(f"Moves: {' '.join(solver.solution_moves)}")
    else:
        print(f"❌ Test failed: {result}")

def scan_and_solve_cube():
    """Full cube scanning and solving experience"""
    print("\n🎲 Starting Cube Scanner...")
    
    # Initialize components
    scanner = CubeScanner()
    solver = CubeSolver()
    
    # Scan all faces
    print("📸 Scanning cube faces...")
    cube_faces = scanner.scan_all_faces()
    
    if cube_faces is None:
        print("❌ Scanning cancelled.")
        return False
    
    # Convert to solver format
    cube_string = scanner.convert_to_kociemba_format(cube_faces)
    print(f"\n🔍 Scanned cube state:")
    print(f"Raw data: {cube_string}")
    
    # Validate cube
    is_valid, validation_message = scanner.validate_cube(cube_string)
    print(f"Validation: {validation_message}")
    
    if not is_valid:
        print("❌ Invalid cube state detected. Please try scanning again.")
        return False
    
    # Solve cube
    print("\n🧠 Solving cube...")
    success, result = solver.solve_cube(cube_string)
    
    if not success:
        print(f"❌ Solving failed: {result}")
        return False
    
    if not solver.solution_moves:
        print("🎉 Your cube is already solved!")
        return True
    
    # Display solution summary
    solver.display_solution_summary()
    
    # Choose guidance method
    print("\n🎯 How would you like to be guided through the solution?")
    print("1. 🖼️  Visual + Audio guidance (Recommended)")
    print("2. 📝 Text-only guidance")
    print("3. 📋 Show full solution and exit")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == '1':
        # Visual + Audio guidance
        tracker = MoveTracker(use_voice=True)
        return tracker.guide_through_solution(solver)
        
    elif choice == '2':
        # Text-only guidance
        tracker = MoveTracker(use_voice=False)
        return tracker.quick_move_guide(solver)
        
    elif choice == '3':
        # Show solution and exit
        print(f"\n📋 Complete solution ({len(solver.solution_moves)} moves):")
        print(' '.join(solver.solution_moves))
        return True
        
    else:
        print("❌ Invalid choice.")
        return False

def main():
    """Main application loop"""
    print_banner()
    
    # Check if camera is available (basic test)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  Warning: Camera not detected. Scanning features may not work.")
        else:
            cap.release()
    except:
        print("⚠️  Warning: OpenCV not properly installed. Camera scanning unavailable.")
    
    while True:
        print_menu()
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            scan_and_solve_cube()
            
        elif choice == '2':
            cube_string = manual_cube_entry()
            if cube_string:
                solver = CubeSolver()
                success, result = solver.solve_cube(cube_string)
                
                if success:
                    if solver.solution_moves:
                        solver.display_solution_summary()
                        
                        # Quick guide
                        tracker = MoveTracker(use_voice=False)
                        tracker.quick_move_guide(solver)
                    else:
                        print("🎉 This cube is already solved!")
                else:
                    print(f"❌ Solving failed: {result}")
                    
        elif choice == '3':
            test_with_solved_cube()
            
        elif choice == '4':
            show_instructions()
            
        elif choice == '5':
            print("\n👋 Thanks for using the Rubik's Cube Solver!")
            print("Happy cubing! 🎲")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-5.")
        
        # Wait for user before showing menu again
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please restart the application.")
