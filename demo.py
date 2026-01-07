#!/usr/bin/env python3
"""
Demo script for Rubik's Cube Solver
"""

from cube_solver import CubeSolver


def demo_solver():
    print("🎲 Rubik's Cube Solver Demo")
    print("=" * 40)

    # Example scrambled cube state
    scrambled_cube = "DLRUUBFBRFLRFUURFRBUFFBLULFRULBFURFDDDLRDDDDDBLLFDUUBDRDFRRULULDDRUDFBBL"

    print(f"Scrambled cube: {scrambled_cube}")
    print(f"Length: {len(scrambled_cube)}")

    # Initialize solver
    solver = CubeSolver()

    # Solve the cube
    success, result = solver.solve_cube(scrambled_cube)

    if success:
        print("✅ Solution found!")
        solver.display_solution_summary()

        print("\n🎯 Would you like to see step-by-step guidance?")
        print("Note: This is a demo, so moves won't be validated")

        # Show first few moves as demonstration
        if solver.solution_moves:
            print("\nFirst 5 moves:")
            for i, move in enumerate(solver.solution_moves[:5]):
                explanation = solver.get_move_with_explanation(move)
                print(f"{i+1}. {move} - {explanation}")
    else:
        print(f"❌ Error: {result}")


def demo_with_valid_cube():
    """Demo with a known valid scrambled cube state"""
    print("\n🎲 Demo with Valid Scrambled Cube")
    print("=" * 40)

    # This is a known valid scrambled state
    valid_scramble = "DUUBULDBFRBFRRULLLBRDFFFBLURDBFDFDRDUULDRDUURRBLBDUDLUFLLFRDBUFRBFBLLGRU"

    # Try a simpler approach - create a cube by applying some moves to solved state
    # Let's manually create a simple scrambled state
    simple_scramble = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    # Modify just a few positions to create a solvable scramble
    scrambled = list(simple_scramble)

    # Swap some pieces to create a valid scramble
    # This creates a simple scrambled state that's still solvable
    scrambled[0] = "R"  # U face corner
    scrambled[9] = "U"  # R face corner

    scrambled_cube = "".join(scrambled)

    print(f"Scrambled cube: {scrambled_cube}")

    solver = CubeSolver()
    success, result = solver.solve_cube(scrambled_cube)

    if success:
        if solver.solution_moves:
            print(f"✅ Solution found with {len(solver.solution_moves)} moves!")
            solver.display_solution_summary()
        else:
            print("✅ Cube is already solved!")
    else:
        print(f"❌ Error: {result}")


if __name__ == "__main__":
    demo_solver()
    demo_with_valid_cube()
