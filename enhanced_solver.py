from collections import Counter

import kociemba


def flatten_face(face):
    """Convert 3x3 face array to flat string"""
    return "".join([color[0].upper() for row in face for color in row])


def solve_cube_string(faces):
    """Convert face colors to cube string and solve"""
    # Face order for Kociemba: U R F D L B
    face_order = ["U", "R", "F", "D", "L", "B"]

    # Get center colors to determine face mapping
    center_colors = []
    for face in faces:
        center_color = face[1][1]  # Center cell
        center_colors.append(center_color)

    # Create mapping from color to face letter
    expected_centers = ["white", "red", "green", "yellow", "orange", "blue"]
    face_letters = dict(zip(expected_centers, face_order, strict=False))

    # Build cube string
    cube_str = ""
    for face in faces:
        for row in face:
            for color in row:
                if color in face_letters:
                    cube_str += face_letters[color]
                else:
                    # Handle unknown colors - try to infer from context
                    cube_str += "U"  # Default fallback

    # Validate cube string
    if "?" in cube_str:
        return None, "Invalid cube string - contains unknown colors"

    # Check color counts
    color_counts = Counter(cube_str)
    for face_letter in face_order:
        if color_counts[face_letter] != 9:
            return (
                None,
                f"Invalid cube: Face {face_letter} has {color_counts[face_letter]} squares (should be 9)",
            )

    # Attempt to solve
    try:
        solution = kociemba.solve(cube_str)
        if solution == "Error":
            return None, "Cube state is invalid or unsolvable"
        return solution.split(), None
    except Exception as e:
        return None, f"Solver error: {str(e)}"


def validate_cube_state(faces):
    """Validate that the scanned faces form a valid cube"""
    # Check we have 6 faces
    if len(faces) != 6:
        return False, f"Need 6 faces, got {len(faces)}"

    # Check each face is 3x3
    for i, face in enumerate(faces):
        if len(face) != 3 or any(len(row) != 3 for row in face):
            return False, f"Face {i} is not 3x3"

    # Count all colors
    all_colors = []
    for face in faces:
        for row in face:
            for color in row:
                if color != "?":
                    all_colors.append(color)

    # Check color distribution
    color_counts = Counter(all_colors)
    expected_colors = ["white", "red", "green", "yellow", "orange", "blue"]

    for color in expected_colors:
        count = color_counts.get(color, 0)
        if count != 9:
            return False, f"Color '{color}' appears {count} times (should be 9)"

    return True, "Cube state is valid"


def get_move_explanation(move):
    """Get human-readable explanation of a move"""
    explanations = {
        "R": "Turn Right face clockwise",
        "R'": "Turn Right face counter-clockwise",
        "R2": "Turn Right face 180 degrees",
        "L": "Turn Left face clockwise",
        "L'": "Turn Left face counter-clockwise",
        "L2": "Turn Left face 180 degrees",
        "U": "Turn Upper face clockwise",
        "U'": "Turn Upper face counter-clockwise",
        "U2": "Turn Upper face 180 degrees",
        "D": "Turn Down face clockwise",
        "D'": "Turn Down face counter-clockwise",
        "D2": "Turn Down face 180 degrees",
        "F": "Turn Front face clockwise",
        "F'": "Turn Front face counter-clockwise",
        "F2": "Turn Front face 180 degrees",
        "B": "Turn Back face clockwise",
        "B'": "Turn Back face counter-clockwise",
        "B2": "Turn Back face 180 degrees",
    }

    return explanations.get(move, f"Unknown move: {move}")
