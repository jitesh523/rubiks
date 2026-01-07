"""Pytest configuration and shared fixtures for Rubik's Cube Solver tests."""

import pytest


@pytest.fixture
def solved_cube_state():
    """Return a solved cube state string."""
    return "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


@pytest.fixture
def scrambled_cube_state():
    """Return a valid scrambled cube state string."""
    # This is a known working scrambled state
    return "FLUFFURRRLRUFRUFBRDBFBRUBRDRRFLLBLDUURBUBLDUBDDLFBDFUF"


@pytest.fixture
def invalid_cube_state():
    """Return an invalid cube state with wrong colors."""
    return "UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture
def sample_solution_moves():
    """Return a sample list of solution moves."""
    return ["R", "U", "R'", "U'", "R'", "F", "R2", "U'", "R'", "U'", "R", "U", "R'", "F'"]


@pytest.fixture
def sample_cube_faces():
    """Return a sample list of cube faces as would come from the scanner."""
    return [
        # U face (white)
        [["white", "white", "white"], ["white", "white", "white"], ["white", "white", "white"]],
        # R face (red)
        [["red", "red", "red"], ["red", "red", "red"], ["red", "red", "red"]],
        # F face (green)
        [["green", "green", "green"], ["green", "green", "green"], ["green", "green", "green"]],
        # D face (yellow)
        [["yellow", "yellow", "yellow"], ["yellow", "yellow", "yellow"], ["yellow", "yellow", "yellow"]],
        # L face (orange)
        [["orange", "orange", "orange"], ["orange", "orange", "orange"], ["orange", "orange", "orange"]],
        # B face (blue)
        [["blue", "blue", "blue"], ["blue", "blue", "blue"], ["blue", "blue", "blue"]],
    ]


@pytest.fixture
def scrambled_cube_faces():
    """Return a valid scrambled cube as faces."""
    return [
        # U face
        [["Y", "W", "W"], ["B", "W", "L"], ["Y", "B", "F"]],
        # R face
        [["R", "B", "F"], ["R", "R", "R"], ["W", "L", "Y"]],
        # F face
        [["L", "R", "L"], ["F", "F", "W"], ["F", "R", "F"]],
        # D face
        [["F", "R", "Y"], ["Y", "Y", "Y"], ["L", "Y", "R"]],
        # L face
        [["L", "L", "L"], ["Y", "L", "B"], ["F", "B", "Y"]],
        # B face
        [["B", "F", "R"], ["W", "B", "B"], ["U", "B", "T"]],
    ]
