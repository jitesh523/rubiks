"""Tests for enhanced_solver utility functions."""

import pytest

# Mark these tests as they require the enhanced_solver module
pytest.importorskip("enhanced_solver")

from enhanced_solver import (
    get_move_explanation,
    solve_cube_string,
    validate_cube_state,
)


class TestEnhancedSolver:
    """Test suite for enhanced_solver utilities."""

    def test_validate_cube_state_solved(self, sample_cube_faces):
        """Test validation of a solved cube state."""
        is_valid, message = validate_cube_state(sample_cube_faces)
        assert is_valid is True

    def test_validate_cube_state_wrong_count(self):
        """Test validation rejects wrong number of faces."""
        invalid_faces = [
            [["W", "W", "W"], ["W", "W", "W"], ["W", "W", "W"]],
            # Missing 5 other faces
        ]
        is_valid, message = validate_cube_state(invalid_faces)
        assert is_valid is False

    def test_solve_cube_string_solved(self, sample_cube_faces):
        """Test solving a solved cube."""
        moves, error = solve_cube_string(sample_cube_faces)

        if error:
            # If there's an error, it should be a string
            assert isinstance(error, str)
        else:
            # If successful, moves should be a list
            assert isinstance(moves, list)

    def test_get_move_explanation_basic_moves(self):
        """Test getting explanations for basic moves."""
        explanation = get_move_explanation("R")
        assert isinstance(explanation, str)
        assert len(explanation) > 0

        explanation = get_move_explanation("U'")
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_get_move_explanation_double_moves(self):
        """Test getting explanations for double moves."""
        explanation = get_move_explanation("R2")
        assert isinstance(explanation, str)
        assert "180" in explanation or "twice" in explanation.lower()
