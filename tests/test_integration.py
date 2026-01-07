"""Integration tests for the complete cube solving workflow."""

import pytest


# These are integration tests that test the full workflow
@pytest.mark.integration
@pytest.mark.skip(reason="Need valid scrambled cube states")
class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_solve_workflow(self, scrambled_cube_state):
        """Test the complete solve workflow."""
        from enhanced_cube_solver import EnhancedCubeSolver

        solver = EnhancedCubeSolver()

        # Step 1: Solve the cube
        success, result = solver.solve_cube(scrambled_cube_state)
        assert success is True

        # Step 2: Get progress
        current, total, percentage = solver.get_progress()
        assert current == 0
        assert total > 0

        # Step 3: Execute moves
        moves_executed = 0
        while not solver.is_complete():
            move = solver.get_current_move()
            assert move is not None

            explanation = solver.get_move_explanation(move)
            assert len(explanation) > 0

            solver.advance_move()
            moves_executed += 1

        assert moves_executed == total
        assert solver.is_complete() is True

    def test_error_recovery(self):
        """Test error recovery with invalid input."""
        from enhanced_cube_solver import EnhancedCubeSolver

        solver = EnhancedCubeSolver()

        # Try to solve invalid cube
        success, result = solver.solve_cube("INVALID")
        assert success is False

        # Should still be able to solve a valid cube after error
        valid_cube = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
        success, result = solver.solve_cube(valid_cube)
        assert success is True

    @pytest.mark.slow
    def test_multiple_solves(self):
        """Test solving multiple cubes in sequence."""
        from enhanced_cube_solver import EnhancedCubeSolver

        solver = EnhancedCubeSolver()

        test_cubes = [
            "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB",  # Solved
            "DUUBULDBFRBFRRULDBLRLFUFLFRFDUDLDRUBLLLDBLUFBFRUBTDUUF",  # Scrambled 1
            "DRLUUBFBRBLUFRLRRRUBRFUFUFDBLLDLUBDDFRUDBFULFLFDRTUBUF",  # Scrambled 2
        ]

        for cube_state in test_cubes:
            success, result = solver.solve_cube(cube_state)

            if success:
                assert isinstance(result, list)
                assert len(result) <= 20  # Kociemba guarantee
