"""
Solver service

Business logic for cube solving operations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhanced_cube_solver import EnhancedCubeSolver
from typing import Dict, List, Any


class SolverService:
    """Service for cube solving operations"""

    def __init__(self):
        self.solver = EnhancedCubeSolver()

    async def solve_cube_string(self, cube_string: str) -> Dict[str, Any]:
        """Solve cube from string notation"""
        try:
            success, result = self.solver.solve_cube(cube_string)

            if success:
                return {
                    "success": True,
                    "solution": result if result else [],
                    "error": None,
                }
            else:
                return {"success": False, "solution": None, "error": str(result)}
        except Exception as e:
            return {"success": False, "solution": None, "error": str(e)}

    async def solve_cube_faces(
        self, faces: List[List[List[str]]], use_ml: bool = False
    ) -> Dict[str, Any]:
        """Solve cube from face arrays"""
        try:
            # For now, just use the solver's existing functionality
            # In production, would convert faces to cube string properly
            success, result = self.solver.solve_cube(faces)

            if success:
                return {
                    "success": True,
                    "solution": result if result else [],
                    "error": None,
                }
            else:
                return {"success": False, "solution": None, "error": str(result)}
        except Exception as e:
            return {"success": False, "solution": None, "error": str(e)}

    def get_move_explanation(self, move: str) -> str:
        """Get human-readable move explanation"""
        return self.solver.get_move_explanation(move)


# Singleton instance
solver_service = SolverService()
