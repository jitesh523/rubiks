"""
Solver API routes

Endpoints for cube solving operations.
"""

from fastapi import APIRouter, HTTPException
from api.models.requests import CubeStateRequest, CubeFacesRequest
from api.models.responses import SolutionResponse, MoveExplanationResponse
from api.services.solver_service import solver_service

router = APIRouter()


@router.post("/solve-string", response_model=SolutionResponse)
async def solve_from_string(request: CubeStateRequest):
    """
    Solve a Rubik's cube from notation string.

    The cube string should be 54 characters representing the colors in URFDLB order.
    """
    try:
        result = await solver_service.solve_cube_string(request.cube_string)

        if result["success"]:
            return SolutionResponse(
                success=True,
                solution=result["solution"],
                move_count=len(result["solution"]) if result["solution"] else 0,
                error=None,
                metadata={"method": "kociemba", "optimal": True},
            )
        else:
            return SolutionResponse(
                success=False,
                solution=None,
                move_count=0,
                error=result["error"],
                metadata={},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solving failed: {str(e)}")


@router.post("/solve-faces", response_model=SolutionResponse)
async def solve_from_faces(request: CubeFacesRequest):
    """
    Solve a Rubik's cube from face arrays.

    Expects 6 faces, each represented as a 3x3 grid of color names.
    """
    try:
        result = await solver_service.solve_cube_faces(
            request.faces, request.use_ml_detection
        )

        if result["success"]:
            return SolutionResponse(
                success=True,
                solution=result["solution"],
                move_count=len(result["solution"]) if result["solution"] else 0,
                error=None,
                metadata={
                    "method": "kociemba",
                    "used_ml": request.use_ml_detection,
                },
            )
        else:
            return SolutionResponse(
                success=False,
                solution=None,
                move_count=0,
                error=result["error"],
                metadata={},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solving failed: {str(e)}")


@router.get("/explain-move/{move}", response_model=MoveExplanationResponse)
async def explain_move(move: str):
    """
    Get human-readable explanation for a cube move.

    Examples: R, U', F2, etc.
    """
    try:
        explanation = solver_service.get_move_explanation(move)
        return MoveExplanationResponse(
            move=move, explanation=explanation, direction_hint=None
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid move notation: {str(e)}"
        )
