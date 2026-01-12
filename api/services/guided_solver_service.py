"""
Guided Solver Service

Manages real-time solving sessions with move tracking and validation.
"""

from datetime import datetime
from uuid import uuid4

from api.services.realtime_detector import realtime_detector
from api.services.solver_service import solver_service


class SolvingSession:
    """Represents an active solving session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.cube_state: str | None = None
        self.solution: list[str] = []
        self.current_step: int = 0
        self.scanned_faces: dict[str, list[list[str]]] = {}
        self.face_scan_order = ["U", "R", "F", "D", "L", "B"]
        self.current_scan_index: int = 0
        self.is_scanning: bool = True
        self.is_solving: bool = False
        self.last_face_state: list[list[str]] | None = None

    def to_dict(self) -> dict:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "cube_state": self.cube_state,
            "solution": self.solution,
            "current_step": self.current_step,
            "total_steps": len(self.solution),
            "scanned_faces": self.scanned_faces,
            "current_scan_index": self.current_scan_index,
            "current_face": self.face_scan_order[self.current_scan_index]
            if self.current_scan_index < len(self.face_scan_order)
            else None,
            "is_scanning": self.is_scanning,
            "is_solving": self.is_solving,
            "progress": self._calculate_progress(),
        }

    def _calculate_progress(self) -> float:
        """Calculate overall progress percentage"""
        if self.is_scanning:
            return (self.current_scan_index / len(self.face_scan_order)) * 50
        elif self.is_solving and self.solution:
            scan_progress = 50
            solve_progress = (self.current_step / len(self.solution)) * 50
            return scan_progress + solve_progress
        return 0.0


class GuidedSolverService:
    """Service for managing guided solving sessions"""

    def __init__(self):
        self.sessions: dict[str, SolvingSession] = {}
        self.session_timeout = 3600  # 1 hour

    def create_session(self) -> SolvingSession:
        """Create a new solving session"""
        session_id = str(uuid4())
        session = SolvingSession(session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SolvingSession | None:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    async def process_frame(
        self, session_id: str, frame_data: str, grid_region: dict[str, int]
    ) -> dict[str, any]:
        """
        Process a camera frame for color detection

        Args:
            session_id: Session identifier
            frame_data: Base64 encoded image
            grid_region: Detection region coordinates

        Returns:
            Detection results with session state
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        try:
            # Decode frame
            frame = realtime_detector.decode_image_from_base64(frame_data)

            # Detect face grid
            detection = realtime_detector.detect_face_grid(frame, grid_region)

            # Store last face state for move validation
            if detection["is_valid"]:
                session.last_face_state = detection["colors"]

            return {
                "success": True,
                "detection": detection,
                "session_state": session.to_dict(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def confirm_face_scan(
        self, session_id: str, face_colors: list[list[str]]
    ) -> dict[str, any]:
        """
        Confirm scanned face and move to next

        Args:
            session_id: Session identifier
            face_colors: 3x3 grid of detected colors

        Returns:
            Updated session state
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        if not session.is_scanning:
            return {"success": False, "error": "Not in scanning mode"}

        # Get current face name
        current_face = session.face_scan_order[session.current_scan_index]

        # Store scanned face
        session.scanned_faces[current_face] = face_colors

        # Move to next face
        session.current_scan_index += 1

        # Check if all faces scanned
        if session.current_scan_index >= len(session.face_scan_order):
            # All faces scanned, generate solution
            success = await self._generate_solution(session)
            if success:
                session.is_scanning = False
                session.is_solving = True
                return {
                    "success": True,
                    "all_faces_scanned": True,
                    "solution_ready": True,
                    "session_state": session.to_dict(),
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate solution. Check cube state.",
                }

        return {
            "success": True,
            "all_faces_scanned": False,
            "next_face": session.face_scan_order[session.current_scan_index],
            "session_state": session.to_dict(),
        }

    async def _generate_solution(self, session: SolvingSession) -> bool:
        """Generate solution from scanned faces"""
        try:
            # Convert faces to cube string
            # Order: U, R, F, D, L, B
            cube_string = ""
            for face_name in session.face_scan_order:
                face_colors = session.scanned_faces.get(face_name, [])
                for row in face_colors:
                    for color in row:
                        # Map color names to single letters
                        color_map = {
                            "white": "U",
                            "red": "R",
                            "green": "F",
                            "yellow": "D",
                            "orange": "L",
                            "blue": "B",
                        }
                        cube_string += color_map.get(color.lower(), "U")

            session.cube_state = cube_string

            # Solve using existing solver service
            result = await solver_service.solve_cube_string(cube_string)

            if result["success"]:
                session.solution = result["solution"] or []
                return True
            else:
                return False

        except Exception as e:
            print(f"Error generating solution: {e}")
            return False

    async def validate_move(
        self, session_id: str, current_face_state: list[list[str]]
    ) -> dict[str, any]:
        """
        Validate if the user performed the correct move

        Args:
            session_id: Session identifier
            current_face_state: Current face state after user's move

        Returns:
            Validation result
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        if not session.is_solving:
            return {"success": False, "error": "Not in solving mode"}

        if not session.last_face_state:
            # No previous state to compare
            session.last_face_state = current_face_state
            return {
                "success": True,
                "is_valid": True,
                "message": "Move tracking started",
            }

        # Get expected move
        if session.current_step >= len(session.solution):
            return {
                "success": True,
                "is_valid": True,
                "message": "Cube solved!",
                "cube_solved": True,
            }

        expected_move = session.solution[session.current_step]

        # Validate move execution
        validation = realtime_detector.validate_move_execution(
            session.last_face_state, current_face_state, expected_move
        )

        if validation["is_valid"]:
            # Move to next step
            session.current_step += 1
            session.last_face_state = current_face_state

        return {
            "success": True,
            "is_valid": validation["is_valid"],
            "message": validation["message"],
            "expected_move": expected_move,
            "current_step": session.current_step,
            "total_steps": len(session.solution),
            "cube_solved": session.current_step >= len(session.solution),
        }

    async def get_current_instruction(self, session_id: str) -> dict[str, any]:
        """Get current solving instruction"""
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        if session.is_scanning:
            current_face = session.face_scan_order[session.current_scan_index]
            return {
                "success": True,
                "mode": "scanning",
                "instruction": f"Scan the {current_face} face",
                "current_face": current_face,
                "progress": session.current_scan_index,
                "total": len(session.face_scan_order),
            }

        elif session.is_solving:
            if session.current_step >= len(session.solution):
                return {
                    "success": True,
                    "mode": "complete",
                    "instruction": "Cube solved! 🎉",
                }

            current_move = session.solution[session.current_step]
            next_move = (
                session.solution[session.current_step + 1]
                if session.current_step + 1 < len(session.solution)
                else None
            )

            return {
                "success": True,
                "mode": "solving",
                "instruction": f"Perform move: {current_move}",
                "current_move": current_move,
                "next_move": next_move,
                "step": session.current_step + 1,
                "total_steps": len(session.solution),
            }

        return {"success": True, "mode": "idle", "instruction": "Initializing..."}


# Singleton instance
guided_solver_service = GuidedSolverService()
