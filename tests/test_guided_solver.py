"""
Tests for Guided Solver Service

Tests session management, move validation, and solving workflow.
"""

import pytest

from api.services.guided_solver_service import GuidedSolverService, SolvingSession


@pytest.fixture
def service():
    """Create a fresh service instance for each test"""
    return GuidedSolverService()


@pytest.fixture
def sample_face_colors():
    """Sample 3x3 color grid"""
    return [
        ["white", "white", "white"],
        ["white", "white", "white"],
        ["white", "white", "white"],
    ]


class TestSolvingSession:
    """Test SolvingSession class"""

    def test_session_creation(self):
        """Test creating a new session"""
        session = SolvingSession("test-123")
        assert session.session_id == "test-123"
        assert session.is_scanning is True
        assert session.is_solving is False
        assert session.current_step == 0
        assert len(session.scanned_faces) == 0

    def test_session_to_dict(self):
        """Test session serialization"""
        session = SolvingSession("test-123")
        data = session.to_dict()

        assert data["session_id"] == "test-123"
        assert data["is_scanning"] is True
        assert data["is_solving"] is False
        assert "current_face" in data
        assert data["progress"] == 0.0


class TestGuidedSolverService:
    """Test GuidedSolverService class"""

    def test_create_session(self, service):
        """Test creating a new session"""
        session = service.create_session()

        assert session is not None
        assert session.session_id in service.sessions
        assert session.is_scanning is True

    def test_get_session(self, service):
        """Test retrieving a session"""
        session = service.create_session()
        retrieved = service.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_nonexistent_session(self, service):
        """Test retrieving a non-existent session"""
        result = service.get_session("nonexistent")
        assert result is None

    def test_delete_session(self, service):
        """Test deleting a session"""
        session = service.create_session()
        session_id = session.session_id

        assert service.delete_session(session_id) is True
        assert service.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_confirm_face_scan(self, service, sample_face_colors):
        """Test confirming a scanned face"""
        session = service.create_session()

        # Confirm first face
        result = await service.confirm_face_scan(session.session_id, sample_face_colors)

        assert result["success"] is True
        assert result["all_faces_scanned"] is False
        assert "next_face" in result

    @pytest.mark.asyncio
    async def test_get_current_instruction_scanning(self, service):
        """Test getting instruction during scanning"""
        session = service.create_session()

        result = await service.get_current_instruction(session.session_id)

        assert result["success"] is True
        assert result["mode"] == "scanning"
        assert "instruction" in result
        assert "current_face" in result

    @pytest.mark.asyncio
    async def test_validate_move(self, service, sample_face_colors):
        """Test move validation"""
        session = service.create_session()
        session.is_scanning = False
        session.is_solving = True
        session.solution = ["R", "U", "R'"]
        session.last_face_state = sample_face_colors

        # Different face state (simulating a move)
        new_state = [
            ["red", "white", "white"],
            ["red", "white", "white"],
            ["red", "white", "white"],
        ]

        result = await service.validate_move(session.session_id, new_state)

        assert result["success"] is True
        assert "is_valid" in result
        assert "message" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
