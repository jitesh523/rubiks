"""
Guided Solver WebSocket API routes

Real-time endpoints for guided cube solving with camera feed.
"""


from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.services.guided_solver_service import guided_solver_service

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """Accept and store connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        """Remove connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        """Send message to specific session"""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = ConnectionManager()


@router.post("/create-session")
async def create_session():
    """Create a new guided solving session"""
    session = guided_solver_service.create_session()
    return {
        "success": True,
        "session_id": session.session_id,
        "session_state": session.to_dict(),
    }


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session state"""
    session = guided_solver_service.get_session(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}

    return {"success": True, "session_state": session.to_dict()}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    success = guided_solver_service.delete_session(session_id)
    return {"success": success}


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time guided solving

    Messages from client:
    - {"type": "process_frame", "frame_data": "base64...", "grid_region": {...}}
    - {"type": "confirm_face", "face_colors": [[...]]}
    - {"type": "validate_move", "current_face_state": [[...]]}
    - {"type": "get_instruction"}
    - {"type": "ping"}

    Messages to client:
    - {"type": "detection_result", "data": {...}}
    - {"type": "face_confirmed", "data": {...}}
    - {"type": "move_validated", "data": {...}}
    - {"type": "instruction", "data": {...}}
    - {"type": "error", "error": "..."}
    - {"type": "pong"}
    """
    await manager.connect(session_id, websocket)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "ping":
                await manager.send_message(session_id, {"type": "pong"})

            elif message_type == "process_frame":
                frame_data = data.get("frame_data")
                grid_region = data.get("grid_region", {"x": 0, "y": 0, "width": 300, "height": 300})

                result = await guided_solver_service.process_frame(
                    session_id, frame_data, grid_region
                )

                await manager.send_message(
                    session_id, {"type": "detection_result", "data": result}
                )

            elif message_type == "confirm_face":
                face_colors = data.get("face_colors")

                result = await guided_solver_service.confirm_face_scan(session_id, face_colors)

                await manager.send_message(
                    session_id, {"type": "face_confirmed", "data": result}
                )

            elif message_type == "validate_move":
                current_face_state = data.get("current_face_state")

                result = await guided_solver_service.validate_move(
                    session_id, current_face_state
                )

                await manager.send_message(
                    session_id, {"type": "move_validated", "data": result}
                )

            elif message_type == "get_instruction":
                result = await guided_solver_service.get_current_instruction(session_id)

                await manager.send_message(session_id, {"type": "instruction", "data": result})

            else:
                await manager.send_message(
                    session_id, {"type": "error", "error": f"Unknown message type: {message_type}"}
                )

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        await manager.send_message(session_id, {"type": "error", "error": str(e)})
        manager.disconnect(session_id)
