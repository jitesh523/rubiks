import cv2
import numpy as np
import math

class CubeVisualizer:
    def __init__(self, size=300):
        self.size = size
        self.scale = 100
        self.vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
        ], dtype=float)
        
        # Define faces using vertex indices (for coloring)
        # Order: U, R, F, D, L, B
        self.faces = {
            'U': [0, 1, 5, 4],  # Top (actually -Y in this coord system if Y is down) -> Let's adjust
            'D': [3, 2, 6, 7],  # Bottom
            'F': [4, 5, 6, 7],  # Front
            'B': [1, 0, 3, 2],  # Back
            'L': [0, 4, 7, 3],  # Left
            'R': [5, 1, 2, 6]   # Right
        }
        
        # Colors (BGR)
        self.colors = {
            'U': (255, 255, 255), # White
            'D': (0, 255, 255),   # Yellow
            'F': (0, 255, 0),     # Green
            'B': (255, 0, 0),     # Blue
            'L': (0, 165, 255),   # Orange
            'R': (0, 0, 255)      # Red
        }

        # Target rotations for each face to face the camera (+Z)
        # Assuming camera looks at +Z face
        # We rotate the cube so the target face aligns with +Z
        self.face_rotations = {
            'F': (0, 0, 0),             # Identity
            'R': (0, -90, 0),           # Rotate Y -90
            'L': (0, 90, 0),            # Rotate Y +90
            'B': (0, 180, 0),           # Rotate Y 180
            'U': (90, 0, 0),            # Rotate X +90
            'D': (-90, 0, 0)            # Rotate X -90
        }

    def project(self, point, width, height):
        """Project 3D point to 2D screen coordinates"""
        # Orthographic projection
        x = int(point[0] * self.scale + width / 2)
        y = int(point[1] * self.scale + height / 2)
        return (x, y)

    def rotate_x(self, angle):
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    def rotate_y(self, angle):
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def rotate_z(self, angle):
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    def get_rotation_matrix(self, rx, ry, rz):
        return self.rotate_x(rx) @ self.rotate_y(ry) @ self.rotate_z(rz)

    def draw_cube(self, frame, rotation=(0,0,0)):
        """Draw the cube on the frame with given rotation"""
        h, w = frame.shape[:2]
        
        # Calculate rotation matrix
        R = self.get_rotation_matrix(*rotation)
        
        # Rotate vertices
        rotated_verts = np.dot(self.vertices, R.T)
        
        # Project points
        points = [self.project(v, w, h) for v in rotated_verts]
        
        # Draw edges
        edges = [
            (0,1), (1,2), (2,3), (3,0), # Back face
            (4,5), (5,6), (6,7), (7,4), # Front face
            (0,4), (1,5), (2,6), (3,7)  # Connecting lines
        ]
        
        # Draw faces (painter's algorithm - sort by Z depth)
        # Calculate centers of faces
        face_depths = []
        for name, indices in self.faces.items():
            # Average Z of vertices
            z_avg = np.mean([rotated_verts[i][2] for i in indices])
            face_depths.append((z_avg, name, indices))
        
        # Sort by depth (furthest first)
        face_depths.sort(key=lambda x: x[0])
        
        # Draw faces
        for _, name, indices in face_depths:
            pts = np.array([points[i] for i in indices], np.int32)
            color = self.colors[name]
            cv2.fillPoly(frame, [pts], color)
            cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
            
            # Draw label on face center
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(frame, name, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    def animate_transition(self, from_face, to_face, duration=1.5):
        """Generate a sequence of frames animating from one face to another"""
        start_rot = np.array(self.face_rotations.get(from_face, (0,0,0)))
        end_rot = np.array(self.face_rotations.get(to_face, (0,0,0)))
        
        # Handle shortest path for rotation (e.g. 0 to 270 vs 0 to -90)
        # Simple linear interpolation for now
        
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames + 1):
            t = i / total_frames
            # Ease in-out
            t = t * t * (3 - 2 * t)
            
            current_rot = start_rot + (end_rot - start_rot) * t
            
            # Create frame
            frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            self.draw_cube(frame, current_rot)
            
            # Add instruction text
            cv2.putText(frame, f"Turn to {to_face}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            frames.append(frame)
            
        return frames
