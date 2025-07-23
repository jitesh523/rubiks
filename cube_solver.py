import kociemba
import time

class CubeSolver:
    def __init__(self):
        self.solution_moves = []
        self.current_move_index = 0
        self.move_explanations = {
            'R': "Turn Right face clockwise",
            'R\'': "Turn Right face counter-clockwise",
            'R2': "Turn Right face 180 degrees",
            'L': "Turn Left face clockwise", 
            'L\'': "Turn Left face counter-clockwise",
            'L2': "Turn Left face 180 degrees",
            'U': "Turn Upper face clockwise",
            'U\'': "Turn Upper face counter-clockwise", 
            'U2': "Turn Upper face 180 degrees",
            'D': "Turn Down face clockwise",
            'D\'': "Turn Down face counter-clockwise",
            'D2': "Turn Down face 180 degrees",
            'F': "Turn Front face clockwise",
            'F\'': "Turn Front face counter-clockwise",
            'F2': "Turn Front face 180 degrees",
            'B': "Turn Back face clockwise",
            'B\'': "Turn Back face counter-clockwise", 
            'B2': "Turn Back face 180 degrees"
        }
    
    def solve_cube(self, cube_string):
        """Solve the cube using Kociemba algorithm"""
        try:
            print("🔄 Solving cube...")
            print(f"Cube state: {cube_string}")
            
            # Solve using Kociemba
            solution = kociemba.solve(cube_string)
            
            if solution == "Error":
                return False, "Invalid cube state - cannot solve"
            
            # Parse solution into individual moves
            self.solution_moves = solution.split() if solution else []
            self.current_move_index = 0
            
            print(f"✅ Solution found!")
            print(f"Number of moves: {len(self.solution_moves)}")
            print(f"Solution: {solution}")
            
            return True, self.solution_moves
            
        except Exception as e:
            return False, f"Error solving cube: {str(e)}"
    
    def get_current_move(self):
        """Get the current move to perform"""
        if self.current_move_index < len(self.solution_moves):
            return self.solution_moves[self.current_move_index]
        return None
    
    def get_move_explanation(self, move):
        """Get human-readable explanation of a move"""
        return self.move_explanations.get(move, f"Unknown move: {move}")
    
    def advance_move(self):
        """Move to the next step in the solution"""
        if self.current_move_index < len(self.solution_moves):
            self.current_move_index += 1
            return True
        return False
    
    def get_progress(self):
        """Get current progress through the solution"""
        if not self.solution_moves:
            return 0, 0, 0  # current, total, percentage
        
        total = len(self.solution_moves)
        current = self.current_move_index
        percentage = int((current / total) * 100) if total > 0 else 0
        
        return current, total, percentage
    
    def reset_solution(self):
        """Reset to the beginning of the solution"""
        self.current_move_index = 0
    
    def is_complete(self):
        """Check if all moves have been completed"""
        return self.current_move_index >= len(self.solution_moves)
    
    def get_remaining_moves(self):
        """Get the remaining moves to complete"""
        if self.current_move_index < len(self.solution_moves):
            return self.solution_moves[self.current_move_index:]
        return []
    
    def display_solution_summary(self):
        """Display a nice summary of the solution"""
        if not self.solution_moves:
            print("No solution available.")
            return
        
        print("\n" + "="*50)
        print("🎯 SOLUTION SUMMARY")
        print("="*50)
        print(f"Total moves: {len(self.solution_moves)}")
        print(f"Estimated time: {len(self.solution_moves) * 2} seconds")
        print("\nMove sequence:")
        
        # Display moves in groups of 10
        for i in range(0, len(self.solution_moves), 10):
            group = self.solution_moves[i:i+10]
            print(f"{i+1:2d}-{min(i+10, len(self.solution_moves)):2d}: {' '.join(group)}")
        
        print("\n" + "="*50)
    
    def get_move_with_explanation(self, move):
        """Get a detailed explanation for a move"""
        explanation = self.get_move_explanation(move)
        
        # Add visual direction hints
        direction_hints = {
            'R': "👉 (when looking at right face)",
            'R\'': "👈 (when looking at right face)", 
            'R2': "↻ (180°)",
            'L': "👈 (when looking at left face)",
            'L\'': "👉 (when looking at left face)",
            'L2': "↻ (180°)",
            'U': "↻ (clockwise from above)",
            'U\'': "↺ (counter-clockwise from above)",
            'U2': "↻ (180°)",
            'D': "↻ (clockwise from below)",
            'D\'': "↺ (counter-clockwise from below)",
            'D2': "↻ (180°)",
            'F': "↻ (clockwise when facing front)",
            'F\'': "↺ (counter-clockwise when facing front)",
            'F2': "↻ (180°)",
            'B': "↻ (clockwise when facing back)",
            'B\'': "↺ (counter-clockwise when facing back)",
            'B2': "↻ (180°)"
        }
        
        hint = direction_hints.get(move, "")
        return f"{explanation} {hint}"
