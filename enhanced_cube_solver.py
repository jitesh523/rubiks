import kociemba
import time
from enhanced_solver import solve_cube_string, get_move_explanation, validate_cube_state

class EnhancedCubeSolver:
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
    
    def solve_cube(self, cube_input):
        """Solve the cube using enhanced solver with validation"""
        try:
            print("🔄 Solving cube with enhanced solver...")
            
            if isinstance(cube_input, list):
                # Handle list of faces (from scanner)
                is_valid, validation_message = validate_cube_state(cube_input)
                if not is_valid:
                    return False, f"Invalid cube state: {validation_message}"
                
                # solve_cube_string returns (solution_list, error_message)
                solution_moves, error = solve_cube_string(cube_input)
                
                if error:
                    return False, error
                
                self.solution_moves = solution_moves
                result_str = ' '.join(solution_moves)
                
            else:
                # Handle string input (manual/test)
                print(f"Cube state: {cube_input}")
                if len(cube_input) != 54:
                    return False, f"Invalid cube string length: {len(cube_input)}"
                
                # Check for invalid characters
                valid_chars = set('URFDLB')
                if not all(c in valid_chars for c in cube_input.upper()):
                    return False, "Invalid characters in cube string"

                # Check if already solved
                solved_state = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
                if cube_input == solved_state:
                    self.solution_moves = []
                    print("🎉 Cube is already solved!")
                    return True, []

                try:
                    result_str = kociemba.solve(cube_input)
                    self.solution_moves = result_str.split()
                except Exception as e:
                    return False, f"Kociemba solver error: {str(e)}"

            self.current_move_index = 0
            
            print(f"✅ Enhanced solution found!")
            print(f"Number of moves: {len(self.solution_moves)}")
            print(f"Solution: {result_str}")
            
            return True, self.solution_moves
            
        except Exception as e:
            return False, f"Error solving cube: {str(e)}"
    
    def get_current_move(self):
        """Get the current move to perform"""
        if self.current_move_index < len(self.solution_moves):
            return self.solution_moves[self.current_move_index]
        return None
    
    def get_move_explanation(self, move):
        """Get human-readable explanation of a move using enhanced explanation"""
        try:
            # Try to use enhanced move explanation first
            enhanced_explanation = get_move_explanation(move)
            if enhanced_explanation:
                return enhanced_explanation
        except:
            # Fall back to basic explanation if enhanced fails
            pass
        
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
        print("🎯 ENHANCED SOLUTION SUMMARY")
        print("="*50)
        print(f"Total moves: {len(self.solution_moves)}")
        print(f"Estimated time: {len(self.solution_moves) * 2} seconds")
        print("\nMove sequence:")
        
        # Display moves in groups of 10
        for i in range(0, len(self.solution_moves), 10):
            group = self.solution_moves[i:i+10]
            print(f"{i+1:2d}-{min(i+10, len(self.solution_moves)):2d}: {' '.join(group)}")
        
        # Show detailed explanations for first few moves
        print("\n📋 First few moves explained:")
        for i, move in enumerate(self.solution_moves[:5]):
            explanation = self.get_move_explanation(move)
            print(f"  {i+1}. {move}: {explanation}")
        
        if len(self.solution_moves) > 5:
            print(f"  ... and {len(self.solution_moves) - 5} more moves")
        
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
