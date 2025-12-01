import cv2
import numpy as np
import time
import pyttsx3
import threading

class MoveTracker:
    def __init__(self, use_voice=True):
        self.use_voice = use_voice
        self.tts_engine = None
        
        # Initialize text-to-speech if requested
        if self.use_voice:
            try:
                self.tts_engine = pyttsx3.init()
                # Set properties
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 150)  # Slower speech
                self.tts_engine.setProperty('volume', 0.8)
            except:
                print("Warning: Text-to-speech not available")
                self.use_voice = False
    
    def speak(self, text):
        """Speak text using TTS"""
        if self.use_voice and self.tts_engine:
            try:
                # Run TTS in a separate thread to avoid blocking
                def speak_thread():
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
                thread = threading.Thread(target=speak_thread)
                thread.daemon = True
                thread.start()
            except:
                print(f"TTS Error - Text: {text}")
    
        # Instructions
        cv2.putText(canvas, "Press SPACE when move is complete", (100, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
        cv2.putText(canvas, "Press 'r' to repeat instruction", (100, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
        cv2.putText(canvas, "Press 'q' to quit", (100, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
                   
        return canvas

    def display_move_instruction(self, move, next_move, current_step, total_steps):
        """Display visual instruction for a move"""
        # Create a black canvas
        canvas = np.zeros((450, 800, 3), dtype=np.uint8)
        
        # Colors
        white = (255, 255, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        yellow = (0, 255, 255)
        gray = (150, 150, 150)
        
        # Title
        title = f"Step {current_step} of {total_steps}"
        cv2.putText(canvas, title, (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)
        
        # Progress bar
        progress = current_step / total_steps
        bar_width = 600
        bar_height = 20
        bar_x = 100
        bar_y = 70
        
        # Background bar
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), white, 1)
        # Progress fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), green, -1)
        
        # Move instruction
        move_text = f"Move: {move}"
        cv2.putText(canvas, move_text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, yellow, 3)
        
        # Next move preview
        if next_move:
            next_text = f"Next: {next_move}"
            cv2.putText(canvas, next_text, (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, gray, 2)
        
        # Move explanation
        explanations = {
            'R': "Turn RIGHT face clockwise",
            'R\'': "Turn RIGHT face counter-clockwise",
            'R2': "Turn RIGHT face 180 degrees",
            'L': "Turn LEFT face clockwise",
            'L\'': "Turn LEFT face counter-clockwise", 
            'L2': "Turn LEFT face 180 degrees",
            'U': "Turn TOP face clockwise",
            'U\'': "Turn TOP face counter-clockwise",
            'U2': "Turn TOP face 180 degrees", 
            'D': "Turn BOTTOM face clockwise",
            'D\'': "Turn BOTTOM face counter-clockwise",
            'D2': "Turn BOTTOM face 180 degrees",
            'F': "Turn FRONT face clockwise",
            'F\'': "Turn FRONT face counter-clockwise",
            'F2': "Turn FRONT face 180 degrees",
            'B': "Turn BACK face clockwise", 
            'B\'': "Turn BACK face counter-clockwise",
            'B2': "Turn BACK face 180 degrees"
        }
        
        explanation = explanations.get(move, "Unknown move")
        
        # Split explanation into multiple lines if needed
        lines = [explanation]
        y_pos = 220
        for line in lines:
            cv2.putText(canvas, line, (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)
            y_pos += 40
        
        # Instructions
        cv2.putText(canvas, "Press SPACE when move is complete", (100, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
        cv2.putText(canvas, "Press 'r' to repeat instruction", (100, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
        cv2.putText(canvas, "Press 'q' to quit", (100, 410), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2)
        
        return canvas
    
    def guide_through_solution(self, solver):
        """Guide user through the complete solution"""
        if not solver.solution_moves:
            print("No solution available!")
            return False
        
        print("\n🎯 Starting guided solution!")
        print("Follow the on-screen instructions for each move.")
        
        # Speak initial instruction
        self.speak("Let's solve your Rubik's cube. Follow the instructions on screen.")
        
        while not solver.is_complete():
            current_move = solver.get_current_move()
            if not current_move:
                break
            
            current, total, percentage = solver.get_progress()
            
            # Get next move for preview
            remaining = solver.get_remaining_moves()
            next_move = remaining[1] if len(remaining) > 1 else None
            
            # Create visual instruction
            instruction_canvas = self.display_move_instruction(current_move, next_move, current + 1, total)
            
            # Speak the move
            move_speech = self.get_move_speech(current_move)
            self.speak(move_speech)
            
            # Show instruction and wait for user input
            waiting_for_input = True
            while waiting_for_input:
                cv2.imshow('Cube Solver - Move Instruction', instruction_canvas)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):  # Space - move completed
                    solver.advance_move()
                    waiting_for_input = False
                    print(f"✅ Move {current + 1} completed: {current_move}")
                    
                elif key == ord('r'):  # Repeat instruction
                    self.speak(move_speech)
                    
                elif key == ord('q'):  # Quit
                    cv2.destroyAllWindows()
                    return False
        
        # Solution complete
        cv2.destroyAllWindows()
        
        # Show completion message
        self.show_completion_screen()
        self.speak("Congratulations! Your Rubik's cube is now solved!")
        
        return True
    
    def get_move_speech(self, move):
        """Convert move notation to speech-friendly text"""
        move_speech = {
            'R': "Turn the right face clockwise",
            'R\'': "Turn the right face counter-clockwise", 
            'R2': "Turn the right face 180 degrees",
            'L': "Turn the left face clockwise",
            'L\'': "Turn the left face counter-clockwise",
            'L2': "Turn the left face 180 degrees", 
            'U': "Turn the top face clockwise",
            'U\'': "Turn the top face counter-clockwise",
            'U2': "Turn the top face 180 degrees",
            'D': "Turn the bottom face clockwise", 
            'D\'': "Turn the bottom face counter-clockwise",
            'D2': "Turn the bottom face 180 degrees",
            'F': "Turn the front face clockwise",
            'F\'': "Turn the front face counter-clockwise",
            'F2': "Turn the front face 180 degrees",
            'B': "Turn the back face clockwise",
            'B\'': "Turn the back face counter-clockwise", 
            'B2': "Turn the back face 180 degrees"
        }
        
        return move_speech.get(move, f"Perform move {move}")
    
    def show_completion_screen(self):
        """Show congratulations screen"""
        canvas = np.zeros((400, 800, 3), dtype=np.uint8)
        
        # Colors
        white = (255, 255, 255)
        green = (0, 255, 0)
        gold = (0, 215, 255)
        
        # Congratulations text
        cv2.putText(canvas, "CONGRATULATIONS!", (180, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, gold, 3)
        cv2.putText(canvas, "Your Rubik's Cube is SOLVED!", (150, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
        
        # Additional messages
        cv2.putText(canvas, "Well done! You followed all the moves correctly.", (120, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)
        cv2.putText(canvas, "Your cube should now be in the solved state.", (130, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)
        
        cv2.putText(canvas, "Press any key to exit", (280, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
        
        cv2.imshow('Cube Solver - Complete!', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def quick_move_guide(self, solver):
        """Quick text-based guide through solution"""
        if not solver.solution_moves:
            print("No solution available!")
            return False
        
        print(f"\n🎯 Quick Solution Guide ({len(solver.solution_moves)} moves):")
        print("="*60)
        
        move_count = 1
        for move in solver.solution_moves:
            explanation = solver.get_move_with_explanation(move)
            print(f"{move_count:2d}. {move:3s} - {explanation}")
            
            # Wait for user confirmation
            user_input = input(f"    Press Enter when move {move_count} is complete (or 'q' to quit): ")
            if user_input.lower() == 'q':
                return False
            
            move_count += 1
        
        print("\n🎉 Congratulations! Your cube should now be solved!")
        return True
