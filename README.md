# 🎲 AI-Powered Rubik's Cube Solver with Computer Vision

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete AI-powered Rubik's Cube solver that uses computer vision to scan the cube and provides step-by-step visual and audio guidance to solve it!

![Rubik's Cube Solver Demo](https://via.placeholder.com/800x400/1a1a1a/ffffff?text=Rubik%27s+Cube+Solver+Demo)

## ✨ Features

### 🎯 **Core Functionality**
- 📸 **Camera-based cube scanning** - Point your camera at each face of the cube
- 🧠 **Kociemba algorithm solving** - Fast and optimal solution finding (≤20 moves)
- 🎯 **Step-by-step guidance** - Visual instructions with progress tracking
- 🔊 **Text-to-speech feedback** - Spoken instructions for each move
- ✅ **Move validation** - Tracks your progress through the solution

### 🚀 **Advanced Features**
- 🎯 **Real-Time Guided Solver** - Live camera-based solving with step-by-step validation!
  - 📹 **Continuous camera feed** - Real-time cube state detection
  - ✅ **Move validation** - Automatically checks if you performed the move correctly
  - 🎲 **3D visualization** - Interactive Three.js cube showing moves
  - 🔄 **WebSocket communication** - Ultra-low latency real-time updates
- 🤖 **ML Color Detection** - Machine learning-based color recognition with 95%+ accuracy
- ⚡ **Auto-Calibration** - Scan a solved cube once to train the ML model automatically
- 🎯 **Confidence Scoring** - Smart fallback to traditional detection when confidence is low
- 🎨 **Real-time color detection** - Live preview with HSV color analysis
- 📊 **Stability tracking** - Ensures faces remain steady before capture
- 🤖 **Auto-capture mode** - Automatically captures when cube is stable
- 🎪 **Semi-transparent overlays** - Visual feedback on detected colors
- 📱 **Professional AR-style interface** - Clean, intuitive user experience

### 🛠️ **Technical Features**
- 🔬 **LAB+HSV+RGB Features** - 9-dimensional feature extraction for robust classification
- 🔍 **Color validation** - Prevents invalid cube states
- 📈 **Progress visualization** - Real-time stability and progress bars
- 🎵 **Voice guidance** - Complete audio walkthrough
- 🧪 **Multiple scanning modes** - Manual, auto-capture, and test modes
- 🔧 **Diagnostic tools** - Built-in troubleshooting utilities (Option 5 in menu)
- ✅ **Comprehensive Testing** - 22+ tests with pytest ensuring reliability

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Webcam/camera
- Good lighting conditions
- A scrambled Rubik's cube

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jitesh523/rubiks.git
   cd rubiks
   ```

3. **Run the application:**

   We've included a helper script to set up the environment and run the app automatically:

   ```bash
   ./start.sh
   ```

   Or manually:
   ```bash
   source venv/bin/activate
   python main.py
   ```

4. **Optional: Auto-Calibrate ML Color Detection (Recommended)**

   For best color detection accuracy, calibrate the ML model with your cube and lighting:

   ```bash
   python auto_calibrator.py
   ```

   - Have a **solved cube** ready
   - Follow on-screen instructions to scan each face
   - Model trains automatically in ~2 minutes
   - Enjoy 95%+ color detection accuracy!

## 🤖 ML Color Detection

Phase 2 of the project introduces advanced machine learning-based color detection with smart fallback mechanisms.

### ✨ Features

- **95%+ Accuracy**: KNN classifier trained on your specific cube and lighting conditions
- **Auto-Calibration**: Quick 2-minute setup with a solved cube
- **Smart Fallback**: Automatically uses HSV detection when ML confidence is low (<70%)
- **Confidence Scoring**: Know when predictions are reliable
- **9D Feature Extraction**: LAB + HSV + RGB ratios for robust classification
- **Persistent Model**: Train once, use forever across sessions

### 🚀 Quick Setup

**Option 1: Auto-Calibration (Recommended)**
```bash
python auto_calibrator.py
```
Follow on-screen instructions to scan a solved cube. The model trains automatically!

**Option 2: Manual Training (Advanced)**
```bash
python ml_data_collector.py
```
- Point camera at cube stickers
- Press 1-6 to label each color
- Collect 10+ samples per color
- Press 't' to train model
- Press 's' to save training data

### 🔬 How It Works

1. **Feature Extraction**: Converts RGB to 9 features (LAB color space, HSV, RGB ratios)
2. **KNN Classification**: Uses k=5 neighbors with distance weighting
3. **Confidence Scoring**: Returns probability of prediction
4. **Smart Fallback**: Falls back to HSV if confidence < 70%
5. **Model Persistence**: Saves to `ml_color_model.pkl`, `ml_color_scaler.pkl`

### 🎯 Why ML vs HSV?

- **HSV Detection**: Fast but lighting-dependent, struggles with red/orange
- **ML Detection**: Learns your specific cube's colors, adapts to lighting
- **Best of Both**: High confidence → ML, Low confidence → HSV fallback

## 🌐 Web API

The Rubik's Cube Solver includes a REST API for web and mobile integrations!

### Quick Start

**Start the API server**:
```bash
uvicorn api.main:app --reload
```

Server runs at: **http://localhost:8000**

### API Documentation

- **Swagger UI**: http://localhost:8000/docs (interactive testing)
- **ReDoc**: http://localhost:8000/redoc (clean documentation)
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Example Usage

**Solve a Cube**:
```bash
curl -X POST "http://localhost:8000/api/v1/solver/solve-string" \
  -H "Content-Type: application/json" \
  -d '{"cube_string": "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"}'
```

**Check ML Model**:
```bash
curl "http://localhost:8000/api/v1/ml/info"
```

**Predict Color**:
```bash
curl -X POST "http://localhost:8000/api/v1/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{"rgb": [200, 30, 30]}'
```

### Available Endpoints

**Solver Endpoints**:
- `POST /api/v1/solver/solve-string` - Solve from 54-char string
- `POST /api/v1/solver/solve-faces` - Solve from face arrays
- `GET /api/v1/solver/explain-move/{move}` - Get move explanation

**ML Model Endpoints**:
- `GET /api/v1/ml/info` - Get model status
- `POST /api/v1/ml/predict` - Predict color from RGB
- `POST /api/v1/ml/train` - Train model

**System Endpoints**:
- `GET /` - API info
- `GET /health` - Health check

**Guided Solver Endpoints** (WebSocket):
- `POST /api/v1/guided/create-session` - Create new solving session
- `GET /api/v1/guided/session/{session_id}` - Get session state
- `DELETE /api/v1/guided/session/{session_id}` - Delete session
- `WebSocket /api/v1/guided/ws/{session_id}` - Real-time guided solving

### WebSocket Protocol

The guided solver uses WebSocket for real-time communication:

**Client → Server Messages**:
```json
{"type": "process_frame", "frame_data": "base64...", "grid_region": {...}}
{"type": "confirm_face", "face_colors": [[...]]}
{"type": "validate_move", "current_face_state": [[...]]}
{"type": "get_instruction"}
{"type": "ping"}
```

**Server → Client Messages**:
```json
{"type": "detection_result", "data": {...}}
{"type": "face_confirmed", "data": {...}}
{"type": "move_validated", "data": {...}}
{"type": "instruction", "data": {...}}
{"type": "error", "error": "..."}
{"type": "pong"}
```

## 🌐 React Frontend

The project includes a modern React-based web interface!

### Quick Start

**Start frontend development server**:
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

### Features

- 🎯 **Manual Scanner** - Color picker interface for cube input
- 🤖 **Guided Solver** - Real-time camera-based solving with validation
- 🎲 **3D Visualization** - Interactive Three.js cube models
- 📱 **Responsive Design** - Works on desktop and mobile
- 🎨 **Modern UI** - Glassmorphism design with smooth animations

### Pages

- **Home** - Welcome page with features overview
- **Scanner** - Manual cube input with color picker
- **Guided Solver** - Real-time camera-based guided solving
- **Solver** - Solution playback with step-by-step instructions
- **About** - Project information and documentation

## 🎮 How to Use

### Option 1: Full Camera Experience (Recommended)

1. **Select option 1** from the main menu
2. **Scan all 6 faces** in this order:
   - U (Up/White) - Top face
   - R (Right/Red) - Right face
   - F (Front/Green) - Front face
   - D (Down/Yellow) - Bottom face
   - L (Left/Orange) - Left face
   - B (Back/Blue) - Back face

3. **For each face:**
   - Hold cube steady with the face centered in view
   - Align with the green grid overlay
   - Press **SPACE** when colors are clear
   - Press **Enter** to continue to next face

4. **Choose guidance method:**
   - Visual + Audio (recommended)
   - Text-only
   - Show solution only

5. **Follow the moves:**
   - Watch on-screen instructions
   - Listen to spoken directions
   - Press **SPACE** after completing each move
   - Press **R** to repeat current instruction

### Option 2: Manual Entry

If camera scanning doesn't work well, you can manually enter the cube state as a 54-character string using face notation (URFDLB).

### Option 3: Test Mode

Test the solver with a solved cube to verify everything works correctly.

## 🎨 Cube Color Mapping

Standard Rubik's cube colors:
- **U (Up)**: White ⚪
- **R (Right)**: Red 🔴
- **F (Front)**: Green 🟢
- **D (Down)**: Yellow 🟡
- **L (Left)**: Orange 🟠
- **B (Back)**: Blue 🔵

## 🔄 Move Notation

The solver uses standard Rubik's cube notation:
- **R** = Right face clockwise
- **R'** = Right face counter-clockwise
- **R2** = Right face 180 degrees
- Same pattern for L, U, D, F, B faces

## 💡 Tips for Best Results

### Camera Scanning:
- **Good lighting** - Avoid shadows and harsh lights
- **Clean cube** - Wipe faces for better color detection
- **Steady hands** - Keep cube still during capture
- **Proper distance** - Fill the grid but don't overfill
- **Correct orientation** - Follow the face order exactly

### Solving:
- **Take your time** - No need to rush through moves
- **Double-check** - Verify each move before pressing SPACE
- **Ask for repeats** - Press R if you missed an instruction
- **Stay organized** - Keep track of which face is which

## 🏗️ Architecture

The system consists of 4 main modules:

### 1. `cube_scanner.py` - Computer Vision
- Camera interface and video capture
- Color detection using HSV color space
- 3x3 grid overlay and guidance
- Color-to-notation conversion
- Cube state validation

### 2. `cube_solver.py` - Algorithm Engine
- Kociemba algorithm integration
- Solution optimization
- Move parsing and explanation
- Progress tracking

### 3. `move_tracker.py` - User Guidance
- Visual instruction display
- Text-to-speech integration
- Progress visualization
- Move-by-move guidance
- Completion celebration

### 4. `main.py` - Application Controller
- Menu system and user interface
- Component orchestration
- Error handling
- Multiple input methods

## 🔧 Troubleshooting

### Camera Issues:
- **No camera detected**: Check camera permissions and connections
- **Poor color detection**: Improve lighting, clean cube faces
- **Colors wrong**: Try different lighting or manual color correction

### Solving Issues:
- **Invalid cube state**: Re-scan more carefully or check for stickers
- **Solution not working**: Verify you're following moves correctly
- **Audio not working**: Check system volume and TTS installation

### Performance:
- **Slow scanning**: Close other camera applications
- **Lag during guidance**: Reduce other running applications

## 🤝 Contributing

Ideas for improvements:
- [ ] Advanced color calibration
- [ ] Cube state validation with physics
- [ ] Animated move demonstrations
- [ ] Multiple language support
- [ ] Mobile app version
- [ ] Competition timing mode

## 📚 Technical Details

### Dependencies:
- **OpenCV**: Computer vision and camera handling
- **NumPy**: Numerical computations and image processing
- **Kociemba**: Two-phase algorithm for cube solving
- **pyttsx3**: Text-to-speech synthesis
- **Threading**: Non-blocking audio playback

### Color Detection:
Uses HSV color space for robust color recognition under varying lighting conditions. Each color has calibrated hue, saturation, and value ranges.

### Solving Algorithm:
Implements Herbert Kociemba's two-phase algorithm:
- **Phase 1**: Get to a state solvable using only R, L, F2, B2, U, D
- **Phase 2**: Solve using only those moves
- Guarantees solution in ≤20 moves

## 🎯 Future Enhancements

- **AR overlay**: Augmented reality move visualization
- **Pattern recognition**: Automatic cube state detection
- **Machine learning**: Improved color recognition
- **Mobile support**: iOS/Android versions
- **Online sharing**: Share solutions and times
- **Tutorial mode**: Learn solving algorithms

## 📄 License

This project is created for educational purposes. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- Herbert Kociemba for the two-phase algorithm
- OpenCV community for computer vision tools
- Python community for excellent libraries
- Rubik's cube community for notation standards

---

**Happy Cubing! 🎲✨**

*"Every expert was once a beginner. Every pro was once an amateur."*
