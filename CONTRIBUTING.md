# Contributing to Rubik's Cube Solver

Thank you for your interest in contributing to the AI-Powered Rubik's Cube Solver! This document provides guidelines and instructions for developers.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A webcam (for testing camera features)

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jitesh523/rubiks.git
   cd rubiks
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### Code Quality Tools

We use several tools to maintain code quality:

#### **Black** - Code Formatting
Automatically formats code to a consistent style:
```bash
# Format all files
black .

# Check without modifying
black --check .
```

#### **Ruff** - Linting
Fast Python linter that checks for errors and style issues:
```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

#### **MyPy** - Type Checking
Static type checker for Python:
```bash
# Check types in main modules
mypy enhanced_cube_solver.py move_tracker.py cube_visualizer.py
```

#### **Pytest** - Testing
Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_cube_solver.py

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

**Testing ML Components:**
```bash
# Run ML detector tests specifically
pytest tests/test_ml_color_detector.py -v

# Check ML detector coverage
pytest tests/test_ml_color_detector.py --cov=ml_color_detector --cov-report=term-missing

# Run only fallback-related tests
pytest tests/test_ml_color_detector.py -k "fallback" -v
```

**Manual Testing of ML Data Collector:**
```bash
# Test the interactive data collection tool
python ml_data_collector.py

# Expected behavior:
# - Camera opens with crosshair
# - Shows live ML predictions (if model exists)
# - Allows labeling with keys 1-6
# - Can save/load training data
# - Can train model interactively
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:
- Trailing whitespace removal
- File ending fixes
- Black formatting
- Ruff linting
- MyPy type checking

**Run manually on all files:**
```bash
pre-commit run --all-files
```

**Skip hooks (not recommended):**
```bash
git commit --no-verify
```

## 📝 Code Style Guidelines

### General Principles

1. **Follow PEP 8** - Python's style guide
2. **Use type hints** - Add type annotations to function signatures
3. **Write docstrings** - Document all public functions and classes
4. **Keep functions small** - Each function should do one thing well
5. **Use descriptive names** - Variables and functions should be self-documenting

### Type Hints Example

```python
def solve_cube(self, cube_input: str | list) -> tuple[bool, list[str] | str]:
    """
    Solve a Rubik's cube from its current state.

    Args:
        cube_input: Either a 54-char string or list of face arrays

    Returns:
        Tuple of (success: bool, result: list of moves or error message)
    """
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def get_move_explanation(move: str) -> str:
    """Get human-readable explanation of a move.

    Args:
        move: Move notation (e.g., 'R', 'U\'', 'F2')

    Returns:
        Human-readable explanation of the move

    Examples:
        >>> get_move_explanation('R')
        'Turn Right face clockwise'
    """
    pass
```

## 🧪 Testing Guidelines

### Writing Tests

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test complete workflows
- **Use fixtures**: Leverage pytest fixtures from `conftest.py`
- **Test edge cases**: Invalid inputs, boundary conditions
- **Mock external dependencies**: Camera, TTS, etc.

### Test Structure

```python
class TestFeatureName:
    """Test suite for FeatureName."""

    def test_basic_functionality(self):
        """Test that basic feature works."""
        # Arrange
        solver = EnhancedCubeSolver()

        # Act
        result = solver.solve_cube("...")

        # Assert
        assert result is not None
```

### Coverage Goals

- Aim for **>80% code coverage**
- Focus on critical paths
- Don't sacrifice test quality for coverage numbers

## 🔄 Git Workflow

### Branching Strategy

1. **main** - Production-ready code
2. **feature/*** - New features
3. **bugfix/*** - Bug fixes
4. **refactor/*** - Code improvements

### Commit Messages

Use clear, descriptive commit messages:

```
Add ML-based color detection

- Implement scikit-learn classifier
- Add training data collection
- Improve accuracy in low light
- Add fallback to HSV detection
```

**Format:**
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description with bullet points

### Pull Request Process

1. Create a feature branch
2. Make your changes
3. Run all tests and quality checks
4. Submit PR with clear description
5. Address review feedback
6. Squash commits if requested

## 🐛 Debugging Tips

### Running in Debug Mode

```python
# Add this at the top of main.py for verbose output
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Camera Issues

```python
# Test camera independently
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened())"
```

### Color Detection Issues

```bash
# Run color calibration
python color_calibrator.py
```

## 📚 Documentation

### Updating README

When adding features:
1. Update the features list
2. Add usage examples
3. Update architecture section if needed
4. Add troubleshooting tips

### Generating API Docs

```bash
# Install sphinx if not already installed
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs
sphinx-build -b html source build
```

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] Improve color detection accuracy
- [ ] Add more comprehensive tests
- [ ] Mobile app development
- [ ] Performance optimizations

### Medium Priority
- [ ] Multi-language support
- [ ] Advanced visualizations
- [ ] Tutorial mode
- [ ] Cloud sync features

### Nice to Have
- [ ] AR overlay features
- [ ] Pattern library
- [ ] Competition modes
- [ ] Solution comparison

## ❓ Getting Help

- **Issues**: Check existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Refer to README.md and code comments

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

---

**Happy Coding! 🎲✨**
