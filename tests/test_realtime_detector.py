"""
Tests for Realtime Detector

Tests color detection and move validation.
"""

import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from api.services.realtime_detector import RealtimeDetector, realtime_detector


@pytest.fixture
def detector():
    """Create a fresh detector instance"""
    return RealtimeDetector()


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple 100x100 white image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    return img


@pytest.fixture
def sample_base64_image(sample_image):
    """Convert sample image to base64"""
    # Convert to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

    # Convert to base64
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{img_str}"


class TestRealtimeDetector:
    """Test RealtimeDetector class"""

    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert len(detector.color_ranges) > 0
        assert detector.stabilization_window == 5

    def test_decode_image_from_base64(self, detector, sample_base64_image):
        """Test decoding base64 image"""
        frame = detector.decode_image_from_base64(sample_base64_image)

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # Should be BGR image

    def test_detect_color_with_confidence(self, detector, sample_image):
        """Test color detection with confidence"""
        color, confidence = detector.detect_color_with_confidence(sample_image)

        assert color is not None
        assert isinstance(color, str)
        assert 0.0 <= confidence <= 1.0

    def test_detect_face_grid(self, detector, sample_image):
        """Test detecting 3x3 face grid"""
        # Create a larger test image
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255

        grid_region = {"x": 50, "y": 50, "width": 300, "height": 300}

        result = detector.detect_face_grid(test_image, grid_region)

        assert result is not None
        assert "colors" in result
        assert "confidences" in result
        assert "avg_confidence" in result
        assert "is_valid" in result

        # Check grid is 3x3
        assert len(result["colors"]) == 3
        assert len(result["colors"][0]) == 3

    def test_validate_move_execution(self, detector):
        """Test move validation logic"""
        before_state = [
            ["white", "white", "white"],
            ["white", "white", "white"],
            ["white", "white", "white"],
        ]

        # Different state (move executed)
        after_state = [
            ["red", "white", "white"],
            ["red", "white", "white"],
            ["red", "white", "white"],
        ]

        result = detector.validate_move_execution(before_state, after_state, "R")

        assert result is not None
        assert "is_valid" in result
        assert "states_different" in result
        assert "center_preserved" in result

    def test_validate_move_same_state(self, detector):
        """Test validation when state hasn't changed"""
        same_state = [
            ["white", "white", "white"],
            ["white", "white", "white"],
            ["white", "white", "white"],
        ]

        result = detector.validate_move_execution(same_state, same_state, "R")

        assert result["states_different"] is False


class TestRealtimeDetectorSingleton:
    """Test the singleton instance"""

    def test_singleton_exists(self):
        """Test that singleton exists"""
        assert realtime_detector is not None
        assert isinstance(realtime_detector, RealtimeDetector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
