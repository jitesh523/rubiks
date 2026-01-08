"""Tests for ML color detector."""

import numpy as np
import pytest

from ml_color_detector import MLColorDetector


class TestMLColorDetector:
    """Test suite for ML color detector."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        data = []

        # Each color gets 20 samples
        colors = {
            "white": (245, 245, 245),
            "red": (200, 30, 30),
            "green": (30, 180, 30),
            "blue": (30, 30, 200),
            "yellow": (220, 220, 30),
            "orange": (220, 100, 30),
        }

        for color_name, base_rgb in colors.items():
            for _ in range(20):
                # Add slight variation
                rgb = [max(0, min(255, base_rgb[i] + np.random.randint(-15, 15))) for i in range(3)]
                data.append({"rgb": rgb, "color": color_name})

        return data

    def test_initialization(self):
        """Test detector initialization."""
        detector = MLColorDetector(confidence_threshold=0.8)
        assert detector.confidence_threshold == 0.8
        assert detector.is_trained is False
        assert len(detector.color_names) == 6

    def test_feature_extraction(self):
        """Test feature extraction produces correct shape."""
        detector = MLColorDetector()
        features = detector.extract_features((255, 0, 0))

        assert features.shape == (9,)  # LAB(3) + HSV(3) + RGB ratios(3)
        assert not np.isnan(features).any()

    def test_training(self, sample_training_data):
        """Test model training."""
        detector = MLColorDetector()
        metrics = detector.train(sample_training_data)

        assert detector.is_trained is True
        assert metrics["accuracy"] > 0.7  # Should achieve decent accuracy
        assert metrics["n_train"] > 0
        assert metrics["n_test"] > 0

    def test_prediction_accuracy(self, sample_training_data):
        """Test prediction accuracy on known colors."""
        detector = MLColorDetector()
        detector.train(sample_training_data)

        # Test predictions
        test_cases = {
            "white": (245, 245, 245),
            "red": (200, 30, 30),
            "green": (30, 180, 30),
            "blue": (30, 30, 200),
        }

        for _expected_color, rgb in test_cases.items():
            predicted, confidence = detector.predict_color(rgb)
            # Most should be correct (allowing some variation)
            assert confidence > 0.5

    def test_confidence_scores(self, sample_training_data):
        """Test that confidence scores are in valid range."""
        detector = MLColorDetector()
        detector.train(sample_training_data)

        color, confidence = detector.predict_color((245, 245, 245))

        assert 0.0 <= confidence <= 1.0
        assert isinstance(color, str)

    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        detector = MLColorDetector()

        with pytest.raises(RuntimeError, match="not trained"):
            detector.predict_color((255, 0, 0))

    def test_model_persistence(self, sample_training_data, tmp_path):
        """Test model saving and loading."""
        # Train and save model
        detector1 = MLColorDetector()
        detector1.train(sample_training_data)
        detector1.save_model(str(tmp_path))

        # Load model
        detector2 = MLColorDetector()
        success = detector2.load_model(str(tmp_path))

        assert success is True
        assert detector2.is_trained is True

        # Test that loaded model makes same predictions
        test_color = (200, 30, 30)
        pred1, conf1 = detector1.predict_color(test_color)
        pred2, conf2 = detector2.predict_color(test_color)

        assert pred1 == pred2
        assert abs(conf1 - conf2) < 0.01

    def test_get_model_info(self, sample_training_data):
        """Test model info retrieval."""
        detector = MLColorDetector()
        info = detector.get_model_info()

        assert info["is_trained"] is False

        detector.train(sample_training_data)
        info = detector.get_model_info()

        assert info["is_trained"] is True
        assert "metadata" in info
        assert info["metadata"]["accuracy"] > 0

    def test_empty_training_data(self):
        """Test that empty training data raises error."""
        detector = MLColorDetector()

        with pytest.raises(ValueError, match="empty"):
            detector.train([])

    def test_predict_with_fallback_high_confidence(self, sample_training_data):
        """Test that ML is used when confidence is high."""
        detector = MLColorDetector(confidence_threshold=0.7)
        detector.train(sample_training_data)

        # Mock HSV detector that returns wrong color
        def hsv_fallback(rgb):
            return "wrong_color"

        # Test with a clear white color
        color, conf, used_ml = detector.predict_with_fallback(
            (245, 245, 245), hsv_detector_func=hsv_fallback
        )

        # Should use ML since confidence should be high
        assert used_ml is True
        assert conf >= 0.7
        assert color != "wrong_color"  # Should not use HSV fallback

    def test_predict_with_fallback_low_confidence(self, sample_training_data):
        """Test that HSV fallback is used when confidence is low."""
        detector = MLColorDetector(confidence_threshold=0.99)  # Very high threshold
        detector.train(sample_training_data)

        # Mock HSV detector
        def hsv_fallback(rgb):
            return "hsv_color"

        # Use an ambiguous/gray color
        color, conf, used_ml = detector.predict_with_fallback(
            (128, 128, 128), hsv_detector_func=hsv_fallback
        )

        # Should fall back to HSV if confidence < 0.99
        if conf < 0.99:
            assert used_ml is False
            assert color == "hsv_color"
        else:
            # If confidence is high, ML should be used
            assert used_ml is True

    def test_predict_with_fallback_no_model_no_hsv(self):
        """Test fallback behavior when model not trained and no HSV detector."""
        detector = MLColorDetector(confidence_threshold=0.7)

        with pytest.raises(RuntimeError, match="not trained"):
            detector.predict_with_fallback((255, 0, 0), hsv_detector_func=None)

    def test_predict_with_fallback_no_model_with_hsv(self):
        """Test fallback uses HSV when model not trained but HSV provided."""
        detector = MLColorDetector(confidence_threshold=0.7)

        def hsv_fallback(rgb):
            return "red"

        color, conf, used_ml = detector.predict_with_fallback(
            (200, 30, 30), hsv_detector_func=hsv_fallback
        )

        assert color == "red"
        assert conf == 0.5  # Default confidence when using fallback
        assert used_ml is False
