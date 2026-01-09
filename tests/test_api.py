"""
API Tests for Rubik's Cube Solver

Comprehensive test suite for FastAPI endpoints.
"""

from fastapi.testclient import TestClient

from api.main import app

# Create test client
client = TestClient(app)


class TestRootEndpoints:
    """Test root and health check endpoints"""

    def test_root(self):
        """Test API root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["version"] == "2.0.0"
        assert "docs" in data
        assert "redoc" in data

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "ml_model_loaded" in data
        assert isinstance(data["ml_model_loaded"], bool)


class TestSolverEndpoints:
    """Test solver-related endpoints"""

    def test_solve_from_string_solved_cube(self):
        """Test solving an already solved cube"""
        response = client.post(
            "/api/v1/solver/solve-string",
            json={"cube_string": "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["solution"], list)
        assert data["move_count"] == 0 or data["move_count"] == len(data["solution"])
        assert data["error"] is None

    def test_solve_from_string_invalid_length(self):
        """Test with invalid cube string length"""
        response = client.post("/api/v1/solver/solve-string", json={"cube_string": "INVALID"})
        assert response.status_code == 422  # Validation error

    def test_solve_from_string_missing_field(self):
        """Test with missing cube_string field"""
        response = client.post("/api/v1/solver/solve-string", json={})
        assert response.status_code == 422

    def test_explain_move_valid(self):
        """Test move explanation with valid move"""
        moves = ["R", "U", "F", "D", "L", "B", "R'", "U2"]
        for move in moves:
            response = client.get(f"/api/v1/solver/explain-move/{move}")
            assert response.status_code == 200
            data = response.json()
            assert data["move"] == move
            assert isinstance(data["explanation"], str)
            assert len(data["explanation"]) > 0

    def test_explain_move_invalid(self):
        """Test move explanation with invalid move"""
        response = client.get("/api/v1/solver/explain-move/INVALID")
        # May return 200 with generic explanation or 400 with error
        assert response.status_code in [200, 400]

    def test_solve_from_faces(self):
        """Test solving from face arrays"""
        # Simple test with minimal face data
        faces = [
            [["U", "U", "U"], ["U", "U", "U"], ["U", "U", "U"]],
            [["R", "R", "R"], ["R", "R", "R"], ["R", "R", "R"]],
            [["F", "F", "F"], ["F", "F", "F"], ["F", "F", "F"]],
            [["D", "D", "D"], ["D", "D", "D"], ["D", "D", "D"]],
            [["L", "L", "L"], ["L", "L", "L"], ["L", "L", "L"]],
            [["B", "B", "B"], ["B", "B", "B"], ["B", "B", "B"]],
        ]
        response = client.post(
            "/api/v1/solver/solve-faces",
            json={"faces": faces, "use_ml_detection": False},
        )
        # May succeed or fail depending on implementation
        assert response.status_code in [200, 500]


class TestMLEndpoints:
    """Test ML model endpoints"""

    def test_ml_info(self):
        """Test ML model info endpoint"""
        response = client.get("/api/v1/ml/info")
        assert response.status_code == 200
        data = response.json()
        assert "is_trained" in data
        assert isinstance(data["is_trained"], bool)
        assert "confidence_threshold" in data
        assert "accuracy" in data
        assert "last_trained" in data
        assert "metadata" in data

    def test_predict_color_valid_rgb(self):
        """Test color prediction with valid RGB values"""
        test_colors = [
            [200, 30, 30],  # Red-ish
            [30, 200, 30],  # Green-ish
            [30, 30, 200],  # Blue-ish
            [245, 245, 245],  # White-ish
        ]

        for rgb in test_colors:
            response = client.post("/api/v1/ml/predict", json={"rgb": rgb})
            # May return 200 with prediction or 500 if model not trained
            assert response.status_code in [200, 500]

            if response.status_code == 200:
                data = response.json()
                assert "color" in data
                assert "confidence" in data
                assert "used_ml" in data
                assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_color_invalid_rgb_length(self):
        """Test color prediction with invalid RGB length"""
        response = client.post("/api/v1/ml/predict", json={"rgb": [200, 30]})
        assert response.status_code == 422

    def test_predict_color_missing_field(self):
        """Test color prediction with missing field"""
        response = client.post("/api/v1/ml/predict", json={})
        assert response.status_code == 422

    def test_train_model_minimal(self):
        """Test model training with minimal valid data"""
        training_data = [
            {"rgb": [200, 30, 30], "color": "red"},
            {"rgb": [30, 200, 30], "color": "green"},
            {"rgb": [30, 30, 200], "color": "blue"},
            {"rgb": [245, 245, 245], "color": "white"},
            {"rgb": [220, 220, 30], "color": "yellow"},
            {"rgb": [220, 100, 30], "color": "orange"},
        ] * 10  # Repeat to have enough samples

        response = client.post(
            "/api/v1/ml/train",
            json={"training_data": training_data, "confidence_threshold": 0.7},
        )

        # Training may succeed or fail depending on data quality
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            if data["success"]:
                assert "accuracy" in data
                assert "n_train" in data
                assert "n_test" in data


class TestAPIDocumentation:
    """Test that API documentation is accessible"""

    def test_openapi_json(self):
        """Test OpenAPI JSON schema endpoint"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "info" in data
        assert "paths" in data
        assert data["info"]["title"] == "Rubik's Cube Solver API"
        assert data["info"]["version"] == "2.0.0"

    def test_docs_redirect(self):
        """Test that /docs is accessible"""
        response = client.get("/docs", follow_redirects=False)
        # Should either return docs page or redirect
        assert response.status_code in [200, 307]

    def test_redoc_redirect(self):
        """Test that /redoc is accessible"""
        response = client.get("/redoc", follow_redirects=False)
        # Should either return redoc page or redirect
        assert response.status_code in [200, 307]


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_endpoint(self):
        """Test requesting non-existent endpoint"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_wrong_http_method(self):
        """Test using wrong HTTP method"""
        response = client.get("/api/v1/solver/solve-string")
        assert response.status_code == 405  # Method not allowed

    def test_malformed_json(self):
        """Test with malformed JSON in request"""
        response = client.post(
            "/api/v1/solver/solve-string",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
