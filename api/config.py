"""
Configuration Management

Centralized configuration using environment variables with Pydantic
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # CORS Configuration
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost"

    # ML Model Configuration
    ml_model_path: str = "./ml_color_model.pkl"
    ml_scaler_path: str = "./ml_color_scaler.pkl"
    ml_metadata_path: str = "./ml_color_metadata.json"
    ml_confidence_threshold: float = 0.7

    # Logging
    log_level: str = "INFO"

    # Application
    app_name: str = "Rubik's Cube Solver API"
    app_version: str = "2.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
