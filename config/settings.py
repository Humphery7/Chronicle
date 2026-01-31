"""Application settings using Pydantic Settings for type-safe configuration."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


    app_name: str = Field(default="AudioDiary", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "production"] = Field(
        default="development", description="Runtime environment"
    )
    debug: bool = Field(default=False, description="Debug mode")


    api_v1_prefix: str = Field(default="/api/v1", description="API v1 path prefix")
    
 
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )


    huggingface_api_key: str = Field(default="", description="Hugging Face API token")
    whisper_model: str = Field(
        default="openai/whisper-large-v3", description="Whisper ASR model"
    )
    tts_model: str = Field(
        default="facebook/mms-tts-eng", description="Text-to-speech model"
    )
    hf_timeout: int = Field(default=60, description="HuggingFace API timeout in seconds")


    llm_provider: Literal["openai", "anthropic", "cohere", "huggingface"] = Field(
        default="openai", description="LLM provider for chat"
    )
    llm_api_key: str = Field(default="", description="LLM API key")
    llm_model: str = Field(
        default="gpt-4o-mini", description="LLM model name"
    )
    llm_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature"
    )
    llm_max_tokens: int = Field(
        default=500, ge=1, description="Maximum tokens in LLM response"
    )


    conversation_memory_size: int = Field(
        default=5, ge=1, le=20, description="Number of previous messages to keep"
    )

    # File Upload Limitation
    max_audio_file_size_mb: int = Field(
        default=25, ge=1, le=100, description="Maximum audio file size in MB"
    )
    allowed_audio_formats: set[str] = Field(
        default={"audio/wav", "audio/mpeg", "audio/mp4", "audio/x-m4a"},
        description="Allowed audio MIME types",
    )


    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    @property
    def max_audio_file_size_bytes(self) -> int:
        """Convert MB to bytes for file size validation."""
        return self.max_audio_file_size_mb * 1024 * 1024

    def validate_required_keys(self) -> list[str]:
        """Validate that required API keys are set. Returns list of missing keys."""
        missing = []
        
        if not self.huggingface_api_key:
            missing.append("HUGGINGFACE_API_KEY")
        
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        
        return missing


# Global settings instance
settings = Settings()
