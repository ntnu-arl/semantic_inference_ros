"""Configurations for the FastAPI and OpenAI clients."""
from dataclasses import dataclass
from typing import Optional

from semantic_inference_python.config import Config


@dataclass
class FastAPIClientConfig(Config):
    """Configuration for the FastAPI client."""

    base_url: str = ""
    timeout: int = 3600  # seconds
    wait_interval: int = 20  # seconds
    submit_endpoint: str = "generate"
    result_endpoint: str = "result"
    verify_ssl: bool = True
    logging: bool = True
    deterministic: bool = False

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@dataclass
class OpenAIClientConfig(Config):
    """Configuration for the OpenAI client."""

    model: str = "o4-mini"
    max_tokens: Optional[int] = None

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
