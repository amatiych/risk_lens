"""Configuration for LLM providers.

This module provides configuration options for selecting and configuring
LLM providers (Claude, OpenAI) used in risk analysis.
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class LLMConfig:
    """Configuration for LLM provider selection and settings.

    Attributes:
        provider: Which LLM provider to use ("claude" or "openai").
        claude_model: Model ID for Claude API.
        openai_model: Model ID for OpenAI API.

    Example:
        # Default (Claude)
        config = LLMConfig()

        # OpenAI
        config = LLMConfig(provider="openai")

        # From environment
        config = LLMConfig.from_env()
    """
    provider: Literal["claude", "openai"] = "claude"
    claude_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables.

        Environment variables:
            LLM_PROVIDER: "claude" or "openai" (default: "claude")
            CLAUDE_MODEL: Claude model ID (optional)
            OPENAI_MODEL: OpenAI model ID (optional)

        Returns:
            LLMConfig instance configured from environment.
        """
        provider = os.environ.get("LLM_PROVIDER", "claude").lower()
        if provider not in ("claude", "openai"):
            provider = "claude"

        claude_model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")

        return cls(
            provider=provider,  # type: ignore
            claude_model=claude_model,
            openai_model=openai_model
        )

    @property
    def model(self) -> str:
        """Get the model ID for the currently selected provider."""
        if self.provider == "openai":
            return self.openai_model
        return self.claude_model
