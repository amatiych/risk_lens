"""LLM provider abstraction layer.

This module provides a unified interface for different LLM providers
(Claude, OpenAI) with automatic format conversion and consistent response handling.

Example:
    from backend.llm.providers import LLMConfig, get_provider

    # Default (Claude)
    config = LLMConfig()
    provider = get_provider(config)

    # OpenAI
    config = LLMConfig(provider="openai")
    provider = get_provider(config)

    # Use the provider
    response = provider.create_message(
        system_prompt="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

from backend.llm.providers.base import LLMProvider, ProviderResponse, ToolCall
from backend.llm.providers.config import LLMConfig
from backend.llm.providers.claude_provider import ClaudeProvider
from backend.llm.providers.openai_provider import OpenAIProvider


def get_provider(config: LLMConfig | None = None) -> LLMProvider:
    """Factory function to get the appropriate LLM provider.

    Args:
        config: LLM configuration specifying which provider to use.
                Defaults to LLMConfig() (Claude).

    Returns:
        LLMProvider instance for the configured provider.

    Raises:
        ValueError: If an unknown provider is specified.

    Example:
        # Default Claude provider
        provider = get_provider()

        # OpenAI provider
        provider = get_provider(LLMConfig(provider="openai"))

        # From environment
        provider = get_provider(LLMConfig.from_env())
    """
    if config is None:
        config = LLMConfig()

    if config.provider == "claude":
        return ClaudeProvider(config=config)
    elif config.provider == "openai":
        return OpenAIProvider(config=config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


__all__ = [
    "LLMProvider",
    "ProviderResponse",
    "ToolCall",
    "LLMConfig",
    "ClaudeProvider",
    "OpenAIProvider",
    "get_provider",
]
