"""Claude (Anthropic) LLM provider implementation.

This module provides the Claude-specific implementation of the LLMProvider interface,
wrapping the Anthropic API client with the unified provider abstraction.
"""

import os
from typing import Any, Dict, List, Optional

import anthropic

from backend.llm.providers.base import LLMProvider, ProviderResponse, ToolCall
from backend.llm.providers.config import LLMConfig


class ClaudeProvider(LLMProvider):
    """Claude API provider implementation.

    Uses the Anthropic Python SDK to interact with Claude models.
    Supports tool calling and structured outputs via tool use.

    Attributes:
        client: Anthropic API client instance.
        model: Model ID to use for completions.

    Example:
        provider = ClaudeProvider()
        response = provider.create_message(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ):
        """Initialize the Claude provider.

        Args:
            api_key: Anthropic API key. Uses CLAUDE_API_KEY env var if not provided.
            config: LLM configuration. Uses defaults if not provided.
        """
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.config = config or LLMConfig()
        self.model = self.config.claude_model
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "No API key provided. Set CLAUDE_API_KEY environment variable."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2000,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Create a message using Claude API.

        Args:
            system_prompt: System-level instructions.
            messages: Conversation history.
            tools: Tool definitions in Anthropic format.
            max_tokens: Maximum response tokens.
            tool_choice: Optional tool choice constraint.

        Returns:
            ProviderResponse with content, tool calls, and stop reason.
        """
        # Build system prompt with cache control
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]

        # Build request parameters
        request_params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages
        }

        if tools:
            request_params["tools"] = tools

        if tool_choice:
            request_params["tool_choice"] = tool_choice

        # Make API call
        response = self.client.messages.create(**request_params)

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: anthropic.types.Message) -> ProviderResponse:
        """Parse Anthropic API response into unified format.

        Args:
            response: Raw Anthropic API response.

        Returns:
            ProviderResponse with extracted content and tool calls.
        """
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))

        # Map stop reason
        stop_reason = response.stop_reason
        if stop_reason == "tool_use":
            stop_reason = "tool_use"
        elif stop_reason == "end_turn":
            stop_reason = "end_turn"
        elif stop_reason == "max_tokens":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_response=response
        )

    def format_tool_result(
        self,
        tool_call_id: str,
        result: str,
        is_error: bool = False
    ) -> Dict[str, Any]:
        """Format tool result for Claude API.

        Args:
            tool_call_id: ID of the tool call.
            result: JSON string result.
            is_error: Whether this is an error result.

        Returns:
            Tool result in Claude message format.
        """
        tool_result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result
        }
        if is_error:
            tool_result["is_error"] = True
        return tool_result

    def format_assistant_message(
        self,
        content: Optional[str],
        tool_calls: List[ToolCall]
    ) -> Dict[str, Any]:
        """Format assistant message with tool calls for conversation history.

        Args:
            content: Text content from assistant.
            tool_calls: Tool calls made by assistant.

        Returns:
            Assistant message in Claude format.
        """
        message_content: List[Dict[str, Any]] = []

        if content:
            message_content.append({
                "type": "text",
                "text": content
            })

        for tc in tool_calls:
            message_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments
            })

        return {
            "role": "assistant",
            "content": message_content
        }
