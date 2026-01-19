"""OpenAI LLM provider implementation.

This module provides the OpenAI-specific implementation of the LLMProvider interface,
wrapping the OpenAI API client with the unified provider abstraction.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend.llm.providers.base import LLMProvider, ProviderResponse, ToolCall
from backend.llm.providers.config import LLMConfig


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation.

    Uses the OpenAI Python SDK to interact with GPT models.
    Converts tools from Anthropic format to OpenAI function calling format.

    Attributes:
        client: OpenAI API client instance.
        model: Model ID to use for completions.

    Example:
        provider = OpenAIProvider()
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
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            config: LLM configuration. Uses defaults if not provided.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.config = config or LLMConfig(provider="openai")
        self.model = self.config.openai_model
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "No API key provided. Set OPENAI_API_KEY environment variable."
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools from Anthropic format to OpenAI function calling format.

        Anthropic format:
            {"name": "...", "description": "...", "input_schema": {...}}

        OpenAI format:
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

        Args:
            tools: List of tool definitions in Anthropic format.

        Returns:
            List of tool definitions in OpenAI format.
        """
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
                }
            }
            # OpenAI uses "strict" at the function level for structured outputs
            if tool.get("strict"):
                openai_tool["function"]["strict"] = True
            openai_tools.append(openai_tool)
        return openai_tools

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages from Anthropic format to OpenAI format.

        Handles tool results which have different formats between providers.

        Args:
            messages: Messages in Anthropic-compatible format.

        Returns:
            Messages in OpenAI format.
        """
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")

            if role == "user":
                # Check if content is a list (could contain tool_results)
                if isinstance(content, list):
                    # Extract tool results and convert to OpenAI format
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item.get("content", "")
                            })
                        elif isinstance(item, dict) and item.get("type") == "text":
                            openai_messages.append({
                                "role": "user",
                                "content": item["text"]
                            })
                else:
                    openai_messages.append({
                        "role": "user",
                        "content": content if isinstance(content, str) else str(content)
                    })

            elif role == "assistant":
                # Handle assistant messages with tool calls
                if isinstance(content, list):
                    text_content = None
                    tool_calls = []

                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_content = item["text"]
                            elif item.get("type") == "tool_use":
                                tool_calls.append({
                                    "id": item["id"],
                                    "type": "function",
                                    "function": {
                                        "name": item["name"],
                                        "arguments": json.dumps(item["input"])
                                    }
                                })

                    assistant_msg: Dict[str, Any] = {"role": "assistant"}
                    if text_content:
                        assistant_msg["content"] = text_content
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls

                    openai_messages.append(assistant_msg)
                else:
                    openai_messages.append({
                        "role": "assistant",
                        "content": content if isinstance(content, str) else str(content) if content else ""
                    })

            elif role == "tool":
                # Already in OpenAI format
                openai_messages.append(msg)

            else:
                # Pass through other message types
                openai_messages.append(msg)

        return openai_messages

    def create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2000,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Create a message using OpenAI API.

        Args:
            system_prompt: System-level instructions.
            messages: Conversation history in Anthropic-compatible format.
            tools: Tool definitions in Anthropic format (will be converted).
            max_tokens: Maximum response tokens.
            tool_choice: Optional tool choice constraint (will be converted).

        Returns:
            ProviderResponse with content, tool calls, and stop reason.
        """
        # Build messages with system prompt
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(self._convert_messages(messages))

        # Build request parameters
        request_params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": openai_messages
        }

        if tools:
            request_params["tools"] = self.convert_tools(tools)

        if tool_choice:
            # Convert Anthropic tool_choice format to OpenAI
            if tool_choice.get("type") == "tool":
                request_params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]}
                }
            elif tool_choice.get("type") == "any":
                request_params["tool_choice"] = "required"
            elif tool_choice.get("type") == "auto":
                request_params["tool_choice"] = "auto"

        # Make API call
        response = self.client.chat.completions.create(**request_params)

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> ProviderResponse:
        """Parse OpenAI API response into unified format.

        Args:
            response: Raw OpenAI API response.

        Returns:
            ProviderResponse with extracted content and tool calls.
        """
        choice = response.choices[0]
        message = choice.message

        content = message.content
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments
                ))

        # Map finish reason
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
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
        """Format tool result for OpenAI API.

        Args:
            tool_call_id: ID of the tool call.
            result: JSON string result.
            is_error: Whether this is an error result (not used by OpenAI).

        Returns:
            Tool result in OpenAI message format.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }

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
            Assistant message in OpenAI format.
        """
        message: Dict[str, Any] = {"role": "assistant"}

        if content:
            message["content"] = content

        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]

        return message
