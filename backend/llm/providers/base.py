"""Base classes for LLM provider abstraction.

This module defines the abstract interface that all LLM providers must implement,
along with shared data types for consistent response handling across providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM.

    Attributes:
        id: Unique identifier for this tool call (used for result matching).
        name: Name of the tool/function to execute.
        arguments: Dictionary of arguments to pass to the tool.
    """
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ProviderResponse:
    """Unified response format from LLM providers.

    Attributes:
        content: Text content from the response (may be None if only tool calls).
        tool_calls: List of tool calls requested by the LLM.
        stop_reason: Why the response ended - "end_turn", "tool_use", or "max_tokens".
        raw_response: Original provider-specific response object for debugging.
    """
    content: Optional[str]
    tool_calls: List[ToolCall]
    stop_reason: str
    raw_response: Any = field(default=None, repr=False)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations (Claude, OpenAI, etc.) must implement this interface
    to ensure consistent behavior in the RiskAnalyzer and other LLM-dependent code.
    """

    @abstractmethod
    def create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2000,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Create a message/completion with the LLM.

        Args:
            system_prompt: System-level instructions for the LLM.
            messages: Conversation history in provider-agnostic format.
            tools: List of tool definitions in Anthropic format (will be converted).
            max_tokens: Maximum tokens in the response.
            tool_choice: Optional constraint on which tool to use.

        Returns:
            ProviderResponse with content, tool calls, and stop reason.
        """
        pass

    @abstractmethod
    def format_tool_result(
        self,
        tool_call_id: str,
        result: str,
        is_error: bool = False
    ) -> Dict[str, Any]:
        """Format a tool execution result for the next message.

        Args:
            tool_call_id: ID of the tool call this result corresponds to.
            result: JSON string result from tool execution.
            is_error: Whether the tool execution resulted in an error.

        Returns:
            Provider-specific message format for tool results.
        """
        pass

    @abstractmethod
    def format_assistant_message(
        self,
        content: Optional[str],
        tool_calls: List[ToolCall]
    ) -> Dict[str, Any]:
        """Format an assistant message with optional tool calls for conversation history.

        Args:
            content: Text content from the assistant (may be None).
            tool_calls: List of tool calls made by the assistant.

        Returns:
            Provider-specific message format for assistant responses.
        """
        pass

    def convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools from Anthropic format to provider-specific format.

        Default implementation returns tools unchanged (for Anthropic-compatible providers).
        Override in providers that use different tool formats.

        Args:
            tools: List of tool definitions in Anthropic format.

        Returns:
            List of tool definitions in provider-specific format.
        """
        return tools
