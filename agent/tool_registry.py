"""Flexible tool registry for the Portfolio Analysis Agent.

This module provides a decorator-based tool registration system that makes
it easy to add new tools to the agent. Tools are automatically registered
with their schemas and handlers.

Example:
    from agent.tool_registry import tool, ToolRegistry

    @tool(
        name="my_tool",
        description="Does something useful",
        parameters={
            "param1": {"type": "string", "description": "A parameter"}
        },
        required=["param1"]
    )
    def my_tool_handler(param1: str) -> dict:
        return {"result": param1}

    # Get all tools for LLM
    registry = ToolRegistry()
    tools = registry.get_tool_definitions()
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the agent.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description for the LLM.
        parameters: JSON Schema for the tool's parameters.
        required: List of required parameter names.
        handler: Function to execute when the tool is called.
        category: Optional category for grouping tools.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]
    handler: Callable[..., Any]
    category: str = "general"


class ToolRegistry:
    """Central registry for all agent tools.

    Provides methods to register tools, get tool definitions for the LLM,
    and execute tools by name.

    Attributes:
        _tools: Dictionary mapping tool names to their definitions.
        _context: Shared context available to all tools (e.g., portfolio).
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, ToolDefinition] = {}
    _context: Dict[str, Any] = {}

    def __new__(cls) -> "ToolRegistry":
        """Singleton pattern to ensure one registry across the application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._context = {}
        return cls._instance

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool with the registry.

        Args:
            tool_def: The tool definition to register.
        """
        self._tools[tool_def.name] = tool_def

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name.

        Args:
            name: The tool name to look up.

        Returns:
            The tool definition or None if not found.
        """
        return self._tools.get(name)

    def get_tool_definitions(self, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get tool definitions in Anthropic API format.

        Args:
            categories: Optional list of categories to filter by.

        Returns:
            List of tool definitions ready for the LLM API.
        """
        definitions = []
        for tool_def in self._tools.values():
            if categories and tool_def.category not in categories:
                continue
            definitions.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "input_schema": {
                    "type": "object",
                    "properties": tool_def.parameters,
                    "required": tool_def.required
                }
            })
        return definitions

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with the given arguments.

        Args:
            name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            Result from the tool execution.

        Raises:
            ValueError: If the tool is not found.
        """
        tool_def = self._tools.get(name)
        if not tool_def:
            raise ValueError(f"Unknown tool: {name}")

        # Inject context into arguments if the handler accepts it
        try:
            return tool_def.handler(**arguments, _context=self._context)
        except TypeError:
            # Handler doesn't accept _context
            return tool_def.handler(**arguments)

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value available to all tools.

        Args:
            key: Context key.
            value: Context value.
        """
        self._context[key] = value

    def get_context(self, key: str) -> Optional[Any]:
        """Get a context value.

        Args:
            key: Context key to retrieve.

        Returns:
            The context value or None if not set.
        """
        return self._context.get(key)

    def clear_context(self) -> None:
        """Clear all context values."""
        self._context.clear()

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def list_by_category(self) -> Dict[str, List[str]]:
        """Get tools grouped by category.

        Returns:
            Dictionary mapping categories to lists of tool names.
        """
        by_category: Dict[str, List[str]] = {}
        for name, tool_def in self._tools.items():
            category = tool_def.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(name)
        return by_category


def tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None,
    category: str = "general"
) -> Callable:
    """Decorator to register a function as an agent tool.

    Args:
        name: Unique identifier for the tool.
        description: Description for the LLM to understand when to use this tool.
        parameters: JSON Schema properties for the tool's parameters.
        required: List of required parameter names.
        category: Category for grouping related tools.

    Returns:
        Decorator function that registers the tool.

    Example:
        @tool(
            name="calculate_var",
            description="Calculate Value at Risk for the portfolio",
            parameters={
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level (e.g., 0.95 for 95%)"
                }
            },
            required=["confidence_level"],
            category="risk"
        )
        def calculate_var(confidence_level: float) -> dict:
            # Implementation
            return {"var": 0.05}
    """
    def decorator(func: Callable) -> Callable:
        registry = ToolRegistry()
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            required=required or [],
            handler=func,
            category=category
        )
        registry.register(tool_def)
        return func
    return decorator
