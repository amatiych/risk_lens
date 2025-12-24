"""Tests for the MCP server module."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

from backend.mcp.server import (
    MCPServer,
    Tool,
    ToolParameter,
    ToolResult,
    Resource,
    MCPErrorCode,
)
from backend.mcp.tools import (
    register_portfolio_tools,
    register_risk_tools,
    register_market_tools,
    create_risk_lens_server,
)


class TestToolParameter:
    """Tests for ToolParameter class."""

    def test_parameter_creation(self):
        """Should create parameter with required fields."""
        param = ToolParameter(
            name="ticker",
            type="string",
            description="Stock ticker symbol",
        )

        assert param.name == "ticker"
        assert param.type == "string"
        assert param.required is True

    def test_parameter_with_defaults(self):
        """Should support default values."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )

        assert param.required is False
        assert param.default == 10

    def test_parameter_with_enum(self):
        """Should support enum values."""
        param = ToolParameter(
            name="period",
            type="string",
            description="Time period",
            enum=["1d", "1w", "1m"],
        )

        assert param.enum == ["1d", "1w", "1m"]

    def test_to_schema(self):
        """Should convert to JSON Schema."""
        param = ToolParameter(
            name="ticker",
            type="string",
            description="Stock ticker",
            enum=["AAPL", "MSFT"],
        )

        schema = param.to_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "Stock ticker"
        assert schema["enum"] == ["AAPL", "MSFT"]


class TestTool:
    """Tests for Tool class."""

    def test_tool_creation(self):
        """Should create tool with required fields."""
        tool = Tool(
            name="get_price",
            description="Get stock price",
        )

        assert tool.name == "get_price"
        assert tool.description == "Get stock price"

    def test_tool_with_parameters(self):
        """Should include parameters."""
        tool = Tool(
            name="get_price",
            description="Get stock price",
            parameters=[
                ToolParameter("ticker", "string", "Ticker symbol"),
            ],
        )

        assert len(tool.parameters) == 1

    def test_to_schema(self):
        """Should convert to MCP schema format."""
        tool = Tool(
            name="get_price",
            description="Get stock price",
            parameters=[
                ToolParameter("ticker", "string", "Ticker symbol", required=True),
                ToolParameter("date", "string", "Date", required=False),
            ],
        )

        schema = tool.to_schema()

        assert schema["name"] == "get_price"
        assert schema["description"] == "Get stock price"
        assert "inputSchema" in schema
        assert "ticker" in schema["inputSchema"]["properties"]
        assert "ticker" in schema["inputSchema"]["required"]
        assert "date" not in schema["inputSchema"]["required"]


class TestToolResult:
    """Tests for ToolResult class."""

    def test_result_string_content(self):
        """Should handle string content."""
        result = ToolResult("Success message")

        mcp = result.to_mcp()

        assert mcp["content"][0]["type"] == "text"
        assert mcp["content"][0]["text"] == "Success message"
        assert mcp["isError"] is False

    def test_result_dict_content(self):
        """Should serialize dict content to JSON."""
        result = ToolResult({"value": 123, "name": "test"})

        mcp = result.to_mcp()

        assert "123" in mcp["content"][0]["text"]
        assert "test" in mcp["content"][0]["text"]

    def test_error_result(self):
        """Should mark error results."""
        result = ToolResult("Error occurred", is_error=True)

        mcp = result.to_mcp()

        assert mcp["isError"] is True


class TestResource:
    """Tests for Resource class."""

    def test_resource_creation(self):
        """Should create resource."""
        resource = Resource(
            uri="portfolio://100",
            name="Portfolio 100",
            description="Test portfolio",
        )

        assert resource.uri == "portfolio://100"
        assert resource.mime_type == "application/json"

    def test_to_schema(self):
        """Should convert to MCP schema."""
        resource = Resource(
            uri="portfolio://100",
            name="Portfolio 100",
            description="Test portfolio",
            mime_type="text/csv",
        )

        schema = resource.to_schema()

        assert schema["uri"] == "portfolio://100"
        assert schema["mimeType"] == "text/csv"


class TestMCPServer:
    """Tests for MCPServer class."""

    @pytest.fixture
    def server(self):
        """Create a test server."""
        return MCPServer(
            name="test-server",
            version="1.0.0",
            description="Test server",
        )

    def test_server_creation(self, server):
        """Should create server with metadata."""
        assert server.name == "test-server"
        assert server.version == "1.0.0"

    def test_register_tool_decorator(self, server):
        """Should register tools via decorator."""
        @server.tool(
            name="test_tool",
            description="A test tool",
            parameters=[
                ToolParameter("arg1", "string", "First argument"),
            ],
        )
        async def test_tool(arg1: str):
            return ToolResult(f"Got: {arg1}")

        assert "test_tool" in server._tools
        assert server._tools["test_tool"].handler is not None

    def test_register_tool_direct(self, server):
        """Should register tools directly."""
        tool = Tool(
            name="direct_tool",
            description="Directly registered",
        )
        server.register_tool(tool)

        assert "direct_tool" in server._tools

    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        """Should handle initialize request."""
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })

        assert response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "serverInfo" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "test-server"

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        """Should list registered tools."""
        @server.tool(name="tool1", description="First tool")
        async def tool1():
            return ToolResult("ok")

        @server.tool(name="tool2", description="Second tool")
        async def tool2():
            return ToolResult("ok")

        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        })

        tools = response["result"]["tools"]
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert "tool1" in names
        assert "tool2" in names

    @pytest.mark.asyncio
    async def test_handle_tools_call(self, server):
        """Should execute tool and return result."""
        @server.tool(
            name="greet",
            description="Greet someone",
            parameters=[ToolParameter("name", "string", "Name to greet")],
        )
        async def greet(name: str):
            return ToolResult(f"Hello, {name}!")

        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "greet",
                "arguments": {"name": "World"},
            },
        })

        result = response["result"]
        assert "Hello, World!" in result["content"][0]["text"]
        assert result["isError"] is False

    @pytest.mark.asyncio
    async def test_handle_tools_call_error(self, server):
        """Should handle tool errors gracefully."""
        @server.tool(name="failing_tool", description="Always fails")
        async def failing_tool():
            raise ValueError("Intentional failure")

        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "failing_tool", "arguments": {}},
        })

        result = response["result"]
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, server):
        """Should return error for unknown tool."""
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        })

        result = response["result"]
        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        """Should return error for unknown method."""
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
            "params": {},
        })

        assert "error" in response
        assert response["error"]["code"] == MCPErrorCode.METHOD_NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_handle_resources_list(self, server):
        """Should list resources."""
        server.register_resource(
            Resource("test://resource", "Test", "A test resource"),
            handler=lambda: {"data": "value"},
        )

        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/list",
            "params": {},
        })

        resources = response["result"]["resources"]
        assert len(resources) == 1
        assert resources[0]["uri"] == "test://resource"

    @pytest.mark.asyncio
    async def test_handle_resources_read(self, server):
        """Should read resource content."""
        server.register_resource(
            Resource("test://data", "Data", "Test data"),
            handler=lambda: {"value": 42},
        )

        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "test://data"},
        })

        contents = response["result"]["contents"]
        assert len(contents) == 1
        assert "42" in contents[0]["text"]

    def test_get_tools_for_claude(self, server):
        """Should format tools for Claude API."""
        @server.tool(
            name="test",
            description="Test tool",
            parameters=[ToolParameter("arg", "string", "Argument")],
        )
        async def test(arg: str):
            return ToolResult("ok")

        claude_tools = server.get_tools_for_claude()

        assert len(claude_tools) == 1
        assert claude_tools[0]["name"] == "test"
        assert "input_schema" in claude_tools[0]


class TestRiskLensTools:
    """Tests for Risk Lens specific tools."""

    @pytest.fixture
    def server(self):
        """Create server with all tools registered."""
        return create_risk_lens_server()

    def test_all_tools_registered(self, server):
        """Should register all expected tools."""
        expected_tools = [
            "get_portfolio",
            "list_portfolios",
            "get_portfolio_holdings",
            "calculate_var",
            "get_risk_contributors",
            "get_factor_exposures",
            "lookup_stock",
            "get_stock_price",
            "get_historical_prices",
            "get_market_news",
        ]

        for tool_name in expected_tools:
            assert tool_name in server._tools, f"Missing tool: {tool_name}"

    def test_resources_registered(self, server):
        """Should register expected resources."""
        assert "risk-lens://portfolios" in server._resources

    @pytest.mark.asyncio
    async def test_list_portfolios_tool(self, server):
        """Should list portfolios."""
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "list_portfolios",
                "arguments": {},
            },
        })

        result = response["result"]
        # Should not error even if no portfolios exist
        assert "content" in result

    @pytest.mark.asyncio
    @patch('yfinance.Ticker')
    async def test_lookup_stock_tool(self, mock_ticker, server):
        """Should look up stock info."""
        mock_instance = Mock()
        mock_instance.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "longBusinessSummary": "Apple designs, manufactures...",
            "marketCap": 3000000000000,
            "currency": "USD",
        }
        mock_ticker.return_value = mock_instance

        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "lookup_stock",
                "arguments": {"ticker": "AAPL"},
            },
        })

        result = response["result"]
        assert result["isError"] is False
        content = json.loads(result["content"][0]["text"])
        assert content["ticker"] == "AAPL"
        assert content["name"] == "Apple Inc."

    @pytest.mark.asyncio
    @patch('yfinance.Ticker')
    async def test_get_stock_price_tool(self, mock_ticker, server):
        """Should get stock price."""
        mock_instance = Mock()
        mock_instance.info = {
            "currentPrice": 150.00,
            "previousClose": 148.50,
        }
        mock_ticker.return_value = mock_instance

        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_stock_price",
                "arguments": {"ticker": "AAPL"},
            },
        })

        result = response["result"]
        assert result["isError"] is False


class TestIntegration:
    """Integration tests for MCP server."""

    def test_full_server_workflow(self):
        """Should handle complete server workflow."""
        server = create_risk_lens_server()

        # Verify tools are accessible
        assert len(server._tools) >= 10

        # Verify tool schemas are valid
        for tool in server._tools.values():
            schema = tool.to_schema()
            assert "name" in schema
            assert "description" in schema
            assert "inputSchema" in schema

        # Verify Claude format export
        claude_tools = server.get_tools_for_claude()
        assert len(claude_tools) >= 10

        for tool in claude_tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_protocol_sequence(self):
        """Should handle standard MCP protocol sequence."""
        server = create_risk_lens_server()

        # 1. Initialize
        init_response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        })
        assert "result" in init_response
        assert server._initialized is True

        # 2. Initialized notification
        await server.handle_request({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {},
        })

        # 3. List tools
        tools_response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        assert len(tools_response["result"]["tools"]) > 0

        # 4. List resources
        resources_response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {},
        })
        assert "resources" in resources_response["result"]
