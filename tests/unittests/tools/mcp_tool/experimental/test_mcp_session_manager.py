# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import hashlib
from io import StringIO
import json
import sys
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.mcp_tool.experimental.mcp_session_manager import (
    MCPSessionManager,
)
from mcp import StdioServerParameters
import pytest


class MockClientSession:
  """Mock ClientSession for testing."""

  def __init__(self):
    self._read_stream = Mock()
    self._write_stream = Mock()
    self._read_stream._closed = False
    self._write_stream._closed = False
    self.initialize = AsyncMock()


class MockSessionLifecycle:
  """Mock SessionLifecycle for testing."""

  def __init__(self, session=None):
    self._session = session or MockClientSession()
    self.start = AsyncMock(return_value=self._session)
    self.close = AsyncMock()

  @property
  def session(self):
    return self._session


class MockAsyncExitStack:
  """Mock AsyncExitStack for testing."""

  def __init__(self):
    self.aclose = AsyncMock()
    self.enter_async_context = AsyncMock()

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.aclose()


class TestMCPSessionManager:
  """Test suite for MCPSessionManager class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_stdio_params = StdioServerParameters(
        command="test_command", args=[]
    )
    self.mock_stdio_connection_params = StdioConnectionParams(
        server_params=self.mock_stdio_params, timeout=5.0
    )

  def test_init_with_stdio_server_parameters(self):
    """Test initialization with StdioServerParameters (deprecated)."""
    with patch(
        "google.adk.tools.mcp_tool.experimental.mcp_session_manager.logger"
    ) as mock_logger:
      manager = MCPSessionManager(self.mock_stdio_params)

      # Should log deprecation warning
      mock_logger.warning.assert_called_once()
      assert "StdioServerParameters is not recommended" in str(
          mock_logger.warning.call_args
      )

      # Should convert to StdioConnectionParams
      assert isinstance(manager._connection_params, StdioConnectionParams)
      assert manager._connection_params.server_params == self.mock_stdio_params
      assert manager._connection_params.timeout == 5

  def test_init_with_stdio_connection_params(self):
    """Test initialization with StdioConnectionParams."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    assert manager._connection_params == self.mock_stdio_connection_params
    assert manager._errlog == sys.stderr
    assert manager._sessions == {}

  def test_init_with_sse_connection_params(self):
    """Test initialization with SseConnectionParams."""
    sse_params = SseConnectionParams(
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token"},
        timeout=10.0,
    )
    manager = MCPSessionManager(sse_params)

    assert manager._connection_params == sse_params

  def test_init_with_streamable_http_params(self):
    """Test initialization with StreamableHTTPConnectionParams."""
    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp", timeout=15.0
    )
    manager = MCPSessionManager(http_params)

    assert manager._connection_params == http_params

  def test_generate_session_key_stdio(self):
    """Test session key generation for stdio connections."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # For stdio, headers should be ignored and return constant key
    key1 = manager._generate_session_key({"Authorization": "Bearer token"})
    key2 = manager._generate_session_key(None)

    assert key1 == "stdio_session"
    assert key2 == "stdio_session"
    assert key1 == key2

  def test_generate_session_key_sse(self):
    """Test session key generation for SSE connections."""
    sse_params = SseConnectionParams(url="https://example.com/mcp")
    manager = MCPSessionManager(sse_params)

    headers1 = {"Authorization": "Bearer token1"}
    headers2 = {"Authorization": "Bearer token2"}

    key1 = manager._generate_session_key(headers1)
    key2 = manager._generate_session_key(headers2)
    key3 = manager._generate_session_key(headers1)

    # Different headers should generate different keys
    assert key1 != key2
    # Same headers should generate same key
    assert key1 == key3

    # Should be deterministic hash
    headers_json = json.dumps(headers1, sort_keys=True)
    expected_hash = hashlib.md5(headers_json.encode()).hexdigest()
    assert key1 == f"session_{expected_hash}"

  def test_merge_headers_stdio(self):
    """Test header merging for stdio connections."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Stdio connections don't support headers
    headers = manager._merge_headers({"Authorization": "Bearer token"})
    assert headers is None

  def test_merge_headers_sse(self):
    """Test header merging for SSE connections."""
    base_headers = {"Content-Type": "application/json"}
    sse_params = SseConnectionParams(
        url="https://example.com/mcp", headers=base_headers
    )
    manager = MCPSessionManager(sse_params)

    # With additional headers
    additional = {"Authorization": "Bearer token"}
    merged = manager._merge_headers(additional)

    expected = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
    }
    assert merged == expected

  def test_is_session_disconnected(self):
    """Test session disconnection detection."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Create mock session
    session = MockClientSession()

    # Not disconnected
    assert not manager._is_session_disconnected(session)

    # Disconnected - read stream closed
    session._read_stream._closed = True
    assert manager._is_session_disconnected(session)

  @pytest.mark.asyncio
  async def test_create_session_stdio_new(self):
    """Test creating a new stdio session."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    mock_session = MockClientSession()
    mock_lifecycle = MockSessionLifecycle(mock_session)

    with patch(
        "google.adk.tools.mcp_tool.experimental.mcp_session_manager.stdio_client"
    ) as mock_stdio:
      with patch(
          "google.adk.tools.mcp_tool.experimental.mcp_session_manager.SessionLifecycle"
      ) as mock_lifecycle_class:
        # Setup mocks
        mock_stdio.return_value = AsyncMock()
        mock_lifecycle_class.return_value = mock_lifecycle

        # Create session
        session = await manager.create_session()

        # Verify session creation
        assert session == mock_session
        assert len(manager._sessions) == 1
        assert "stdio_session" in manager._sessions

        # Verify SessionLifecycle was created with correct params
        mock_lifecycle_class.assert_called_once()
        call_kwargs = mock_lifecycle_class.call_args[1]
        assert call_kwargs["timeout"] == 5.0
        assert call_kwargs["is_stdio"] is True

        # Verify lifecycle.start() was called
        mock_lifecycle.start.assert_called_once()

  @pytest.mark.asyncio
  async def test_create_session_reuse_existing(self):
    """Test reusing an existing connected session."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Create mock existing session and lifecycle
    existing_session = MockClientSession()
    existing_lifecycle = MockSessionLifecycle(existing_session)
    manager._sessions["stdio_session"] = existing_lifecycle

    # Session is connected
    existing_session._read_stream._closed = False
    existing_session._write_stream._closed = False

    session = await manager.create_session()

    # Should reuse existing session
    assert session == existing_session
    assert len(manager._sessions) == 1

    # Should not create new lifecycle
    existing_lifecycle.start.assert_not_called()

  @pytest.mark.asyncio
  async def test_create_session_replace_disconnected(self):
    """Test replacing a disconnected session."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Create mock existing disconnected session
    existing_session = MockClientSession()
    existing_session._read_stream._closed = True  # Disconnected
    existing_lifecycle = MockSessionLifecycle(existing_session)
    manager._sessions["stdio_session"] = existing_lifecycle

    # New session mock
    new_session = MockClientSession()
    new_lifecycle = MockSessionLifecycle(new_session)

    with patch(
        "google.adk.tools.mcp_tool.experimental.mcp_session_manager.stdio_client"
    ) as mock_stdio:
      with patch(
          "google.adk.tools.mcp_tool.experimental.mcp_session_manager.SessionLifecycle"
      ) as mock_lifecycle_class:
        mock_stdio.return_value = AsyncMock()
        mock_lifecycle_class.return_value = new_lifecycle

        session = await manager.create_session()

        # Old lifecycle should be closed
        existing_lifecycle.close.assert_called_once()

        # Should return new session
        assert session == new_session
        assert manager._sessions["stdio_session"] == new_lifecycle

  @pytest.mark.asyncio
  async def test_create_session_error(self):
    """Test session creation error handling."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    mock_lifecycle = MockSessionLifecycle()
    mock_lifecycle.start.side_effect = Exception("Connection failed")

    with patch(
        "google.adk.tools.mcp_tool.experimental.mcp_session_manager.stdio_client"
    ) as mock_stdio:
      with patch(
          "google.adk.tools.mcp_tool.experimental.mcp_session_manager.SessionLifecycle"
      ) as mock_lifecycle_class:
        mock_stdio.return_value = AsyncMock()
        mock_lifecycle_class.return_value = mock_lifecycle

        with pytest.raises(ConnectionError, match="Failed to create MCP session"):
          await manager.create_session()

        # Session should not be added to pool
        assert not manager._sessions

  @pytest.mark.asyncio
  async def test_close_success(self):
    """Test successful cleanup of all sessions."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Add mock sessions
    lifecycle1 = MockSessionLifecycle()
    lifecycle2 = MockSessionLifecycle()

    manager._sessions["session1"] = lifecycle1
    manager._sessions["session2"] = lifecycle2

    await manager.close()

    # All lifecycles should be closed
    lifecycle1.close.assert_called_once()
    lifecycle2.close.assert_called_once()
    assert len(manager._sessions) == 0

  @pytest.mark.asyncio
  async def test_close_with_errors(self):
    """Test cleanup when some sessions fail to close."""
    manager = MCPSessionManager(self.mock_stdio_connection_params)

    # Add mock sessions
    lifecycle1 = MockSessionLifecycle()
    lifecycle1.close.side_effect = Exception("Close error 1")

    lifecycle2 = MockSessionLifecycle()

    manager._sessions["session1"] = lifecycle1
    manager._sessions["session2"] = lifecycle2

    custom_errlog = StringIO()
    manager._errlog = custom_errlog

    # Should not raise exception
    await manager.close()

    # Good session should still be closed
    lifecycle2.close.assert_called_once()
    assert len(manager._sessions) == 0

    # Error should be logged
    error_output = custom_errlog.getvalue()
    assert "Warning: Error during MCP session cleanup" in error_output
    assert "Close error 1" in error_output

  @pytest.mark.asyncio
  async def test_create_session_sse(self):
    """Test creating an SSE session."""
    sse_params = SseConnectionParams(
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token"},
    )
    manager = MCPSessionManager(sse_params)

    mock_session = MockClientSession()
    mock_lifecycle = MockSessionLifecycle(mock_session)

    with patch(
        "google.adk.tools.mcp_tool.experimental.mcp_session_manager.sse_client"
    ) as mock_sse:
      with patch(
          "google.adk.tools.mcp_tool.experimental.mcp_session_manager.SessionLifecycle"
      ) as mock_lifecycle_class:
        mock_sse.return_value = AsyncMock()
        mock_lifecycle_class.return_value = mock_lifecycle

        session = await manager.create_session()

        # Verify sse_client was called with correct params
        mock_sse.assert_called_once_with(
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer token"},
            timeout=5.0,
            sse_read_timeout=300.0,
        )

        # Verify SessionLifecycle was created with is_stdio=False
        call_kwargs = mock_lifecycle_class.call_args[1]
        assert call_kwargs["is_stdio"] is False

        assert session == mock_session

  @pytest.mark.asyncio
  async def test_create_session_streamable_http(self):
    """Test creating a Streamable HTTP session."""
    from datetime import timedelta

    http_params = StreamableHTTPConnectionParams(
        url="https://example.com/mcp",
        timeout=10.0,
        sse_read_timeout=120.0,
        terminate_on_close=False,
    )
    manager = MCPSessionManager(http_params)

    mock_session = MockClientSession()
    mock_lifecycle = MockSessionLifecycle(mock_session)

    with patch(
        "google.adk.tools.mcp_tool.experimental.mcp_session_manager.streamablehttp_client"
    ) as mock_http:
      with patch(
          "google.adk.tools.mcp_tool.experimental.mcp_session_manager.SessionLifecycle"
      ) as mock_lifecycle_class:
        mock_http.return_value = AsyncMock()
        mock_lifecycle_class.return_value = mock_lifecycle

        session = await manager.create_session()

        # Verify streamablehttp_client was called with correct params
        mock_http.assert_called_once_with(
            url="https://example.com/mcp",
            headers={},
            timeout=timedelta(seconds=10.0),
            sse_read_timeout=timedelta(seconds=120.0),
            terminate_on_close=False,
        )

        assert session == mock_session

  @pytest.mark.asyncio
  @patch("google.adk.tools.mcp_tool.experimental.mcp_session_manager.stdio_client")
  @patch("google.adk.tools.mcp_tool.experimental.session_lifecycle.AsyncExitStack")
  async def test_create_session_cancelled_error(
          self, mock_exit_stack_class, mock_stdio
  ):
      """Test session creation when CancelledError is raised (e.g., HTTP 403).

      When an MCP server returns an HTTP error (e.g., 401, 403), the MCP SDK
      uses anyio cancel scopes internally, which raise CancelledError. This
      test verifies that CancelledError is caught and converted to a
      ConnectionError with proper cleanup.
      """
      manager = MCPSessionManager(self.mock_stdio_connection_params)

      mock_exit_stack = MockAsyncExitStack()

      mock_exit_stack_class.return_value = mock_exit_stack
      mock_stdio.return_value = AsyncMock()

      # Simulate CancelledError during session creation (e.g., HTTP 403)
      mock_exit_stack.enter_async_context.side_effect = asyncio.CancelledError(
          "Cancelled by cancel scope"
      )

      # Expect ConnectionError due to CancelledError
      with pytest.raises(ConnectionError, match="Failed to create MCP session"):
          await manager.create_session()

      # Verify session was not added to pool
      assert not manager._sessions
      # Verify cleanup was called
      mock_exit_stack.aclose.assert_called_once()

