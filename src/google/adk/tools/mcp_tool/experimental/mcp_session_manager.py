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

"""Lifecycle-based MCP session manager for AnyIO CancelScope compatibility.

This module provides a MCPSessionManager that ensures MCP client
lifecycle operations (creation and cleanup) occur within the same asyncio task.
This is required because MCP clients use AnyIO internally, and AnyIO's
TaskGroup/CancelScope requires that the start and end of a scope occur within
the same task.

The manager uses SessionLifecycle to manage each session's lifecycle within
a dedicated task. SessionLifecycle spawns a background task that handles the
entire lifecycle (creation, usage, and cleanup) within a single task, ensuring
CancelScope constraints are satisfied regardless of which task calls
create_session() or close().

Use this manager instead of the standard MCPSessionManager when working with
StreamableHTTPConnectionParams to avoid CancelScope constraint violations.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
import hashlib
import json
import logging
import sys
from typing import Any
from typing import Dict
from typing import Optional
from typing import TextIO
from typing import Union

from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from ..mcp_session_manager import SseConnectionParams
from ..mcp_session_manager import StdioConnectionParams
from ..mcp_session_manager import StreamableHTTPConnectionParams
from .session_lifecycle import SessionLifecycle

logger = logging.getLogger('google_adk.' + __name__)


class MCPSessionManager:
  """Lifecycle-based MCP session manager for AnyIO CancelScope compatibility.

  This class provides the same functionality as the standard MCPSessionManager
  but ensures that MCP client lifecycle operations (creation and cleanup)
  occur within the same asyncio task. This is required because MCP clients use
  AnyIO internally, and AnyIO's TaskGroup/CancelScope requires that the start
  and end of a scope occur within the same task.

  The session lifecycle is managed by SessionLifecycle, which spawns a
  dedicated background task for each session. This background task:
  1. Enters the MCP client's async context and initializes the session
  2. Signals readiness via an asyncio.Event
  3. Waits for a close signal
  4. Cleans up the client within the same task

  This ensures CancelScope constraints are satisfied regardless of which
  task calls create_session() or close().
  """

  def __init__(
      self,
      connection_params: Union[
          StdioServerParameters,
          StdioConnectionParams,
          SseConnectionParams,
          StreamableHTTPConnectionParams,
      ],
      errlog: TextIO = sys.stderr,
  ):
    """Initializes the lifecycle-based MCP session manager.

    Args:
        connection_params: Parameters for the MCP connection (Stdio, SSE or
          Streamable HTTP). Stdio by default also has a 5s read timeout as other
          parameters but it's not configurable for now.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.
    """
    if isinstance(connection_params, StdioServerParameters):
      # So far timeout is not configurable. Given MCP is still evolving, we
      # would expect stdio_client to evolve to accept timeout parameter like
      # other client.
      logger.warning(
          'StdioServerParameters is not recommended. Please use'
          ' StdioConnectionParams.'
      )
      self._connection_params = StdioConnectionParams(
          server_params=connection_params,
          timeout=5,
      )
    else:
      self._connection_params = connection_params
    self._errlog = errlog

    # Session pool: maps session keys to lifecycle managers
    self._sessions: Dict[str, SessionLifecycle] = {}

    # Lock to prevent race conditions in session creation
    self._session_lock = asyncio.Lock()

  def _generate_session_key(
      self, merged_headers: Optional[Dict[str, str]] = None
  ) -> str:
    """Generates a session key based on connection params and merged headers.

    For StdioConnectionParams, returns a constant key since headers are not
    supported. For SSE and StreamableHTTP connections, generates a key based
    on the provided merged headers.

    Args:
        merged_headers: Already merged headers (base + additional).

    Returns:
        A unique session key string.
    """
    if isinstance(self._connection_params, StdioConnectionParams):
      return 'stdio_session'

    if merged_headers:
      headers_json = json.dumps(merged_headers, sort_keys=True)
      headers_hash = hashlib.md5(headers_json.encode()).hexdigest()
      return f'session_{headers_hash}'
    else:
      return 'session_no_headers'

  def _merge_headers(
      self, additional_headers: Optional[Dict[str, str]] = None
  ) -> Optional[Dict[str, str]]:
    """Merges base connection headers with additional headers.

    Args:
        additional_headers: Optional headers to merge with connection headers.

    Returns:
        Merged headers dictionary, or None if no headers are provided.
    """
    if isinstance(self._connection_params, StdioConnectionParams) or isinstance(
        self._connection_params, StdioServerParameters
    ):
      return None

    base_headers = {}
    if (
        hasattr(self._connection_params, 'headers')
        and self._connection_params.headers
    ):
      base_headers = self._connection_params.headers.copy()

    if additional_headers:
      base_headers.update(additional_headers)

    return base_headers

  def _is_session_disconnected(self, session: ClientSession) -> bool:
    """Checks if a session is disconnected or closed.

    Args:
        session: The ClientSession to check.

    Returns:
        True if the session is disconnected, False otherwise.
    """
    return session._read_stream._closed or session._write_stream._closed

  def _create_client(self, merged_headers: Optional[Dict[str, str]] = None):
    """Creates an MCP client based on the connection parameters.

    Args:
        merged_headers: Optional headers to include in the connection.
                       Only applicable for SSE and StreamableHTTP connections.

    Returns:
        The appropriate MCP client instance.

    Raises:
        ValueError: If the connection parameters are not supported.
    """
    if isinstance(self._connection_params, StdioConnectionParams):
      client = stdio_client(
          server=self._connection_params.server_params,
          errlog=self._errlog,
      )
    elif isinstance(self._connection_params, SseConnectionParams):
      client = sse_client(
          url=self._connection_params.url,
          headers=merged_headers,
          timeout=self._connection_params.timeout,
          sse_read_timeout=self._connection_params.sse_read_timeout,
      )
    elif isinstance(self._connection_params, StreamableHTTPConnectionParams):
      client = streamablehttp_client(
          url=self._connection_params.url,
          headers=merged_headers,
          timeout=timedelta(seconds=self._connection_params.timeout),
          sse_read_timeout=timedelta(
              seconds=self._connection_params.sse_read_timeout
          ),
          terminate_on_close=self._connection_params.terminate_on_close,
      )
    else:
      raise ValueError(
          'Unable to initialize connection. Connection should be'
          ' StdioServerParameters or SseServerParams, but got'
          f' {self._connection_params}'
      )
    return client

  async def create_session(
      self, headers: Optional[Dict[str, str]] = None
  ) -> ClientSession:
    """Creates and initializes an MCP client session.

    This method will check if an existing session for the given headers
    is still connected. If it's disconnected, it will be cleaned up and
    a new session will be created.

    The session lifecycle is managed by SessionLifecycle, which spawns a
    dedicated background task to handle the entire lifecycle (creation, usage,
    and cleanup) within a single task. This is required because MCP clients
    use AnyIO internally, and AnyIO's TaskGroup/CancelScope requires that the
    start and end of a scope occur within the same task.

    Args:
        headers: Optional headers to include in the session. These will be
                merged with any existing connection headers. Only applicable
                for SSE and StreamableHTTP connections.

    Returns:
        ClientSession: The initialized MCP client session.
    """
    merged_headers = self._merge_headers(headers)
    session_key = self._generate_session_key(merged_headers)

    async with self._session_lock:
      # Check if we have an existing session
      if session_key in self._sessions:
        lifecycle_manager = self._sessions[session_key]

        if not self._is_session_disconnected(lifecycle_manager.session):
          return lifecycle_manager.session
        else:
          # Session is disconnected, clean it up
          logger.info('Cleaning up disconnected session: %s', session_key)
          try:
            await lifecycle_manager.close()
          except Exception as e:
            logger.warning('Error during disconnected session cleanup: %s', e)
          finally:
            del self._sessions[session_key]

      # Create a new session
      timeout_in_seconds = (
          self._connection_params.timeout
          if hasattr(self._connection_params, 'timeout')
          else None
      )

      is_stdio = isinstance(self._connection_params, StdioConnectionParams)

      # Use SessionLifecycle to ensure client lifecycle operations
      # happen in the same task (required by AnyIO's CancelScope)
      client = self._create_client(merged_headers)
      lifecycle_manager = SessionLifecycle(
          client=client,
          timeout=timeout_in_seconds,
          is_stdio=is_stdio,
      )

      try:
        session = await lifecycle_manager.start()
        self._sessions[session_key] = lifecycle_manager
        logger.debug('Created new session: %s', session_key)
        return session

      except Exception as e:
        raise ConnectionError(f'Failed to create MCP session: {e}') from e

  async def close(self):
    """Closes all sessions and cleans up resources.

    Each session's cleanup is performed by its SessionLifecycle,
    which ensures that the cleanup happens in the same task where the
    client was created (required by AnyIO's CancelScope).
    """
    async with self._session_lock:
      for session_key in list(self._sessions.keys()):
        lifecycle_manager = self._sessions[session_key]
        try:
          await lifecycle_manager.close()
        except Exception as e:
          print(
              'Warning: Error during MCP session cleanup for'
              f' {session_key}: {e}',
              file=self._errlog,
          )
        finally:
          del self._sessions[session_key]

