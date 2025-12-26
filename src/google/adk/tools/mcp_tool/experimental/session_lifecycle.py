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

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
import logging
from typing import Any
from typing import Optional

from mcp import ClientSession

logger = logging.getLogger('google_adk.' + __name__)


class SessionLifecycle:
  """Represents the lifecycle of a single MCP session within a dedicated task.

  AnyIO's TaskGroup/CancelScope requires that the start and end of a scope
  occur within the same task. Since MCP clients use AnyIO internally, we need
  to ensure that the client's entire lifecycle (creation, usage, and cleanup)
  happens within a single dedicated task.

  This class spawns a background task that:
  1. Enters the MCP client's async context and initializes the session
  2. Signals readiness via an asyncio.Event
  3. Waits for a close signal
  4. Cleans up the client within the same task

  This ensures CancelScope constraints are satisfied regardless of which
  task calls start() or close().

  Can be used in two ways:
  1. Direct method calls: start() and close()
  2. As an async context manager: async with lifecycle as session: ...
  """

  def __init__(
      self,
      client: Any,
      timeout: Optional[float],
      is_stdio: bool = False,
  ):
    """Initializes the SessionLifecycle.

    Args:
        client: An MCP client context manager (e.g., from streamablehttp_client,
            sse_client, or stdio_client). The client's TaskGroup won't start
            until its async context is entered.
        timeout: Timeout in seconds for connection and initialization.
        is_stdio: Whether this is a stdio connection (affects read timeout).
    """
    self._client = client
    self._timeout = timeout
    self._is_stdio = is_stdio
    self._session: Optional[ClientSession] = None
    self._ready_event = asyncio.Event()
    self._close_event = asyncio.Event()
    self._task: Optional[asyncio.Task] = None

  @property
  def session(self) -> Optional[ClientSession]:
    """Returns the managed ClientSession, if available."""
    return self._session

  async def start(self) -> ClientSession:
    """Starts the lifecycle task and waits for the session to be ready.

    Returns:
        The initialized ClientSession.

    Raises:
        ConnectionError: If session creation fails.
    """
    self._task = asyncio.create_task(self._run_lifecycle())
    await self._ready_event.wait()

    if self._task.cancelled():
      raise ConnectionError('Error during session creation: task cancelled')

    if self._task.done() and self._task.exception():
      raise ConnectionError(
          f'Error during session creation: {self._task.exception()}'
      ) from self._task.exception()

    return self._session

  async def close(self):
    """Signals the lifecycle task to close and waits for cleanup."""
    self._close_event.set()
    if self._task:
      try:
        await self._task
      except asyncio.CancelledError:
        pass
      except Exception as e:
        logger.warning('Error during session lifecycle cleanup: %s', e)

  async def __aenter__(self) -> ClientSession:
    """Async context manager entry point."""
    return await self.start()

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit point."""
    await self.close()

  async def _run_lifecycle(self):
    """Runs the complete session lifecycle within a single task.

    This method:
    1. Enters the client's async context (which starts its TaskGroup)
    2. Creates and initializes the ClientSession
    3. Signals that the session is ready
    4. Waits for the close signal
    5. Exits the async contexts (cleanup happens in the same task)
    """
    try:
      async with AsyncExitStack() as exit_stack:
        transports = await asyncio.wait_for(
            exit_stack.enter_async_context(self._client),
            timeout=self._timeout,
        )
        # The streamable http client returns a GetSessionCallback in addition
        # to the read/write MemoryObjectStreams needed to build the
        # ClientSession. We limit to the first two values to be compatible
        # with all clients.
        if self._is_stdio:
          session = await exit_stack.enter_async_context(
              ClientSession(
                  *transports[:2],
                  read_timeout_seconds=timedelta(seconds=self._timeout),
              )
          )
        else:
          session = await exit_stack.enter_async_context(
              ClientSession(*transports[:2])
          )
        await asyncio.wait_for(session.initialize(), timeout=self._timeout)

        self._session = session
        self._ready_event.set()

        # Wait for close signal - the session remains valid while we wait
        await self._close_event.wait()
    except BaseException as e:
      logger.error('Error during session lifecycle: %s', e)
      raise
    finally:
      self._ready_event.set()
      self._close_event.set()
