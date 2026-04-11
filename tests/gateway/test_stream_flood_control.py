"""Tests for adaptive flood control backoff in GatewayStreamConsumer.

Tests the new features from upstream commit d7607292:
- Adaptive backoff on flood-control edit failures
- Cursor strip when entering fallback mode
- Fallback final-send retry on flood-control
- _send_or_edit return values
- _is_flood_error detection
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class TestIsFloodError:
    """Verify _is_flood_error detects various flood/rate-limit error patterns."""

    def _make_consumer(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        config = StreamConsumerConfig(edit_interval=0.01)
        return GatewayStreamConsumer(adapter, "chat_123", config)

    def test_flood_keyword(self):
        c = self._make_consumer()
        result = SimpleNamespace(success=False, error="Too many requests: flood control")
        assert c._is_flood_error(result) is True

    def test_retry_after_keyword(self):
        c = self._make_consumer()
        result = SimpleNamespace(success=False, error="Retry after 5 seconds")
        assert c._is_flood_error(result) is True

    def test_rate_keyword(self):
        c = self._make_consumer()
        result = SimpleNamespace(success=False, error="rate limit exceeded")
        assert c._is_flood_error(result) is True

    def test_non_flood_error(self):
        c = self._make_consumer()
        result = SimpleNamespace(success=False, error="Internal server error")
        assert c._is_flood_error(result) is False

    def test_no_error_attribute(self):
        c = self._make_consumer()
        result = SimpleNamespace(success=False)
        assert c._is_flood_error(result) is False

    def test_none_error(self):
        c = self._make_consumer()
        result = SimpleNamespace(success=False, error=None)
        assert c._is_flood_error(result) is False

    def test_empty_error(self):
        c = self._make_consumer()
        result = SimpleNamespace(success=False, error="")
        assert c._is_flood_error(result) is False


class TestSendOrEditReturnValues:
    """Verify _send_or_edit returns correct bool values in all scenarios."""

    @pytest.mark.asyncio
    async def test_returns_true_on_successful_send(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        result = await consumer._send_or_edit("Hello world")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_on_successful_edit(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # First send to get a message_id
        await consumer._send_or_edit("Hello")
        # Edit should succeed
        result = await consumer._send_or_edit("Hello world")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_flood_failure_under_max(self):
        """Flood failure below MAX_FLOOD_STRIKES returns False (backoff, not fallback)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood control"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        result = await consumer._send_or_edit("Hello world")
        assert result is False
        # Should NOT be in fallback mode yet (only 1 flood strike)
        assert consumer._edit_supported is True

    @pytest.mark.asyncio
    async def test_returns_false_on_non_flood_edit_failure(self):
        """Non-flood edit failure enters fallback immediately."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="message not found"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        result = await consumer._send_or_edit("Hello world")
        assert result is False
        assert consumer._edit_supported is False
        assert consumer._flood_strikes == 0

    @pytest.mark.asyncio
    async def test_returns_true_for_empty_text(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        result = await consumer._send_or_edit("   ")
        assert result is True
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_true_for_duplicate_text(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        result = await consumer._send_or_edit("Hello")  # same text
        assert result is True
        adapter.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_false_on_initial_send_failure(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=False, error="forbidden"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        result = await consumer._send_or_edit("Hello")
        assert result is False
        assert consumer._edit_supported is False

    @pytest.mark.asyncio
    async def test_returns_true_on_send_success_no_message_id(self):
        """Platform accepts but returns no message_id (e.g. Signal)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id=None))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        result = await consumer._send_or_edit("Hello")
        assert result is True
        assert consumer._edit_supported is False

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(side_effect=RuntimeError("connection lost"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        result = await consumer._send_or_edit("Hello")
        assert result is False


class TestAdaptiveBackoffDirect:
    """Test adaptive backoff via direct _send_or_edit calls (bypassing run() loop)."""

    @pytest.mark.asyncio
    async def test_single_flood_failure_doubles_interval(self):
        """First flood failure doubles edit interval, keeps edits enabled."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood control"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=1.0)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # Initial send
        await consumer._send_or_edit("Hello")
        # First edit attempt fails with flood
        result = await consumer._send_or_edit("Hello world")
        assert result is False
        assert consumer._flood_strikes == 1
        assert consumer._current_edit_interval == 2.0  # doubled from 1.0
        assert consumer._edit_supported is True  # still trying edits

    @pytest.mark.asyncio
    async def test_second_flood_failure_quadruples_interval(self):
        """Second flood failure doubles again (4x original)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood control"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=1.0)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        await consumer._send_or_edit("Hello world")  # strike 1
        await consumer._send_or_edit("Hello world!")  # strike 2

        assert consumer._flood_strikes == 2
        assert consumer._current_edit_interval == 4.0  # 1.0 -> 2.0 -> 4.0
        assert consumer._edit_supported is True

    @pytest.mark.asyncio
    async def test_third_flood_failure_enters_fallback(self):
        """Third flood failure (MAX_FLOOD_STRIKES) enters fallback mode."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood control"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=1.0)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        await consumer._send_or_edit("Hello world")   # strike 1
        await consumer._send_or_edit("Hello world!")   # strike 2
        result = await consumer._send_or_edit("Hello world!!")  # strike 3 -> fallback

        assert result is False
        assert consumer._flood_strikes == 3
        assert consumer._edit_supported is False
        assert consumer._fallback_final_send is True

    @pytest.mark.asyncio
    async def test_successful_edit_resets_flood_strikes(self):
        """A successful edit resets the flood strike counter to 0."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(side_effect=[
            SimpleNamespace(success=False, error="flood control"),  # strike 1
            SimpleNamespace(success=True),  # success -> reset
            SimpleNamespace(success=False, error="flood control"),  # new strike 1
        ])
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=1.0)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        await consumer._send_or_edit("Hello world")  # strike 1
        await consumer._send_or_edit("Hello world!")  # success, reset to 0
        assert consumer._flood_strikes == 0

        await consumer._send_or_edit("Hello world!!")  # new strike 1
        assert consumer._flood_strikes == 1

    @pytest.mark.asyncio
    async def test_backoff_interval_capped_at_10(self):
        """Adaptive backoff interval is capped at 10.0 seconds."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="flood"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=8.0)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        await consumer._send_or_edit("Hello world")
        # 8.0 -> min(16.0, 10.0) = 10.0
        assert consumer._current_edit_interval == 10.0

    @pytest.mark.asyncio
    async def test_non_flood_edit_failure_enters_fallback_immediately(self):
        """Non-flood edit failure enters fallback immediately, no strikes."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="message not found"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=1.0)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello")
        result = await consumer._send_or_edit("Hello world")
        assert result is False
        assert consumer._edit_supported is False
        assert consumer._flood_strikes == 0
        assert consumer._fallback_final_send is True


class TestTryStripCursor:
    """Verify _try_strip_cursor removes cursor from last visible message."""

    @pytest.mark.asyncio
    async def test_strips_cursor_on_fallback_entry(self):
        """When entering fallback, _try_strip_cursor is called and edits the message."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
        # Non-flood error to trigger immediate fallback
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="server error"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, cursor=" \u2589")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        await consumer._send_or_edit("Hello\u2589")
        # This edit fails -> fallback mode -> _try_strip_cursor called
        await consumer._send_or_edit("Hello world\u2589")

        # edit_message should have been called twice:
        # 1st: the actual edit that failed
        # 2nd: the _try_strip_cursor attempt to clean up the cursor
        assert adapter.edit_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_no_strip_without_message_id(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        consumer._message_id = None

        await consumer._try_strip_cursor()
        adapter.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_strip_with_sentinel_id(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        consumer._message_id = "__no_edit__"

        await consumer._try_strip_cursor()
        adapter.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_strip_failure_is_swallowed(self):
        """If _try_strip_cursor's edit raises, it's caught silently."""
        adapter = MagicMock()
        adapter.edit_message = AsyncMock(side_effect=RuntimeError("network error"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(cursor=" \u2589")
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        consumer._message_id = "msg_1"
        # Need a prefix for the strip to proceed
        consumer._fallback_prefix = "Some text"

        # Should not raise
        await consumer._try_strip_cursor()


class TestFallbackFinalRetry:
    """Verify fallback final-send retries on flood control."""

    @pytest.mark.asyncio
    async def test_fallback_retries_on_flood_then_succeeds(self):
        """Fallback send retries once on flood error, then delivers the chunk."""
        adapter = MagicMock()
        # Initial send succeeds, then edit fails (non-flood -> immediate fallback)
        adapter.send = AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            # Fallback chunk 1: flood then success on retry
            SimpleNamespace(success=False, error="flood control: retry after 3"),
            SimpleNamespace(success=True, message_id="msg_2"),
        ])
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, error="message too old"))
        adapter.MAX_MESSAGE_LENGTH = 4096

        config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5000)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)

        # First call: initial send succeeds (gets message_id)
        await consumer._send_or_edit("Hello")
        assert consumer._message_id == "msg_1"
        # Second call: edit fails with non-flood error -> fallback mode
        await consumer._send_or_edit("Hello world")
        assert consumer._edit_supported is False
        assert consumer._fallback_final_send is True

        # Now call _send_fallback_final with the continuation
        consumer._accumulated = "Hello world continuation"
        await consumer._send_fallback_final("Hello world continuation")

        # Should have retried: initial send + flood fail + retry success = 3 total
        assert adapter.send.call_count == 3
        assert consumer.already_sent


class TestDefaultEditInterval:
    """Verify default config changed from 0.3s to 1.0s."""

    def test_default_edit_interval_is_1_second(self):
        config = StreamConsumerConfig()
        assert config.edit_interval == 1.0

    def test_custom_interval_overrides(self):
        config = StreamConsumerConfig(edit_interval=0.5)
        assert config.edit_interval == 0.5

    def test_initial_backoff_matches_config(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        config = StreamConsumerConfig(edit_interval=2.0)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        assert consumer._current_edit_interval == 2.0


class TestFloodStrrikesState:
    """Verify flood state is properly initialized and tracked."""

    def test_initial_flood_strikes_zero(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        config = StreamConsumerConfig()
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        assert consumer._flood_strikes == 0

    def test_max_flood_strikes_is_3(self):
        assert GatewayStreamConsumer._MAX_FLOOD_STRIKES == 3

    def test_initial_interval_matches_config(self):
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        config = StreamConsumerConfig(edit_interval=2.5)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config)
        assert consumer._current_edit_interval == 2.5
