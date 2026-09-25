"""
Tests for the judge-call retry logic added 2026-09-25 (Llama branch's Update 9 Fix A),
after a real OpenAI 5xx outage window silently dropped 5 scores across 2 runs with zero
retry. These tests are scoped entirely to the retry/backoff/error-surfacing machinery --
they never touch parse_judge_score or the actual API call parameters (model, messages,
max_completion_tokens, temperature), which Update 9 explicitly required be left unchanged.
"""

import asyncio

import httpx2
import openai
import pytest

from assistant_axis.judge import (
    JUDGE_MAX_RETRY_ATTEMPTS,
    RateLimiter,
    _call_judge_single_with_retry,
    _is_retryable_judge_error,
    call_judge_batch,
    call_judge_single,
)


def _status_error(cls, status_code: int, message: str = "error"):
    response = httpx2.Response(status_code, request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return cls("error" if message is None else message, response=response, body=None)


class _FakeCompletions:
    """Stands in for client.chat.completions -- .create is an AsyncMock-like callable
    that either returns a canned response or raises, per a queue of behaviors."""

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, behaviors):
        self.chat = _FakeChat(_FakeCompletions(behaviors))


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retries would otherwise really sleep for up to ~31s across a full 6-attempt
    exhaustion; patch asyncio.sleep in the judge module so tests run fast."""
    async def _fast_sleep(_seconds):
        return None

    monkeypatch.setattr("assistant_axis.judge.asyncio.sleep", _fast_sleep)


class TestIsRetryableJudgeError:
    def test_429_is_retryable(self):
        assert _is_retryable_judge_error(_status_error(openai.RateLimitError, 429)) is True

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    def test_5xx_is_retryable(self, status_code):
        assert _is_retryable_judge_error(_status_error(openai.InternalServerError, status_code)) is True

    def test_400_is_not_retryable(self):
        assert _is_retryable_judge_error(_status_error(openai.BadRequestError, 400)) is False

    def test_401_is_not_retryable(self):
        assert _is_retryable_judge_error(_status_error(openai.AuthenticationError, 401)) is False

    def test_timeout_is_retryable(self):
        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        assert _is_retryable_judge_error(openai.APITimeoutError(request=request)) is True

    def test_connection_error_is_retryable(self):
        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        assert _is_retryable_judge_error(openai.APIConnectionError(request=request)) is True

    def test_generic_exception_is_not_retryable(self):
        assert _is_retryable_judge_error(ValueError("unrelated")) is False


class TestCallJudgeSingleWithRetry:
    def test_succeeds_on_first_attempt_no_retry_needed(self):
        client = _FakeClient([_FakeResponse("2")])
        text, error = asyncio.run(
            _call_judge_single_with_retry(client, "prompt", "gpt-4.1-mini", 10, RateLimiter(100))
        )
        assert text == "2"
        assert error is None
        assert len(client.chat.completions.calls) == 1

    def test_retries_a_503_then_succeeds(self):
        client = _FakeClient([_status_error(openai.InternalServerError, 503), _FakeResponse("3")])
        text, error = asyncio.run(
            _call_judge_single_with_retry(client, "prompt", "gpt-4.1-mini", 10, RateLimiter(100))
        )
        assert text == "3"
        assert error is None
        assert len(client.chat.completions.calls) == 2

    def test_exhausts_all_retries_and_returns_the_last_error(self):
        behaviors = [_status_error(openai.InternalServerError, 503) for _ in range(JUDGE_MAX_RETRY_ATTEMPTS)]
        client = _FakeClient(behaviors)
        text, error = asyncio.run(
            _call_judge_single_with_retry(client, "prompt", "gpt-4.1-mini", 10, RateLimiter(100))
        )
        assert text is None
        assert error is not None
        assert "503" in error or "InternalServerError" in error
        assert len(client.chat.completions.calls) == JUDGE_MAX_RETRY_ATTEMPTS

    def test_non_retryable_error_fails_on_the_first_attempt_no_retry(self):
        client = _FakeClient([_status_error(openai.BadRequestError, 400, "bad request")])
        text, error = asyncio.run(
            _call_judge_single_with_retry(client, "prompt", "gpt-4.1-mini", 10, RateLimiter(100))
        )
        assert text is None
        assert error is not None
        assert len(client.chat.completions.calls) == 1  # never retried

    def test_empty_response_is_not_an_error_but_returns_none_with_a_message(self):
        client = _FakeClient([_FakeResponse(None)])
        text, error = asyncio.run(
            _call_judge_single_with_retry(client, "prompt", "gpt-4.1-mini", 10, RateLimiter(100))
        )
        assert text is None
        assert error is not None
        assert len(client.chat.completions.calls) == 1  # not a retryable exception path

    def test_api_call_parameters_are_unchanged(self):
        """The actual request shape (model, messages, max_completion_tokens, temperature)
        must be byte-for-byte the same as before the retry wrapper was added -- Update 9's
        explicit constraint."""
        client = _FakeClient([_FakeResponse("1")])
        asyncio.run(_call_judge_single_with_retry(client, "my prompt text", "gpt-4.1-mini", 10, RateLimiter(100)))
        call = client.chat.completions.calls[0]
        assert call == {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "my prompt text"}],
            "max_completion_tokens": 10,
            "temperature": 1,
        }


class TestCallJudgeSingleBackwardCompatibility:
    def test_same_signature_and_return_type_as_before(self):
        """call_judge_single's own public contract (Optional[str] return, same params) must
        be unchanged -- only its internals gained retries."""
        client = _FakeClient([_status_error(openai.InternalServerError, 500), _FakeResponse("0")])
        result = asyncio.run(call_judge_single(client, "prompt", "gpt-4.1-mini", 10, RateLimiter(100)))
        assert result == "0"

    def test_returns_none_after_exhausting_retries_same_as_pre_retry_behavior(self):
        behaviors = [_status_error(openai.InternalServerError, 503) for _ in range(JUDGE_MAX_RETRY_ATTEMPTS)]
        client = _FakeClient(behaviors)
        result = asyncio.run(call_judge_single(client, "prompt", "gpt-4.1-mini", 10, RateLimiter(100)))
        assert result is None


class TestCallJudgeBatch:
    def test_return_type_unchanged_when_errors_out_not_given(self):
        client = _FakeClient([_FakeResponse("1"), _FakeResponse("2")])
        results = asyncio.run(
            call_judge_batch(client, ["p1", "p2"], "gpt-4.1-mini", 10, RateLimiter(100), batch_size=50)
        )
        assert results == ["1", "2"]

    def test_errors_out_is_populated_index_aligned_with_results(self):
        behaviors = [
            _FakeResponse("2"),
            *[_status_error(openai.InternalServerError, 503) for _ in range(JUDGE_MAX_RETRY_ATTEMPTS)],
        ]
        client = _FakeClient(behaviors)
        errors_out = []
        results = asyncio.run(
            call_judge_batch(
                client, ["ok prompt", "failing prompt"], "gpt-4.1-mini", 10, RateLimiter(100),
                batch_size=50, errors_out=errors_out,
            )
        )
        assert results == ["2", None]
        assert errors_out[0] is None
        assert errors_out[1] is not None

    def test_errors_out_all_none_on_full_success(self):
        client = _FakeClient([_FakeResponse("1"), _FakeResponse("2"), _FakeResponse("3")])
        errors_out = []
        asyncio.run(
            call_judge_batch(
                client, ["p1", "p2", "p3"], "gpt-4.1-mini", 10, RateLimiter(100),
                batch_size=50, errors_out=errors_out,
            )
        )
        assert errors_out == [None, None, None]
