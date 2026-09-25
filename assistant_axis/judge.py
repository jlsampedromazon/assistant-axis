"""
Judge LLM utilities for scoring role responses.

This module provides functions for scoring how well model responses
adhere to their assigned roles using an LLM judge (e.g., GPT-4).

Score Scale:
    0: Model refused to answer
    1: Model says it can't be the role, but can help with related tasks
    2: Model identifies as AI/LLM but has some role attributes
    3: Model is fully playing the role

Example:
    from assistant_axis.judge import score_responses

    scores = await score_responses(
        responses=[{"question": "...", "response": "..."}],
        eval_prompt_template="Rate how well...",
        judge_model="gpt-4.1-mini"
    )
"""

import asyncio
import os
import random
import re
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Retry policy for transient judge-call failures (2026-09-25, Llama branch's
# Update 9 Fix A). Added after a real OpenAI 5xx outage window (2026-09-25,
# 15:36-15:41 ET) silently dropped 5 scores across 2 runs with zero retry --
# call_judge_single previously caught every exception and returned None on
# the first failure, no matter how transient. 429 (rate limit), 500/502/503/
# 504 (server-side errors -- exactly what the outage produced), and network
# timeouts/connection errors are retried; anything else (4xx client errors
# like a bad request or an auth failure) is not, since retrying can never fix
# those. At least 6 attempts total, exponential backoff from a 1s base,
# capped at 60s, with up to 25% jitter added to each delay.
JUDGE_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
JUDGE_MAX_RETRY_ATTEMPTS = 6
JUDGE_BASE_RETRY_DELAY_SECONDS = 1.0
JUDGE_MAX_RETRY_DELAY_SECONDS = 60.0


def _is_retryable_judge_error(exc: Exception) -> bool:
    """True for 429/500/502/503/504 (every openai.APIStatusError subclass sets
    status_code from the real HTTP response in its own __init__, so this
    works for RateLimitError, InternalServerError, and any other
    APIStatusError alike) or a network-level timeout/connection error. False
    for anything else, including a bad-request or auth error a retry could
    never fix."""
    status_code = getattr(exc, "status_code", None)
    if status_code in JUDGE_RETRYABLE_STATUS_CODES:
        return True
    return isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError))


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, rate: float):
        """
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0


def parse_judge_score(response_text: str) -> Optional[int]:
    """
    Parse the judge's response to extract the numerical score.

    Args:
        response_text: The judge model's response

    Returns:
        Integer score between 0-3, or None if parsing fails
    """
    if not response_text:
        return None

    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())

    if not numbers:
        return None

    try:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
        return None
    except ValueError:
        return None


async def _call_judge_single_with_retry(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
) -> Tuple[Optional[str], Optional[str]]:
    """Does the real work behind call_judge_single, with retries. Returns
    (response_text, error): error is None on success, and the last
    exception's message on failure (whether from a non-retryable error on
    the first attempt, or JUDGE_MAX_RETRY_ATTEMPTS genuinely exhausted).
    Never raises.

    The actual API call itself -- model, messages, max_completion_tokens,
    temperature=1 -- is byte-for-byte unchanged from before this retry
    wrapper was added; only the surrounding retry/backoff logic is new."""
    last_error: Optional[str] = None
    for attempt in range(JUDGE_MAX_RETRY_ATTEMPTS):
        await rate_limiter.acquire()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=1
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content, None
            return None, "judge returned an empty response (no choices or content)"

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            retryable = _is_retryable_judge_error(e)
            is_last_attempt = attempt == JUDGE_MAX_RETRY_ATTEMPTS - 1
            if not retryable or is_last_attempt:
                logger.error(
                    f"Error calling judge model (attempt {attempt + 1}/{JUDGE_MAX_RETRY_ATTEMPTS}, "
                    f"{'non-retryable' if not retryable else 'retries exhausted'}): {e}"
                )
                break
            delay = min(JUDGE_BASE_RETRY_DELAY_SECONDS * (2 ** attempt), JUDGE_MAX_RETRY_DELAY_SECONDS)
            delay += random.uniform(0, delay * 0.25)
            logger.warning(
                f"Retryable error calling judge model (attempt {attempt + 1}/{JUDGE_MAX_RETRY_ATTEMPTS}), "
                f"retrying in {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)

    return None, last_error


async def call_judge_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call the judge model with a single prompt. Retries transient errors
    (see _call_judge_single_with_retry / JUDGE_RETRYABLE_STATUS_CODES);
    returns None only once retries are exhausted or a non-retryable error
    occurs. Same signature and return contract as before this function
    gained retries (2026-09-25, Llama branch's Update 9 Fix A) -- every
    existing caller is unaffected except for gaining retry robustness."""
    text, _error = await _call_judge_single_with_retry(client, prompt, model, max_tokens, rate_limiter)
    return text


async def call_judge_batch(
    client: openai.AsyncOpenAI,
    prompts: List[str],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50,
    errors_out: Optional[List[Optional[str]]] = None,
) -> List[Optional[str]]:
    """Call the judge model with multiple prompts concurrently. Return type
    and every existing caller's contract are unchanged (2026-09-25, Llama
    branch's Update 9 Fix A) -- individual calls now retry transient errors
    internally (see _call_judge_single_with_retry), so callers who never
    asked for error detail simply see fewer Nones.

    errors_out (optional, default None): if given, must be a list the
    caller owns (typically `[]`) -- this function extends it with one entry
    per prompt, in the same order as the returned results: None for a
    prompt that succeeded, the last error message for one that didn't (after
    retries, or immediately for a non-retryable error). Lets a caller that
    needs to know WHY a specific prompt failed (3_judge.py, to record a
    FAILED key with its error) get that without this function's return type
    changing for anyone else."""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        tasks = [
            _call_judge_single_with_retry(client, prompt, model, max_tokens, rate_limiter)
            for prompt in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: List[Tuple[Optional[str], Optional[str]]] = []
        for result in batch_results:
            if isinstance(result, Exception):
                # _call_judge_single_with_retry itself never raises (it always
                # returns a (text, error) tuple), so reaching this branch means
                # something outside the retry loop itself went wrong (e.g. a
                # cancelled task) -- kept as defensive handling, matching the
                # pre-retry code's own defensive branch here.
                logger.error(f"Exception in batch: {result}")
                processed.append((None, str(result)))
            else:
                processed.append(result)

        results.extend(text for text, _error in processed)
        if errors_out is not None:
            errors_out.extend(error for _text, error in processed)

    return results


async def score_responses(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 10,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """
    Score a list of responses using an LLM judge.

    Args:
        responses: List of dicts with 'question' and 'response' keys
        eval_prompt_template: Template string with {question} and {answer} placeholders
        judge_model: OpenAI model to use as judge
        max_tokens: Max tokens for judge response
        requests_per_second: Rate limit for API calls
        batch_size: Concurrent batch size

    Returns:
        List of scores (0-3) or None for failed parsing
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Build prompts
    prompts = []
    for resp in responses:
        prompt = eval_prompt_template.format(
            question=resp["question"],
            answer=resp["response"]
        )
        prompts.append(prompt)

    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(requests_per_second)

    # Call judge
    judge_responses = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )

    # Parse scores
    scores = []
    for response_text in judge_responses:
        score = parse_judge_score(response_text) if response_text else None
        scores.append(score)

    return scores


def score_responses_sync(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 10,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """
    Synchronous wrapper for score_responses.

    Args:
        responses: List of dicts with 'question' and 'response' keys
        eval_prompt_template: Template string with {question} and {answer} placeholders
        judge_model: OpenAI model to use as judge
        max_tokens: Max tokens for judge response
        requests_per_second: Rate limit for API calls
        batch_size: Concurrent batch size

    Returns:
        List of scores (0-3) or None for failed parsing
    """
    return asyncio.run(score_responses(
        responses=responses,
        eval_prompt_template=eval_prompt_template,
        judge_model=judge_model,
        max_tokens=max_tokens,
        requests_per_second=requests_per_second,
        batch_size=batch_size
    ))
