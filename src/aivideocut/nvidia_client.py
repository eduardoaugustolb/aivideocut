# pyright: basic
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

load_dotenv()

NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_REQUESTS_PER_MINUTE = 30
RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_RETRIES = 4


class RateLimiter:
    """Client-side sliding-window limiter, so we back off before the API
    ever returns a 429 instead of reacting to it after the fact."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            self._drop_expired()

            if len(self._calls) >= self._max_calls:
                sleep_for = self._window_seconds - (
                    time.monotonic() - self._calls[0]
                )
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self._drop_expired()

            self._calls.append(time.monotonic())

    def _drop_expired(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self._window_seconds:
            self._calls.popleft()


_rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, RATE_LIMIT_WINDOW_SECONDS)


@dataclass
class NvidiaResponse:
    text: str
    model_version: str


def _get_client() -> OpenAI:
    api_key = os.environ["NVIDIA_API_KEY"]
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def ask_nvidia(
    prompt: str, *, model: str = NVIDIA_MODEL, max_retries: int = MAX_RETRIES
) -> NvidiaResponse:
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries):
        _rate_limiter.wait()

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                top_p=0.95,
                max_tokens=16384,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 16384,
                },
                stream=True,
            )

            full_text = ""
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    full_text += delta_content

            return NvidiaResponse(text=full_text.strip(), model_version=model)

        except (RateLimitError, APIError) as exc:
            last_error = exc
            backoff_seconds = min(2**attempt, 30)
            time.sleep(backoff_seconds)

    error_message = f"NVIDIA API: still failing after {max_retries} retries"
    raise RuntimeError(error_message) from last_error
