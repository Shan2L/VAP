from __future__ import annotations

import os

from vap.agent.runtime import AGENT_BASE_URL, AGENT_MODEL, AGENT_TIMEOUT_SEC


def user_header() -> str:
    try:
        return os.getlogin()
    except OSError:
        return os.getenv("USER") or os.getenv("USERNAME") or "unknown"


def create_client(subscription_key: str):
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed. Run install.sh again."
        ) from exc

    return openai.OpenAI(
        base_url=AGENT_BASE_URL,
        api_key="dummy",
        timeout=AGENT_TIMEOUT_SEC,
        default_headers={
            "Ocp-Apim-Subscription-Key": subscription_key,
            "user": user_header(),
        },
    )


def chat_completion(
    client,
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_completion_tokens: int,
    stream: bool = False,
):
    try:
        kwargs = {
            "model": AGENT_MODEL,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        if "timed out" in str(exc).lower():
            raise TimeoutError(
                f"LLM request exceeded {AGENT_TIMEOUT_SEC:.0f}s. Try again, ask a narrower question, or increase VAP_LLM_TIMEOUT_SEC."
            ) from exc
        raise
