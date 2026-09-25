"""Talk to the models the router can pick.

Both Ollama and OpenRouter speak the OpenAI chat-completions dialect, so one
streaming function covers every text lane.  The image lane is separate:
OpenRouter returns images from /chat/completions when you ask for the
"image" modality; a local server is called through the OpenAI images API.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from .router import ModelRef
from .settings import settings

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


@dataclass
class StreamResult:
    """Filled in as the stream progresses; read it after the generator finishes."""
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finished: bool = False


def _endpoint(ref: ModelRef) -> tuple[str, dict]:
    if ref.provider == "ollama":
        return f"{settings.ollama_base_url}/v1/chat/completions", {"Authorization": "Bearer ollama"}
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Jev model router demo",
    }
    return f"{settings.openrouter_base_url}/chat/completions", headers


async def stream_chat(client: httpx.AsyncClient, ref: ModelRef, messages: list[dict],
                      result: StreamResult) -> AsyncIterator[str]:
    """Yield text deltas from the model; usage (when the provider sends it) lands in `result`."""
    url, headers = _endpoint(ref)
    body: dict = {"model": ref.model, "messages": messages, "stream": True}
    if ref.provider == "openrouter":
        body["stream_options"] = {"include_usage": True}

    in_think = False
    async with client.stream("POST", url, headers=headers, json=body, timeout=httpx.Timeout(300, connect=15)) as r:
        if r.status_code >= 400:
            detail = (await r.aread()).decode(errors="replace")[:400]
            raise RuntimeError(f"{ref.label} returned HTTP {r.status_code}: {detail}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                result.input_tokens = usage.get("prompt_tokens", result.input_tokens)
                result.output_tokens = usage.get("completion_tokens", result.output_tokens)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content") or ""
                if not delta:
                    continue
                # Hide <think> blocks from thinking-style local models.
                if "<think>" in delta:
                    in_think = True
                if in_think:
                    if "</think>" in delta:
                        in_think = False
                        delta = delta.split("</think>", 1)[1]
                    else:
                        continue
                result.text += delta
                yield delta
    result.text = _THINK_RE.sub("", result.text)
    result.finished = True


async def complete(client: httpx.AsyncClient, ref: ModelRef, messages: list[dict]) -> str:
    """Non-streaming helper (used for prompt rewriting)."""
    url, headers = _endpoint(ref)
    r = await client.post(url, headers=headers, json={"model": ref.model, "messages": messages, "stream": False},
                          timeout=120)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"] or ""
    return _THINK_RE.sub("", text).strip()


# ── Image lane ──────────────────────────────────────────────────────────────────
_REWRITE_SYSTEM = (
    "You turn short image requests into one vivid, concrete image-generation prompt. "
    "Describe subject, setting, lighting, style and composition in under 70 words. "
    "Output only the prompt, nothing else."
)


async def rewrite_image_prompt(client: httpx.AsyncClient, user_prompt: str) -> str:
    """Like every image service: expand the user's one-liner with a small local model first."""
    small = ModelRef("ollama", settings.local_small_model, settings.local_small_model, True)
    try:
        text = await complete(client, small, [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])
        return text.strip().strip('"') or user_prompt
    except Exception:
        return user_prompt


async def generate_image(client: httpx.AsyncClient, prompt: str) -> str:
    """Return a data: URL (or https URL) for the generated image."""
    if settings.image_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        r = await client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.image_model,
                "messages": [{"role": "user", "content": prompt}],
                "modalities": ["image", "text"],
                "stream": False,
            },
            timeout=180,
        )
        r.raise_for_status()
        message = r.json()["choices"][0]["message"]
        images = message.get("images") or []
        if not images:
            raise RuntimeError("model returned no image (check that IMAGE_MODEL lists 'image' in its output modalities)")
        return images[0]["image_url"]["url"]

    if settings.image_provider == "local":
        r = await client.post(
            settings.local_image_url,
            json={"prompt": prompt, "n": 1, "size": "1024x1024", "response_format": "b64_json"},
            timeout=300,
        )
        r.raise_for_status()
        item = r.json()["data"][0]
        if item.get("b64_json"):
            return "data:image/png;base64," + item["b64_json"]
        return item["url"]

    raise RuntimeError("image lane is disabled (IMAGE_PROVIDER=off)")


async def ollama_models(client: httpx.AsyncClient) -> list[str]:
    r = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
