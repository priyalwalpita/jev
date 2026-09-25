# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service that sends every prompt to a **judge** (TypeSafe's Jev, or a local Ollama model emulating it) which answers four typed questions in one call — `category` (choice), `difficulty` (score), `private` (noul/probability), `needs_web` (noul) — and then uses plain if/else on those numbers to pick a **lane** (local / cloud / web / frontier / image) and stream the answer back over SSE. Every request is logged to SQLite for a cost/savings stats panel. See `README.md` for the full demo walkthrough and the rules table.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest    # pytest is not in requirements.txt
cp .env.example .env                    # fill TYPESAFE_API_KEY / OPENROUTER_API_KEY, or set JUDGE=local

uvicorn app.main:app --reload           # server + UI at http://127.0.0.1:8000
python -m pytest -q                     # routing-rule tests, no network / no Ollama needed
python -m pytest tests/test_router.py::test_web_lane -q   # single test
python scripts/demo.py                  # judge-only table for the demo prompts (needs running server)
python scripts/demo.py --chat           # full round trip with streamed answers
python scripts/demo.py --judge local    # force the local judge for that run
```

There is no linter or formatter configured. Running the server needs Ollama at `OLLAMA_BASE_URL`; the tests do not.

## Architecture

Data flow for `POST /chat` (`app/main.py`):

1. `judge.decide(messages)` → `Decision` (or `forced_route()` if the UI picked a model explicitly, skipping the judge)
2. `router.choose_route(decision, prompt_tokens, settings)` → `Route(lane, candidates[], reason)`
3. Text lanes: iterate `route.candidates` in order via `providers.stream_chat`; if a candidate fails **before** its first token, fall back to the next and emit a `status` event. A failure **mid-stream** is fatal (no fallback).
4. Image lane: `rewrite_image_prompt` (small local model expands the prompt) → `generate_image` (OpenRouter `modalities: ["image","text"]` or a local OpenAI-images server).
5. `db.log_request` records the decision, model, tokens, `est_cost` and `est_cloud_cost` (what the default cloud model would have cost) — `/stats` derives % local and savings from these.

SSE event names the UI and `scripts/demo.py` both depend on: `decision`, `status`, `start`, `token`, `image`, `done`, `error`.

### Key design points

- **`app/questions.py` is the single source of truth for the judge.** `CATEGORIES` and `DIFFICULTY_LEVELS` feed both `JevJudge` (wire format for TypeSafe `/v1/systemone`) and `LocalJudge` (which builds an Ollama structured-output JSON schema from them). Adding a category or level there updates both judges automatically; you then need to handle it in `router.choose_route` and the README rules table.
- **Both judges return the same `Decision` dataclass**, so `router.py` is judge-agnostic and pure (no I/O). That's why `tests/test_router.py` can build `Decision` objects by hand and test every rule without a network.
- **Rule order in `router.choose_route` is the policy.** The numbered comments in `router.py`, the module docstring, and the README table must stay in sync when rules change. Rule 1 (private ≥ threshold → local) must remain first.
- **`settings` is a mutable module-level singleton** (`app/settings.py`). `POST /config` mutates `settings.judge` and `settings.thresholds` in place at runtime — the UI's threshold sliders rely on this. Tests construct a fresh `Settings()` and override fields rather than touching the global. Dataclass defaults are evaluated from env at import time (`load_dotenv()` runs on import).
- **Two different Ollama endpoints are used on purpose:** `/v1/chat/completions` (OpenAI-compatible) for answer generation so the same `stream_chat` covers Ollama and OpenRouter, but `/api/chat` with `format=<json schema>` for `LocalJudge`, since structured outputs are only on the native API.
- `providers.stream_chat` strips `<think>…</think>` blocks from thinking-style local models before yielding tokens.
- Ollama's OpenAI endpoint reports no usage in streams, so local token counts are estimated at 4 chars/token (`router.estimate_tokens`); `done` events carry `tokens_estimated` to flag this.
- `ModelRef.provider` is `"ollama" | "openrouter" | "image"`; `model_catalog()` keys (`local_small`, `local_big`, `cloud`, `web`, `frontier`, `image`) are the ids the UI sends in `ChatRequest.model` to force a lane.
