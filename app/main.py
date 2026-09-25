"""Jev model router — FastAPI server.

    POST /chat    one call: judge → route → stream from the winner (SSE)
    POST /decide  judge only, so you can inspect what Jev decided and why
    GET  /stats   requests, % answered locally, estimated spend and savings
    GET  /config  current thresholds & judge   (POST to change them live)
    GET  /health  is Ollama up? are the keys present?

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import db
from .judge import Decision, make_judge
from .providers import StreamResult, generate_image, ollama_models, rewrite_image_prompt, stream_chat
from .router import ModelRef, Route, choose_route, estimate_tokens, forced_route, model_catalog
from .settings import settings

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    app.state.http = httpx.AsyncClient()
    yield
    await app.state.http.aclose()


app = FastAPI(title="Jev model router", lifespan=lifespan)


# ── Schemas ─────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None          # a catalog id (see /models) to bypass the judge; None = auto
    judge: str | None = None          # "jev" | "local"; None = server default
    system_prompt: str | None = None


class ConfigUpdate(BaseModel):
    judge: str | None = None
    private: float | None = None
    web: float | None = None
    local_max_difficulty: float | None = None
    frontier_difficulty: float | None = None
    min_category_confidence: float | None = None
    long_prompt_tokens: int | None = None
    frontier_enabled: bool | None = None


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _price(ref: ModelRef, in_tokens: int, out_tokens: int) -> float:
    if ref.is_local:
        return 0.0
    if ref.model == settings.frontier_model:
        return (in_tokens * settings.frontier_price_in + out_tokens * settings.frontier_price_out) / 1e6
    return (in_tokens * settings.cloud_price_in + out_tokens * settings.cloud_price_out) / 1e6


def _cloud_equivalent(in_tokens: int, out_tokens: int) -> float:
    """What this request would have cost had it gone to the default cloud model."""
    return (in_tokens * settings.cloud_price_in + out_tokens * settings.cloud_price_out) / 1e6


async def _judge(messages: list[dict], kind: str | None) -> Decision:
    kind = kind or settings.judge
    judge = make_judge(kind, app.state.http)
    return await judge.decide(messages)


def _plain(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    out = {
        "judge": settings.judge,
        "typesafe_key": bool(settings.typesafe_api_key),
        "openrouter_key": bool(settings.openrouter_api_key),
        "ollama": False,
        "ollama_models": [],
    }
    try:
        out["ollama_models"] = await ollama_models(app.state.http)
        out["ollama"] = True
    except Exception as exc:  # noqa: BLE001
        out["ollama_error"] = str(exc)[:200]
    return out


@app.get("/models")
async def models():
    return {"auto": "auto (judge decides)", "catalog": {k: v.to_dict() for k, v in model_catalog(settings).items()}}


@app.get("/config")
async def get_config():
    return {"judge": settings.judge, "thresholds": settings.thresholds.as_dict(),
            "judge_models": {"jev": settings.jev_model, "local": settings.local_judge_model}}


@app.post("/config")
async def set_config(update: ConfigUpdate):
    data = update.model_dump(exclude_none=True)
    if "judge" in data:
        if data["judge"] not in {"jev", "local"}:
            raise HTTPException(400, "judge must be 'jev' or 'local'")
        settings.judge = data.pop("judge")
    for key, value in data.items():
        setattr(settings.thresholds, key, value)
    return await get_config()


@app.post("/decide")
async def decide(req: ChatRequest):
    messages = _plain(req.messages)
    try:
        decision = await _judge(messages, req.judge)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"judge failed: {exc}") from exc
    tokens = estimate_tokens("\n".join(m["content"] for m in messages))
    route = choose_route(decision, tokens, settings)
    return {"decision": decision.to_dict(), "route": route.to_dict(), "prompt_tokens": tokens}


@app.get("/stats")
async def stats():
    return db.stats()


@app.get("/requests")
async def requests_(limit: int = 20):
    return db.recent(limit)


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = _plain(req.messages)
    if req.system_prompt:
        messages = [{"role": "system", "content": req.system_prompt}] + messages
    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    latest_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    prompt_tokens = estimate_tokens("\n".join(m["content"] for m in messages))

    async def gen():
        t_start = time.perf_counter()
        decision: Decision | None = None
        route: Route

        # 1. Decide — either the judge, or the user forced a model.
        if req.model and req.model != "auto":
            try:
                route = forced_route(req.model, settings)
            except KeyError as exc:
                yield _sse("error", {"message": str(exc)})
                return
        else:
            try:
                decision = await _judge(messages, req.judge)
            except Exception as exc:  # noqa: BLE001
                yield _sse("error", {"message": f"judge failed: {exc}"})
                return
            route = choose_route(decision, prompt_tokens, settings)

        yield _sse("decision", {
            "decision": decision.to_dict() if decision else None,
            "route": route.to_dict(),
            "prompt_tokens": prompt_tokens,
        })

        judge_cost = (decision.input_tokens * settings.jev_price_in / 1e6) if (decision and decision.judge == "jev") else 0.0
        log = {
            "prompt_preview": latest_user[:160],
            "judge": decision.judge if decision else None,
            "judge_model": decision.judge_model if decision else None,
            "judge_ms": decision.latency_ms if decision else None,
            "judge_tokens": decision.input_tokens if decision else None,
            "category": decision.category if decision else None,
            "category_conf": decision.category_confidence if decision else None,
            "difficulty": decision.difficulty if decision else None,
            "private": decision.private if decision else None,
            "needs_web": decision.needs_web if decision else None,
            "lane": route.lane,
            "reason": route.reason,
            "forced": int(route.forced),
            "decision_json": decision.to_dict() if decision else {},
        }

        # 2. Image lane.
        if route.lane == "image":
            ref = route.candidates[0]
            t0 = time.perf_counter()
            try:
                rewritten = await rewrite_image_prompt(app.state.http, latest_user)
                yield _sse("status", {"message": f"prompt rewritten by {settings.local_small_model}", "rewritten": rewritten})
                data_url = await generate_image(app.state.http, rewritten)
            except Exception as exc:  # noqa: BLE001
                db.log_request({**log, "model": ref.model, "provider": ref.provider, "is_local": int(ref.is_local),
                                "ok": 0, "error": str(exc)[:300], "est_cost": judge_cost, "est_cloud_cost": judge_cost})
                yield _sse("error", {"message": f"image generation failed: {exc}"})
                return
            gen_ms = int((time.perf_counter() - t0) * 1000)
            rid = db.log_request({**log, "model": ref.model, "provider": ref.provider, "is_local": int(ref.is_local),
                                  "gen_ms": gen_ms, "in_tokens": 0, "out_tokens": 0, "ok": 1,
                                  "est_cost": judge_cost, "est_cloud_cost": judge_cost})
            db.log_image(rid, latest_user, rewritten, data_url)
            yield _sse("image", {"data_url": data_url, "rewritten_prompt": rewritten})
            yield _sse("done", {"model": ref.label, "provider": ref.provider, "is_local": ref.is_local,
                                "gen_ms": gen_ms, "total_ms": int((time.perf_counter() - t_start) * 1000),
                                "est_cost_usd": round(judge_cost, 6)})
            return

        # 3. Text lanes: try the candidates in order, fall back if one fails before streaming.
        last_error = "no candidates"
        for ref in route.candidates:
            result = StreamResult()
            t0 = time.perf_counter()
            started = False
            try:
                async for delta in stream_chat(app.state.http, ref, messages, result):
                    if not started:
                        started = True
                        yield _sse("start", {"model": ref.label, "provider": ref.provider, "is_local": ref.is_local})
                    yield _sse("token", {"text": delta})
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if started:  # failed mid-stream: nothing sensible to fall back to
                    db.log_request({**log, "model": ref.model, "provider": ref.provider, "is_local": int(ref.is_local),
                                    "ok": 0, "error": last_error[:300], "est_cost": judge_cost, "est_cloud_cost": judge_cost})
                    yield _sse("error", {"message": f"{ref.label} failed mid-stream: {exc}"})
                    return
                yield _sse("status", {"message": f"{ref.label} unavailable ({str(exc)[:120]}) — falling back"})
                continue

            gen_ms = int((time.perf_counter() - t0) * 1000)
            in_tokens = result.input_tokens or estimate_tokens(user_text)
            out_tokens = result.output_tokens or estimate_tokens(result.text)
            est_cost = _price(ref, in_tokens, out_tokens) + judge_cost
            est_cloud = _cloud_equivalent(in_tokens, out_tokens) + judge_cost
            db.log_request({**log, "model": ref.model, "provider": ref.provider, "is_local": int(ref.is_local),
                            "gen_ms": gen_ms, "in_tokens": in_tokens, "out_tokens": out_tokens, "ok": 1,
                            "est_cost": est_cost, "est_cloud_cost": est_cloud})
            yield _sse("done", {"model": ref.label, "provider": ref.provider, "is_local": ref.is_local,
                                "gen_ms": gen_ms, "total_ms": int((time.perf_counter() - t_start) * 1000),
                                "in_tokens": in_tokens, "out_tokens": out_tokens,
                                "est_cost_usd": round(est_cost, 6), "tokens_estimated": not result.input_tokens})
            return

        db.log_request({**log, "ok": 0, "error": last_error[:300], "est_cost": judge_cost, "est_cloud_cost": judge_cost})
        yield _sse("error", {"message": f"all candidates failed: {last_error}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
