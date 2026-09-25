"""Judges: something that turns (state, typed questions) into typed answers.

Two implementations share one interface:

  JevJudge    -> TypeSafe's hosted Jev.  POST /v1/systemone.  70-500 ms, output free.
  LocalJudge  -> an Ollama model forced into a JSON schema that mimics Jev's
                 Choice / Score / Noul answers.  Slower and less calibrated,
                 but the prompt never leaves your machine — which matters when
                 the question you are asking is "is this private?"

Both return a Decision with the same fields, so the router does not care
which one answered.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict

import httpx

from .questions import CATEGORIES, DIFFICULTY_LEVELS, build_questions, build_state
from .settings import settings


# ── Decision ───────────────────────────────────────────────────────────────────
@dataclass
class Decision:
    category: str
    category_probs: dict[str, float]
    category_confidence: float
    difficulty: float                # probability-weighted level, 0.0 .. 4.0
    difficulty_probs: dict[str, float]
    difficulty_confidence: float
    private: float                   # p(yes)
    needs_web: float                 # p(yes)
    judge: str                       # "jev" | "local"
    judge_model: str
    latency_ms: int
    input_tokens: int = 0
    raw_request: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)

    @property
    def difficulty_label(self) -> str:
        idx = min(len(DIFFICULTY_LEVELS) - 1, max(0, round(self.difficulty)))
        return DIFFICULTY_LEVELS[idx].split(":")[0]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["difficulty_label"] = self.difficulty_label
        return d


def _confidence(probs: dict[str, float]) -> float:
    """1 - normalised entropy: 1.0 when one option owns all the mass, 0.0 when flat."""
    vals = [max(1e-9, float(v)) for v in probs.values()]
    total = sum(vals)
    if total <= 0 or len(vals) < 2:
        return 1.0
    ps = [v / total for v in vals]
    entropy = -sum(p * math.log(p) for p in ps)
    return round(max(0.0, 1.0 - entropy / math.log(len(ps))), 3)


def _normalise(probs: dict[str, float], keys: list[str]) -> dict[str, float]:
    out = {k: max(0.0, float(probs.get(k, 0.0))) for k in keys}
    total = sum(out.values()) or 1.0
    return {k: round(v / total, 4) for k, v in out.items()}


# ── Jev (TypeSafe) ──────────────────────────────────────────────────────────────
class JevJudge:
    name = "jev"

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def decide(self, messages: list[dict]) -> Decision:
        if not settings.typesafe_api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set (or switch JUDGE=local)")

        body = {
            "model": settings.jev_model,
            "state": build_state(messages),
            "questions": build_questions(),
        }
        t0 = time.perf_counter()
        r = await self.client.post(
            f"{settings.typesafe_base_url}/v1/systemone",
            headers={"Authorization": f"Bearer {settings.typesafe_api_key}"},
            json=body,
            timeout=30,
        )
        latency = int((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        data = r.json()
        a = data["answers"]

        cat = a["category"]
        diff = a["difficulty"]
        cat_probs = _normalise(cat.get("probabilities", {}), list(CATEGORIES))
        diff_probs = _normalise(diff.get("probabilities", {}), [str(i) for i in range(len(DIFFICULTY_LEVELS))])

        return Decision(
            category=cat.get("choice") or max(cat_probs, key=cat_probs.get),
            category_probs=cat_probs,
            category_confidence=float(cat.get("confidence", _confidence(cat_probs))),
            difficulty=float(diff.get("score", sum(int(k) * v for k, v in diff_probs.items()))),
            difficulty_probs=diff_probs,
            difficulty_confidence=float(diff.get("confidence", _confidence(diff_probs))),
            private=float(a["private"]["noul"]),
            needs_web=float(a["needs_web"]["noul"]),
            judge=self.name,
            judge_model=data.get("model", settings.jev_model),
            latency_ms=latency,
            input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
            raw_request=body,
            raw_response=data,
        )


# ── Local judge (Ollama, structured output) ─────────────────────────────────────
_LOCAL_SYSTEM = """You are a decision model, not a chat assistant. You never write prose.
You read the STATE and answer each QUESTION with probabilities.

Rules:
- For "category" give a probability for EVERY option; they must sum to 1.
- For "difficulty" give a probability for EVERY level "0".."4"; they must sum to 1.
- For "private" and "needs_web" give a single probability (0..1) that the answer is YES.
- Judge only what the criteria say. Do not answer the user's request.
Return JSON only."""


def _local_schema() -> dict:
    num = {"type": "number", "minimum": 0, "maximum": 1}
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "object",
                "properties": {k: num for k in CATEGORIES},
                "required": list(CATEGORIES),
            },
            "difficulty": {
                "type": "object",
                "properties": {str(i): num for i in range(len(DIFFICULTY_LEVELS))},
                "required": [str(i) for i in range(len(DIFFICULTY_LEVELS))],
            },
            "private": num,
            "needs_web": num,
        },
        "required": ["category", "difficulty", "private", "needs_web"],
    }


class LocalJudge:
    name = "local"

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def decide(self, messages: list[dict]) -> Decision:
        state = build_state(messages)
        questions = build_questions()
        user = (
            "STATE:\n" + json.dumps(state, ensure_ascii=False, indent=1) +
            "\n\nQUESTIONS:\n" + json.dumps(questions, ensure_ascii=False, indent=1)
        )
        body = {
            "model": settings.local_judge_model,
            "messages": [{"role": "system", "content": _LOCAL_SYSTEM}, {"role": "user", "content": user}],
            "format": _local_schema(),   # Ollama structured outputs
            "stream": False,
            "think": False,              # ignored by models without a thinking mode
            "options": {"temperature": 0},
        }
        t0 = time.perf_counter()
        r = await self.client.post(f"{settings.ollama_base_url}/api/chat", json=body, timeout=120)
        latency = int((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        data = r.json()
        parsed = json.loads(data["message"]["content"])

        cat_probs = _normalise(parsed.get("category", {}), list(CATEGORIES))
        diff_probs = _normalise(parsed.get("difficulty", {}), [str(i) for i in range(len(DIFFICULTY_LEVELS))])

        return Decision(
            category=max(cat_probs, key=cat_probs.get),
            category_probs=cat_probs,
            category_confidence=_confidence(cat_probs),
            difficulty=round(sum(int(k) * v for k, v in diff_probs.items()), 3),
            difficulty_probs=diff_probs,
            difficulty_confidence=_confidence(diff_probs),
            private=round(float(parsed.get("private", 0.0)), 4),
            needs_web=round(float(parsed.get("needs_web", 0.0)), 4),
            judge=self.name,
            judge_model=settings.local_judge_model,
            latency_ms=latency,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            raw_request={"model": settings.local_judge_model, "state": state, "questions": questions},
            raw_response=parsed,
        )


def make_judge(kind: str, client: httpx.AsyncClient):
    return LocalJudge(client) if kind == "local" else JevJudge(client)
