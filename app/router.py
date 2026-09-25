"""Turn a Decision into a lane and an ordered list of candidate models.

This is the "boring part" that ordinary code does well: plain if/else on the
probabilities the judge returned.  Every rule is deliberately explicit and
ordered, because the order IS the policy:

  1. private data      -> never leaves the machine, whatever else is true
  2. image request     -> image lane
  3. needs the web     -> web-enabled cloud model
  4. very long prompt  -> long-context cloud model
  5. judge unsure      -> safe default (general cloud model)
  6. hard/frontier     -> frontier model if enabled, else cloud
  7. code              -> cloud code model
  8. easy chat/QA      -> small local model
  9. everything else   -> cloud general model
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .judge import Decision
from .settings import Settings, Thresholds


@dataclass(frozen=True)
class ModelRef:
    provider: str      # "ollama" | "openrouter" | "image"
    model: str
    label: str
    is_local: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Route:
    lane: str
    candidates: list[ModelRef]
    reason: str
    forced: bool = False

    def to_dict(self) -> dict:
        return {
            "lane": self.lane,
            "candidates": [c.to_dict() for c in self.candidates],
            "reason": self.reason,
            "forced": self.forced,
        }


def model_catalog(s: Settings) -> dict[str, ModelRef]:
    """Every model the router can pick, keyed by a short id used by the UI."""
    cat = {
        "local_small": ModelRef("ollama", s.local_small_model, f"{s.local_small_model} (local)", True),
        "cloud": ModelRef("openrouter", s.cloud_model, f"{s.cloud_model} (OpenRouter)", False),
        "web": ModelRef("openrouter", s.web_model, f"{s.web_model} (OpenRouter + web)", False),
        "frontier": ModelRef("openrouter", s.frontier_model, f"{s.frontier_model} (OpenRouter)", False),
    }
    if s.local_big_model:
        cat["local_big"] = ModelRef("ollama", s.local_big_model, f"{s.local_big_model} (local)", True)
    if s.image_provider == "openrouter":
        cat["image"] = ModelRef("image", s.image_model, f"{s.image_model} (image, OpenRouter)", False)
    elif s.image_provider == "local":
        cat["image"] = ModelRef("image", "local-image-server", "local image server", True)
    return cat


def choose_route(decision: Decision, prompt_tokens: int, s: Settings, t: Thresholds | None = None) -> Route:
    t = t or s.thresholds
    cat = model_catalog(s)
    local_small = cat["local_small"]
    local_big = cat.get("local_big")
    cloud, web, frontier = cat["cloud"], cat["web"], cat["frontier"]
    image = cat.get("image")

    local_chain = [local_small]
    if local_big and decision.difficulty >= 2.0:
        local_chain = [local_big, local_small]

    # 1. Private data stays local, no matter what else the judge said.
    if decision.private >= t.private:
        if decision.category == "image" and image and image.is_local:
            return Route("image", [image], f"private={decision.private:.2f} but the image server is local")
        return Route("local", local_chain, f"private={decision.private:.2f} ≥ {t.private} → must stay on this machine")

    # 2. Image requests go to the image lane (if it is switched on).
    if decision.category == "image":
        if image:
            return Route("image", [image], "category=image")
        return Route("cloud", [cloud, local_small], "category=image but IMAGE_PROVIDER=off → text answer instead")

    # 3. Fresh information → web-enabled model.
    if decision.needs_web >= t.web:
        return Route("web", [web, cloud], f"needs_web={decision.needs_web:.2f} ≥ {t.web}")

    # 4. Very long prompts → long-context cloud model.
    if prompt_tokens > t.long_prompt_tokens:
        return Route("cloud", [cloud, frontier] if t.frontier_enabled else [cloud], f"prompt ≈{prompt_tokens} tokens > {t.long_prompt_tokens}")

    # 5. The judge is not sure what this is → safe default.
    if decision.category_confidence < t.min_category_confidence:
        return Route("cloud", [cloud, local_small], f"category confidence {decision.category_confidence:.2f} < {t.min_category_confidence} → safe default")

    # 6. Hard problems → frontier model when enabled.
    if decision.difficulty >= t.frontier_difficulty:
        if t.frontier_enabled:
            return Route("frontier", [frontier, cloud], f"difficulty={decision.difficulty:.2f} ≥ {t.frontier_difficulty} and frontier enabled")
        return Route("cloud", [cloud], f"difficulty={decision.difficulty:.2f} ≥ {t.frontier_difficulty} (frontier disabled)")

    # 7. Code always goes to the cloud code model.
    if decision.category == "code":
        return Route("cloud", [cloud, local_small], "category=code")

    # 8. Easy conversational work stays local.
    if decision.category in {"chit_chat", "simple_question", "rewrite_summarize"} and decision.difficulty <= t.local_max_difficulty:
        return Route("local", [local_small, cloud], f"{decision.category} with difficulty {decision.difficulty:.2f} ≤ {t.local_max_difficulty}")

    # 9. Everything else.
    return Route("cloud", [cloud, local_small], f"{decision.category} with difficulty {decision.difficulty:.2f}")


def forced_route(model_id: str, s: Settings) -> Route:
    """The user picked a model explicitly — skip the judge."""
    cat = model_catalog(s)
    if model_id not in cat:
        raise KeyError(f"unknown model id '{model_id}'")
    ref = cat[model_id]
    lane = "image" if ref.provider == "image" else ("local" if ref.is_local else "cloud")
    return Route(lane, [ref], f"forced by user: {model_id}", forced=True)


def estimate_tokens(text: str) -> int:
    """Rough and cheap: ~4 characters per token for English."""
    return max(1, len(text) // 4)
