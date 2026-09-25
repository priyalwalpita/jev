"""Configuration for the Jev model router.

Everything is read from environment variables (see .env.example).  Thresholds
live in a mutable object so the UI can change them at runtime without a
restart — that is the whole point of a router driven by probabilities: the
judgement stays probabilistic, the risk tolerance lives in numbers you own.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


@dataclass
class Thresholds:
    private: float = _float("PRIVATE_THRESHOLD", 0.5)
    web: float = _float("WEB_THRESHOLD", 0.6)
    local_max_difficulty: float = _float("LOCAL_MAX_DIFFICULTY", 1.5)
    frontier_difficulty: float = _float("FRONTIER_DIFFICULTY", 3.5)
    min_category_confidence: float = _float("MIN_CATEGORY_CONFIDENCE", 0.5)
    long_prompt_tokens: int = int(_float("LONG_PROMPT_TOKENS", 16000))
    frontier_enabled: bool = _bool("FRONTIER_ENABLED", False)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Settings:
    # judge
    judge: str = _env("JUDGE", "jev")  # "jev" | "local"
    typesafe_api_key: str = _env("TYPESAFE_API_KEY")
    typesafe_base_url: str = _env("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
    jev_model: str = _env("JEV_MODEL", "jev-latest")
    local_judge_model: str = _env("LOCAL_JUDGE_MODEL", "qwen3:4b")

    # local generation (Ollama)
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    local_small_model: str = _env("LOCAL_SMALL_MODEL", "llama3.2:3b")
    local_big_model: str = _env("LOCAL_BIG_MODEL", "")

    # cloud generation (OpenRouter)
    openrouter_api_key: str = _env("OPENROUTER_API_KEY")
    openrouter_base_url: str = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    cloud_model: str = _env("CLOUD_MODEL", "deepseek/deepseek-v4.1-flash")
    web_model: str = _env("WEB_MODEL", "deepseek/deepseek-v4.1-flash:online")
    frontier_model: str = _env("FRONTIER_MODEL", "anthropic/claude-opus-5")

    # image lane
    image_provider: str = _env("IMAGE_PROVIDER", "openrouter")  # openrouter | local | off
    image_model: str = _env("IMAGE_MODEL", "openai/gpt-5.4-image-2")
    local_image_url: str = _env("LOCAL_IMAGE_URL", "http://localhost:8188/v1/images/generations")

    # prices (USD per million tokens) — only used for the stats panel
    cloud_price_in: float = _float("CLOUD_PRICE_IN", 0.12)
    cloud_price_out: float = _float("CLOUD_PRICE_OUT", 0.48)
    frontier_price_in: float = _float("FRONTIER_PRICE_IN", 5.0)
    frontier_price_out: float = _float("FRONTIER_PRICE_OUT", 25.0)
    jev_price_in: float = _float("JEV_PRICE_IN", 0.042)

    # server
    host: str = _env("HOST", "127.0.0.1")
    port: int = int(_float("PORT", 8000))
    db_path: str = _env("DB_PATH", "router.sqlite3")

    thresholds: Thresholds = field(default_factory=Thresholds)


settings = Settings()
