"""Routing rules, tested with hand-made decisions (no network needed).

    python -m pytest -q
"""
from app.judge import Decision
from app.router import choose_route, forced_route
from app.settings import Settings, Thresholds


def decision(**kw) -> Decision:
    base = dict(
        category="chit_chat", category_probs={"chit_chat": 0.9}, category_confidence=0.9,
        difficulty=0.3, difficulty_probs={"0": 0.8, "1": 0.2}, difficulty_confidence=0.8,
        private=0.02, needs_web=0.05, judge="test", judge_model="test", latency_ms=1,
    )
    base.update(kw)
    return Decision(**base)


def settings(**thr) -> Settings:
    s = Settings()
    s.local_small_model = "small"
    s.local_big_model = "big"
    s.cloud_model = "cloud"
    s.web_model = "cloud:online"
    s.frontier_model = "frontier"
    s.image_provider = "openrouter"
    s.thresholds = Thresholds(**thr)
    return s


def test_chit_chat_goes_local():
    r = choose_route(decision(), 20, settings())
    assert r.lane == "local" and r.candidates[0].model == "small"


def test_code_goes_to_cloud():
    r = choose_route(decision(category="code", difficulty=1.8), 60, settings())
    assert r.lane == "cloud" and r.candidates[0].model == "cloud"


def test_private_overrides_everything():
    r = choose_route(decision(category="code", difficulty=3.9, needs_web=0.95, private=0.96), 60, settings(frontier_enabled=True))
    assert r.lane == "local" and all(c.is_local for c in r.candidates)
    assert r.candidates[0].model == "big"          # harder + private → bigger local model first


def test_web_lane():
    r = choose_route(decision(category="simple_question", needs_web=0.8), 30, settings())
    assert r.lane == "web" and r.candidates[0].model == "cloud:online"


def test_frontier_only_when_enabled():
    d = decision(category="reasoning_analysis", difficulty=3.7)
    assert choose_route(d, 100, settings()).lane == "cloud"
    assert choose_route(d, 100, settings(frontier_enabled=True)).candidates[0].model == "frontier"


def test_unsure_judge_uses_safe_default():
    r = choose_route(decision(category_confidence=0.3), 20, settings())
    assert r.lane == "cloud" and "safe default" in r.reason


def test_long_prompt_goes_to_cloud():
    r = choose_route(decision(), 20_000, settings())
    assert r.lane == "cloud"


def test_image_lane():
    r = choose_route(decision(category="image"), 20, settings())
    assert r.lane == "image"


def test_forced_model_skips_judge():
    r = forced_route("local_small", settings())
    assert r.forced and r.lane == "local"
