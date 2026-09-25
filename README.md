# Jev model router

A local endpoint that reads every prompt with **Jev** (TypeSafe AI's System One model) and decides,
in one ~100 ms call, **which model should answer it** — a tiny local model on Ollama, a cloud
model on OpenRouter, a web-enabled model, a frontier model, or an image model — and whether the
prompt contains **private data that must never leave your machine**.

```
 browser ──► FastAPI /chat ──► judge (Jev, or a local Ollama judge)
                                  │  one request, four typed questions, evaluated in parallel:
                                  │    choice  category    → chit_chat | simple_question | rewrite_summarize | code | reasoning_analysis | image
                                  │    score   difficulty  → trivial · easy · moderate · hard · frontier
                                  │    noul    private     → p(contains PII / secrets)
                                  │    noul    needs_web   → p(needs fresh information)
                                  ▼
                            router.py (plain if/else on the probabilities)
                                  │
        ┌───────────┬─────────────┼──────────────┬─────────────┐
        ▼           ▼             ▼              ▼             ▼
   local (Ollama)  cloud       web (:online)   frontier      image
   llama3.2:3b     DeepSeek    DeepSeek+web    Claude Opus 5 GPT-5.4 Image 2 / local server
        └───────────┴─────────────┴──────────────┴─────────────┘
                                  │  stream back from the winner, fall back if it fails
                                  ▼
                           SQLite log → /stats (requests, % local, spend, savings)
```

## 1. Setup (10 minutes)

```bash
# models on your machine
ollama pull llama3.2:3b        # the small answerer (swap for gemma3:4b / qwen3:4b if you prefer)
ollama pull qwen3:4b           # the local judge (only needed for JUDGE=local)

# the router
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill in the keys below
```

Keys in `.env`:

| Variable | Where to get it | Needed for |
|---|---|---|
| `TYPESAFE_API_KEY` | console.typesafe.ai (waitlist at typesafe.ai) | `JUDGE=jev` |
| `OPENROUTER_API_KEY` | openrouter.ai | cloud, web, frontier and image lanes |

No TypeSafe key yet? Set `JUDGE=local` — the router still works, it just uses an Ollama model
forced into Jev's answer shapes (see *Swapping the judge* below).

```bash
uvicorn app.main:app --reload          # http://127.0.0.1:8000
python scripts/demo.py                 # judge-only table for the demo prompts
python scripts/demo.py --chat          # full round trip with streamed answers
python -m pytest -q                    # routing rules, no network needed
```

## 2. The demo, step by step (what to show on camera)

1. **Health dots** in the header: Ollama up, TypeSafe key present, OpenRouter key present.
2. Pick a model manually first (`llama3.2:3b (local)`) — instant, guaranteed local. Then pick
   DeepSeek — slower, goes to the cloud. Then switch back to **auto** so the judge decides.
3. `Hey, how's it going?` → category *chit_chat*, difficulty ≈ 0.3, private ≈ 0.02 → **local lane**.
   Open *Last decision* and show the judge latency (Jev: a few hundred ms).
4. `Write a Python function that removes duplicates…` → *code* at ~1.0 probability → **cloud lane**
   (DeepSeek V4.1 Flash on OpenRouter). Expand *raw request → judge* to show the exact Jev call.
5. `My OpenRouter key is sk-or-v1-…` → **private ≈ 0.96** → forced **local**, whatever the category.
   Same with a name + phone number.
6. `Draw a sandy-coloured Maine Coon…` → *image* → the local model rewrites the prompt, the image
   lane renders it.
7. `What is the latest news about … today?` → *needs_web* ≥ 0.6 → DeepSeek with OpenRouter's
   `:online` web plugin.
8. Flip **Judge → Local (Ollama)** and repeat the API-key prompt: same decision, ~100 ms on a decent
   GPU, and the prompt never left the machine — not even to be judged.
9. Drag the **threshold sliders** live (e.g. lower `private ≥` to 0.3) and re-send.
10. **Session stats**: requests, % answered locally, estimated spend and what an all-cloud setup
    would have cost.

## 3. How the routing works

`app/questions.py` is the single source of truth for the questions. `app/router.py` applies the
rules in this order — the order *is* the policy:

| # | Rule | Goes to |
|---|---|---|
| 1 | `private ≥ 0.5` | local, always (bigger local model if configured and difficulty ≥ 2) |
| 2 | category = image | image lane (OpenRouter image model, or your local server) |
| 3 | `needs_web ≥ 0.6` | web-enabled cloud model |
| 4 | prompt > 16k tokens | long-context cloud model |
| 5 | category confidence < 0.5 | safe default (general cloud model) |
| 6 | difficulty ≥ 3.5 | frontier model if enabled, else cloud |
| 7 | category = code | cloud code model |
| 8 | chat / simple question / rewrite and difficulty ≤ 1.5 | small local model |
| 9 | everything else | general cloud model |

Every lane is an ordered list of candidates; if the first model errors before it starts streaming,
the router falls back to the next one and tells the UI why.

### The one Jev request

```json
{
  "model": "jev-latest",
  "state": { "conversation": [...], "latest_user_message": "..." },
  "questions": {
    "category":   { "type": "choice", "instructions": "...", "criteria": { "chit_chat": "...", "code": "...", "image": "..." } },
    "difficulty": { "type": "score",  "instructions": "...", "criteria": ["Trivial: ...", "Easy: ...", "Moderate: ...", "Hard: ...", "Frontier: ..."] },
    "private":    { "type": "noul",   "instructions": "...", "criteria": { "true": "...", "false": "..." } },
    "needs_web":  { "type": "noul",   "instructions": "...", "criteria": { "true": "...", "false": "..." } }
  }
}
```

All four questions are evaluated in parallel against the same state, so asking four costs about
the same time as asking one. At $0.042 per million input tokens with output free, a 400-token
router call costs ~$0.000017.

## 4. Swapping the judge

`JUDGE=jev` sends the prompt to TypeSafe to ask, among other things, whether it is private — which
means you have already leaked it. That is fine if you trust TypeSafe and just don't want the big
frontier labs to see your data; it is not fine for a hospital.

`JUDGE=local` (or the toggle in the UI) replaces Jev with an Ollama model constrained to a JSON
schema that mimics Jev's Choice / Score / Noul answers (`app/judge.py → LocalJudge`). It is slower
and less calibrated than the real thing, but nothing leaves the machine. A good pattern: local
judge for the privacy gate, Jev for everything else.

## 5. Gotchas

- **Thresholds are yours, not Jev's.** The 0.5 / 0.6 / 1.5 / 3.5 values are starting points. Run
  the router in shadow mode against your real prompts, look at where confidence and correctness
  diverge, then move the sliders.
- **Typed ≠ correct.** Jev cannot return a lane you did not define, but it can pick the wrong one.
  That is what the category-confidence rule (#5) is for.
- **Long chats**: if a conversation started on the frontier model, you probably want to *stay*
  there for prompt-cache hits instead of re-routing every turn. Easy to add: check
  `state.messages` length in `main.py` before calling the judge.
- **Cost figures are estimates.** Ollama's OpenAI endpoint does not report usage in streams, so
  local token counts use a 4-chars-per-token estimate; cloud prices come from `.env`.
- **Image lane** defaults to OpenRouter (`openai/gpt-5.4-image-2`, requested with
  `modalities: ["image","text"]`). Point `IMAGE_PROVIDER=local` at any server that speaks the
  OpenAI `/v1/images/generations` API if you run Qwen-Image or Stable Diffusion locally.

## 6. Files

```
app/settings.py     every knob, read from .env (thresholds are live-editable)
app/questions.py    the typed questions — edit these, not the prompt
app/judge.py        JevJudge (TypeSafe HTTP API) and LocalJudge (Ollama structured output)
app/router.py       lanes, ordered fallbacks, the rules table above
app/providers.py    streaming for Ollama + OpenRouter, image generation, prompt rewrite
app/db.py           SQLite log + /stats
app/main.py         FastAPI: /chat (SSE) /decide /config /stats /health
app/static/index.html   the UI
scripts/demo.py     the demo prompts, judge-only or full round trip
tests/test_router.py    routing rules
```
