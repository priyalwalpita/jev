"""Fire the demo prompts at the router and print what the judge decided.

    python scripts/demo.py                 # judge only (POST /decide) — fast, no generation
    python scripts/demo.py --chat          # full round trip, streams each answer
    python scripts/demo.py --judge local   # force the local judge for this run
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Hey, how's it going?",
    "Write a Python function that removes duplicates from a list while preserving order.",
    "My OpenRouter key is sk-or-v1-8f3ab12c9d... can you help me rotate it safely?",
    "Nimal Perera, +94 77 123 4567, owes us LKR 240,000 — draft a polite reminder.",
    "What is the latest news about TypeSafe AI's Jev model today?",
    "Draw a sandy-coloured Maine Coon cat sitting in a sunlit meadow.",
    "Compare event sourcing with CRUD for a banking ledger and recommend one for a 5-person team.",
]


def _request(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def post(url: str, body: dict) -> dict:
    return json.loads(_request(url, body).read())


def post_stream(url: str, body: dict):
    """Yield (event, data) pairs from the router's SSE stream."""
    event = "message"
    for raw in _request(url, body):
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            yield event, json.loads(line[5:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--judge", choices=["jev", "local"], default=None)
    ap.add_argument("--chat", action="store_true", help="run the full /chat round trip instead of /decide")
    args = ap.parse_args()

    if not args.chat:
        print(f"{'prompt':<62} {'category':<19} {'diff':>5} {'priv':>5} {'web':>5} {'ms':>5}  lane → model")
        print("-" * 140)
    for p in PROMPTS:
        body = {"messages": [{"role": "user", "content": p}], "judge": args.judge}
        if not args.chat:
            out = post(f"{args.base}/decide", body)
            d, r = out["decision"], out["route"]
            print(f"{p[:60]:<62} {d['category']:<19} {d['difficulty']:>5.2f} {d['private']:>5.2f} {d['needs_web']:>5.2f} "
                  f"{d['latency_ms']:>5}  {r['lane']} → {r['candidates'][0]['model']}")
            continue
        print(f"\n▶ {p}")
        for event, data in post_stream(f"{args.base}/chat", body):
            if event == "decision":
                d, r = data["decision"], data["route"]
                print(f"  judge {d['judge']} {d['latency_ms']} ms · {d['category']} · diff {d['difficulty']:.2f} · "
                      f"private {d['private']:.2f} · web {d['needs_web']:.2f} → {r['lane']} ({r['reason']})")
            elif event == "token":
                sys.stdout.write(data["text"]); sys.stdout.flush()
            elif event == "image":
                print(f"  [image generated · prompt: {data['rewritten_prompt'][:80]}…]")
            elif event == "status":
                print(f"  · {data['message']}")
            elif event == "done":
                print(f"\n  ✓ {data['model']} · {data['gen_ms']} ms · ≈${data['est_cost_usd']}")
            elif event == "error":
                print(f"  ✗ {data['message']}")


if __name__ == "__main__":
    main()
