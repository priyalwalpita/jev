"""Log every routed request so the stats panel can answer the only questions that
matter for a router: how many stayed local, what did it cost, what did it save?"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing

from .settings import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL NOT NULL,
    prompt_preview TEXT,
    judge          TEXT,
    judge_model    TEXT,
    judge_ms       INTEGER,
    judge_tokens   INTEGER,
    category       TEXT,
    category_conf  REAL,
    difficulty     REAL,
    private        REAL,
    needs_web      REAL,
    lane           TEXT,
    reason         TEXT,
    forced         INTEGER,
    model          TEXT,
    provider       TEXT,
    is_local       INTEGER,
    gen_ms         INTEGER,
    in_tokens      INTEGER,
    out_tokens     INTEGER,
    est_cost       REAL,
    est_cloud_cost REAL,
    ok             INTEGER,
    error          TEXT,
    decision_json  TEXT
);
CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER,
    prompt     TEXT,
    rewritten  TEXT,
    data_url   TEXT
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(connect()) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def log_request(row: dict) -> int:
    row = dict(row)
    row.setdefault("ts", time.time())
    row["decision_json"] = json.dumps(row.get("decision_json") or {})
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with closing(connect()) as conn:
        cur = conn.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", list(row.values()))
        conn.commit()
        return int(cur.lastrowid)


def log_image(request_id: int, prompt: str, rewritten: str, data_url: str) -> None:
    with closing(connect()) as conn:
        conn.execute("INSERT INTO images (request_id, prompt, rewritten, data_url) VALUES (?,?,?,?)",
                     (request_id, prompt, rewritten, data_url))
        conn.commit()


def recent(limit: int = 20) -> list[dict]:
    with closing(connect()) as conn:
        rows = conn.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["decision_json"] = json.loads(d.get("decision_json") or "{}")
        out.append(d)
    return out


def stats() -> dict:
    with closing(connect()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM requests WHERE ok=1").fetchone()[0]
        local = conn.execute("SELECT COUNT(*) FROM requests WHERE ok=1 AND is_local=1").fetchone()[0]
        private = conn.execute("SELECT COUNT(*) FROM requests WHERE ok=1 AND private >= ?",
                               (settings.thresholds.private,)).fetchone()[0]
        spend = conn.execute("SELECT COALESCE(SUM(est_cost),0) FROM requests WHERE ok=1").fetchone()[0]
        would_have = conn.execute("SELECT COALESCE(SUM(est_cloud_cost),0) FROM requests WHERE ok=1").fetchone()[0]
        judge_ms = conn.execute(
            "SELECT judge, AVG(judge_ms) AS ms, COUNT(*) AS n FROM requests WHERE judge_ms IS NOT NULL GROUP BY judge"
        ).fetchall()
        lanes = conn.execute("SELECT lane, COUNT(*) AS n FROM requests WHERE ok=1 GROUP BY lane").fetchall()
    return {
        "requests": total,
        "local": local,
        "local_pct": round(100 * local / total, 1) if total else 0.0,
        "private": private,
        "est_spend_usd": round(spend, 6),
        "est_saved_usd": round(max(0.0, would_have - spend), 6),
        "judge_latency_ms": {r["judge"]: {"avg_ms": round(r["ms"] or 0), "n": r["n"]} for r in judge_ms},
        "lanes": {r["lane"]: r["n"] for r in lanes},
    }
