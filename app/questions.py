"""The one Jev request that drives the whole router.

Three (well, four) typed questions, all evaluated in parallel against the same
state.  Keep every question and its criteria in this file: it is the part a
human needs to review, and the part you will tune.

  choice  -> which lane (category) does this prompt belong to?
  score   -> how much model capability does a good answer need?
  noul    -> does it contain private / confidential data?
  noul    -> does it need fresh information from the web?
"""
from __future__ import annotations

CATEGORIES: dict[str, str] = {
    "chit_chat": "Greetings, small talk, thanks, jokes or casual conversation with no real task",
    "simple_question": "A factual or how-to question that a small model can answer in a few sentences",
    "rewrite_summarize": "Rewrite, rephrase, translate, proofread, summarize or shorten text the user supplied",
    "code": "Write, fix, explain, refactor or review code, shell commands, SQL, regex or configuration files",
    "reasoning_analysis": "Multi-step reasoning, analysis, planning, maths, comparisons or long structured answers",
    "image": "Asks to generate, draw, create, render, paint or edit an image or picture",
}

DIFFICULTY_LEVELS: list[str] = [
    "Trivial: a tiny 2-3B local model handles it easily (greetings, one-line facts, tiny edits)",
    "Easy: a small local model handles it with a short, straightforward answer",
    "Moderate: needs a capable general model (multi-part answer, moderate code, careful explanation)",
    "Hard: needs a strong frontier-class model (complex code, deep analysis, tricky edge cases)",
    "Frontier: needs the best model available (research-grade reasoning, large systems, novel problems)",
]

PRIVATE_CRITERIA = {
    "true": (
        "The conversation contains real personal, confidential, client, financial, medical or credential "
        "data: names with contact details, phone numbers, addresses, ID numbers, card numbers, medical "
        "details, internal company documents, API keys, tokens, passwords or secrets"
    ),
    "false": "No sensitive data; generic questions, public information or placeholder examples only",
}

WEB_CRITERIA = {
    "true": (
        "A good answer needs information newer than a model's training data or live data: today's news, "
        "current prices, latest versions, live scores, weather, what a specific URL says right now"
    ),
    "false": "A good answer needs no fresh information: general knowledge, reasoning, code, rewriting",
}


def build_questions() -> dict:
    """Return the questions in TypeSafe's wire format (see docs.typesafe.ai)."""
    return {
        "category": {
            "type": "choice",
            "instructions": (
                "What is `latest_user_message` asking the assistant to do? Judge the latest message in the "
                "context of `conversation`."
            ),
            "criteria": CATEGORIES,
        },
        "difficulty": {
            "type": "score",
            "instructions": (
                "How much model capability does a good answer to `latest_user_message` need?"
            ),
            "criteria": DIFFICULTY_LEVELS,
        },
        "private": {
            "type": "noul",
            "instructions": (
                "Does `conversation` contain personal, confidential, client, financial, medical or credential "
                "data (such as API keys, passwords or tokens) that must not be sent to a third-party cloud?"
            ),
            "criteria": PRIVATE_CRITERIA,
        },
        "needs_web": {
            "type": "noul",
            "instructions": "Does a good answer to `latest_user_message` require current information from the web?",
            "criteria": WEB_CRITERIA,
        },
    }


def build_state(messages: list[dict]) -> dict:
    """Send only what the questions need: the conversation and the latest user turn."""
    latest = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest = m.get("content", "")
            break
    # Keep the state small — accuracy drops as irrelevant context grows.
    trimmed = [{"role": m["role"], "content": m["content"][-4000:]} for m in messages[-8:]]
    return {"conversation": trimmed, "latest_user_message": latest}
