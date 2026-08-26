"""LLM-first natural-language-understanding layer, sitting IN FRONT OF the
existing rule-based intent matcher (intents.py) and query pipeline
(queries.py) — added as an explicitly-approved, deliberate reversal of this
project's original "no LLM" design constraint.

HARD SAFETY CONTRACT (do not violate when editing this file):
  - Gemini's job ends at producing {intent, entities, confidence}. It NEVER
    sees the database, NEVER writes SQL, and NEVER produces a final numeric
    answer. Every number the user sees still comes from queries.py running
    against public.pace_chatbot_view / public.pace_1, exactly as before this
    layer was added.
  - `intent` returned by Gemini is validated against the real intent-name
    set derived from intents._INTENTS (not hand-duplicated) before it is
    trusted at all. Anything else (typo, hallucinated intent, malformed
    JSON) is treated as "no LLM opinion" and the caller falls back to the
    pre-existing rule-based matcher.
  - Any Gemini call failure (bad/missing key, network error, timeout,
    malformed response) must be caught here and turned into `None` — callers
    must never crash or block on an LLM outage; the whole app must keep
    working via the rule-based matcher alone.
"""

import json
import logging
import os
import time

from . import intents

logger = logging.getLogger("pace_chatbot.llm_nlu")

GEMINI_MODEL = "gemini-3.5-flash-lite"
_TIMEOUT_SECONDS = 20.0  # Gemini API rejects a manually-set deadline below 10s; raised from 12s to reduce timeout rate

# Derived from the SAME source of truth the rule-based matcher uses, so this
# list can never drift out of sync with intents.py._INTENTS.
_VALID_INTENTS = sorted(set(n for n, _ in intents._INTENTS))

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": _VALID_INTENTS + ["none"]},
        "entities": {
            "type": "object",
            "properties": {
                "employee": {"type": "string", "nullable": True},
                "department": {"type": "string", "nullable": True},
                "manager": {"type": "string", "nullable": True},
                "month": {"type": "string", "nullable": True},
                "metric": {"type": "string", "nullable": True},
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["intent", "confidence"],
}

_SYSTEM_PROMPT = f"""You are the natural-language-understanding front end for an internal
HR/attendance analytics chatbot called PACE. Your ONLY job is to classify the
user's message into exactly one of a fixed list of intents, and to pull out
any named entities (employee name, department name, manager name, month,
metric) mentioned in the text, AS WRITTEN by the user (do not invent, guess,
correct spelling heavily, or normalize names beyond obvious capitalization —
downstream code does its own careful fuzzy-matching against real data).

You NEVER answer the question yourself. You NEVER know any real attendance
numbers, scores, or employee data. You only output structured JSON describing
what the user is asking for.

Valid intent names (choose EXACTLY one, or "none" if nothing fits):
{', '.join(_VALID_INTENTS)}

Guidance on what some intent families mean:
- attendance_best / attendance_worst: who has the best/worst attendance (org, dept, or team scoped)
- pace_score_best / pace_score_worst: ranking by overall PACE score
- improving / declining: org-wide ranking of who is improving/declining
- emp_trend: is a SPECIFIC named person improving or declining
- emp_pace_score: a specific person's current PACE score/status
- day_count / day_list: "how many were X yesterday" / "who was X on [day]" for a flag (attendance, WFH, leave, late, etc.)
- status_list / status_count: filter/count by Black/Red/Amber/Green PACE status category
- wfh_ranking / wfh_emp / wfh_by_dept: work-from-home questions (ranking vs specific employee vs department breakdown)
- score_drop_ranking / score_improvement_alltime: whose score dropped/improved the most (opposite pair — be careful about direction)
- subscore_compare_emp: which area (engagement/effectiveness/discipline) a specific person is weakest/strongest in
- team_how_doing / team_lowest_scorers / team_compare: "my team" / manager's-team scoped questions
- dept_best / dept_worst / dept_compare / dept_summary / dept_avg: department-level questions
- full_trend_emp: a SPECIFIC named person's month-by-month/monthly breakdown for ANY metric — not just PACE score. This covers "X month wise", "X month by month", "how many times did [person] [do X] each month" for PACE score, WFH days, visit days, leave days, late-comings, early-leavings, deficient-hour days, OT days/hours, engagement %, effectiveness %, discipline %, or working-hours % of a NAMED employee. Prefer full_trend_emp over the single-day/status lookup intents (wfh_emp, leave_emp_check, emp_late_comings, etc.) whenever the message asks for a MONTHLY breakdown/trend rather than a one-off status check.

CRITICAL DIRECTION SAFETY: many intents come in opposite pairs (best/worst,
improving/declining, most/fewest, drop/improvement). Read the user's wording
carefully for the ACTUAL direction they mean — do not default to one side.
If the phrasing is genuinely ambiguous about direction, lower your confidence
score rather than guessing.

If the message doesn't match any listed intent (e.g. small talk, an
unrelated question, or something this system genuinely has no capability
for), return intent "none".

Respond with JSON matching the given schema. `confidence` is your own 0.0-1.0
estimate of how sure you are about the intent choice."""

_FEW_SHOT_EXAMPLES = [
    ("pace score of Aarna Jain", {"intent": "emp_pace_score", "entities": {"employee": "Aarna Jain"}, "confidence": 0.95}),
    ("who barely showed up for work", {"intent": "attendance_worst", "entities": {}, "confidence": 0.7}),
    ("anyone been remote a lot lately", {"intent": "wfh_ranking", "entities": {}, "confidence": 0.65}),
    ("is Megha Sharma improving", {"intent": "emp_trend", "entities": {"employee": "Megha Sharma"}, "confidence": 0.95}),
    ("who is in red category in scm department", {"intent": "status_list", "entities": {"department": "SCM"}, "confidence": 0.9}),
    ("how many employees were on wfh yesterday", {"intent": "day_count", "entities": {"metric": "wfh"}, "confidence": 0.9}),
    ("aryan is declining in which discipline eng or eff", {"intent": "subscore_compare_emp", "entities": {"employee": "Aryan"}, "confidence": 0.8}),
    ("worst attendance in Accounts this month", {"intent": "attendance_worst", "entities": {"department": "Accounts", "month": "this month"}, "confidence": 0.9}),
    ("top 5 by pace score in IT-Development", {"intent": "pace_score_best", "entities": {"department": "IT-Development"}, "confidence": 0.9}),
    ("compare Accounts vs Billing", {"intent": "dept_compare", "entities": {}, "confidence": 0.9}),
    ("how is my team doing", {"intent": "team_how_doing", "entities": {}, "confidence": 0.9}),
    ("new joiners in my team", {"intent": "new_joiners", "entities": {}, "confidence": 0.85}),
    ("whose score dropped the most", {"intent": "score_drop_ranking", "entities": {}, "confidence": 0.9}),
    ("who improved the most overall", {"intent": "score_improvement_alltime", "entities": {}, "confidence": 0.9}),
    ("was Rahul on leave last week", {"intent": "leave_emp_check", "entities": {"employee": "Rahul"}, "confidence": 0.85}),
    ("who made the most client visits", {"intent": "visit_ranking", "entities": {}, "confidence": 0.85}),
    ("what's rudhi's score?", {"intent": "emp_pace_score", "entities": {"employee": "rudhi"}, "confidence": 0.85}),
    ("who is the most disciplined employee", {"intent": "most_disciplined", "entities": {}, "confidence": 0.85}),
    ("give me a rundown of absenteeism", {"intent": "attendance_worst", "entities": {}, "confidence": 0.55}),
    ("who's crushing it on pace score this month", {"intent": "pace_score_best", "entities": {"month": "this month"}, "confidence": 0.7}),
    ("is anyone racking up overtime hours", {"intent": "ot_ranking", "entities": {}, "confidence": 0.75}),
    ("month on month pace score trend for Rudhi", {"intent": "full_trend_emp", "entities": {"employee": "Rudhi"}, "confidence": 0.9}),
    ("how many times did Aman Kawadia take wfh month wise", {"intent": "full_trend_emp", "entities": {"employee": "Aman Kawadia", "metric": "wfh"}, "confidence": 0.9}),
    ("visit days for Rahul month by month", {"intent": "full_trend_emp", "entities": {"employee": "Rahul", "metric": "visit"}, "confidence": 0.88}),
    ("leave days month wise for Priya", {"intent": "full_trend_emp", "entities": {"employee": "Priya", "metric": "leave"}, "confidence": 0.88}),
    ("late comings each month for Aryan Gupta", {"intent": "full_trend_emp", "entities": {"employee": "Aryan Gupta", "metric": "late_comings"}, "confidence": 0.85}),
    ("engagement % trend by month for Megha Sharma", {"intent": "full_trend_emp", "entities": {"employee": "Megha Sharma", "metric": "engagement"}, "confidence": 0.88}),
    ("hey what's the weather like", {"intent": "none", "entities": {}, "confidence": 0.95}),
]


def _build_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        logger.exception("Failed to construct Gemini client")
        return None


_client = None
_client_init_attempted = False


def _get_client():
    global _client, _client_init_attempted
    if not _client_init_attempted:
        _client_init_attempted = True
        _client = _build_client()
    return _client


def classify(raw_message, timeout=_TIMEOUT_SECONDS):
    """Calls Gemini to classify `raw_message` into {intent, entities,
    confidence}. Returns None on ANY failure (missing/invalid key, network
    error, timeout, malformed response, hallucinated intent name) so callers
    can transparently fall back to the rule-based matcher alone — the app
    must never break or hang because Gemini is unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        from google.genai import types
    except Exception:
        logger.exception("google-genai SDK not available")
        return None

    contents = [_SYSTEM_PROMPT, "\nExamples:"]
    for text, out in _FEW_SHOT_EXAMPLES:
        contents.append(f'User: "{text}"\nJSON: {json.dumps(out)}')
    contents.append(f'\nNow classify this message.\nUser: "{raw_message}"\nJSON:')
    prompt = "\n".join(contents)

    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_ENTITY_SCHEMA,
                temperature=0.0,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            ),
        )
    except Exception as e:
        logger.warning("Gemini call failed (%s) after %.2fs — falling back to rule-based matcher", e, time.time() - t0)
        return None

    try:
        data = json.loads(resp.text)
    except Exception:
        logger.warning("Gemini returned non-JSON / unparseable response: %r", getattr(resp, "text", None))
        return None

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        # Covers "none", hallucinated names, or missing field — no usable
        # LLM opinion; caller falls back to the rule-based matcher.
        return None

    entities = data.get("entities") or {}
    confidence = data.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    usage = getattr(resp, "usage_metadata", None)
    latency = time.time() - t0
    logger.info(
        "Gemini classify: %.2fs intent=%s confidence=%.2f prompt_tokens=%s total_tokens=%s",
        latency, intent, confidence,
        getattr(usage, "prompt_token_count", None), getattr(usage, "total_token_count", None),
    )

    return {
        "intent": intent,
        "entities": {k: v for k, v in entities.items() if v},
        "confidence": confidence,
        "_latency": latency,
        "_usage": usage,
    }
