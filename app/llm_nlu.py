"""LLM-first natural-language-understanding layer, sitting IN FRONT OF the
existing rule-based intent matcher (intents.py) and query pipeline
(queries.py) — added as an explicitly-approved, deliberate reversal of this
project's original "no LLM" design constraint.

PROVIDER ABSTRACTION (added in the GPT-5-mini migration round):
  - The active provider is a pure runtime config choice, controlled by the
    env var LLM_PROVIDER ("openai" or "gemini"). Default: "openai".
  - Both providers implement the exact same contract: given a raw user
    message (+ optional session context), return
    {intent, entities, confidence, _latency, _usage} or None on any failure.
  - To roll back to Gemini instantly with NO code change: set
    LLM_PROVIDER=gemini in the environment (Render dashboard env var, or
    local shell) and restart the process. The Gemini integration code below
    is kept fully intact for exactly this reason — do not delete or degrade
    it while adding/maintaining the OpenAI path.
  - Model id for the OpenAI path: "gpt-5-mini" (resolves server-side to a
    dated snapshot, e.g. gpt-5-mini-2025-08-07 — confirmed via a live test
    call against the Responses API during this round, not guessed from
    memory).

HARD SAFETY CONTRACT (do not violate when editing this file):
  - The LLM's job ends at producing {intent, entities, confidence}. It NEVER
    sees the database, NEVER writes SQL, and NEVER produces a final numeric
    answer. Every number the user sees still comes from queries.py running
    against public.pace_chatbot_view / public.pace_1, exactly as before this
    layer was added. (The separate SQL-generation FALLBACK path added
    alongside this file, in sql_fallback.py, is a distinct, explicitly-
    authorized code path with its own safety checks — see that module.)
  - `intent` returned by either provider is validated against the real
    intent-name set derived from intents._INTENTS (not hand-duplicated)
    before it is trusted at all. Anything else (typo, hallucinated intent,
    malformed JSON) is treated as "no LLM opinion" and the caller falls back
    to the pre-existing rule-based matcher.
  - Any LLM call failure (bad/missing key, network error, timeout,
    malformed response) must be caught here and turned into `None` — callers
    must never crash or block on an LLM outage; the whole app must keep
    working via the rule-based matcher alone.

PROMPT CACHING (OpenAI path only — see Part 2 of the migration task):
  OpenAI's automatic prompt caching keys off the request's token PREFIX
  being byte-identical across calls. For the Responses API, the top-level
  `instructions` field is placed ahead of `input` internally by the API, so
  putting 100% of the STABLE content (system prompt + few-shot examples,
  identical on every call) in `instructions`, and 100% of the VARIABLE
  content (the user's own message, different every call) in `input`,
  guarantees the stable/cacheable part comes first. This ordering is
  verified by inspecting `resp.usage.input_tokens_details.cached_tokens` on
  consecutive calls (see sql_fallback.py / DEPLOY.md notes for the same
  pattern applied to the SQL-fallback prompt, which additionally prepends
  the condensed schema reference doc — also stable — inside `instructions`).
"""

import json
import logging
import os
import time

from . import intents
from .usage_log import log_usage

logger = logging.getLogger("pace_chatbot.llm_nlu")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

GEMINI_MODEL = "gemini-3.5-flash-lite"
OPENAI_MODEL = "gpt-5-mini"
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

# Strict-mode-compatible variant (OpenAI's json_schema strict mode requires
# every property listed in "required" and forbids the Gemini-style
# "nullable" keyword — nullable fields use a ["string","null"] type union
# instead, and additionalProperties:false must be set at every object level).
_OPENAI_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": _VALID_INTENTS + ["none"]},
        "entities": {
            "type": "object",
            "properties": {
                "employee": {"type": ["string", "null"]},
                "department": {"type": ["string", "null"]},
                "manager": {"type": ["string", "null"]},
                "month": {"type": ["string", "null"]},
                "metric": {"type": ["string", "null"]},
            },
            "required": ["employee", "department", "manager", "month", "metric"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number"},
    },
    "required": ["intent", "entities", "confidence"],
    "additionalProperties": False,
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
    ("how many employees have ps not installed yesterday", {"intent": "day_count", "entities": {"metric": "ps_not_installed"}, "confidence": 0.85}),
    ("who has ps not installed today", {"intent": "day_list", "entities": {"metric": "ps_not_installed"}, "confidence": 0.85}),
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
    ("top 10 gainer and loser last 4 weeks", {"intent": "gainer_loser_ranking", "entities": {}, "confidence": 0.9}),
    ("who improved or declined the most in the last 4 weeks", {"intent": "gainer_loser_ranking", "entities": {}, "confidence": 0.9}),
    ("top 10 gainers on wfh only in last 4 weeks", {"intent": "gainer_loser_ranking", "entities": {"metric": "wfh"}, "confidence": 0.85}),
    # --- Round: closing the few-shot coverage gap (Part 2, this round) ---
    # Attendance
    ("who is chronically late", {"intent": "chronic_late", "entities": {}, "confidence": 0.85}),
    ("who has perfect attendance", {"intent": "perfect_attendance", "entities": {}, "confidence": 0.9}),
    ("top defaulters this month", {"intent": "defaulter_ranking", "entities": {"month": "this month"}, "confidence": 0.85}),
    ("who has the most deficit hours", {"intent": "deficit_hours_ranking", "entities": {}, "confidence": 0.85}),
    ("attendance summary for Rahul Kanwaria", {"intent": "emp_attendance_summary", "entities": {"employee": "Rahul Kanwaria"}, "confidence": 0.85}),
    ("how many late comings does Priya have", {"intent": "emp_late_comings", "entities": {"employee": "Priya"}, "confidence": 0.85}),
    ("early leavings for Aman this month", {"intent": "emp_early_leavings", "entities": {"employee": "Aman", "month": "this month"}, "confidence": 0.85}),
    ("who has the most late comings", {"intent": "most_late_comings", "entities": {}, "confidence": 0.85}),
    ("who has the fewest late comings", {"intent": "fewest_late_comings", "entities": {}, "confidence": 0.8}),
    ("who leaves early the most", {"intent": "most_early_leavings", "entities": {}, "confidence": 0.85}),
    ("who has the most deficient hours", {"intent": "most_deficient_hours", "entities": {}, "confidence": 0.85}),
    # PACE score / sub-scores
    ("worst pace score this month", {"intent": "pace_score_worst", "entities": {"month": "this month"}, "confidence": 0.9}),
    ("highest engagement in Accounts", {"intent": "engagement_high", "entities": {"department": "Accounts"}, "confidence": 0.85}),
    ("lowest engagement this month", {"intent": "engagement_low", "entities": {"month": "this month"}, "confidence": 0.85}),
    ("who is the most effective employee", {"intent": "effectiveness_high", "entities": {}, "confidence": 0.85}),
    ("lowest effectiveness in IT-Development", {"intent": "effectiveness_low", "entities": {"department": "IT-Development"}, "confidence": 0.85}),
    ("who is least disciplined", {"intent": "least_disciplined", "entities": {}, "confidence": 0.85}),
    ("discipline percentage for Megha Sharma", {"intent": "emp_discipline", "entities": {"employee": "Megha Sharma"}, "confidence": 0.9}),
    ("how engaged is Rudhi", {"intent": "emp_engagement", "entities": {"employee": "Rudhi"}, "confidence": 0.85}),
    ("how effective is Aryan Gupta", {"intent": "emp_effectiveness", "entities": {"employee": "Aryan Gupta"}, "confidence": 0.85}),
    ("deficient hours for Vaibhav Bansal", {"intent": "emp_deficient_hours", "entities": {"employee": "Vaibhav Bansal"}, "confidence": 0.85}),
    ("working hours percentage for Priya", {"intent": "emp_working_pct", "entities": {"employee": "Priya"}, "confidence": 0.85}),
    ("who has the lowest working hours percentage", {"intent": "lowest_working_pct", "entities": {}, "confidence": 0.8}),
    ("who has the highest working hours percentage", {"intent": "highest_working_pct", "entities": {}, "confidence": 0.8}),
    ("who uses whatsapp the most", {"intent": "most_whatsapp", "entities": {}, "confidence": 0.85}),
    # Status categories
    ("which employees moved from red to black", {"intent": "status_transitions", "entities": {}, "confidence": 0.85}),
    ("which department has the most red employees", {"intent": "status_distribution", "entities": {}, "confidence": 0.8}),
    ("how many employees are black right now", {"intent": "status_count", "entities": {}, "confidence": 0.85}),
    ("what status is Rahul currently in", {"intent": "status_emp", "entities": {"employee": "Rahul"}, "confidence": 0.85}),
    ("which black status employees are improving", {"intent": "status_improving", "entities": {}, "confidence": 0.8}),
    # Comparisons
    ("compare Aryan Gupta and Megha Sharma", {"intent": "employee_compare", "entities": {}, "confidence": 0.85}),
    ("is my team better than Accounts", {"intent": "team_compare", "entities": {}, "confidence": 0.85}),
    ("which day was better 2 sept or 7 sept", {"intent": "day_compare", "entities": {}, "confidence": 0.8}),
    # Department-level
    ("which department is improving", {"intent": "dept_trend", "entities": {}, "confidence": 0.85}),
    ("is my team improving", {"intent": "team_improving", "entities": {}, "confidence": 0.85}),
    ("average pace score for Accounts", {"intent": "dept_avg", "entities": {"department": "Accounts"}, "confidence": 0.85}),
    ("best department this month", {"intent": "dept_best", "entities": {"month": "this month"}, "confidence": 0.85}),
    ("worst department this month", {"intent": "dept_worst", "entities": {"month": "this month"}, "confidence": 0.85}),
    ("how many employees are in SCM department", {"intent": "dept_count", "entities": {"department": "SCM"}, "confidence": 0.85}),
    ("give me a summary of the Accounts department", {"intent": "dept_summary", "entities": {"department": "Accounts"}, "confidence": 0.8}),
    ("lowest scorers in my team", {"intent": "team_lowest_scorers", "entities": {}, "confidence": 0.85}),
    # Leave / calls / visits / WFH
    ("who is on leave today", {"intent": "leave_who", "entities": {}, "confidence": 0.85}),
    ("who took the most half days", {"intent": "half_day_ranking", "entities": {}, "confidence": 0.8}),
    ("leave days by department", {"intent": "leave_by_dept", "entities": {}, "confidence": 0.8}),
    ("who has zero leave this month", {"intent": "zero_leave", "entities": {"month": "this month"}, "confidence": 0.8}),
    ("who made the most calls", {"intent": "call_most", "entities": {}, "confidence": 0.85}),
    ("who made the fewest calls", {"intent": "call_fewest", "entities": {}, "confidence": 0.85}),
    ("call duration for Priya", {"intent": "call_duration", "entities": {"employee": "Priya"}, "confidence": 0.8}),
    ("how many calls did Rahul make", {"intent": "call_emp", "entities": {"employee": "Rahul"}, "confidence": 0.85}),
    ("who made zero client visits", {"intent": "zero_visit", "entities": {}, "confidence": 0.8}),
    ("did Aman go on any visits this month", {"intent": "visit_emp", "entities": {"employee": "Aman", "month": "this month"}, "confidence": 0.8}),
    ("did Priya take wfh yesterday", {"intent": "wfh_emp", "entities": {"employee": "Priya"}, "confidence": 0.8}),
    ("wfh days by department", {"intent": "wfh_by_dept", "entities": {}, "confidence": 0.8}),
    ("who has the fewest wfh days", {"intent": "fewest_wfh", "entities": {}, "confidence": 0.8}),
    # Tasks / todos / meetings / d_score
    ("who created the most todos", {"intent": "todos_created_ranking", "entities": {}, "confidence": 0.8}),
    ("who was assigned the most todos", {"intent": "todos_assigned_ranking", "entities": {}, "confidence": 0.8}),
    ("who created the most tasks", {"intent": "tasks_created_ranking", "entities": {}, "confidence": 0.75}),
    ("who was assigned the most tasks", {"intent": "tasks_assigned_ranking", "entities": {}, "confidence": 0.75}),
    ("highest ontime completion rate", {"intent": "ontime_completion_ranking", "entities": {}, "confidence": 0.7}),
    ("who has the best responsiveness score", {"intent": "responsiveness_ranking", "entities": {}, "confidence": 0.7}),
    ("who has the best extension adherence score", {"intent": "extension_adherence_ranking", "entities": {}, "confidence": 0.7}),
    ("who has the most meetings", {"intent": "meeting_count_ranking", "entities": {}, "confidence": 0.8}),
    ("meeting ratio for Aryan Gupta", {"intent": "meeting_ratio_emp", "entities": {"employee": "Aryan Gupta"}, "confidence": 0.75}),
    ("d score trend for Rudhi", {"intent": "d_score_trend", "entities": {"employee": "Rudhi"}, "confidence": 0.7}),
    ("highest d score ranking", {"intent": "d_score_ranking", "entities": {}, "confidence": 0.7}),
    ("d score for Priya", {"intent": "d_score_emp", "entities": {"employee": "Priya"}, "confidence": 0.7}),
    # Roster / OT / device
    ("what shift type is Rahul on", {"intent": "shift_type_emp", "entities": {"employee": "Rahul"}, "confidence": 0.75}),
    ("is Priya on a break shift", {"intent": "breakshift_emp", "entities": {"employee": "Priya"}, "confidence": 0.7}),
    ("who is offline the most", {"intent": "offline_ranking", "entities": {}, "confidence": 0.75}),
    ("is Aman offline today", {"intent": "offline_emp", "entities": {"employee": "Aman"}, "confidence": 0.75}),
    ("ps install rate by department", {"intent": "ps_install_rate", "entities": {}, "confidence": 0.8}),
    # Org info
    ("what grade is Megha Sharma", {"intent": "grade_lookup", "entities": {"employee": "Megha Sharma"}, "confidence": 0.8}),
    ("designation breakdown for Accounts", {"intent": "designation_breakdown", "entities": {"department": "Accounts"}, "confidence": 0.75}),
    ("average tenure in SCM", {"intent": "avg_tenure", "entities": {"department": "SCM"}, "confidence": 0.75}),
    # Misc / overview / PS-worked
    ("give me an overview of Rahul Kanwaria", {"intent": "emp_overview", "entities": {"employee": "Rahul Kanwaria"}, "confidence": 0.75}),
    ("what is Priya's ps worked ratio", {"intent": "ps_worked_emp", "entities": {"employee": "Priya"}, "confidence": 0.7}),
    ("ranking by ps worked ratio", {"intent": "ps_worked_ranking", "entities": {}, "confidence": 0.7}),
    ("subscore trend for Aryan Gupta", {"intent": "subscore_trend_emp", "entities": {"employee": "Aryan Gupta"}, "confidence": 0.7}),
    ("Rahul's OT subscore breakdown", {"intent": "ot_subscore", "entities": {"employee": "Rahul"}, "confidence": 0.7}),
    ("Priya's wfh subscore breakdown", {"intent": "wfh_subscore", "entities": {"employee": "Priya"}, "confidence": 0.7}),
    ("month wise pace score trend for the whole team", {"intent": "full_trend_team", "entities": {}, "confidence": 0.7}),
    ("month wise pace score trend for Accounts department", {"intent": "full_trend_dept", "entities": {"department": "Accounts"}, "confidence": 0.75}),
    ("who is slacking off", {"intent": "productive_low", "entities": {}, "confidence": 0.7}),
    ("who is working hard", {"intent": "productive_high", "entities": {}, "confidence": 0.7}),
    ("best attendance this month", {"intent": "attendance_best", "entities": {"month": "this month"}, "confidence": 0.85}),
    ("who is declining the most", {"intent": "declining", "entities": {}, "confidence": 0.85}),
    ("who is improving the most", {"intent": "improving", "entities": {}, "confidence": 0.85}),
    ("ai usage minutes for Rahul", {"intent": "emp_ai_usage", "entities": {"employee": "Rahul"}, "confidence": 0.8}),
    ("what department is Priya in", {"intent": "emp_department", "entities": {"employee": "Priya"}, "confidence": 0.85}),
    ("who is Aman's manager", {"intent": "emp_manager", "entities": {"employee": "Aman"}, "confidence": 0.85}),
    ("productive minutes for Megha Sharma", {"intent": "emp_productive_time", "entities": {"employee": "Megha Sharma"}, "confidence": 0.8}),
    ("is Rudhi improving or declining over the last 2 months", {"intent": "emp_trend_2month", "entities": {"employee": "Rudhi"}, "confidence": 0.75}),
    ("whatsapp minutes for Aryan Gupta", {"intent": "emp_whatsapp", "entities": {"employee": "Aryan Gupta"}, "confidence": 0.8}),
    ("meeting minutes ranking", {"intent": "meeting_min_ranking", "entities": {}, "confidence": 0.75}),
    ("how many days is Rahul excluded for ps not installed", {"intent": "ps_exclude_metric", "entities": {"employee": "Rahul"}, "confidence": 0.75}),
    ("what is the ps working ratio", {"intent": "ps_ratio_info", "entities": {}, "confidence": 0.7}),
    ("what does ps not installed mean", {"intent": "ps_explain", "entities": {}, "confidence": 0.75}),
]

# ---------------------------------------------------------------------------
# Shared: stable instructions text (system prompt + few-shot), built ONCE at
# import time, reused byte-identical on every call for BOTH providers. For
# the OpenAI path this is what goes in `instructions` (ahead of the variable
# `input`) so automatic prompt caching can key off it.
# ---------------------------------------------------------------------------
_FEWSHOT_LINES = []
for _text, _out in _FEW_SHOT_EXAMPLES:
    _FEWSHOT_LINES.append(f'User: "{_text}"\nJSON: {json.dumps(_out)}')
_STABLE_INSTRUCTIONS = _SYSTEM_PROMPT + "\n\nExamples:\n" + "\n".join(_FEWSHOT_LINES)


# ---------------------------------------------------------------------------
# Gemini path (kept fully intact from the pre-migration version)
# ---------------------------------------------------------------------------

def _build_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        logger.exception("Failed to construct Gemini client")
        return None


_gemini_client = None
_gemini_client_init_attempted = False


def _get_gemini_client():
    global _gemini_client, _gemini_client_init_attempted
    if not _gemini_client_init_attempted:
        _gemini_client_init_attempted = True
        _gemini_client = _build_gemini_client()
    return _gemini_client


def _classify_gemini(raw_message, timeout):
    client = _get_gemini_client()
    if client is None:
        return None

    try:
        from google.genai import types
    except Exception:
        logger.exception("google-genai SDK not available")
        return None

    prompt = _STABLE_INSTRUCTIONS + f'\n\nNow classify this message.\nUser: "{raw_message}"\nJSON:'

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
        return None

    entities = data.get("entities") or {}
    confidence = data.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    usage = getattr(resp, "usage_metadata", None)
    latency = time.time() - t0
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    logger.info(
        "Gemini classify: %.2fs intent=%s confidence=%.2f prompt_tokens=%s total_tokens=%s",
        latency, intent, confidence,
        input_tokens, getattr(usage, "total_token_count", None),
    )
    log_usage(
        path="classify", provider="gemini", model=GEMINI_MODEL,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cached_tokens=None, latency=latency,
    )

    return {
        "intent": intent,
        "entities": {k: v for k, v in entities.items() if v},
        "confidence": confidence,
        "_latency": latency,
        "_usage": usage,
    }


# ---------------------------------------------------------------------------
# OpenAI (GPT-5 mini) path
# ---------------------------------------------------------------------------

_openai_client = None
_openai_client_init_attempted = False


def _get_openai_client():
    global _openai_client, _openai_client_init_attempted
    if not _openai_client_init_attempted:
        _openai_client_init_attempted = True
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            _openai_client = None
        else:
            try:
                import openai
                _openai_client = openai.OpenAI(api_key=api_key)
            except Exception:
                logger.exception("Failed to construct OpenAI client")
                _openai_client = None
    return _openai_client


def _classify_openai(raw_message, timeout):
    client = _get_openai_client()
    if client is None:
        return None

    t0 = time.time()
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            # STABLE content first (system prompt + few-shot) so OpenAI's
            # automatic prompt-caching prefix-match can hit on repeat calls.
            instructions=_STABLE_INSTRUCTIONS,
            # VARIABLE content last (this call's user message only).
            input=f'Now classify this message.\nUser: "{raw_message}"\nJSON:',
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pace_intent_classification",
                    "schema": _OPENAI_ENTITY_SCHEMA,
                    "strict": True,
                }
            },
            reasoning={"effort": "minimal"},
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("OpenAI call failed (%s) after %.2fs — falling back to rule-based matcher", e, time.time() - t0)
        return None

    try:
        data = json.loads(resp.output_text)
    except Exception:
        logger.warning("OpenAI returned non-JSON / unparseable response: %r", getattr(resp, "output_text", None))
        return None

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        return None

    entities = data.get("entities") or {}
    confidence = data.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    usage = getattr(resp, "usage", None)
    latency = time.time() - t0
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    itd = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(itd, "cached_tokens", None) if itd is not None else None
    logger.info(
        "OpenAI classify: %.2fs intent=%s confidence=%.2f input_tokens=%s output_tokens=%s cached_tokens=%s",
        latency, intent, confidence, input_tokens, output_tokens, cached_tokens,
    )
    log_usage(
        path="classify", provider="openai", model=getattr(resp, "model", OPENAI_MODEL),
        input_tokens=input_tokens, output_tokens=output_tokens,
        cached_tokens=cached_tokens, latency=latency,
    )

    return {
        "intent": intent,
        "entities": {k: v for k, v in entities.items() if v},
        "confidence": confidence,
        "_latency": latency,
        "_usage": usage,
    }


# ---------------------------------------------------------------------------
# Public entry point — dispatches on LLM_PROVIDER
# ---------------------------------------------------------------------------

def classify(raw_message, timeout=_TIMEOUT_SECONDS):
    """Calls the active LLM provider (LLM_PROVIDER env var: "openai" default,
    or "gemini") to classify `raw_message` into {intent, entities,
    confidence}. Returns None on ANY failure (missing/invalid key, network
    error, timeout, malformed response, hallucinated intent name) so callers
    can transparently fall back to the rule-based matcher alone — the app
    must never break or hang because the LLM is unavailable.
    """
    if LLM_PROVIDER == "gemini":
        return _classify_gemini(raw_message, timeout)
    return _classify_openai(raw_message, timeout)
