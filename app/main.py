import datetime
import logging
import re

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import entities, queries, intents, team, session_store, spellcheck, llm_nlu, sql_fallback

app = FastAPI(title="Pace Chatbot (Phase 1)")

logger = logging.getLogger("pace_chatbot")


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Safety net so a bug anywhere in the query/formatting pipeline never
    surfaces to the browser as FastAPI's default plain-text "Internal Server
    Error" body — that response isn't valid JSON, and chat-widget.js's
    `await resp.json()` throws a raw SyntaxError on it ("Unexpected token
    'I', 'Internal S'... is not valid JSON"), which is confusing and hides
    the real problem. The full traceback is still logged server-side (via
    `logger.exception`, same as uvicorn would print) for debugging - this
    handler only changes what's sent back over the wire, it does not swallow
    or hide the error from developers."""
    logger.exception("Unhandled exception while handling %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=200,
        content={
            "reply": "Sorry, something went wrong answering that — please try rephrasing your question, "
                     "or ask something else.",
            "rows": [],
            "needs_clarification": False,
            "clarification_options": [],
        },
    )

_CONFIRM_PATTERN = re.compile(r"\b(yes|confirm|full company|entire company|whole company|all employees)\b", re.I)

# Matches a same-session affirmative reply to the "want this broken down by
# week instead?" offer that follows an individual monthly PACE-trend answer
# (see session["awaiting_weekly_breakdown"] in handle_message below).
_WEEKLY_FOLLOWUP_PATTERN = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok(ay)?|please do|show (me )?(the )?weekly|weekly( breakdown)?|"
    r"week by week|by week|break(\s*it)?\s*down\s*by\s*week)\b",
    re.I,
)

# --- Bug 1 fix: vague "list the thing I was just shown" follow-up resolution ---
# A vague follow-up ("show me their names", "list them", "who are they",
# "give me the list") needs to expand whatever the MOST RECENT list/count/
# ranking-producing answer was about - tracked separately in
# session["last_list"] (see session_store.set_last_list) - rather than
# falling back to the unrelated whole-session sticky department/employee/
# time-period context (which exists to fill in MISSING scope on a NEW
# question, not to resolve "expand the thing I was just shown"). Explicitly
# asks for a list/names/expansion - NOT just any short "who/what" question,
# so it doesn't swallow genuinely fresh queries that happen to be short.
_VAGUE_LIST_EXPAND_STRICT = re.compile(
    r"("
    r"show (me )?(their |the )?names\b"
    r"|list (them|everyone|their names|the names)\b"
    r"|who (are|were|was) (they|them)\b"
    r"|can (you )?share (their |the )?names\b"
    r"|share (their |the )?names\b"
    r"|share names list\b"
    r"|their names list\b"
    r"|names please\b"
    r"|^names\??\s*$"
    r"|give me (the )?(full |whole )?list\b(?!\s+of\s+\w)"
    r"|give me the rest of the list\b"
    r"|^full list\b"
    r"|full list please\b"
    r"|show more\b"
    r"|who else\b"
    r"|expand this\b"
    r"|show \d+ instead\b"
    r"|not just \d+\b"
    r"|who'?s next after them\b"
    r"|who is next after them\b"
    r"|list everyone\b"
    r"|the (black|red|amber|green) ones instead\b"
    r")",
    re.IGNORECASE,
)

# A looser set that OVERLAPS with phrasing a genuinely fresh, standalone
# query could also use (e.g. "who was offline" / "show me who" / "who took
# leave" name an actual flag/category, so they're plausible first messages
# in a session, not just follow-ups). These are only ever treated as a
# vague-list follow-up when a prior last_list actually exists (see
# handle_message) - when there's no prior list, they fall through untouched
# to the normal intent pipeline instead of being force-clarified, so a
# fresh "who was offline yesterday" still works exactly as before.
_VAGUE_LIST_EXPAND_LOOSE = re.compile(
    r"("
    r"who (are|were) the \w+"
    r"|show me who\b"
    r"|who was offline\b"
    r"|who took leave\b"
    r"|how many total\b"
    r"|employee[- ]wise\b"
    r"|by employee\b"
    r"|employees? list\b"
    r")",
    re.IGNORECASE,
)

_VAGUE_LIST_EXPAND_PATTERN = re.compile(
    r"(" + _VAGUE_LIST_EXPAND_STRICT.pattern + "|" + _VAGUE_LIST_EXPAND_LOOSE.pattern + r")",
    re.IGNORECASE,
)

# --- Bare superlative direction follow-up ("least", "most", "highest",
# "lowest", ...) right after a ranking (item #63). Matched ONLY when the
# ENTIRE message (after stripping whitespace/punctuation) is just one of
# these words - a real sentence that happens to contain "least" (e.g.
# "least productive employees") is untouched, since this only fires via
# handle_message's exact-match check below, not a bare \b...\b search.
# Root cause this exists to fix: a bare direction word matches NOTHING in
# the rule-based regex patterns (they all require an accompanying metric
# word), so it fell through to intents._fuzzy_match_intent(), where
# rapidfuzz's token_set_ratio scores a single contained word as a perfect
# 100 match against EVERY multi-word canonical phrase that contains it
# (confirmed: "least" scores 100 against "least productive time",
# "least engaged", "least wfh days", etc. simultaneously) - Python's
# max(scores, key=scores.get) then deterministically picks whichever
# intent happens to be inserted first in _CANONICAL_PHRASES
# ("productive_low"/"productive_high"), regardless of what the prior
# ranking's real metric/dimension actually was. This interception runs
# BEFORE that fuzzy fallback can ever be reached, so it can't misfire this
# way, and mirrors the ACTUAL prior ranking's metric/dept/employee scope
# via session_store's rerun_opposite/rerun_same/ascending fields instead.
_BARE_DIRECTION_LOW = re.compile(r"^(least|lowest|worst|fewest|bottom|smallest)$", re.IGNORECASE)
_BARE_DIRECTION_HIGH = re.compile(r"^(most|highest|best|top|largest|greatest)$", re.IGNORECASE)


def _handle_bare_direction_followup(message, session):
    """See the comment above _BARE_DIRECTION_LOW/_BARE_DIRECTION_HIGH.
    Returns a ChatResponse if this message is a bare direction-word
    follow-up (handled here, one way or another - a correct re-ranked
    result, a repeat of the same ranking, or a clean clarification), or
    None if this message isn't shaped like one at all (falls through to
    normal routing untouched)."""
    stripped = message.strip().strip("?!.").strip()
    is_low = _BARE_DIRECTION_LOW.match(stripped) is not None
    is_high = (not is_low) and _BARE_DIRECTION_HIGH.match(stripped) is not None
    if not is_low and not is_high:
        return None
    if session is None:
        return ChatResponse(
            reply="I don't have a prior ranking to flip the direction on — could you ask a ranking question "
                  "first (e.g. \"top 10 by pace score\"), then say \"least\"/\"most\"?",
            needs_clarification=True,
        )
    last_list = session_store.get_last_list(session)
    if last_list is None or last_list.get("kind") != "ranking":
        return ChatResponse(
            reply="I don't have a prior ranking to flip the direction on — could you ask a ranking question "
                  "first (e.g. \"top 10 by pace score\"), then say \"least\"/\"most\"?",
            needs_clarification=True,
        )
    requested_ascending = is_low  # low direction == ascending sort (smallest first)
    last_ascending = last_list.get("ascending")
    rerun_same = last_list.get("rerun_same") or last_list.get("rerun_list")
    rerun_opposite = last_list.get("rerun_opposite")
    if last_ascending is not None and requested_ascending == last_ascending:
        rerun = rerun_same
    elif last_ascending is not None:
        rerun = rerun_opposite
        if rerun is None:
            return ChatResponse(
                reply="I can show that ranking again, but I don't have a way to flip its direction for this "
                      "metric yet — could you ask a fresh ranking question instead (e.g. \"lowest engagement\")?",
                needs_clarification=True,
            )
    else:
        # The prior ranking's direction wasn't tracked (an older/未-wired
        # ranking type) - flipping blind would risk silently repeating the
        # wrong direction, so ask rather than guess.
        return ChatResponse(
            reply="I can show that ranking again, but I'm not sure which direction it was in to flip it — "
                  "could you ask a fresh ranking question instead (e.g. \"lowest engagement\")?",
            needs_clarification=True,
        )
    reply, rows = rerun()
    # Whichever closure we just displayed becomes the new "same direction"
    # (so a repeated bare word just re-shows it), and whichever we DIDN'T
    # use becomes the new "opposite direction" (so flipping back and forth
    # keeps working correctly across multiple bare-direction turns in a
    # row, not just the first flip).
    if rerun is rerun_opposite:
        new_rerun_same, new_rerun_opposite = rerun_opposite, rerun_same
    else:
        new_rerun_same, new_rerun_opposite = rerun_same, rerun_opposite
    session_store.set_last_list(
        session, kind="ranking", rerun_list=new_rerun_same, rerun_same=new_rerun_same,
        rerun_opposite=new_rerun_opposite,
        answer_kind="list", ascending=requested_ascending,
        dept_name=last_list.get("dept_name"), employee_ids=last_list.get("employee_ids"),
        team_label=last_list.get("team_label"), month=last_list.get("month"), date_range=last_list.get("date_range"),
    )
    return ChatResponse(reply=reply, rows=rows)

# --- List-population pronoun follow-up ("their pace score" / "there pace
# score" / "what about them") (item #67) ---
# Extends the item #61/#62 sticky-context meta-followup mechanism to a
# related but distinct case: "there"/"their"/"them" (or a plausible typo)
# referring back to a just-shown LIST answer - a department/RM-team roster
# (roster_list, item #66), a ranking, a gainer/loser list, a filtered
# subset - asking for a DIFFERENT metric for that SAME population, e.g.
# "give me complete list of ai labs employees" -> "their pace score". This
# is deliberately NOT the same mechanism as _PRONOUN_PATTERN/
# _extract_employee_ctx above, which only ever inherits a single employee_id
# pushed via push_context (set for an individual lookup, never for a
# department/RM-team list answer) - there's no collision risk between the
# two: this one only fires when session_store.last_list actually carries a
# dept_name/team_label (a population scope), the other only when a single
# employee_id was pushed to recent context.
_LIST_POPULATION_PRONOUN_PATTERN = re.compile(
    r"\b(their|thier|thear|theyre|there|them|thm)\b", re.IGNORECASE
)


def _handle_list_pronoun_metric_followup(message, session):
    """See _LIST_POPULATION_PRONOUN_PATTERN above. Checked BEFORE intent
    classification, same slot as _handle_filter_meta_followup/
    _handle_bare_direction_followup, so a plural population reference can
    never be misrouted to an unrelated fresh single-employee or
    company-wide query. Reuses build_query_overview_reply() (item #64/#66's
    engine) - no parallel query mechanism is built here.

    Returns None (falls through to normal routing) when: no pronoun match,
    no metric keyword named (too ambiguous to guess at, same non-guessing
    discipline as item #61's topic gate), no prior list, or the prior list's
    scope carries neither a dept_name nor a team_label (e.g. a bare
    company-wide roster, or a single-employee context, or an untracked
    answer shape) - a known, narrow scope limit, not a silent guess.
    """
    if session is None:
        return None
    text_l = (message or "").lower().strip()
    if not _LIST_POPULATION_PRONOUN_PATTERN.search(text_l):
        return None
    if not any(re.search(pat, text_l) for _, pat in _BUILD_QUERY_METRIC_PATTERNS):
        return None
    last_list = session_store.get_last_list(session)
    if last_list is None:
        return None
    dept_name = last_list.get("dept_name")
    team_label = last_list.get("team_label")
    if dept_name:
        dimension, name = "department", dept_name
    elif team_label:
        # team_label is stored as "<Manager Name>'s team" (see
        # build_query_overview_reply/answer_intent) - strip the suffix back
        # to the bare manager name build_query_overview_reply("rm", ...)
        # expects.
        dimension, name = "rm", re.sub(r"'s team$", "", team_label)
    else:
        return None
    # Prior list's own period (only ever populated for build_query-based
    # answers - see build_query_overview_reply's set_last_list call). Ranking-
    # kind answers store a bare "month" string instead, which isn't the
    # (start, end) shape build_query()'s period param expects, so it's
    # deliberately not threaded through here - the new metric query falls
    # back to build_query()'s own last-60-days default (item #65) in that
    # case, a reasonable secondary approximation given the population/scope
    # (the primary correctness concern here) is still exactly preserved.
    period = last_list.get("date_range")
    # Force the per-employee list shape (not a single aggregate row) - the
    # user just saw a roster of NAMES, so "their pace score" reads as "show
    # me each of their pace scores," matching build_query_overview_reply's
    # own wants_list branch (dimension != "employee" + a "list" cue).
    synthetic_message = f"{message} list"
    reply, rows = build_query_overview_reply(dimension, name, synthetic_message, period=period, session=session)
    return ChatResponse(reply=reply, rows=rows)

# A narrower "re-scope only" follow-up ("what about last month", "what about
# next week") - no explicit ask for names/a list, just a change of time
# period/department applied to the SAME prior answer, kept in its ORIGINAL
# answer shape (e.g. still a bare count) rather than force-expanded to a
# list. Deliberately gated to short messages starting with this phrase, to
# minimize collision with unrelated fresh queries.
_VAGUE_RESCOPE_PATTERN = re.compile(r"^\s*what about\b", re.IGNORECASE)

# --- Filter/methodology meta-follow-up ("in this have u removed ps not
# working days?") ---
# Detects a message that (a) refers back to "this"/"that"/"it" AND (b) asks
# a filter/methodology question, scoped ONLY to the 4 known population
# filters (PS status, visit status, shift type, work mode) - see
# session_store.set_last_answer_filters/get_last_answer_filters and
# _handle_filter_meta_followup below. Deliberately loose on grammar (the
# real repro was "in this have u removed ps not working days?") but requires
# both an exclude/remove/filter/include VERB and a this/that/it REFERENT
# somewhere in the message, not just either alone - so an unrelated message
# that happens to contain "this" doesn't get swept in.
_FILTER_META_SHAPE_PATTERN = re.compile(
    r"(\b(this|that|it)\b.{0,40}\b(exclud\w*|remov\w*|filter\w*|includ\w*|appl(y|ied)\w*)\b"
    r"|\b(exclud\w*|remov\w*|filter\w*|includ\w*|appl(y|ied)\w*)\b.{0,40}\b(this|that|it)\b)",
    re.IGNORECASE,
)

_FILTER_META_TOPIC_PATTERNS = [
    ("ps_status", re.compile(r"\bps\b|\bpace\s*sync\b|\boffline\s*attendance\b", re.IGNORECASE)),
    ("visit_status", re.compile(r"\bvisit(s|ed|ing)?\b", re.IGNORECASE)),
    ("shift_type", re.compile(r"\bshift(s)?\b|\bstandard\b|\bovertime\b|\bot\s*days?\b", re.IGNORECASE)),
    ("work_mode", re.compile(r"\bwfh\b|work(ing)? from home|\bremote\b|\bwork\s*mode\b|\boffice\b", re.IGNORECASE)),
]

_FILTER_META_TOPIC_LABELS = {
    "ps_status": "PS (offline-attendance-system) working status",
    "visit_status": "client-visit status",
    "shift_type": "shift type",
    "work_mode": "work mode (WFH vs office)",
}


def _detect_filter_meta_topic(text_l):
    """Returns one of 'ps_status'/'visit_status'/'shift_type'/'work_mode' if
    the message clearly names one of the 4 known filters, else None (which
    means: let this fall through to normal routing rather than guessing)."""
    for topic, pat in _FILTER_META_TOPIC_PATTERNS:
        if pat.search(text_l):
            return topic
    return None


def _strip_leading_the(label):
    """subject_label is sometimes already 'the full company' - avoid a
    double 'the the ...' when composing a filter-meta-followup label."""
    return label[4:] if label.lower().startswith("the ") else label


def _default_filters_from_message(message):
    """Same override-detection regexes as _resolve_population_filter, but
    returns a structured dict instead of SQL - used to record what was
    ACTUALLY applied to a filter-driven answer (day/month compare,
    gainer/loser ranking, ...) for later filter-meta follow-ups."""
    text_l = (message or "").lower()
    ot = bool(re.search(_OT_PATTERN, text_l))
    standard = (not ot) and bool(re.search(_STANDARD_PATTERN, text_l))
    visit_no = bool(re.search(_VISIT_NO_PATTERN, text_l))
    visit_yes = (not visit_no) and bool(re.search(_VISIT_YES_PATTERN, text_l))
    ps_not = bool(re.search(_PS_NOT_WORKING_PATTERN, text_l))
    ps_yes = (not ps_not) and bool(re.search(_PS_WORKING_PATTERN, text_l))
    wfh = bool(re.search(r"\bwfh\b|work(ing)? from home|\bremote\b", text_l))
    return {
        "ps_status": "not_working" if ps_not else "working",
        "visit_status": "yes" if visit_yes else "no",
        "shift_type": "ot" if ot else "standard",
        "work_mode": "wfh" if wfh else None,
    }


def _filter_meta_answer_text(topic, ctx):
    """Plain yes/no explanation of whether `topic` was applied to the
    tracked prior answer described by `ctx` (a session_store
    last_answer_filters dict), plus a recompute offer when it wasn't applied
    the way a user asking to exclude something would want."""
    label = ctx["label"]
    applied = ctx.get(topic)
    topic_label = _FILTER_META_TOPIC_LABELS[topic]

    if topic == "ps_status":
        if applied == "working":
            return (f"Yes — {label} excluded PS-not-working days: only rows where PS was "
                     f"working (ps_worked_flag_day = 1) were counted.")
        if applied == "not_working":
            return (f"The other way round, actually — {label} was restricted to PS-NOT-working "
                     f"days only (ps_worked_flag_day = 0); PS-working days were the ones excluded.")
        return (f"No — {label} did not filter on PS status at all; it includes days regardless of "
                f"whether PS was working or not. Want me to recompute it excluding PS-not-working days?")

    if topic == "visit_status":
        if applied == "no":
            return f"Yes — {label} excluded client-visit days (visit_flag = 'No' only)."
        if applied == "yes":
            return f"The other way round — {label} was restricted to visit days only (visit_flag = 'Yes')."
        return (f"No — {label} did not filter on visit status at all; visit and non-visit days are "
                f"both included. Want me to recompute it excluding visit days?")

    if topic == "shift_type":
        if applied == "standard":
            return f"Yes — {label} is restricted to Standard-shift days only."
        if applied == "ot":
            return f"That one was restricted to Overtime (OT) days only, not Standard shift."
        return (f"No — {label} did not restrict by shift type; Standard and Overtime days are both "
                f"included. Want me to recompute it for Standard shift only?")

    if topic == "work_mode":
        if applied == "wfh":
            return f"Yes — {label} was restricted to WFH (work-from-home) days only."
        return (f"No — {label} did not filter by work mode at all (WFH and office days are both "
                f"included, since there's no default work-mode filter in this system). Want me to "
                f"recompute it for one specific work mode?")

    return f"I'm not sure whether {topic_label} was applied to {label} — could you rephrase?"


def _handle_filter_meta_followup(message, session):
    """Intercepts a "did you exclude X in this?" style meta-question about
    the most recent substantive answer, BEFORE it can be misrouted to an
    unrelated fresh-query intent. Returns a ChatResponse if handled, else
    None (meaning: fall through to normal routing)."""
    text_l = (message or "").lower()
    if not _FILTER_META_SHAPE_PATTERN.search(text_l):
        return None
    topic = _detect_filter_meta_topic(text_l)
    if topic is None:
        # Matches the "this/that/it + exclude/filter" shape but doesn't
        # clearly name one of the 4 known filters (e.g. "why did you say
        # that", "what does this mean") - out of scope for this round, let
        # it fall through to normal routing rather than guessing.
        return None
    ctx = session_store.get_last_answer_filters(session) if session is not None else None
    if ctx is None:
        # Nothing tracked yet this session (fresh session, or the last
        # substantive answer wasn't one that tracks filter metadata) -
        # don't fabricate an answer to a meta-question with nothing to
        # point at.
        return None
    return ChatResponse(reply=_filter_meta_answer_text(topic, ctx))

# Explicit scope-BROADENING override ("in the whole company", "overall",
# "company-wide", "in general", "all departments", "everyone") - a follow-up
# that carries no list-expand wording of its own but explicitly signals the
# user wants to drop whatever department/team scope is currently sticky
# (e.g. "AI Labs" locked in from an earlier ranking) and re-run the SAME
# prior answer shape company-wide instead. Precedence: explicit scope
# signal in the current message > sticky scope from recent turns > default.
# Routed through the same rescope-only path as _VAGUE_RESCOPE_PATTERN
# ("what about ...") - keeps the prior answer's shape (still a ranking,
# still a count, etc.) rather than force-expanding to a list.
_SCOPE_OVERRIDE_PATTERN = re.compile(
    r"("
    r"\bwhole company\b"
    r"|\bentire company\b"
    r"|\bfull company\b"
    r"|\bcompany[- ]wide\b"
    r"|\boverall\b"
    r"|\bin general\b"
    r"|\ball departments\b"
    r"|\beveryone\b"
    r"|\bacross the company\b"
    r")",
    re.IGNORECASE,
)


def _scope_note_generic(team_label, dept_name, month, date_range):
    note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")
    note += _period_note(month, date_range)
    return note


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    rows: list = []
    needs_clarification: bool = False
    clarification_options: list = []


def _esc(s):
    """HTML-escape a value for safe inclusion as table cell text. Every cell
    that may contain DB-sourced strings (employee/department names etc.)
    MUST go through this before being placed in HTML, since this app has no
    auth and untrusted names/departments should never be injected raw."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_table(headers, rows):
    """Builds an HTML table from `headers` (list of strings) and `rows`
    (list of lists/tuples of raw cell values — escaping happens here, callers
    must NOT pre-escape). Only called with 2+ rows; single-row/no-row cases
    are handled by callers as plain text so single-value answers stay plain."""
    thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        tds = "".join(f"<td>{_esc(c)}</td>" for c in row)
        body_rows.append(f"<tr>{tds}</tr>")
    return (
        '<table class="pace-table"><thead><tr>' + thead + "</tr></thead><tbody>"
        + "".join(body_rows) + "</tbody></table>"
    )


def format_attendance_rows(rows):
    if not rows:
        return "No matching data found for that filter."
    if len(rows) == 1:
        r = rows[0]
        return (
            f"{r['emp_name']} ({r['dept_name']}) — "
            f"LC:{r['total_lc']} EL:{r['total_el']} DH:{r['total_dh']} "
            f"Defaulter days:{r['total_defaulter']} (total flags: {r['flag_sum']})"
        )
    headers = ["#", "Employee", "Department", "LC", "EL", "DH", "Defaulter days", "Total flags"]
    data = [
        [i, r["emp_name"], r["dept_name"], r["total_lc"], r["total_el"], r["total_dh"], r["total_defaulter"], r["flag_sum"]]
        for i, r in enumerate(rows, 1)
    ]
    return _render_table(headers, data)


def format_productive_rows(rows):
    if not rows:
        return "No matching data found for that filter."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — {r['total_productive_min']} productive minutes"
    headers = ["#", "Employee", "Department", "Productive minutes"]
    data = [[i, r["emp_name"], r["dept_name"], r["total_productive_min"]] for i, r in enumerate(rows, 1)]
    return _render_table(headers, data)


def format_trend_rows(rows, meta):
    note = ""
    if meta["partial_month"]:
        note = (
            "\n\nNote: this month is still in progress, so its average — and therefore this "
            "change — may shift as more days come in."
        )
    if not rows:
        return (
            f"No employees had enough data in both this month and {meta['prev_month']} "
            f"(at least {meta['min_days']} Standard-shift days in each) to show a reliable trend."
        ) + note
    if len(rows) == 1:
        r = rows[0]
        sign = "+" if r["pace_score_delta"] and r["pace_score_delta"] > 0 else ""
        return (
            f"{r['emp_name']} ({r['dept_name']}) — {sign}{_fmt(r['pace_score_delta'])} pts "
            f"(prior month avg {_fmt(r['pace_score_prev_month'])}, based on "
            f"{r['days_current_month']} days this month vs {r['days_prev_month']} days last month)"
        ) + note
    headers = ["#", "Employee", "Department", "Change (pts)", "Prior month avg", "Days this month", "Days last month"]
    data = []
    for i, r in enumerate(rows, 1):
        sign = "+" if r["pace_score_delta"] and r["pace_score_delta"] > 0 else ""
        data.append([i, r["emp_name"], r["dept_name"], f"{sign}{_fmt(r['pace_score_delta'])}", _fmt(r["pace_score_prev_month"]), r["days_current_month"], r["days_prev_month"]])
    return _render_table(headers, data) + note


def _fmt(v, nd=0):
    """Formats a numeric value for display. Defaults to whole-number
    rounding (round-half-to-even via Python's round(), which agrees with
    round-half-up for the non-.5 values these metrics actually produce,
    e.g. 81.9 -> 82) — this is a DISPLAY-only rounding, it never touches
    the underlying stored/query precision. Pass nd explicitly for the rare
    case a caller still wants decimal places."""
    if v is None:
        return "N/A"
    if isinstance(v, (int,)):
        return str(v)
    try:
        fv = float(v)
        if nd == 0:
            return str(int(round(fv)))
        return f"{round(fv, nd):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def format_metric_rows(rows, metric_key):
    if not rows:
        return "No matching data found for that filter."
    label = queries.METRICS[metric_key][1]
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — {label}: {_fmt(r['metric_value'])}"
    headers = ["#", "Employee", "Department", label]
    data = [[i, r["emp_name"], r["dept_name"], _fmt(r["metric_value"])] for i, r in enumerate(rows, 1)]
    return _render_table(headers, data)


def format_dept_rows(rows, metric_key):
    if not rows:
        return "No matching data found."
    label = queries.METRICS[metric_key][1]
    if len(rows) == 1:
        r = rows[0]
        return f"{r['dept_name']} ({r['n_employees']} employees) — {label}: {_fmt(r['metric_value'])}"
    headers = ["#", "Department", "Employees", label]
    data = [[i, r["dept_name"], r["n_employees"], _fmt(r["metric_value"])] for i, r in enumerate(rows, 1)]
    return _render_table(headers, data)


def format_rm_rows(rows, metric_key):
    if not rows:
        return "No matching data found."
    label = queries.METRICS[metric_key][1]
    if len(rows) == 1:
        r = rows[0]
        return f"{r['reporting_manager_name']} ({r['n_employees']} employees) — {label}: {_fmt(r['metric_value'])}"
    headers = ["#", "Reporting Manager", "Employees", label]
    data = [[i, r["reporting_manager_name"], r["n_employees"], _fmt(r["metric_value"])] for i, r in enumerate(rows, 1)]
    return _render_table(headers, data)


def _avg_per_day(total, days_worked):
    """total minutes / worked days-in-period -> current daily average, used
    for every minute-based metric so they read as a meaningful per-day rate
    instead of a raw period total that gets bigger the longer the period is."""
    if not days_worked:
        return "N/A"
    try:
        return _fmt(float(total) / days_worked)
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"


def format_employee_detail(row, emp_name):
    if not row:
        return f"No data found for {emp_name} for that period."
    days = row["days_worked"]
    return (
        f"{row['emp_name']} ({row['dept_name']}, reports to {row['reporting_manager_name']}, "
        f"{row['designation']}) — {days} days worked.\n"
        f"PACE score: {_fmt(row['avg_pace_score'])} (changed by {_fmt(row['pace_score_delta'])} pts vs prior month avg {_fmt(row['pace_score_prev_month'])})\n"
        f"Attendance: LC {row['total_lc']}, EL {row['total_el']}, DH {row['total_dh']}, Defaulter days {row['total_defaulter']}\n"
        f"Avg productive min/day: {_avg_per_day(row['total_productive_min'], days)}, "
        f"Avg WhatsApp min/day: {_avg_per_day(row['total_whatsapp_min'], days)}, "
        f"Avg AI min/day: {_avg_per_day(row['total_ai_min'], days)}, "
        f"Avg tools min/day: {_avg_per_day(row['total_tools_min'], days)}\n"
        f"Discipline %: {_fmt(row['avg_discipline_pct'])}, Engagement %: {_fmt(row['avg_engagement_pct'])}, "
        f"Effectiveness %: {_fmt(row['avg_effectiveness_pct'])}, Working %: {_fmt(row['avg_working_pct'])}"
    )


def format_dept_summary(row, dept_name):
    if not row:
        return f"No data found for {dept_name} for that period."
    return (
        f"{row['dept_name']} — {row['n_employees']} employees\n"
        f"Avg PACE score: {_fmt(row['avg_pace_score'])}, Avg engagement %: {_fmt(row['avg_engagement_pct'])}, "
        f"Avg effectiveness %: {_fmt(row['avg_effectiveness_pct'])}, Avg discipline %: {_fmt(row['avg_discipline_pct'])}\n"
        f"LC {row['total_lc']}, EL {row['total_el']}, DH {row['total_dh']}, Productive minutes {row['total_productive_min']}"
    )


def format_compare(rows, labels, fmt_row_fn):
    if not rows or any(r is None for r in rows):
        return "Couldn't find data for one or both of those — please check the names/departments."
    parts = []
    for label, r in zip(labels, rows):
        parts.append(f"--- {label} ---\n{fmt_row_fn(r, label)}")
    return "\n\n".join(parts)


def format_new_joiners(rows):
    if not rows:
        return "No new joiners found in that scope."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — joined {r['doj']}, {r['tenure_days']} days ago [{r['status']}]"
    headers = ["#", "Employee", "Department", "Joined", "Tenure (days)", "Status"]
    data = [[i, r["emp_name"], r["dept_name"], r["doj"], r["tenure_days"], r["status"]] for i, r in enumerate(rows, 1)]
    return _render_table(headers, data)


def format_meeting_rows(rows):
    if not rows:
        return "No matching data found."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — {r['total_meeting_min']} meeting minutes"
    headers = ["#", "Employee", "Department", "Meeting minutes"]
    data = [[i, r["emp_name"], r["dept_name"], r["total_meeting_min"]] for i, r in enumerate(rows, 1)]
    return _render_table(headers, data)


def format_chronic_late(rows, threshold):
    note = f"(threshold: {threshold}+ late-coming days)"
    if not rows:
        return f"No one had {threshold}+ late-coming days in that scope."
    if len(rows) == 1:
        r = rows[0]
        return f"{note}\n{r['emp_name']} ({r['dept_name']}) — {r['total_lc']} late-comings"
    headers = ["#", "Employee", "Department", "Late-comings"]
    data = [[i, r["emp_name"], r["dept_name"], r["total_lc"]] for i, r in enumerate(rows, 1)]
    return note + "\n\n" + _render_table(headers, data)


def format_perfect_attendance(rows):
    if not rows:
        return "No one had perfect attendance (0 defaulter days) in that scope."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — {r['days_worked']} days worked, 0 defaulter days"
    headers = ["#", "Employee", "Department", "Days worked"]
    data = [[i, r["emp_name"], r["dept_name"], r["days_worked"]] for i, r in enumerate(rows, 1)]
    return _render_table(headers, data)


def format_team_summary(row, label):
    if not row:
        return f"No data found for {label}."
    return (
        f"{label} — {row['n_employees']} employees\n"
        f"Avg PACE score: {_fmt(row['avg_pace_score'])}, Avg engagement %: {_fmt(row['avg_engagement_pct'])}, "
        f"Avg effectiveness %: {_fmt(row['avg_effectiveness_pct'])}, Avg discipline %: {_fmt(row['avg_discipline_pct'])}\n"
        f"LC {row['total_lc']}, EL {row['total_el']}, DH {row['total_dh']}"
    )


_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _first_month(month):
    """Collapses a possibly-multi-month value down to a single 'YYYY-MM'
    string — for the handful of trend/delta functions (this-month-vs-prior-
    month comparisons, e.g. pace_score_trend_ranking/dept_delta_ranking)
    where "sum across N months" has no defined meaning; those intentionally
    keep behaving on just the first named month, same as before this fix."""
    if isinstance(month, list):
        return month[0] if month else None
    return month


def _no_period_named_at_all(message, session):
    """True only when THIS message named no month/date reference AND no
    sticky session context (recent dept/month/date_range carry-forward -
    see handle_message's own identical check around 'Time-period fallback')
    supplied one either. Used to distinguish a genuine "nothing specified at
    all" case (item B, SESSION_HANDOFF.md: matrix-wide default changed from
    current-in-progress-month to last-60-days) from a month/date the caller
    already resolved via explicit mention or sticky context, which must be
    left completely alone. Re-derives the same signals handle_message()
    already computed (month_mentioned/date_range_mentioned) directly from
    the raw message text rather than threading extra booleans through
    answer_intent()'s signature - cheap, side-effect-free, and exactly
    mirrors the logic already in handle_message()."""
    date_mentioned = entities.extract_date_range(message)[2]
    if date_mentioned:
        return False
    _, month_mentioned = entities.extract_months(message, default_to_current=False)
    if month_mentioned:
        return False
    if session is not None:
        if session_store.get_recent_context(session, "date_range") is not None:
            return False
        if session_store.get_recent_context(session, "month") is not None:
            return False
    return True


def _format_months(months):
    """'2026-06','2026-07','2026-08' -> 'Jun+Jul+Aug 2026' (or, if the named
    months span different years, 'Jun 2026+Jan 2027')."""
    parts = []
    years = {m.split("-")[0] for m in months}
    same_year = len(years) == 1
    for m in months:
        y, mo = m.split("-")
        label = _MONTH_ABBR[int(mo)]
        if not same_year:
            label += f" {y}"
        parts.append(label)
    joined = "+".join(parts)
    return f"{joined} {next(iter(years))}" if same_year else joined


def _period_note(month, date_range):
    if date_range:
        start, end = date_range
        return f" ({start} to {end})" if start != end else f" ({start})"
    if isinstance(month, list):
        if not month:
            return ""
        if len(month) == 1:
            return f" for {month[0]}"
        return f" for {_format_months(month)} (summed)"
    return f" for {month}" if month else ""


def format_leave_rows(rows):
    if not rows:
        return "No leave records found for that scope."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — {r['worked_day']}: {r['applied_leave_type']} [{r['applied_leave_status']}]"
    headers = ["#", "Employee", "Department", "Date", "Leave type", "Status"]
    data = [
        [i, r["emp_name"], r["dept_name"], r["worked_day"], r["applied_leave_type"], r["applied_leave_status"]]
        for i, r in enumerate(rows, 1)
    ]
    return _render_table(headers, data)


def format_count_rows(rows, count_field, label, name_field=None):
    if not rows:
        return "No matching data found."

    def _name_dept(r):
        if name_field:
            return r.get(name_field), ""
        name = r.get("emp_name") or r.get("dept_name")
        dept = r.get("dept_name") if r.get("emp_name") and "dept_name" in r else ""
        return name, dept

    if len(rows) == 1:
        r = rows[0]
        name, dept = _name_dept(r)
        dept_str = f" ({dept})" if dept else ""
        return f"{name}{dept_str} — {label}: {_fmt(r[count_field])}"
    headers = ["#", "Name", "Department", label]
    data = []
    for i, r in enumerate(rows, 1):
        name, dept = _name_dept(r)
        data.append([i, name, dept, _fmt(r[count_field])])
    return _render_table(headers, data)


# ---------------------------------------------------------------------------
# Item 1: which direction(s) - gainers-only / losers-only / both - a
# gainer/loser question is actually asking for.
# ---------------------------------------------------------------------------
_LOSER_ONLY_PATTERN = r"\blos+er(s)?\b|\bworst perform(er|ance)s?\b|\bdropped the most\b|\bwho dropped\b|\bdeclin(e|ed|ing)\b"
_GAINER_ONLY_PATTERN = r"\bgainer(s)?\b|\bimproved the most\b|\bwho improved\b|\bbest perform(er|ance)s?\b"
_BOTH_PATTERN = r"\bgainers?\s+and\s+los+ers?\b|\blos+ers?\s+and\s+gainers?\b|\bboth\b|\bimproved\s+and\s+(who\s+)?dropped\b|\bdropped\s+and\s+(who\s+)?improved\b"


def _gainer_loser_directions(message):
    """Returns a tuple subset of ('gainers', 'losers') describing which
    direction(s) the question actually asked for. Defaults to both when the
    phrasing is ambiguous or explicitly asks for both."""
    text_l = (message or "").lower()
    if re.search(_BOTH_PATTERN, text_l):
        return ("gainers", "losers")
    wants_loser = bool(re.search(_LOSER_ONLY_PATTERN, text_l))
    wants_gainer = bool(re.search(_GAINER_ONLY_PATTERN, text_l))
    if wants_loser and not wants_gainer:
        return ("losers",)
    if wants_gainer and not wants_loser:
        return ("gainers",)
    return ("gainers", "losers")


# ---------------------------------------------------------------------------
# Item 3: shared default-population filter. shift_type='Standard' is the
# unchanged always-default baseline; visit_flag='No' AND ps_worked_flag_day=1
# are now ALSO applied by default whenever the question doesn't specify a
# filter itself. Used by gainer/loser ranking and day-vs-day comparison only
# this round (see SESSION_HANDOFF.md for the explicit scope boundary).
# ---------------------------------------------------------------------------
_OT_PATTERN = r"\bovertime\b|\bot\b"
_STANDARD_PATTERN = r"\bstandard\b|\bstd\b"
_VISIT_NO_PATTERN = r"\bnon[\s-]?visit\b|\bno visit\b|\bnot on visit\b|\bwithout visit\b|\bnon[\s-]?visiting\b"
_VISIT_YES_PATTERN = r"\bvisit(s|ed|ing)?\b|\bclient visit"
_PS_NOT_WORKING_PATTERN = r"\bps[\s-]?not[\s-]?work(ed|ing)?\b|\bps[\s-]?not[\s-]?installed\b"
_PS_WORKING_PATTERN = r"\bps[\s-]?worked\b|\bps[\s-]?working\b|\bps status\b"


def _resolve_population_filter(message):
    """Detect explicit population-filter overrides in `message` and compose
    the population filter SQL fragment + the dynamic footer text describing
    what's actually in effect. Returns (filter_sql, footer_text).

    Pure default (nothing overridden): shift_type='Standard' AND
    visit_flag='No' AND ps_worked_flag_day=1 -> generic default footer,
    never naming 'Standard' unless the user asked about shift type."""
    text_l = (message or "").lower()

    ot_requested = bool(re.search(_OT_PATTERN, text_l))
    standard_requested = (not ot_requested) and bool(re.search(_STANDARD_PATTERN, text_l))
    visit_no_explicit = bool(re.search(_VISIT_NO_PATTERN, text_l))
    visit_yes = (not visit_no_explicit) and bool(re.search(_VISIT_YES_PATTERN, text_l))
    ps_not_working = bool(re.search(_PS_NOT_WORKING_PATTERN, text_l))
    ps_working_explicit = (not ps_not_working) and bool(re.search(_PS_WORKING_PATTERN, text_l))

    clauses = []
    footer_parts = []

    if ot_requested:
        clauses.append("shift_type = 'Overtime (OT)'")
        footer_parts.append("OT (Overtime) days")
    elif standard_requested:
        clauses.append("shift_type = 'Standard'")
        footer_parts.append("Standard shift days")
    else:
        clauses.append("shift_type = 'Standard'")

    if visit_yes:
        clauses.append("visit_flag = 'Yes'")
        footer_parts.append("visit days")
    elif visit_no_explicit:
        clauses.append("visit_flag = 'No'")
        footer_parts.append("non-visit days")
    else:
        clauses.append("visit_flag = 'No'")

    if ps_not_working:
        clauses.append("ps_worked_flag_day = 0")
        footer_parts.append("PS-not-working days")
    elif ps_working_explicit:
        clauses.append("ps_worked_flag_day = 1")
        footer_parts.append("PS-working days")
    else:
        clauses.append("ps_worked_flag_day = 1")

    is_pure_default = not (ot_requested or standard_requested or visit_yes or visit_no_explicit
                            or ps_not_working or ps_working_explicit)

    if is_pure_default:
        footer = "By default showing data for non visit and PS working days."
    else:
        footer = "Showing data for " + ", ".join(footer_parts) + "."

    filter_sql = " and ".join(clauses)
    return filter_sql, footer


_BUILD_QUERY_ALL_METRICS = ["pace_score", "engagement_pct", "effectiveness_pct", "discipline_pct",
                            "working_pct", "LC", "EL", "DH", "working_hours"]
_BUILD_QUERY_METRIC_LABELS = {
    "pace_score": "PACE score", "engagement_pct": "engagement %", "effectiveness_pct": "effectiveness %",
    "discipline_pct": "discipline %", "working_pct": "working hours %", "LC": "late-comings",
    "EL": "early leavings", "DH": "deficient-hour days", "working_hours": "total working hours",
    "productive_minutes": "avg productive minutes",
    # Item #70 gap-fill metrics (this round) — reachable only via the new
    # extraction-LLM cascade step (_extraction_llm_reply), not via
    # _detect_build_query_metrics()'s keyword patterns, since these are
    # narrower/internal-facing metrics not worth adding regex keywords for.
    "pace_status": "PACE status", "capped_engagement": "capped engagement (internal)",
    "capped_effectiveness": "capped effectiveness (internal)", "capped_discipline": "capped discipline (internal)",
    "pace_score_day_level": "avg day-level PACE score", "dept_score_60_days_precomputed": "precomputed dept score (60d)",
    "dept_status_60_days_derived": "derived dept status (60d)",
    # Item #79 gap-fill (rows 39/40/49, per item #77's plan)
    "engagement_minutes": "avg engagement minutes", "meeting_minutes": "total meeting minutes",
    "meeting_count": "total meetings",
    "tasks_created": "tasks created", "tasks_assigned": "tasks assigned",
    "todos_created": "todos created", "todos_assigned": "todos assigned",
}

_BUILD_QUERY_METRIC_PATTERNS = [
    ("pace_score", r"\bpace score\b|\bscore\b"),
    ("engagement_pct", r"\bengagement\b"),
    ("effectiveness_pct", r"\beffectiveness\b"),
    ("discipline_pct", r"\bdiscipline\b"),
    ("working_pct", r"\bworking (%|percent|percentage)\b|\bworking hours %\b"),
    ("LC", r"\blate[- ]?coming(s)?\b|\blc\b"),
    ("EL", r"\bearly[- ]?leaving(s)?\b|\bel\b"),
    ("DH", r"\bdeficient[- ]?hour(s)?\b|\bdh\b"),
    ("working_hours", r"\bworking hours\b|\bworked hours\b"),
    # Added for the average_metric intent - "prod(uctive) minutes/mins" was
    # previously not detectable by this engine at all (only reachable via
    # dedicated ranking functions like productive_high/productive_low).
    ("productive_minutes", r"\bprod(?:uctive)?\s*(minutes?|mins?)\b|\bproductive\b"),
    # Item #79 gap-fill (rows 39/40/49, per item #77's plan): these 6 must be
    # checked BEFORE the bare "engagement" pattern above in the detection
    # loop below (see _detect_build_query_metrics - the loop excludes
    # engagement_pct from re-matching once engagement_minutes is found, the
    # same "specific wins over bare fallback" precedent as working_pct/
    # working_hours). No existing bare "meeting"/"task"/"todo" pattern
    # exists in this list, so meeting_minutes/tasks_*/todos_* have no
    # collision risk and are added directly.
    ("engagement_minutes", r"\bengagement\s*minutes?\b"),
    ("meeting_minutes", r"\bmeeting\s*minutes?\b"),
    ("meeting_count", r"\bmeeting\s*count\b|\bnumber of meetings\b|\btotal meetings?\b|\bhow many meetings\b"),
    ("tasks_created", r"\btasks?\s*created\b"),
    ("tasks_assigned", r"\btasks?\s*assigned\b"),
    ("todos_created", r"\btodos?\s*created\b"),
    ("todos_assigned", r"\btodos?\s*assigned\b"),
]

# Item #73: generalized capped-vs-percentage business rule for effectiveness/
# engagement/discipline, shared by every caller that needs it (the
# build_query() keyword-fallback detector below, the dept_best/dept_worst
# rule-based handler, and the extraction-LLM cascade's post-validation
# override) - a single source of truth, not 3 separate hand-rolled checks.
# The rule (deliberate, non-obvious - see SESSION_HANDOFF.md item #73, do
# NOT "simplify" this back to naive keyword matching):
#   "X"                              -> "X_pct"            (normal wording)
#   "capped X" (no %/percentage)      -> "capped_X"         (raw internal)
#   "raw capped X"                     -> "capped_X"        (same as above)
#   "capped X %" / "capped X percentage" -> "X_pct"          (DELIBERATE -
#       this is NOT a mistake to "fix" back to capped_X)
_PCT_CAPPED_METRIC_PATTERN = re.compile(
    r"\b(?:(?P<raw>raw)\s+)?(?:(?P<capped>capped)\s+)?(?P<word>effectiveness|engagement|discipline)\b"
    r"(?!\s*minutes?\b)"  # item #79: "engagement minutes" is a distinct raw-minutes
    # metric (engagement_minutes), never engagement_pct/capped_engagement - do not
    # let this shared normalizer swallow it (the exact silent-wrong-metric collision
    # item #77 found live: "average engagement minutes for X" was returning
    # engagement_pct instead of engagement_minutes).
    r"(?:\s*(?P<pct>%|percent|percentage))?",
    re.IGNORECASE,
)
_PCT_CAPPED_METRIC_MAP = {
    "effectiveness": ("effectiveness_pct", "capped_effectiveness"),
    "engagement": ("engagement_pct", "capped_engagement"),
    "discipline": ("discipline_pct", "capped_discipline"),
}
_ALL_PCT_CAPPED_KEYS = {k for pair in _PCT_CAPPED_METRIC_MAP.values() for k in pair}


def _detect_pct_capped_metrics(message):
    """Applies the item #73 business rule documented above to every
    effectiveness/engagement/discipline mention in `message`. Returns a
    de-duplicated, order-preserving list of BUILD_QUERY_METRICS-style keys
    (empty if none of these three words appear at all)."""
    found = []
    for m in _PCT_CAPPED_METRIC_PATTERN.finditer(message or ""):
        word = (m.group("word") or "").lower()
        pair = _PCT_CAPPED_METRIC_MAP.get(word)
        if pair is None:
            continue
        pct_key, capped_key = pair
        capped = bool(m.group("capped"))
        pct = bool(m.group("pct"))
        key = capped_key if (capped and not pct) else pct_key
        if key not in found:
            found.append(key)
    return found


def _detect_build_query_metrics(message):
    """Picks the metric(s) named in `message` from BUILD_QUERY_METRICS + pace_score.
    Falls back to just ["pace_score"] (the overwhelmingly common "how is X
    doing" case) when nothing more specific is named."""
    text_l = (message or "").lower()
    found = []
    # Item #73: effectiveness/engagement/discipline capped-vs-percentage
    # business rule - shared normalizer, checked first so its resolved key
    # (e.g. "capped_effectiveness") wins over the naive substring patterns
    # below rather than being overridden by them.
    for key in _detect_pct_capped_metrics(message):
        if key not in found:
            found.append(key)
    # Working hours: percentage-wording must win over the raw-hours pattern
    # when both match the same phrase (e.g. "working hours percentage") -
    # item #73 finding: the old working_pct regex only matched a literal
    # "%" sign right after "working", never the word "percentage", so
    # "working hours percentage" silently fell through to the raw
    # working_hours (sum) metric instead of working_pct. Fixed here, in the
    # shared detector, not a new parallel mechanism.
    if re.search(r"\bworking hours?\s*(%|percent|percentage)\b", text_l):
        if "working_pct" not in found:
            found.append("working_pct")
    elif re.search(r"\bworking hours\b|\bworked hours\b", text_l):
        if "working_hours" not in found:
            found.append("working_hours")
    for key, pat in _BUILD_QUERY_METRIC_PATTERNS:
        if key in ("engagement_pct", "effectiveness_pct", "discipline_pct", "working_pct", "working_hours"):
            continue  # handled above
        if re.search(pat, text_l) and key not in found:
            found.append(key)
    return found or ["pace_score"]


def _detect_build_query_filters(message):
    """Reuses the same override-detection regexes as _resolve_population_filter
    (ps/visit/shift) plus a WFH/office check, returning a filters dict for
    queries.build_query() - None for anything not explicitly overridden so
    build_query's own confirmed defaults (ps_worked_flag_day=1, visit_flag='No',
    shift_type='Standard', no work-mode filter) apply."""
    text_l = (message or "").lower()
    filters = {}
    if re.search(_PS_NOT_WORKING_PATTERN, text_l):
        filters["ps_status"] = "not_working"
    elif re.search(_PS_WORKING_PATTERN, text_l):
        filters["ps_status"] = "working"
    if re.search(_VISIT_YES_PATTERN, text_l) and not re.search(_VISIT_NO_PATTERN, text_l):
        filters["visit_status"] = "yes"
    elif re.search(_VISIT_NO_PATTERN, text_l):
        filters["visit_status"] = "no"
    if re.search(r"\bwfh\b|\bwork(ing)? from home\b|\bremote\b", text_l):
        filters["work_mode"] = "wfh"
    if re.search(_OT_PATTERN, text_l):
        filters["shift_type"] = "Overtime (OT)"
    elif re.search(_STANDARD_PATTERN, text_l):
        filters["shift_type"] = "Standard"
    return filters


# capped_* metrics are internal 0-1(ish) fractions, not whole-number scores
# or percentages - _fmt()'s default nd=0 rounding turns e.g. 0.887 into the
# misleading "1". Item #72 fix (found live during item #71's own testing,
# flagged there as a cosmetic bug, fixed here): format these with 2 decimal
# places instead, everything else keeps the existing whole-number rounding.
_BUILD_QUERY_DECIMAL_METRICS = {"capped_engagement", "capped_effectiveness", "capped_discipline"}


def _fmt_bq(v, metric_key):
    return _fmt(v, nd=2) if metric_key in _BUILD_QUERY_DECIMAL_METRICS else _fmt(v)


def _format_build_query_rows(rows, dimension, metrics, name_label=None):
    if not rows:
        return f"No data found for {name_label or 'that scope'} in this period."
    # "company" has no dimension column at all (build_query() returns one
    # aggregate row, no GROUP BY) - always formatted via the single-row
    # name_label branch below, never the multi-row table branch.
    dim_col = {"employee": "emp_name", "rm": "reporting_manager_name", "department": "dept_name",
               "company": None}[dimension]
    if len(rows) == 1 and name_label:
        row = rows[0]
        parts = ", ".join(f"{_BUILD_QUERY_METRIC_LABELS[m]}: {_fmt_bq(row.get(m), m)}" for m in metrics)
        return f"{name_label} — {row.get('n_employees')} employee(s)\n{parts}"
    headers = [dimension.capitalize()] + [_BUILD_QUERY_METRIC_LABELS[m] for m in metrics]
    data = []
    for r in rows:
        data.append([r.get(dim_col)] + [_fmt_bq(r.get(m), m) for m in metrics])
    return _render_table(headers, data)


_WANTS_LIST_PATTERN = re.compile(r"\blist\b", re.IGNORECASE)


def build_query_overview_reply(dimension, name, message="", period=None, session=None, display_name=None):
    """Shared helper: runs queries.build_query() for a single named
    employee/RM/department scope and formats a reply - used both by the
    'how is ai labs doing'-style bug fix (redirecting a failed single-
    employee lookup to a department/RM overview when the name actually
    matches a department/RM instead) and by the new general fallback engine
    in handle_message().

    Item #59 fixes (both reuse the EXISTING session_store.last_list/
    sticky-context mechanism the older ranking functions already use -
    nothing new invented here):
      1. Wires this reply into session_store.set_last_list() so a follow-up
         like "give me list" has real context to continue from (previously
         build_query()'s output never registered a last_list at all, so any
         such follow-up fell straight to the generic fallback).
      2. Response-shape switching: if the CURRENT message explicitly asks
         for a "list" (e.g. "give me list of ai labs") and the scope is a
         department/RM (not already a single employee), answer with the
         per-employee breakdown for that scope instead of the aggregate
         summary - same "does this message want a list or a summary" idea
         as the existing dept_summary/ranking switch a few lines below in
         handle_message() (items #52/#53), just applied to build_query()'s
         own summary/list duality.
    """
    metrics = _detect_build_query_metrics(message)
    filters = _detect_build_query_filters(message)
    wants_list = dimension != "employee" and _WANTS_LIST_PATTERN.search(message or "") is not None
    # "company" scope has no name at all (no department/RM/employee named) -
    # label it explicitly rather than printing "None" anywhere in the reply.
    # `name` doubles as build_query()'s own name_filter, which for
    # dimension="employee" is the numeric employee_id (the column build_query()
    # actually filters on), NOT a display-friendly string - every
    # dimension="employee" caller was passing the raw id straight through as
    # the label too, so replies like "average meeting minutes for Rudhi" showed
    # "36014 - 1 employee(s)..." instead of "Rudhi - ...". `display_name` lets
    # a caller supply the real name separately for the label while `name`
    # keeps filtering correctly; falls back to `name` unchanged for every
    # dimension/caller that already passed a proper display string (department/
    # RM/company all already did, only employee-dimension callers had this bug).
    name_label = display_name if display_name else (name if name else "The whole company")

    def _summary_reply():
        rows = queries.build_query(dimension, metrics, filters=filters, period=period, name_filter=name, limit=1)
        return _format_build_query_rows(rows, dimension, metrics, name_label=name_label), rows

    def _list_reply(limit=500):
        # "company" scope has no OTHER-dimension column to filter on (every
        # employee is in scope) - pass scope=None instead of ("company", name),
        # which build_query()'s scope-column lookup doesn't recognize.
        scope = None if dimension == "company" else (dimension, name)
        rows = queries.build_query("employee", metrics, filters=filters, period=period,
                                    scope=scope, limit=limit)
        if not rows:
            return f"No employees found for {name_label} in this period.", rows
        table_reply = _format_build_query_rows(rows, "employee", metrics, name_label=None)
        return f"{name_label} — employee list:\n\n{table_reply}", rows

    if wants_list:
        reply, rows = _list_reply()
        answer_kind = "list"
    else:
        reply, rows = _summary_reply()
        answer_kind = "count"

    if session is not None:
        def _rerun_list(dept_name=None, employee_ids=None, team_label=None, month=None, date_range=None, limit=500):
            return _list_reply(limit=limit or 500)

        def _rerun_same(dept_name=None, employee_ids=None, team_label=None, month=None, date_range=None, limit=500):
            return (_list_reply() if wants_list else _summary_reply())

        session_store.set_last_list(
            session, kind="build_query", rerun_list=_rerun_list, rerun_same=_rerun_same, answer_kind=answer_kind,
            dept_name=(name if dimension == "department" else None),
            team_label=(f"{name}'s team" if dimension == "rm" else None),
            date_range=period,
        )
        if dimension == "department":
            session_store.push_context(session, dept_name=name)

    return reply, rows


def _month_str_to_range(month_str):
    """'YYYY-MM' -> (first_day, last_day) date tuple. Small local helper -
    the extraction-LLM cascade step (item #70) needs to turn a resolved
    month string into build_query()'s (start,end) period shape; every other
    caller in this codebase passes month/date_range separately into
    queries._period_filter() instead, so this conversion didn't exist yet."""
    import calendar
    year, mo = int(month_str[:4]), int(month_str[5:7])
    last_day = calendar.monthrange(year, mo)[1]
    return datetime.date(year, mo, 1), datetime.date(year, mo, last_day)


def _extraction_llm_reply(raw_message, message, session):
    """Item #70: the new extraction-LLM cascade step. Called only when both
    the rule-based matcher and llm_nlu.classify() found nothing usable
    (intent is None), and strictly BEFORE sql_fallback.answer() (see the
    ordering judgment call documented in SESSION_HANDOFF.md item #70/this
    round). Returns (reply, rows) on success, or None to let the cascade
    fall through to sql_fallback.answer() next - on ANY validation failure,
    hallucinated field, or LLM/network error, this returns None rather than
    raising or guessing, per the standing hard-safety contract.
    """
    sticky = (session or {}).get("sticky_context") or {}
    hint_bits = []
    if sticky.get("dept_name"):
        hint_bits.append(f"currently discussing department: {sticky['dept_name']}")
    if sticky.get("employee_name"):
        hint_bits.append(f"currently discussing employee: {sticky['employee_name']}")
    context_hint = "; ".join(hint_bits) or None

    try:
        extracted = llm_nlu.extract_build_query(raw_message, context_hint=context_hint)
    except Exception:
        logging.getLogger("pace_chatbot.main").exception("llm_nlu.extract_build_query() raised unexpectedly")
        return None
    if not extracted:
        return None

    dimension = extracted.get("dimension")
    if dimension not in queries.BUILD_QUERY_DIMENSIONS:
        return None  # hallucinated dimension - fail safe, fall through

    # Validate every extracted metric against the REAL metric-key set (not a
    # hand-duplicated list) - drop anything invalid rather than crashing;
    # empty after filtering -> the same "defaults to pace_score" convention
    # _detect_build_query_metrics() already uses.
    valid_metric_keys = set(queries.BUILD_QUERY_METRICS.keys()) | {
        "pace_score", "pace_status", "dept_status_60_days_derived",
    }
    metrics = [m for m in (extracted.get("metrics") or []) if m in valid_metric_keys]
    # Word order varies in natural phrasing ("Rahul's strongest area" vs
    # "which area is Rahul weakest in") - check both orders and derive the
    # direction word from whichever one actually matched. Computed early
    # (moved up from below, item #76) because the empty-metrics fallback
    # chain right below needs to know NOT to treat "strongest/weakest area"
    # phrasing as an unrecognized-metric case - its metrics are SUPPOSED to
    # be empty at this point (the caller supplies the 4 area metrics itself
    # further down).
    _area_dir_match = (
        re.search(r"\b(strongest|weakest)\b[^.?!]{0,40}\b(?:area|metric|dimension|aspect)\b", raw_message, re.I)
        or re.search(r"\b(?:area|metric|dimension|aspect)\b[^.?!]{0,40}\b(strongest|weakest)\b", raw_message, re.I)
    )
    _area_match = _area_dir_match
    if not metrics:
        # Item #72: don't silently default straight to plain pace_score when
        # the LLM extraction returned an empty metrics list - a raw_message
        # keyword scan for the same new-vocabulary markers that route here
        # in the first place (_NEW_VOCAB_OVERRIDE_PATTERN) is a cheap,
        # deterministic repair that catches exactly the failure mode this
        # round must eliminate (extraction under-confident about which new
        # metric key applies -> would otherwise silently answer with the
        # WRONG metric, "pace_score", rather than the one actually asked
        # about). Falls through to the plain "pace_score" default below only
        # if none of these markers are present either.
        metrics = _repair_new_vocab_metric(raw_message)
    if not metrics and not _area_match:
        # Item #76 (Phase 3): the LAST safety net before the plain pace_score
        # default - extract_build_query() now explicitly flags when the
        # message named a SPECIFIC metric concept it didn't recognize
        # (unrecognized_metric_phrase), as opposed to a genuinely generic
        # "how is X doing" question (which correctly still defaults to
        # pace_score below). Live-verified gap this closes: "what is Aryan
        # Gupta's synergy quotient for last week" previously silently
        # answered with his plain PACE score - exactly the fabricated-
        # specific-number failure mode this project must never produce.
        _unrecognized = extracted.get("unrecognized_metric_phrase")
        if _unrecognized:
            return (
                f"I don't have a metric called \"{_unrecognized}\" — I can answer about PACE score/status, "
                "engagement, effectiveness, discipline, working hours, capped engagement/effectiveness/"
                "discipline, late-comings, early leavings, deficient-hour days, or productive minutes. "
                "Could you rephrase using one of those?",
                [],
            )
    if not metrics:
        metrics = ["pace_score"]
    # Item #73: deterministic override for the capped-vs-percentage business
    # rule - this is a precise, non-obvious mapping (see SESSION_HANDOFF.md
    # item #73) that must never be left to the LLM's own probabilistic
    # judgment, same "narrow deterministic check wins over a probabilistic
    # guess" precedent as _repair_new_vocab_metric() above. Whenever the
    # RAW message actually names effectiveness/engagement/discipline, the
    # shared normalizer's resolved key(s) always replace whatever the LLM
    # guessed for those same 3 concepts - any OTHER metric it also asked
    # for (e.g. combined with pace_status) is left untouched.
    _pct_capped_override = _detect_pct_capped_metrics(raw_message)
    if _pct_capped_override:
        metrics = [m for m in metrics if m not in _ALL_PCT_CAPPED_KEYS]
        for _k in _pct_capped_override:
            if _k not in metrics:
                metrics.append(_k)
    # dept_status_60_days_derived / dept_score_60_days_precomputed only make
    # sense for dimension="department" - drop them otherwise rather than
    # letting build_query() raise a confusing SQL error downstream.
    if dimension != "department":
        metrics = [m for m in metrics if m not in ("dept_status_60_days_derived", "dept_score_60_days_precomputed")]
        if not metrics:
            metrics = ["pace_score"]

    # Never trust the LLM's own filter values blindly beyond the enum the
    # schema already constrains them to - queries.build_query() itself
    # validates/defaults these further. Item #76: merged with two
    # deterministic sources that OVERRIDE the LLM's own guess, same
    # "narrow deterministic regex beats a probabilistic LLM guess" precedent
    # as _repair_new_vocab_metric()/_detect_pct_capped_metrics() above -
    # (1) _detect_build_query_filters() (the same regex detector the
    # rule-based build_query() callers already use for ps/visit/wfh/shift
    # overrides), and (2) a filter implied by the period phrase itself (e.g.
    # "last 10 WFH days" implies work_mode="wfh" even if the LLM's separate
    # `filters` field missed it).
    filters = dict(extracted.get("filters") or {})
    filters.update(_detect_build_query_filters(raw_message))

    # Re-parse period_phrase through the EXISTING, already-tested date
    # parsers - never trust LLM date arithmetic directly. Item #76: bare/
    # qualified "last N days" is checked FIRST (entities.extract_last_n_days) -
    # a strictly more specific shape than extract_date_range's named windows,
    # and the ONLY parser here that supports a raw day-count at all (see
    # entities.py's docstring for the live-verified gap this closes).
    period = None
    latest_n_days = None
    phrase = extracted.get("period_phrase")
    if phrase:
        n_days, n_qualifier, n_mentioned = entities.extract_last_n_days(phrase)
        if n_mentioned:
            if n_qualifier is None:
                # Plain calendar window: "last 40 days" -> (today-39, today).
                _today = datetime.date.today()
                period = (_today - datetime.timedelta(days=n_days - 1), _today)
            else:
                # "last 10 WFH days" etc - qualifying-ROW-count mode
                # (queries.build_query()'s new latest_n_days param): filters
                # define the population FIRST, then the latest N matching
                # rows are taken - never a calendar window filtered
                # afterward. The qualifier also deterministically sets/
                # overrides the matching build_query() filter, same
                # "deterministic wins" precedent as above.
                latest_n_days = n_days
                _qualifier_filter = {
                    "wfh": {"work_mode": "wfh"}, "office": {"work_mode": "office"},
                    "ot": {"shift_type": "Overtime (OT)"}, "standard": {"shift_type": "Standard"},
                    "visit": {"visit_status": "yes"},
                    "ps_working": {"ps_status": "working"}, "ps_not_working": {"ps_status": "not_working"},
                    "non_working": {"ps_status": "not_working"},
                }.get(n_qualifier)
                if _qualifier_filter:
                    filters.update(_qualifier_filter)
        else:
            d_start, d_end, mentioned = entities.extract_date_range(phrase)
            if mentioned:
                period = (d_start, d_end)
            else:
                month_str, mentioned_m = entities.extract_month(phrase, default_to_current=False)
                if mentioned_m and month_str:
                    period = _month_str_to_range(month_str)
            # If the phrase was named but genuinely unparseable, period stays
            # None -> build_query() defaults to last 60 days (documented, not
            # a crash) rather than us guessing.

    # Item #76 (Part B): "<employee/dept/RM>'s strongest/weakest area" - a
    # derived OPERATION (rank the 4 pct sub-metrics for one scope), not a
    # new queryable column. Detected on the raw message (see _area_match,
    # computed earlier above) so it composes with whatever dimension/name/
    # period/filters were already extracted above, rather than being a new
    # parallel intent.

    # Re-resolve dimension_name through the EXISTING fuzzy-safe extraction
    # functions against the ORIGINAL message - never trust the LLM's own
    # name transcription directly, same fallback_text pattern used
    # everywhere else in this file.
    name_text = extracted.get("dimension_name") or message
    name_filter = None
    name_label = None
    if dimension == "employee":
        try:
            emp_id, emp_name = entities.extract_employee(name_text, fallback_text=raw_message)
        except entities.Ambiguous as e:
            return (
                f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                [],
            )
        if not emp_id and extracted.get("dimension_name"):
            # LLM claimed a name but our resolver can't find it at all -
            # don't silently run a company-wide query under that name;
            # fail safe and let the cascade fall through instead.
            return None
        name_filter = emp_id
        name_label = emp_name or sticky.get("employee_name")
    elif dimension == "department":
        dept_name, candidates = entities.extract_department(name_text, fallback_text=raw_message)
        if candidates:
            return (
                f"Multiple departments match that name: {', '.join(candidates)}. Which one did you mean?",
                [],
            )
        if not dept_name and extracted.get("dimension_name"):
            return None
        name_filter = dept_name or sticky.get("dept_name")
        name_label = name_filter
    elif dimension == "rm":
        try:
            mgr_id, mgr_name = entities.extract_manager(name_text, fallback_text=raw_message)
        except entities.Ambiguous:
            mgr_id, mgr_name = None, None
        if not mgr_id and extracted.get("dimension_name"):
            return None
        name_filter = mgr_name
        name_label = mgr_name
    else:  # company
        name_filter = None
        name_label = "The whole company"

    # Item #84 (failures I/J, item #83 Phase 2 design section 4(iii)): a
    # follow-up naming "them"/"both"/"the two" - referring to TWO entities
    # named across the last couple of turns (e.g. "...compare them with the
    # employee who had the highest PACE..." then "what could explain the
    # difference, look at LC/EL/DH for both") - resolves against the new
    # comparison_entities 2-slot tracker (session_store.push_context() now
    # keeps this in sync automatically) instead of failing or silently
    # collapsing to just the single most-recent entity the way
    # sticky_context's one-slot employee_id/dept_name would. Only fires
    # when nothing was explicitly named THIS turn (no name_filter) and both
    # tracked entities are the SAME type as the requested dimension.
    _COMPARISON_PRONOUN = re.compile(r"\b(them|both|the two|either of them)\b", re.IGNORECASE)
    if (not name_filter and dimension in ("employee", "department")
            and _COMPARISON_PRONOUN.search(raw_message) and session is not None):
        _cmp_first, _cmp_second = session_store.get_comparison_entities(session)
        if (_cmp_first and _cmp_second
                and _cmp_first.get("type") == dimension and _cmp_second.get("type") == dimension):
            _cmp_rows = []
            for _ent in (_cmp_first, _cmp_second):
                try:
                    _r = queries.build_query(
                        dimension, metrics, filters=filters, period=period,
                        name_filter=_ent["id"], limit=1, latest_n_days=latest_n_days,
                    )
                except Exception:
                    logging.getLogger("pace_chatbot.main").exception(
                        "build_query() raised inside extraction-LLM cascade step (2-entity comparison)")
                    continue
                if _r:
                    _cmp_rows.append(_r[0])
            if _cmp_rows:
                reply = _format_build_query_rows(_cmp_rows, dimension, metrics, name_label=None)
                return reply, _cmp_rows
            # Both entities tracked but neither had data - fall through to
            # the rest of the cascade rather than returning a confusing
            # empty comparison.

    # Item #84 (failures G/H, item #83 section 1's confirmed conversational-
    # state gap): a SINGULAR referent right after a ranking answer ("the
    # employee with the lowest score", "which department had the highest")
    # should resolve against the query_context this cascade now maintains
    # (see the set_query_context() calls below), not re-run a fresh
    # company-wide ranking from scratch. Only applies when nothing else was
    # already resolved this turn (no name_filter) and the prior answer's
    # dimension matches, so this never overrides an explicit name/ranking
    # request in the CURRENT message.
    _qc = (session or {}).get("query_context") or {}
    _SINGULAR_RANKED_REFERENT = re.compile(
        r"\bthe (employee|department|person)\b[^.?!]{0,30}\bwith the (lowest|highest|best|worst)\b",
        re.IGNORECASE)
    _m_singular_ref = None if name_filter else _SINGULAR_RANKED_REFERENT.search(raw_message)
    if (_m_singular_ref and dimension in ("employee", "department")
            and _qc.get("last_dimension") == dimension and _qc.get("last_result_ids")):
        want_lowest = _m_singular_ref.group(2).lower() in ("lowest", "worst")
        ids = _qc["last_result_ids"]
        qc_ascending = _qc.get("ascending")
        if qc_ascending is True:
            idx = 0 if want_lowest else -1
        elif qc_ascending is False:
            idx = -1 if want_lowest else 0
        else:
            idx = 0
        if ids:
            name_filter = ids[idx]
            name_label = None  # let _format_build_query_rows read the resolved row's own name

    # Item #76 (Part B): RANKING operation - dimension in (employee,
    # department, rm) with NO name resolvable at all (nothing named in this
    # message, nothing in sticky context either) used to be a hard fail
    # (return None -> falls through to sql_fallback). If the raw message
    # actually reads as a ranking/superlative question ("highest"/"lowest"/
    # "best"/"worst"/"top"/"most"/"least"/etc.), that's not missing
    # information - it's a genuine multi-row ranking request the engine
    # already supports (build_query() with no name_filter returns a ranked,
    # LIMIT-ed table) but this cascade never exercised before this round.
    _RANKING_WORDS = re.compile(
        r"\b(highest|lowest|best|worst|top|bottom|most|least|greatest|smallest|"
        r"largest|highest-scoring|lowest-scoring)\b", re.IGNORECASE)
    _ASCENDING_WORDS = re.compile(r"\b(least|lowest|worst|fewest|bottom|smallest)\b", re.IGNORECASE)
    # Item #84 (Finding 2): the raw-message regex above remains the
    # AUTHORITATIVE, deterministic signal (same "narrow deterministic check
    # wins over a probabilistic guess" precedent as every other override in
    # this cascade) - but the extraction LLM's own `operation` field
    # (schema addition this round) is now consulted as an ADDITIONAL signal
    # for the cases the regex alone can't see, e.g. a ranking phrased with
    # no superlative WORD at all in English ("5 employees with the lowest
    # engagement" still matches "lowest" so the regex catches it - but this
    # also gives the LLM a chance to confirm/extend without ever being able
    # to override an explicit non-ranking regex read).
    _op = extracted.get("operation")
    is_ranking = (
        dimension in ("employee", "department", "rm")
        and not name_filter
        and not _area_match
        and (_RANKING_WORDS.search(raw_message) is not None or _op in ("rank_top", "rank_bottom"))
    )

    if dimension in ("employee", "department", "rm") and not name_filter and not is_ranking:
        # Nothing resolvable at all (no name in the message, no sticky
        # context either) and not a ranking question either - not enough to
        # run a scoped query; fall through.
        return None

    _AREA_METRICS = ["engagement_pct", "effectiveness_pct", "discipline_pct", "working_pct"]
    _AREA_LABELS = {
        "engagement_pct": "Engagement", "effectiveness_pct": "Effectiveness",
        "discipline_pct": "Discipline", "working_pct": "Working hours",
    }

    if (_area_match and dimension == "employee" and not name_filter
            and _qc.get("last_dimension") == "employee" and _qc.get("last_result_ids")
            and re.search(r"\b(their|them|those)\b", raw_message, re.IGNORECASE)):
        # Item #84 (failure G): "their weakest areas" right after a ranking
        # ("5 employees with lowest PACE" -> "what are their weakest
        # areas?") - resolves "their" against the query_context ranking
        # this cascade now maintains, and reports EACH employee's own
        # strongest/weakest area (not a single aggregated scope - "strongest/
        # weakest area" is inherently per-employee, and there is no existing
        # multi-employee aggregate version of this operation to reuse).
        want_weakest = _area_match.group(1).lower() == "weakest"
        direction = "weakest" if want_weakest else "strongest"
        _emp_ids = _qc["last_result_ids"][:queries.LIMIT]
        _group_rows = []
        for _eid in _emp_ids:
            try:
                _r = queries.build_query(
                    "employee", _AREA_METRICS, filters=filters, period=period,
                    name_filter=_eid, limit=1, latest_n_days=latest_n_days,
                )
            except Exception:
                logging.getLogger("pace_chatbot.main").exception(
                    "build_query() raised inside extraction-LLM cascade step (group strongest/weakest area)")
                continue
            if not _r:
                continue
            row = _r[0]
            present = [(k, row.get(k)) for k in _AREA_METRICS if row.get(k) is not None]
            if not present:
                continue
            chosen_key, chosen_val = (min if want_weakest else max)(present, key=lambda kv: float(kv[1]))
            _group_rows.append((row.get("emp_name"), _AREA_LABELS[chosen_key], chosen_val))
        if not _group_rows:
            return (f"No data found for that group of employees in this period.", [])
        headers = ["Employee", f"{direction.capitalize()} area", "Value"]
        data = [[name, area, f"{_fmt(val)}%"] for name, area, val in _group_rows]
        reply = f"{direction.capitalize()} area per employee (from the last ranking shown):\n\n" + _render_table(headers, data)
        return reply, []

    if _area_match and dimension in ("employee", "department", "rm") and not name_filter:
        # "strongest/weakest area" needs ONE concrete scope (a named entity,
        # or company-wide) - a bare ranking of "areas" across many
        # entities isn't a supported shape; fall through rather than guess.
        return None

    if _area_match:
        # Force the metrics to exactly the 4 comparable pct sub-metrics,
        # overriding whatever extract_build_query() guessed for `metrics` -
        # this operation is never about any other metric.
        try:
            rows = queries.build_query(dimension, _AREA_METRICS, filters=filters, period=period,
                                        name_filter=name_filter, limit=1, latest_n_days=latest_n_days)
        except Exception:
            logging.getLogger("pace_chatbot.main").exception(
                "build_query() raised inside extraction-LLM cascade step (strongest/weakest area)")
            return None
        if not rows:
            return (f"No data found for {name_label or 'that scope'} in this period.", [])
        row = rows[0]
        present = [(k, row.get(k)) for k in _AREA_METRICS if row.get(k) is not None]
        if not present:
            return (f"No data found for {name_label or 'that scope'} in this period.", [])
        want_weakest = _area_match.group(1).lower() == "weakest"
        best_key, best_val = max(present, key=lambda kv: float(kv[1]))
        worst_key, worst_val = min(present, key=lambda kv: float(kv[1]))
        chosen_key, chosen_val = (worst_key, worst_val) if want_weakest else (best_key, best_val)
        direction = "weakest" if want_weakest else "strongest"
        reply = (
            f"{name_label}'s {direction} area is {_AREA_LABELS[chosen_key]} "
            f"({_fmt(chosen_val)}%).\n"
            "All 4 areas: " + ", ".join(f"{_AREA_LABELS[k]} {_fmt(v)}%" for k, v in present)
        )
        if session is not None and dimension == "department" and name_filter:
            session_store.push_context(session, dept_name=name_filter)
        if session is not None and dimension == "employee" and name_filter:
            session_store.push_context(session, employee_id=name_filter, employee_name=name_label)
        return reply, rows

    if is_ranking:
        # Item #84 (Finding 1/2): the deterministic raw-message regex
        # (entities.extract_limit(), now broadened - see entities.py) is
        # checked FIRST and wins whenever it finds a count; only when it
        # finds nothing at all do we fall back to the LLM's own `limit`
        # field (already sanity-clamped 1-100 in llm_nlu.py), then finally
        # queries.LIMIT - same "narrow deterministic check wins" precedent
        # as every other field in this cascade.
        limit = entities.extract_limit(raw_message, default=None) or extracted.get("limit") or queries.LIMIT
        ascending = _ASCENDING_WORDS.search(raw_message) is not None or _op == "rank_bottom"
        try:
            rows = queries.build_query(dimension, metrics, filters=filters, period=period,
                                        name_filter=None, limit=limit, ascending=ascending,
                                        latest_n_days=latest_n_days)
        except Exception:
            logging.getLogger("pace_chatbot.main").exception(
                "build_query() raised inside extraction-LLM cascade step (ranking)")
            return None
        reply = _format_build_query_rows(rows, dimension, metrics, name_label=None)
        if session is not None:
            # Item #84 (confirmed root cause of failures G/H/L/M/N, item
            # #83 section 1): this branch previously never registered ANY
            # memory of the ranking it just produced - unlike every
            # rule-based ranking intent elsewhere in this file, which all
            # call set_last_list()/push_context(). Wired up the same way
            # here so a later "list them"/"show me their names" follow-up
            # (last_list) and a pronoun/referent follow-up (query_context)
            # can both resolve against this answer instead of a stale or
            # company-wide default.
            _id_key = {"employee": "employee_id", "department": "dept_name", "rm": "reporting_manager_name"}.get(dimension)
            _result_ids = [r.get(_id_key) for r in rows if r.get(_id_key) is not None] if _id_key else []
            session_store.set_last_list(
                session, kind="ranking", answer_kind="list",
                dept_name=None, employee_ids=_result_ids if dimension == "employee" else None,
                team_label=None, month=None, date_range=period, ascending=ascending,
            )
            session_store.set_query_context(
                session,
                last_operation="rank_bottom" if ascending else "rank_top",
                last_dimension=dimension, last_result_ids=_result_ids,
                ascending=ascending, metric=metrics, period_phrase=extracted.get("period_phrase"),
            )
        return reply, rows

    try:
        rows = queries.build_query(dimension, metrics, filters=filters, period=period,
                                    name_filter=name_filter, limit=1, latest_n_days=latest_n_days)
    except Exception:
        logging.getLogger("pace_chatbot.main").exception("build_query() raised inside extraction-LLM cascade step")
        return None

    reply = _format_build_query_rows(rows, dimension, metrics, name_label=name_label)
    if session is not None and dimension == "department" and name_filter:
        session_store.push_context(session, dept_name=name_filter)
    if session is not None and dimension == "employee" and name_filter:
        session_store.push_context(session, employee_id=name_filter, employee_name=name_label)
    if session is not None and name_filter:
        # Item #84 (item #83 Phase 2 design, section 4(iii)): the singular
        # branch previously only ever pushed dept_name/employee_id into
        # sticky_context, never the metric/period_phrase actually used -
        # so a pronoun follow-up like "them"/"their" after a single-entity
        # lookup had no way to recover WHAT was asked, only WHO. Recorded
        # here too (not just the ranking branch above) so both answer
        # shapes populate the same structured state.
        session_store.set_query_context(
            session, last_operation="value", last_dimension=dimension,
            last_result_ids=[name_filter], ascending=None,
            metric=metrics, period_phrase=extracted.get("period_phrase"),
        )
    return reply, rows


def format_gainer_loser_ranking(gainers, losers, meta, scope_note, directions, filter_footer):
    window_note = (
        f"Current period: {meta['cur_start']} to {meta['cur_end']} vs "
        f"prior period: {meta['prior_start']} to {meta['prior_end']}"
    )

    def _table(rows):
        if not rows:
            return "None (no rows matched)."
        headers = ["#", "Employee", "Department", "Current score", "Prior score", "Change (pts)"]
        data = []
        for i, r in enumerate(rows, 1):
            change = r["score_change"]
            sign = "+" if change is not None and change > 0 else ""
            data.append([
                i, r["emp_name"], r["dept_name"],
                _fmt(r["cur_score"]), _fmt(r["prior_score"]), f"{sign}{_fmt(change)}",
            ])
        return _render_table(headers, data)

    sections = []
    if "gainers" in directions:
        sections.append(f"Top Gainers:\n{_table(gainers)}")
    if "losers" in directions:
        sections.append(f"Top Losers:\n{_table(losers)}")
    if directions == ("gainers",):
        title = "Top gainers"
    elif directions == ("losers",):
        title = "Top losers"
    else:
        title = "Top gainers and losers"

    body = (
        f"{title}{scope_note} (last 4 complete weeks vs prior 4 complete weeks):\n"
        f"{window_note}\n\n" + "\n\n".join(sections) + f"\n\n{filter_footer}"
    )
    return body


_DAY_COMPARE_METRIC_KEYWORDS = [
    ("engagement", r"\bengagement\b"),
    ("discipline", r"\bdiscipline\b"),
    ("working", r"\bworking\b"),
    ("effectiveness", r"\beffectiveness\b"),
    ("pace", r"\bpace\b"),
]
_DAY_COMPARE_TIE_THRESHOLD = 1  # points; within this, report "essentially the same"


def _day_compare_metrics(message):
    """Detects which metric(s) are being asked about. 'all scores'/'all
    metrics' -> every metric, reported separately (never a combined/
    composite score, per the confirmed business rule). Otherwise, every
    explicitly-named metric keyword (order-preserving, de-duplicated) - or,
    if none named, the single confirmed default (new_pace_score_7_3_event_level,
    aliased 'pace')."""
    text_l = (message or "").lower()
    if re.search(r"\ball scores?\b|\ball metrics\b|\beverything\b", text_l):
        return list(queries.DAY_COMPARE_METRICS.keys())
    found = []
    for key, pattern in _DAY_COMPARE_METRIC_KEYWORDS:
        if re.search(pattern, text_l) and key not in found:
            found.append(key)
    return found or [queries.DEFAULT_DAY_COMPARE_METRIC]


def format_day_compare(results, date1, date2, subject_label, filter_footer=None):
    lines = [f"Comparing {subject_label} on {date1} vs {date2}:\n"]
    any_data = False
    for r in results:
        v1, v2 = r["val1"], r["val2"]
        if v1 is None or v2 is None:
            lines.append(f"{r['label']}: no data available for one or both of those days "
                         f"(0 matching rows) under the current filters.")
            continue
        any_data = True
        f1, f2 = _fmt(v1), _fmt(v2)
        diff = float(v1) - float(v2)
        if abs(diff) <= _DAY_COMPARE_TIE_THRESHOLD:
            verdict = "essentially the same, no meaningful difference"
        elif diff > 0:
            verdict = f"{date1} was better (+{_fmt(abs(diff))} pts)"
        else:
            verdict = f"{date2} was better (+{_fmt(abs(diff))} pts)"
        lines.append(f"{r['label']}: {date1} = {f1}, {date2} = {f2} — {verdict}")
    if filter_footer:
        lines.append("")
        lines.append(filter_footer)
    return "\n".join(lines)


def _handle_day_compare(message, raw_message, session):
    fb = raw_message if raw_message and raw_message != message else None
    d1, d2, found = entities.extract_two_dates(message)
    if not found:
        # No explicit dates in THIS message (e.g. a same-session follow-up
        # like "which day was more productive" with no restated dates) -
        # fall back to the last day-vs-day date pair explicitly compared
        # this session, if any.
        ctx_dates = session_store.get_recent_context(session, "day_compare_dates")
        if ctx_dates:
            d1, d2 = ctx_dates
        else:
            return ChatResponse(
                reply="Which two dates would you like me to compare — e.g. \"2 Sept vs 7 Sept\"?"
            )
    if d2 < d1:
        d1, d2 = d2, d1

    session_store.push_context(session, day_compare_dates=(d1, d2))

    metric_keys = _day_compare_metrics(message)

    # Employee-specific vs department-fixed vs full-company: try employee
    # first (a named person takes precedence over a department mention in
    # the same message being coincidental), then department, else full
    # company. Both use word-boundary-safe extraction with fallback_text,
    # same as every other entity lookup in this file.
    employee_id = None
    subject_label = "the full company"
    try:
        emp_id, emp_name = entities.extract_employee(message, fallback_text=fb)
    except entities.Ambiguous as e:
        return ChatResponse(
            reply=f"I found multiple matching employees: {', '.join(e.candidates)}. Which one did you mean?",
            needs_clarification=True, clarification_options=e.candidates,
        )
    dept_name, dept_candidates = entities.extract_department(message, fallback_text=fb)
    if dept_candidates:
        return ChatResponse(
            reply=f"I found multiple matching departments: {', '.join(dept_candidates)}. Which one did you mean?",
            needs_clarification=True, clarification_options=dept_candidates,
        )

    if emp_id:
        employee_id = emp_id
        subject_label = emp_name
        dept_name = None  # employee-specific takes precedence; don't also scope by dept
    elif dept_name:
        subject_label = dept_name

    filter_sql, filter_footer = _resolve_population_filter(message)
    results = queries.day_compare(d1, d2, dept_name=dept_name, employee_id=employee_id, metric_keys=metric_keys, filter_sql=filter_sql)
    if session is not None:
        session_store.set_last_answer_filters(
            session, label=f"the {_strip_leading_the(subject_label)} comparison ({d1} vs {d2})",
            **_default_filters_from_message(message))
    return ChatResponse(reply=format_day_compare(results, d1, d2, subject_label, filter_footer), rows=results)


def _month_label(month_str):
    """'2026-08' -> 'August 2026' - a readable label for format_day_compare's
    date1/date2 params, which are printed verbatim ("Comparing X on {date1}
    vs {date2}:") - reused as-is for month comparison rather than writing a
    parallel formatter, per the confirmed instruction to reuse the existing
    comparison response logic wherever it genuinely fits."""
    try:
        y, m = month_str.split("-")
        dt = datetime.date(int(y), int(m), 1)
        return dt.strftime("%B %Y")
    except (ValueError, AttributeError):
        return month_str


def _handle_month_compare(message, raw_message, session):
    """Company-wide (or dept/employee-scoped) month-vs-month comparison -
    the month-granularity sibling of _handle_day_compare above. Deliberately
    mirrors that function's structure almost line-for-line (dept/employee
    resolution, sticky-context fallback, metric detection, population
    filter, tie-threshold formatting) rather than inventing a parallel
    shape, per the explicit instruction to reuse as much of the existing
    day-comparison logic as fits."""
    fb = raw_message if raw_message and raw_message != message else None
    m1, m2, found = entities.extract_two_months(message)
    if not found:
        # Same sticky-follow-up pattern as day_compare: a same-session
        # follow-up naming no months ("was it better this time?") reuses
        # the last month pair explicitly compared this session, if any.
        ctx_months = session_store.get_recent_context(session, "month_compare_months")
        if ctx_months:
            m1, m2 = ctx_months
        else:
            return ChatResponse(
                reply="Which two months would you like me to compare — e.g. \"August vs July\"?"
            )
    if m2 < m1:
        m1, m2 = m2, m1

    session_store.push_context(session, month_compare_months=(m1, m2))

    # Reuses day_compare's own metric-keyword detector/METRICS dict
    # verbatim - same default (new_pace_score_7_3_event_level, aliased
    # "pace") and the same explicit-metric-name mapping, per the confirmed
    # business rule that this feature's default metric follows the exact
    # same rule as day-vs-day comparison.
    metric_keys = _day_compare_metrics(message)

    employee_id = None
    subject_label = "the full company"
    try:
        emp_id, emp_name = entities.extract_employee(message, fallback_text=fb)
    except entities.Ambiguous as e:
        return ChatResponse(
            reply=f"I found multiple matching employees: {', '.join(e.candidates)}. Which one did you mean?",
            needs_clarification=True, clarification_options=e.candidates,
        )
    dept_name, dept_candidates = entities.extract_department(message, fallback_text=fb)
    if dept_candidates:
        return ChatResponse(
            reply=f"I found multiple matching departments: {', '.join(dept_candidates)}. Which one did you mean?",
            needs_clarification=True, clarification_options=dept_candidates,
        )

    if emp_id:
        employee_id = emp_id
        subject_label = emp_name
        dept_name = None
    elif dept_name:
        subject_label = dept_name

    filter_sql, filter_footer = _resolve_population_filter(message)
    results = queries.month_compare(m1, m2, dept_name=dept_name, employee_id=employee_id,
                                     metric_keys=metric_keys, filter_sql=filter_sql)
    label1, label2 = _month_label(m1), _month_label(m2)
    if session is not None:
        session_store.set_last_answer_filters(
            session, label=f"the {_strip_leading_the(subject_label)} comparison ({label1} vs {label2})",
            **_default_filters_from_message(message))
    return ChatResponse(reply=format_day_compare(results, label1, label2, subject_label, filter_footer), rows=results)


def format_score_delta_ranking(rows, meta, header_prefix):
    """Table of employees ranked by CURRENT-MONTH-AVG vs PRIOR-MONTH-AVG
    PACE score change (month-over-month) — same MIN_DAYS_FOR_DELTA
    reliability gate and partial-month caution note as
    format_monthly_pace_trend, just ranking several employees instead of
    describing one. This is a month-over-month comparison, NOT a
    first-scored-day vs last-scored-day comparison within the period."""
    header = f"{header_prefix} (this month's avg vs prior month's avg PACE score)"
    if not rows:
        body = (
            f"No employees had at least {meta['min_days']} reliable Standard-shift days of data in "
            f"both this month and {meta['prev_month']} to measure a month-over-month change."
        )
        return f"{header}:\n\n{body}"

    def _cur_avg(r):
        if r["pace_score_prev_month"] is None or r["pace_score_delta"] is None:
            return None
        return r["pace_score_prev_month"] + r["pace_score_delta"]

    if len(rows) == 1:
        r = rows[0]
        delta = r["pace_score_delta"]
        sign = "+" if delta and delta > 0 else ""
        body = (
            f"{r['emp_name']} ({r['dept_name']}) — {sign}{_fmt(delta)} pts "
            f"(current month avg {_fmt(_cur_avg(r))}, prior month avg {_fmt(r['pace_score_prev_month'])})"
        )
    else:
        headers = ["#", "Employee", "Department", "Current month avg", "Prior month avg", "Change (pts)"]
        data = []
        for i, r in enumerate(rows, 1):
            delta = r["pace_score_delta"]
            sign = "+" if delta and delta > 0 else ""
            data.append([
                i, r["emp_name"], r["dept_name"],
                _fmt(_cur_avg(r)), _fmt(r["pace_score_prev_month"]), f"{sign}{_fmt(delta)}",
            ])
        body = _render_table(headers, data)

    reply = f"{header}:\n\n{body}"
    if meta["partial_month"]:
        reply += (
            "\n\nNote: this month is still in progress, so its average — and therefore this "
            "ranking — may shift as more days come in."
        )
    return reply


def format_subscore_delta_ranking(rows, meta, header_prefix):
    """Item #84 (Finding 3 follow-through): the sub-metric (engagement/
    effectiveness/discipline/working hours) counterpart of
    format_score_delta_ranking — same shape, same MIN_DAYS_FOR_DELTA
    reliability gate and partial-month caution note, but for
    queries.subscore_delta_ranking()'s cur_avg/prev_avg/delta columns
    instead of the precomputed pace_score_delta column."""
    header = f"{header_prefix} (this month's avg vs prior month's avg {meta['label']})"
    if not rows:
        body = (
            f"No employees had at least {meta['min_days']} reliable Standard-shift days of data in "
            f"both this month and {meta['prev_month']} to measure a month-over-month change."
        )
        return f"{header}:\n\n{body}"

    if len(rows) == 1:
        r = rows[0]
        delta = r["delta"]
        sign = "+" if delta and delta > 0 else ""
        body = (
            f"{r['emp_name']} ({r['dept_name']}) — {sign}{_fmt(delta)} pts "
            f"(current month avg {_fmt(r['cur_avg'])}%, prior month avg {_fmt(r['prev_avg'])}%)"
        )
    else:
        headers = ["#", "Employee", "Department", "Current month avg", "Prior month avg", "Change (pts)"]
        data = []
        for i, r in enumerate(rows, 1):
            delta = r["delta"]
            sign = "+" if delta and delta > 0 else ""
            data.append([
                i, r["emp_name"], r["dept_name"],
                f"{_fmt(r['cur_avg'])}%", f"{_fmt(r['prev_avg'])}%", f"{sign}{_fmt(delta)}",
            ])
        body = _render_table(headers, data)

    reply = f"{header}:\n\n{body}"
    if meta["partial_month"]:
        reply += (
            "\n\nNote: this month is still in progress, so its average — and therefore this "
            "ranking — may shift as more days come in."
        )
    return reply


def format_ranking_weekly_trend(rows, label):
    """Weekly counterpart to format_score_delta_ranking's monthly ranking —
    one row per employee per week, mirroring format_weekly_trend's
    single-employee table but across the whole ranked scope."""
    if not rows:
        return f"Not enough weekly data (need at least 2 scored days per week) to show a week-by-week trend for {label}."
    headers = ["Employee", "Week start", "Week end", "Avg score", "Change vs prior week"]
    data = []
    for r in rows:
        if r["delta"] is None:
            change = "N/A"
        else:
            sign = "+" if r["delta"] > 0 else ""
            change = f"{sign}{_fmt(r['delta'])} pts"
        data.append([r["emp_name"], r["week_start"], r["week_end"], _fmt(r["avg_score"]), change])
    return f"Weekly PACE score trend for {label}:\n\n" + _render_table(headers, data)


def _detect_day_flag(message):
    """Which queries.DAY_FLAGS key the message is asking about — checked in
    priority order (most specific keyword first) since several keywords
    could otherwise collide (e.g. 'attendance' as a generic word appearing
    inside an OT/leave question). Returns None if no flag keyword matched."""
    ml = message.lower()
    checks = [
        ("wfh", [r"\bwfh\b", r"work(ed|ing)? from home"]),
        ("visit", [r"\bvisit(s|ed|ing)?\b", r"client visit"]),
        ("overtime", [r"\bovertime\b", r"\bot\b"]),
        ("deficient_hours", [r"\bdeficient hours?\b", r"\bdeficit hours?\b"]),
        ("early_leave", [r"\bearly\b", r"left early"]),
        ("late", [r"\blate\b", r"came late"]),
        ("offline", [r"\boffline\b"]),
        ("ps_not_installed", [r"ps not installed", r"ps missing", r"no ps installed",
                               r"ps not installed on (their|his|her) device", r"don'?t have ps installed",
                               r"didn'?t have ps installed", r"missing ps\b"]),
        ("ps_worked", [r"\bps working\b", r"ps worked", r"worked \(ps"]),
        ("zero_productive", [r"zero productive"]),
        ("called_clients", [r"\bcalled? clients?\b", r"\bcalls?\b"]),
        ("had_meetings", [r"\bmeetings?\b"]),
        ("completed_tasks", [r"\bcompleted tasks?\b", r"\btasks?\b"]),
        ("leave", [r"\bon leave\b", r"\btook leave\b", r"\bleave\b"]),
        ("absent", [r"\babsent\b", r"didn'?t (punch|attend)", r"zero attendance"]),
        ("defaulter", [r"\bdefaulter\b"]),
        ("attendance", [r"\battend(ed|ance)?\b", r"\bpresent\b", r"\bpunch(ed)?\b"]),
    ]
    for key, patterns in checks:
        for p in patterns:
            if re.search(p, ml):
                return key
    return None


_DAY_FLAG_ANSWER_VERB = {
    "attendance": "punched attendance", "absent": "were absent", "leave": "were on leave",
    "wfh": "were on WFH", "visit": "were on a client visit", "late": "came late",
    "early_leave": "left early", "overtime": "did overtime", "deficient_hours": "were marked deficient hours",
    "defaulter": "were marked defaulter", "offline": "were marked offline attendance",
    "ps_not_installed": "did not have PS installed",
    "ps_worked": "had PS working", "zero_productive": "had zero productive minutes",
    "called_clients": "made calls", "had_meetings": "had meetings", "completed_tasks": "had task activity",
}


def _period_label_for_range(date_range):
    """Human label for a resolved (start,end) date_range — 'yesterday'/'today'
    when it matches those special single days, else the literal date(s)."""
    start, end = date_range
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    if start == end == today:
        return "today"
    if start == end == yesterday:
        return "yesterday"
    if start == end:
        return f"on {start}"
    return f"from {start} to {end}"


def format_day_count(result, flag_key, scope_note):
    verb = _DAY_FLAG_ANSWER_VERB.get(flag_key, flag_key)
    return f"{result['n']} employee(s) {verb}{scope_note} (out of {result['total']} total present employees)."


def format_day_list(rows, flag_key, scope_note):
    verb = _DAY_FLAG_ANSWER_VERB.get(flag_key, flag_key)
    # _DAY_FLAG_ANSWER_VERB values are written for a plural subject ("were on
    # WFH", "did overtime") — "No one" is singular, so swap in the matching
    # singular verb form for the empty-result sentence specifically.
    _singular = {
        "punched attendance": "punched attendance", "were absent": "was absent", "were on leave": "was on leave",
        "were on WFH": "was on WFH", "were on a client visit": "was on a client visit", "came late": "came late",
        "left early": "left early", "did overtime": "did overtime", "were marked deficient hours": "was marked deficient hours",
        "were marked defaulter": "was marked defaulter", "were marked offline attendance": "was marked offline attendance",
        "did not have PS installed": "did not have PS installed",
        "had PS working": "had PS working", "had zero productive minutes": "had zero productive minutes",
        "made calls": "made calls", "had meetings": "had meetings", "had task activity": "had task activity",
    }
    if not rows:
        return f"No one {_singular.get(verb, verb)}{scope_note}."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) {verb}{scope_note} ({r['matching_days']} day(s))."
    headers = ["#", "Employee", "Department", "Days"]
    data = [[i, r["emp_name"], r["dept_name"], r["matching_days"]] for i, r in enumerate(rows, 1)]
    return f"Employees who {verb}{scope_note}:\n\n" + _render_table(headers, data)


def format_status_list(rows, statuses, scope_note):
    # statuses=None/empty means NO status filter (item #59) - "all" rather
    # than the old hardcoded Red/Black default.
    label = "/".join(statuses) if statuses else "all statuses"
    subject = f"currently {label}" if statuses else "listed (all statuses)"
    if not rows:
        return f"No one is {subject}{scope_note}."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — {r['overall_std_pace_status']}"
    headers = ["#", "Employee", "Department", "Status"]
    data = [[i, r["emp_name"], r["dept_name"], r["overall_std_pace_status"]] for i, r in enumerate(rows, 1)]
    return f"Employees {subject}{scope_note}:\n\n" + _render_table(headers, data)


def format_status_distribution(rows):
    if not rows:
        return "No data found."
    headers = ["#", "Department", "Employees", "Black", "Red", "Amber", "Green", "Red+Black %"]
    data = []
    for i, r in enumerate(rows, 1):
        red_black_pct = round((r["black_pct"] or 0) + (r["red_pct"] or 0))
        data.append([i, r["dept_name"], r["n_employees"], r["black_n"], r["red_n"], r["amber_n"], r["green_n"], red_black_pct])
    return _render_table(headers, data)


def format_status_transitions(rows, from_status, to_status):
    label = f"{from_status or 'any'} -> {to_status or 'any'}"
    if not rows:
        return f"No employees found with a status transition {label} between last month and this month."
    headers = ["#", "Employee", "Department", "Prior month status", "Current month status"]
    data = [[i, r["emp_name"], r["dept_name"], r["prev_status"], r["cur_status"]] for i, r in enumerate(rows, 1)]
    return f"Status transitions ({label}):\n\n" + _render_table(headers, data)


def format_full_trend(rows, label, metric_label="PACE score"):
    if not rows:
        return f"No monthly history found for {label}."
    headers = ["Month", metric_label, "Days/Employees"]
    data = []
    for r in rows:
        count_field = r.get("days_counted", r.get("n_employees"))
        data.append([r["mo"], _fmt(r["metric_value"]), count_field])
    return f"Month-on-month {metric_label} trend for {label}:\n\n" + _render_table(headers, data)


def _detect_subscore_key(message, default="engagement", include_working_hours=False):
    """Which of engagement/effectiveness/discipline(/working_hours) the user's
    text mentions; falls back to `default` if none named explicitly."""
    ml = message.lower()
    keys = ("engagement", "effectiveness", "discipline") + (("working_hours",) if include_working_hours else ())
    for k in keys:
        term = "working hours" if k == "working_hours" else k
        if term in ml:
            return k
    return default


# Keyword -> metric key for the "month wise" full-trend intent, covering
# every metric this round generalized beyond overall PACE score (see
# queries.employee_full_monthly_trend / queries.employee_monthly_count_trend
# and queries.COUNT_METRICS). Checked in order, most specific phrase first
# within each entry, first match wins - falls back to "pace_score" (the
# pre-existing default) when no metric keyword is named at all.
_FULL_TREND_METRIC_KEYWORDS = [
    ("wfh", ("wfh", "work from home")),
    ("visit", ("visit",)),
    ("leave", ("leave",)),
    ("deficient_hours", ("deficient hour", "deficient-hour")),
    ("late_comings", ("late coming", "late-coming", "late comings", "came late", "come late", "coming late", "was late", "late-comings", "arrive late", "arrived late")),
    ("early_leavings", ("early leaving", "early leavings", "left early", "leaving early")),
    ("ot_hours", ("ot hours", "overtime hours")),
    ("ot_days", ("ot day", "overtime day", " ot ", "overtime")),
    ("engagement", ("engagement",)),
    ("effectiveness", ("effectiveness",)),
    ("discipline", ("discipline",)),
    ("working_pct", ("working hours", "working %", "working percentage", "working pct")),
]


# Maps an _EMP_FIELD_INTENTS intent name to the matching queries.py
# PS_FILTERED_METRICS key, for the Category C proactive PS-off caveat below.
# Only intents with a real PS-filterable metric are included here.
_PS_CAVEAT_METRIC_BY_INTENT = {
    "emp_engagement": "engagement",
    "emp_effectiveness": "effectiveness",
    "emp_discipline": "discipline",
    "emp_working_pct": "working_hours",
    "emp_productive_time": "productive_min",
}

_PS_METRIC_KEYWORDS = [
    ("pace_score", ["pace score", "score"]),
    ("engagement", ["engagement"]),
    ("effectiveness", ["effectiveness"]),
    ("discipline", ["discipline"]),
    ("working_hours", ["working hours", "working %", "working pct"]),
    ("whatsapp_min", ["whatsapp"]),
    ("ai_min", ["ai tool", "ai usage", "ai minutes"]),
    ("tools_and_mails_min", ["tools and mail", "tools & mail", "mail minutes"]),
    ("productive_min", ["productive minutes", "productive time", "productivity"]),
]


def _detect_ps_metric(message):
    """Same 'keyword resolver over a fixed message' pattern as
    _detect_full_trend_metric below, but for PS-exclusion queries
    (queries.PS_FILTERED_METRICS keys). Defaults to engagement, the most
    common metric named in the PS-exclusion test spec, when nothing more
    specific is named."""
    ml = f" {message.lower()} "
    for key, terms in _PS_METRIC_KEYWORDS:
        for t in terms:
            if t in ml:
                return key
    return "engagement"


def _ps_off_caveat(emp_id, month, date_range):
    """Category C: a proactive data-quality note appended to a normal
    (non-PS-filtered) engagement/effectiveness/etc. answer, but ONLY when
    PS-off days are a MEANINGFUL fraction of the period - otherwise this
    would fire on nearly every query and just be noise. Threshold and
    rationale documented alongside queries.PS_OFF_CAVEAT_RATIO /
    PS_OFF_CAVEAT_MIN_DAYS. Never raises/crashes the main answer - a lookup
    failure here just means no caveat is appended."""
    try:
        ratio_row = queries.ps_working_ratio(emp_id, month=month, date_range=date_range)
    except Exception:
        return ""
    if not ratio_row or not ratio_row.get("total_days"):
        return ""
    total = ratio_row["total_days"]
    off = ratio_row.get("ps_off_days") or 0
    if total < queries.PS_OFF_CAVEAT_MIN_DAYS:
        return ""
    if (off / total) < queries.PS_OFF_CAVEAT_RATIO:
        return ""
    pct = round((off / total) * 100)
    return (f"\n\n(Note: PS wasn't working on {off} of {total} days ({pct}%) in this period — "
            f"those days' numbers may be unreliable. Ask me to \"exclude PS non-working days\" "
            f"if you'd like the number recalculated without them.)")


# Maps a metric_key resolved by _detect_full_trend_metric() back to the
# single-value per-employee intent that answers the SAME metric for one
# explicit period (as opposed to a month-by-month breakdown). Used by the
# explicit-month override in handle_message below: only metrics with a
# known single-value counterpart here are eligible to be forced out of
# full_trend_emp when the user names an explicit month/date range. Metrics
# with no defined single-value emp intent (ot_hours/ot_days/pace_score) are
# deliberately left out - full_trend_emp remains their only path, so the
# override leaves those alone rather than guessing a mapping that doesn't
# exist.
_FULL_TREND_METRIC_TO_SINGLE_INTENT = {
    "wfh": "wfh_emp",
    "visit": "visit_emp",
    "leave": "leave_emp_check",
    "deficient_hours": "emp_deficient_hours",
    "late_comings": "emp_late_comings",
    "early_leavings": "emp_early_leavings",
    "engagement": "emp_engagement",
    "effectiveness": "emp_effectiveness",
    "discipline": "emp_discipline",
    "working_pct": "emp_working_pct",
}


def _detect_full_trend_metric(message):
    # Pad with spaces so word-ish tokens like " ot " don't need extra regex
    # machinery to avoid matching inside another word.
    ml = f" {message.lower()} "
    for key, terms in _FULL_TREND_METRIC_KEYWORDS:
        for t in terms:
            if t in ml:
                return key
    return "pace_score"


def format_trend_2month(rows, emp_name):
    if not rows:
        return f"Not enough data to show a 2-month trend for {emp_name}."
    header_line = f"PACE score trend for {emp_name}:"
    if len(rows) == 1:
        r = rows[0]
        sign = "+" if r["delta"] and r["delta"] > 0 else ""
        return f"{header_line}\n{r['mo']}: changed by {sign}{_fmt(r['delta'])} pts (prior-month avg {_fmt(r['prev_avg'])}), {r['days']} days"
    headers = ["Month", "Change (pts)", "Prior month avg", "Days"]
    data = []
    for r in rows:
        sign = "+" if r["delta"] and r["delta"] > 0 else ""
        data.append([r["mo"], f"{sign}{_fmt(r['delta'])}", _fmt(r["prev_avg"]), r["days"]])
    return header_line + "\n\n" + _render_table(headers, data)


def format_monthly_pace_trend(row, meta, emp_name):
    """Individual month-over-month trend answer for 'is X improving' -
    default response before offering the weekly breakdown."""
    if not row:
        return (
            f"Not enough reliable data to judge whether {emp_name} is improving this month "
            f"(needs at least {meta['min_days']} Standard-shift days in both this month and "
            f"{meta['prev_month']})."
        )
    delta = row["pace_score_delta"]
    sign = "+" if delta and delta > 0 else ""
    direction = "improving" if delta > 0 else "declining" if delta < 0 else "flat"
    lines = [
        f"{emp_name} is {direction} this month: {sign}{_fmt(delta)} pts "
        f"(prior month avg {_fmt(row['pace_score_prev_month'])}, based on {row['days_current_month']} "
        f"days this month vs {row['days_prev_month']} days last month)."
    ]
    if meta["partial_month"]:
        lines.append(
            "Note: this month is still in progress, so its average — and therefore this "
            "change — may shift as more days come in."
        )
    return "\n".join(lines)


def format_weekly_trend(rows, emp_name):
    if not rows:
        return f"Not enough weekly data (need at least 2 scored days per week) to show a week-by-week trend for {emp_name}."
    header_line = f"Weekly PACE score trend for {emp_name}:"

    def _week_change(r):
        if r["delta"] is None:
            return "N/A"
        sign = "+" if r["delta"] > 0 else ""
        direction = "up" if r["delta"] > 0 else "down" if r["delta"] < 0 else "flat"
        return f"{direction} {sign}{_fmt(r['delta'])} pts"

    if len(rows) == 1:
        r = rows[0]
        return f"{header_line}\n{r['week_start']} to {r['week_end']}: avg {_fmt(r['avg_score'])} ({_week_change(r)} vs prior week, {r['scored_days']} scored days)"
    headers = ["Week start", "Week end", "Avg score", "Change vs prior week", "Scored days"]
    data = [[r["week_start"], r["week_end"], _fmt(r["avg_score"]), _week_change(r), r["scored_days"]] for r in rows]
    return header_line + "\n\n" + _render_table(headers, data)


# Category B/C metric intents -> (METRICS key, ascending)
_METRIC_INTENTS = {
    "pace_score_best": ("pace_score", False),
    "pace_score_worst": ("pace_score", True),
    "engagement_high": ("engagement", False),
    "engagement_low": ("engagement", True),
    "effectiveness_high": ("effectiveness", False),
    "effectiveness_low": ("effectiveness", True),
    "most_late_comings": ("late_comings", False),
    "most_early_leavings": ("early_leavings", False),
    "most_deficient_hours": ("deficient_hours_days", False),
    "most_disciplined": ("discipline", False),
    "most_whatsapp": ("whatsapp_min", False),
    "defaulter_ranking": ("defaulter_days", False),
    "deficit_hours_ranking": ("deficient_hours_days", False),
    # gap-category-E superlative synonyms (see intents.py additions)
    "fewest_late_comings": ("late_comings", True),
    "least_disciplined": ("discipline", True),
    "lowest_working_pct": ("working_pct", True),
    "highest_working_pct": ("working_pct", False),
}

# Category A single-employee intents -> which field(s) of employee_detail to
# highlight in a short, targeted reply (falls back to the full detail block
# for a couple of intents that don't map to one specific field).
_EMP_FIELD_INTENTS = {
    "emp_pace_score": ("PACE score", lambda r: f"{_fmt(r['avg_pace_score'])} (changed by {_fmt(r['pace_score_delta'])} pts vs prior month)"),
    "emp_late_comings": ("Late-comings", lambda r: str(r['total_lc'])),
    "emp_early_leavings": ("Early leavings", lambda r: str(r['total_el'])),
    "emp_productive_time": ("Avg productive min/day", lambda r: _avg_per_day(r['total_productive_min'], r['days_worked'])),
    "emp_whatsapp": ("Avg WhatsApp min/day", lambda r: _avg_per_day(r['total_whatsapp_min'], r['days_worked'])),
    "emp_ai_usage": ("Avg AI min/day", lambda r: _avg_per_day(r['total_ai_min'], r['days_worked'])),
    "emp_discipline": ("Discipline %", lambda r: _fmt(r['avg_discipline_pct'])),
    "emp_engagement": ("Engagement %", lambda r: _fmt(r['avg_engagement_pct'])),
    "emp_effectiveness": ("Effectiveness %", lambda r: _fmt(r['avg_effectiveness_pct'])),
    "emp_deficient_hours": ("Deficient-hour days", lambda r: str(r['total_dh'])),
    "emp_working_pct": ("Working hours %", lambda r: _fmt(r['avg_working_pct'])),
    "emp_department": ("Department", lambda r: r['dept_name']),
    "emp_manager": ("Reporting manager", lambda r: r['reporting_manager_name']),
}

# Maps an _EMP_FIELD_INTENTS key to its queries.METRICS-family equivalent,
# for the "no named employee, but this is clearly a bulk/all-employees
# request" fallback below (e.g. "give me score of all the employees of AI
# Labs department") - previously these intents ONLY had a single-employee
# lookup path, so a bulk/plural phrasing with no named employee just failed
# with "couldn't find that employee" instead of returning a ranking/list for
# the scope in play. emp_department/emp_manager are deliberately excluded -
# neither has a numeric METRICS counterpart to rank by.
_EMP_FIELD_TO_METRIC_KEY = {
    "emp_pace_score": "pace_score",
    "emp_late_comings": "late_comings",
    "emp_early_leavings": "early_leavings",
    "emp_productive_time": "productive_min",
    "emp_whatsapp": "whatsapp_min",
    "emp_ai_usage": "ai_min",
    "emp_discipline": "discipline",
    "emp_engagement": "engagement",
    "emp_effectiveness": "effectiveness",
    "emp_deficient_hours": "deficient_hours_days",
    "emp_working_pct": "working_pct",
}

_BULK_ALL_EMPLOYEES_PATTERN = re.compile(
    r"\ball\b.*\bemployees?\b|\bemployees?\b.*\ball\b|\beveryone\b|\beach employee\b", re.IGNORECASE
)

# Intents that always answer about ONE named individual employee (resolved
# via entities.extract_employee), never a team/department scope. These must
# bypass the extract_manager()/resolve_named_person_team() routing in
# handle_message entirely: that routing exists for genuine team-scope
# queries ("my team", "[name]'s team", department-less rankings), and
# previously ran unconditionally for every intent - so asking about a single
# employee who ALSO happens to be a manager (or a universal-access admin
# email) incorrectly triggered the "this person has admin access" prompt
# instead of just answering about that person. Keep this list in sync with
# every branch in answer_intent() that starts by calling
# entities.extract_employee(message, ...) directly.
#
# Deliberately EXCLUDES ot_subscore/wfh_subscore/ps_worked_ranking: those
# three intents genuinely serve BOTH an individual lookup ("OT engagement
# for Megha Sharma") AND a team/dept ranking ("OT engagement for Nikhil
# Kumar's team" / "most days PS not working") under the same intent name -
# their handler in answer_intent() already tries extract_employee(message)
# FIRST and only falls back to the dept/employee_ids ranking if no employee
# matched, so routing them through the manager/admin-check path is still
# correct for their ranking phrasing and doesn't reintroduce this bug for
# their individual phrasing (extract_employee still wins there regardless
# of what team-resolution also happened to find).
_INDIVIDUAL_EMP_INTENTS = set(_EMP_FIELD_INTENTS) | {
    "employee_day_summary",
    "emp_attendance_summary", "emp_trend", "emp_trend_2month", "emp_overview",
    "subscore_compare_emp", "subscore_trend_emp", "d_score_trend", "d_score_emp",
    "leave_emp_check", "call_emp", "visit_emp", "wfh_emp",
    "shift_type_emp", "breakshift_emp", "offline_emp", "meeting_ratio_emp", "meeting_had_emp",
    "ps_worked_emp",
    # NEW capability 3: single-employee full multi-month trend and the
    # single-employee "what status is X in" lookup are individual-scoped
    # (extract_employee resolves the person directly) — must bypass the
    # team/admin routing exactly like every other entry in this set.
    "full_trend_emp", "status_emp",
    # Part 3 (PS exclusion): ps_exclude_metric always targets a named
    # employee in this implementation (no department-wide PS-filtered
    # ranking yet), and ps_explain names no employee at all - both must
    # bypass team/manager routing the same as every other individual-scoped
    # intent above.
    "ps_exclude_metric", "ps_explain",
}

# Genuinely dual-purpose: same intent name covers both an individual lookup
# ("OT engagement for Megha Sharma") and a team/dept ranking ("OT engagement
# for Nikhil Kumar's team", "most days PS not working"). These can't be
# bypassed by intent name alone - handle_message instead probes
# extract_employee() on the raw message before deciding whether to skip the
# manager/team routing (see the bypass logic there).
_DUAL_PURPOSE_EMP_INTENTS = {"ot_subscore", "wfh_subscore", "ps_worked_ranking", "ps_ratio_info"}

# Department-vs-department / RM-team-vs-RM-team ranking intents are
# inherently company-wide comparisons across ALL departments (or ALL RM
# teams) — queries.dept_ranking()/rm_ranking() don't even take a dept_name
# scope. A sticky department from a prior turn (e.g. "AI Labs" locked in by
# an earlier employee ranking) must never leak into these — asking "which
# dept has the most score" or "which RM team has the most score" is itself
# a signal of a different ranking DIMENSION, not a continuation of the
# prior scoped employee ranking, so it's excluded from the dept-context
# fallback below exactly like the individual-employee intents are.
_DEPT_LEVEL_RANKING_INTENTS = {"dept_best", "dept_worst", "dept_avg", "rm_ranking_best", "rm_ranking_worst"}

# Pronoun-referring-to-a-person detection ("is he improving?" as a follow-up
# to "aryan gupta score"). Deliberately kept as a RULE-BASED, deterministic
# check rather than relying on Gemini to infer this from conversation
# context: the LLM layer (llm_nlu.py) sees only the raw message, with no
# session history in its prompt, so on a pronoun-only follow-up with no named
# employee it has nothing to disambiguate from and tends to confidently guess
# the generic org-wide ranking intent (e.g. "improving") instead of the
# individual-employee intent (e.g. "emp_trend") - even though the rule-based
# regex matcher (intents.match_intent) already correctly identifies these as
# individual-shaped ("\bis .* (improving|declining)\b" matches "is he
# improving" regardless of whether a name is present). When a pronoun is
# present AND the rule-based matcher landed on an _INDIVIDUAL_EMP_INTENTS
# entry, that rule-based intent is trusted over whatever the LLM proposed -
# see the override in handle_message. This keeps the fix independent of
# Gemini's availability/latency/prompt tuning entirely.
_PRONOUN_PATTERN = re.compile(r"\b(he|she|him|her|his|their|they|them)\b", re.IGNORECASE)

# Item #72: deterministic new-vocabulary override (see handle_message) - a
# message matching any of these should never be answered by an OLD intent
# that has no concept of the distinct new metric being asked for. Kept
# narrow/explicit (same "narrow regex beats probabilistic LLM guess" pattern
# already established by _ps_override/_gainer_loser_override/etc. above) so
# it can't false-positive on unrelated questions.
_NEW_VOCAB_OVERRIDE_PATTERN = re.compile(
    r"\bcapped (engagement|effectiveness|discipline)\b"
    r"|\b(day|event)[- ]level\b.*\bpace score\b"
    r"|\bpace score\b.*\b(day|event)[- ]level\b"
    r"|\bprecomputed\b.*\b(dept|department)?\s*(score|status)\b"
    r"|\bpace status\b.*\b(over|for|last|past|this|next)\b.*\b(day|days|week|weeks|month|months)\b"
    r"|\bderived\b.*\b(dept|department)\b.*\bstatus\b"
    # Item #76 (Phase 3, Part A): bare "last/past N days" phrasing (with or
    # without a filter-word right before "days", e.g. "last 10 WFH days").
    # Live-verified gap (SESSION_HANDOFF item #76): no existing rule-based
    # intent or entities.py parser handled this shape at all before this
    # round - "PACE status for AI Labs last 5 days" and "...last 60 days"
    # previously returned byte-identical replies (silently unparsed, always
    # falling back to build_query()'s 60-day default). A rule_intent/
    # llm_result match on a message like this is therefore never trustworthy
    # for the period it silently applies - null it out so extract_build_query()
    # (which now understands entities.extract_last_n_days()) gets the turn.
    r"|\b(?:last|past)\s+\d{1,3}\s+(?:\w+\s+)?days?\b"
    # Item #76 (Part B): "strongest/weakest area" - a NEW derived operation
    # (ranking the 4 pct sub-metrics for one scope) with no equivalent
    # concept in any of the ~123 existing intents; must reach the
    # extraction cascade, never an old ranking/percentage intent.
    r"|\b(strongest|weakest)\b[^.?!]{0,40}\b(area|metric|dimension|aspect)\b"
    r"|\b(area|metric|dimension|aspect)\b[^.?!]{0,40}\b(strongest|weakest)\b"
    # Item #79: engagement_minutes/tasks_*/todos_*/meeting_count are new
    # BUILD_QUERY_METRICS entries with NO equivalent concept in any of the
    # ~123 existing intents - live-verified classify() (the LLM intent
    # classifier, independent of the rule-based regex matcher and of
    # extract_build_query()) still confidently guesses an old bare-percentage
    # intent for "engagement minutes for X" (emp_engagement, ignoring
    # "minutes" entirely and returning engagement_pct) since nothing
    # previously told it not to. Null both rule_intent (harmless - none of
    # these bare phrasings match any existing rule-based pattern, confirmed)
    # and llm_result so extract_build_query() gets the turn instead. Bare
    # "meeting minutes"/"meeting count" deliberately NOT included here (would
    # incorrectly null the legitimate org-wide meeting_min_ranking/
    # meeting_count_ranking use case that has no named employee at all) -
    # the meeting-minutes collision is handled by the narrower avg-word-
    # conditioned redirect near "meeting_min_ranking" above instead.
    r"|\bengagement\s*minutes?\b"
    r"|\btasks?\s*(created|assigned)\b"
    r"|\btodos?\s*(created|assigned)\b",
    re.IGNORECASE,
)


def _repair_new_vocab_metric(raw_message):
    """Item #72: deterministic keyword-based repair used ONLY when
    extract_build_query() returned an empty/all-invalid metrics list -
    cheap last check for the exact new-vocabulary markers
    _NEW_VOCAB_OVERRIDE_PATTERN is built from, so a message that got here
    BECAUSE it named one of these new metrics doesn't silently end up
    answered with the unrelated default "pace_score" instead. Order matters
    (most specific phrase first) since a message could contain multiple
    markers; returns [] (not a default) if nothing matches, deferring to
    the caller's own ["pace_score"] fallback."""
    text_l = (raw_message or "").lower()
    # Item #73: delegates to the shared capped-vs-percentage normalizer
    # instead of the 3 old hardcoded "capped X" -> capped_X branches, which
    # ignored the "capped X percentage" -> X_pct business rule entirely.
    _pct_capped = _detect_pct_capped_metrics(raw_message)
    if _pct_capped:
        return _pct_capped
    if re.search(r"\bprecomputed\b", text_l) and re.search(r"\bstatus\b", text_l):
        return ["dept_status_60_days_derived"]
    if re.search(r"\bprecomputed\b", text_l):
        return ["dept_score_60_days_precomputed"]
    if re.search(r"\bderived\b.*\bstatus\b", text_l):
        return ["dept_status_60_days_derived"]
    if re.search(r"\b(day|event)[- ]level\b", text_l):
        return ["pace_score_day_level"]
    if re.search(r"\bpace status\b|\bstatus banding\b", text_l):
        return ["pace_status"]
    return []

# Part 3 ("did you mean X?" cascade) confidence threshold: the LLM's own
# self-reported `confidence` (0.0-1.0, see llm_nlu.py's _SYSTEM_PROMPT) is
# informative but not perfectly calibrated - it's a single model's own guess
# about its own guess, not a verified accuracy rate. 0.6 was chosen as a
# middle-of-the-road cutoff: below it, the few-shot examples in llm_nlu.py
# that use confidence < 0.6 are deliberately the genuinely-vague/casual ones
# ("give me a rundown of absenteeism" = 0.55) where the LLM itself signals
# real uncertainty, while the bulk of concrete/well-covered phrasings score
# 0.8+. Treating anything below this as "not confident enough to answer
# directly" errs toward trying the (harmless, clearly-labeled) SQL-fallback
# path more often rather than risking a wrong direct answer on a shaky guess.
_LLM_LOW_CONFIDENCE_THRESHOLD = 0.6

# "beside X"/"except X"/"excluding X"/"other than X" — a query naming one
# employee but asking to EXCLUDE them from an otherwise org/dept-wide list
# ("beside muskan who all did visit yesterday"). This is a genuinely new
# capability (no prior exclusion concept existed anywhere in the intent/
# query layer) built at the day_flag_list level (leave/call/visit/wfh "who"
# list queries), the same query path plural "who all ..." queries already
# reroute to. Distinguished from a plain individual lookup so that naming
# someone here does NOT resolve to an individual-employee answer about them.
_EXCLUDE_PATTERN = re.compile(r"\b(?:beside|besides|except|excluding|other than)\b", re.IGNORECASE)


# --- Conversational context carry-forward (feature) -------------------
# Intents with a fixed "current month vs prior month" trend/delta
# methodology must NOT inherit a carried-forward time period from context -
# their period semantics are intentionally anchored to "now" regardless of
# what was discussed earlier in the conversation. Keeping this as an
# explicit blacklist (rather than trying to guess per-intent) so it's easy
# to audit and matches the regression-safety requirement.
_PERIOD_CONTEXT_BLACKLIST = {
    "emp_trend", "emp_trend_2month", "score_drop_ranking", "score_improvement_alltime",
    "d_score_trend", "dept_trend", "team_improving", "status_transitions",
    "full_trend_emp", "full_trend_dept", "full_trend_team",
}


def _extract_employee_ctx(message, fb, session):
    """Same as entities.extract_employee(message, fallback_text=fb), but
    falls back to the session's recent conversational context (last 3-5
    turns) when the CURRENT message names no employee at all — e.g. "tell
    me about Megha Sharma" followed by "what about their attendance" (no
    name repeated). Only ever used as a fallback: if the current message
    resolves an employee (or raises Ambiguous), that always wins. Only
    called from branches that already know the intent is individual-
    employee-shaped (see _INDIVIDUAL_EMP_INTENTS), so inheriting an
    employee here can never bleed into an unrelated org-wide ranking."""
    emp_id, emp_name = entities.extract_employee(message, fallback_text=fb)
    if emp_id is not None and session is not None:
        # Record this REAL mention (not an inherited one) into the rolling
        # context history, so a later follow-up in this session can inherit
        # this employee if it names none of its own.
        session_store.push_context(session, employee_id=emp_id, employee_name=emp_name)
        return emp_id, emp_name
    # Sticky-context inheritance is ONLY safe when the CURRENT message
    # actually refers back to a person (a pronoun like "he"/"she"/"they",
    # or a possessive "their") - see the docstring example "what about
    # their attendance". An org-wide/plural query with no name and no
    # pronoun at all - "who all are on visit yesterday", "give me name of
    # all the employees that were on visit yesterday" - must NEVER inherit
    # the last-discussed employee, or every such query silently narrows to
    # whoever was last asked about individually (bug: this previously made
    # visit/WFH/leave "who all" list queries answer about only the sticky
    # employee instead of the whole org/dept). Only inherit when there's an
    # explicit backward reference to hang the inheritance on.
    if emp_id is None and session is not None and _PRONOUN_PATTERN.search(message):
        ctx_id = session_store.get_recent_context(session, "employee_id")
        ctx_name = session_store.get_recent_context(session, "employee_name")
        if ctx_id is not None:
            return ctx_id, ctx_name
    return emp_id, emp_name


def answer_intent(intent, dept_name, month, manager_id, manager_name, employee_ids=None, team_label=None, message="", session=None, date_range=None, raw_message=None):
    """Runs one of the 5 query templates. `employee_ids` (if set) scopes to a
    resolved 'my team' list and takes precedence for display purposes over
    dept_name in the scope note, though dept_name/manager filtering logic
    itself is untouched for the explicit (non-self-referential) path.
    `date_range` (start_date, end_date), if set, is a day/week-granularity
    time reference ("yesterday"/"last week"/"today"/"this week") that takes
    precedence over `month` for the new Category A-J query functions below
    that accept both.
    `raw_message`, if given, is the pre-spellcheck text - passed as
    fallback_text to every extract_employee/extract_department call below,
    same rationale as the fix already applied to extract_department's and
    extract_manager's call sites in handle_message: spellcheck can corrupt a
    real name into an unrelated English word (e.g. "offi"->"off",
    "Yadav"->"Adam"), which would otherwise silently fail to find the named
    employee/department here too."""
    scope_note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")
    scope_note += _period_note(month, date_range)
    limit = entities.extract_limit(message)
    period_month = None if date_range else month
    fb = raw_message if raw_message and raw_message != message else None

    if intent == "employee_day_summary":
        # New, additive intent (item #58): single-employee, single-day
        # SNAPSHOT - a completely different output shape from every ranking/
        # aggregate branch in this function (one row, one person, one day -
        # no GROUP BY, no population average). Reuses entities.extract_employee
        # (same fallback_text pattern as every other employee lookup in this
        # function) and the new entities.extract_single_date() helper (which
        # itself reuses extract_date_range()'s existing yesterday/today/
        # on-date/bare-ISO parsing before falling back to the same bare
        # "N Month" token scan extract_two_dates() already uses) - no new
        # date-parsing logic was written from scratch.
        try:
            emp_id, emp_name = entities.extract_employee(message, fallback_text=fb)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if not emp_id:
            return ChatResponse(reply="I couldn't find that employee. Could you check the spelling or give the full name?")
        target_date, date_mentioned = entities.extract_single_date(message)
        if not date_mentioned:
            # No explicit date named - default to yesterday, the most recent
            # day that can possibly have real data (see PROJECT_BACKUP
            # §2: "today" never has data, by permanent design).
            target_date = datetime.date.today() - datetime.timedelta(days=1)
        row = queries.employee_day_summary(emp_id, target_date)
        if row is None:
            return ChatResponse(
                reply=f"No Standard-shift data found for {emp_name} on {target_date} (could be a leave/absent/OT-only day)."
            )
        lc = "Yes" if row["lc"] else "No"
        el = "Yes" if row["el"] else "No"
        dh = "Yes" if row["dh"] else "No"
        reply = (
            f"{row['emp_name']} ({row['employee_id']}) — {row['dept_name']}, reports to {row['reporting_manager_name']}\n"
            f"Date: {row['worked_day']}\n"
            f"LC (late-coming): {lc} | EL (early-leaving): {el} | DH (deficient hours): {dh}\n"
            f"PACE score (day): {_fmt(row['pace_score'])}"
        )
        return ChatResponse(reply=reply, rows=[row])

    if intent == "average_metric":
        # New, additive intent: a genuine "avg"/"average"/"mean" + metric
        # [+ optional scope] request that computes ONE aggregate number (or,
        # if the message explicitly asks for a "list", the per-employee
        # breakdown behind that number - see build_query_overview_reply's
        # own wants_list switch), never a ranking/full-list-of-everyone
        # reply. This is the fix for the root-cause routing bug: a message
        # like "avg prod minutes in whole company" previously matched NO
        # rule-based intent at all (confirmed via a direct intents.match_intent()
        # call before this round), so it fell through to the LLM classifier /
        # SQL-fallback cascade, which could misroute it into an unrelated
        # dept_best/dept_avg-style full department-ranking table instead of
        # ever computing an average. Being a plain rule-based match now
        # means this always wins BEFORE that LLM/SQL-fallback path is ever
        # reached (rule-based intents take precedence unconditionally once
        # non-None - same guarantee every other rule-based intent in this
        # file already relies on).
        #
        # Scope resolution mirrors the bottom-of-cascade build_query()
        # fallback in handle_message() (employee -> department -> RM ->
        # company-wide), reusing the same extract_employee/
        # extract_department/extract_manager calls and fallback_text
        # pattern - nothing new invented here.
        #
        # Exception: when the message explicitly says "team" (e.g. "avg
        # pace score for Nikhil Kumar's team"), manager resolution is tried
        # FIRST, ahead of employee. Several real managers in this dataset
        # are ALSO themselves individual employees (documented elsewhere in
        # this file, e.g. Nikhil Kumar) or a bare first-name fragment of
        # "<Name>'s team" can spuriously match/collide with an unrelated
        # employee's name (confirmed live: "Rahul Yadav's team" ->
        # extract_employee raised Ambiguous on "Rahul" alone, before ever
        # reaching manager resolution) - explicit "team" wording is never
        # ambiguous about intent (it can only mean the named person's team,
        # never their own personal score), so it must not be shadowed by an
        # employee-name false-positive. Every other phrasing (no "team"
        # word - "avg X in whole company"/"avg X in <dept>"/"avg score for
        # <employee>") keeps the original employee-first order unchanged.
        wants_team_scope = re.search(r"\bteam\b", message, re.IGNORECASE) is not None
        avg_emp_id = avg_emp_name = None
        avg_mgr_id = avg_mgr_name = None
        if wants_team_scope:
            try:
                avg_mgr_id, avg_mgr_name = entities.extract_manager(message, fallback_text=fb)
            except entities.Ambiguous as e:
                return ChatResponse(reply=f"Multiple managers match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                     needs_clarification=True, clarification_options=e.candidates)
            if not avg_mgr_id:
                try:
                    avg_emp_id, avg_emp_name = entities.extract_employee(message, fallback_text=fb)
                except entities.Ambiguous:
                    pass  # a bare name-fragment collision on "team" wording - not a real employee lookup, ignore
        else:
            try:
                avg_emp_id, avg_emp_name = entities.extract_employee(message, fallback_text=fb)
            except entities.Ambiguous as e:
                return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                     needs_clarification=True, clarification_options=e.candidates)
            if not avg_emp_id:
                try:
                    avg_mgr_id, avg_mgr_name = entities.extract_manager(message, fallback_text=fb)
                except entities.Ambiguous:
                    avg_mgr_id, avg_mgr_name = None, None
        avg_dept_name, avg_dept_candidates = entities.extract_department(message, fallback_text=fb)
        avg_start, avg_end, avg_date_mentioned = entities.extract_date_range(message)
        avg_period = (avg_start, avg_end) if avg_date_mentioned else date_range
        if avg_mgr_id:
            reply, rows = build_query_overview_reply("rm", avg_mgr_name, message, period=avg_period, session=session)
        elif avg_emp_id:
            reply, rows = build_query_overview_reply("employee", avg_emp_id, message, period=avg_period, session=session, display_name=avg_emp_name)
        elif avg_dept_name and not avg_dept_candidates:
            reply, rows = build_query_overview_reply("department", avg_dept_name, message, period=avg_period, session=session)
        else:
            # No named employee/department/RM at all - "whole company"
            # (explicitly said or simply left unscoped, same convention as
            # every other company-wide default in this file, e.g.
            # gainer_loser_ranking/day_compare/month_compare).
            reply, rows = build_query_overview_reply("company", None, message, period=avg_period, session=session)
        return ChatResponse(reply=reply, rows=rows)

    if manager_id and employee_ids is None and intent in ("attendance_best", "attendance_worst"):
        rows = queries.team_attendance_ranking(manager_id, month, worst=(intent == "attendance_worst"))
        label = "worst" if intent == "attendance_worst" else "best"
        return ChatResponse(
            reply=f"{label.capitalize()} attendance for {manager_name}'s team{_period_note(month, None)}:\n\n{format_attendance_rows(rows)}",
            rows=rows,
        )

    if intent == "attendance_best":
        rows = queries.attendance_ranking(dept_name, month, worst=False, employee_ids=employee_ids, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.attendance_ranking(dept_name, month, worst=False, employee_ids=employee_ids, limit=limit)
                return f"Best attendance{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_attendance_rows(_rows)}", _rows
            def _rerun_opposite(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.attendance_ranking(dept_name, month, worst=True, employee_ids=employee_ids, limit=limit)
                return f"Worst attendance{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_attendance_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, rerun_opposite=_rerun_opposite,
                                         answer_kind="list", ascending=False,
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Best attendance{scope_note}:\n\n{format_attendance_rows(rows)}", rows=rows)

    if intent == "attendance_worst":
        rows = queries.attendance_ranking(dept_name, month, worst=True, employee_ids=employee_ids, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.attendance_ranking(dept_name, month, worst=True, employee_ids=employee_ids, limit=limit)
                return f"Worst attendance{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_attendance_rows(_rows)}", _rows
            def _rerun_opposite(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.attendance_ranking(dept_name, month, worst=False, employee_ids=employee_ids, limit=limit)
                return f"Best attendance{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_attendance_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, rerun_opposite=_rerun_opposite,
                                         answer_kind="list", ascending=True,
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Worst attendance{scope_note}:\n\n{format_attendance_rows(rows)}", rows=rows)

    if intent == "productive_high":
        rows = queries.productive_time_ranking(dept_name, month, lowest=False, employee_ids=employee_ids, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.productive_time_ranking(dept_name, month, lowest=False, employee_ids=employee_ids, limit=limit)
                return f"Most productive time{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_productive_rows(_rows)}", _rows
            def _rerun_opposite(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.productive_time_ranking(dept_name, month, lowest=True, employee_ids=employee_ids, limit=limit)
                return f"Least productive time{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_productive_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, rerun_opposite=_rerun_opposite,
                                         answer_kind="list", ascending=False,
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Most productive time{scope_note}:\n\n{format_productive_rows(rows)}", rows=rows)

    if intent == "productive_low":
        rows = queries.productive_time_ranking(dept_name, month, lowest=True, employee_ids=employee_ids, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.productive_time_ranking(dept_name, month, lowest=True, employee_ids=employee_ids, limit=limit)
                return f"Least productive time{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_productive_rows(_rows)}", _rows
            def _rerun_opposite(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.productive_time_ranking(dept_name, month, lowest=False, employee_ids=employee_ids, limit=limit)
                return f"Most productive time{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_productive_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, rerun_opposite=_rerun_opposite,
                                         answer_kind="list", ascending=True,
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Least productive time{scope_note}:\n\n{format_productive_rows(rows)}", rows=rows)

    # --- Category B/C: generic metric rankings ---
    if intent in _METRIC_INTENTS:
        metric_key, ascending = _METRIC_INTENTS[intent]
        # Item B (SESSION_HANDOFF.md): when NO period was named at all (no
        # explicit mention this turn, no sticky session context either),
        # default to the last 60 days instead of the old current-(partial)-
        # month default - scoped locally to this branch (month/date_range
        # for every OTHER branch in this function are untouched) via
        # queries.metric_ranking()'s new date_range support.
        _mr_month, _mr_date_range = month, date_range
        _mr_scope_note = scope_note
        if _no_period_named_at_all(message, session):
            _mr_month = None
            _mr_date_range = queries.default_period_last_60_days()
            _mr_scope_note = (f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")) + _period_note(None, _mr_date_range)
        rows = queries.metric_ranking(
            metric_key, dept_name, _mr_month, ascending=ascending, employee_ids=employee_ids,
            limit=limit, reporting_user_id=manager_id if employee_ids is None else None, date_range=_mr_date_range,
        )
        label = queries.METRICS[metric_key][1]
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=_mr_month, date_range=_mr_date_range, limit=500,
                       _metric_key=metric_key, _ascending=ascending, _label=label, _rid=manager_id):
                _rows = queries.metric_ranking(_metric_key, dept_name, month, ascending=_ascending, employee_ids=employee_ids,
                                                limit=limit, reporting_user_id=_rid if employee_ids is None else None, date_range=date_range)
                return f"Ranked by {_label}{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_metric_rows(_rows, _metric_key)}", _rows
            # Direction-flip closure for bare superlative follow-ups
            # ("least"/"most"/"highest"/"lowest" with nothing else) - see
            # _handle_bare_direction_followup. Mirrors the SAME metric_key/
            # scope as the ranking just shown, just with `ascending` flipped,
            # so "top pace score this month" -> "least" correctly re-ranks
            # PACE score (not some unrelated metric) in the opposite
            # direction, instead of falling through to the fuzzy intent
            # matcher (which previously always misrouted a bare direction
            # word to "productive_low"/"productive_high" regardless of what
            # the real prior metric was - see SESSION_HANDOFF.md item #63).
            def _rerun_opposite(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=_mr_month, date_range=_mr_date_range, limit=500,
                                 _metric_key=metric_key, _ascending=(not ascending), _label=label, _rid=manager_id):
                _rows = queries.metric_ranking(_metric_key, dept_name, month, ascending=_ascending, employee_ids=employee_ids,
                                                limit=limit, reporting_user_id=_rid if employee_ids is None else None, date_range=date_range)
                _dir_word = "lowest" if _ascending else "highest"
                return (f"Ranked by {_label} ({_dir_word} first)"
                        f"{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n"
                        f"{format_metric_rows(_rows, _metric_key)}", _rows)
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, rerun_opposite=_rerun_opposite,
                                         answer_kind="list", ascending=ascending,
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=_mr_month, date_range=_mr_date_range)
        return ChatResponse(reply=f"Ranked by {label}{_mr_scope_note}:\n\n{format_metric_rows(rows, metric_key)}", rows=rows)

    # --- Category A: single-employee lookups ---
    # --- PS (ps_worked_flag_day) exclusion intents (Part 3) ---
    if intent == "ps_explain":
        return ChatResponse(
            reply=(
                "\"PS not working\" refers to pace_1.ps_worked_flag_day = 0 — days when the "
                "productivity-sensor tracking wasn't active/installed for that employee, so no real "
                "usage data (engagement, effectiveness, minutes, etc.) was captured for that day. "
                "Numbers computed across a day like that can be unreliable, so you can ask me to "
                "\"exclude PS non-working days\" for any metric and I'll drop those rows before "
                "computing the answer."
            )
        )

    if intent == "ps_ratio_info":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(
                reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                needs_clarification=True, clarification_options=e.candidates,
            )
        if emp_id is not None:
            row = queries.ps_working_ratio(emp_id, month=month, date_range=date_range)
            if not row or not row.get("total_days"):
                return ChatResponse(reply=f"No data found for {emp_name} for that period.")
            total, working, off = row["total_days"], row["ps_working_days"] or 0, row["ps_off_days"] or 0
            pct = round((working / total) * 100) if total else 0
            return ChatResponse(
                reply=(f"{emp_name}{scope_note}: PS was working {working} of {total} days ({pct}%), "
                       f"and NOT working {off} of {total} days."),
                rows=[row],
            )
        # No named employee -> department/company-wide ranking of PS-off days.
        ascending = bool(re.search(r"\bfewest\b", message, re.I))
        rows = queries.ps_off_ranking(dept_name, month=month, date_range=date_range,
                                       employee_ids=employee_ids, ascending=ascending, limit=limit)
        if not rows:
            return ChatResponse(reply=f"No PS-tracking data found{scope_note}.")
        direction = "fewest" if ascending else "most"
        headers = ["#", "Employee", "PS non-working days", "Total days"]
        data = [[i, r["emp_name"], r["ps_off_days"], r["total_days"]] for i, r in enumerate(rows, 1)]
        return ChatResponse(reply=f"Employees with the {direction} PS non-working days{scope_note}:\n\n{_render_table(headers, data)}", rows=rows)

    if intent == "ps_exclude_metric":
        metric_key = _detect_ps_metric(message)
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(
                reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                needs_clarification=True, clarification_options=e.candidates,
            )
        if emp_id is None:
            # No named employee - this is a RANKING request ("...then who has
            # the least score?"), not an individual lookup. Same "ranking
            # fallback when no employee is named" shape as ps_ratio_info
            # above; previously this branch just gave up with "couldn't find
            # that employee" even for a plainly ranking-shaped message.
            ranking_ascending = bool(re.search(
                r"\b(least|lowest|worst|fewest|bottom|smallest)\b", message, re.I))
            rows = queries.metric_ranking_ps_filtered(
                metric_key, dept_name, month=month, date_range=date_range,
                ascending=ranking_ascending, employee_ids=employee_ids, limit=limit,
                exclude_ps_off=True,
            )
            direction = "lowest" if ranking_ascending else "highest"
            label = queries.PS_FILTERED_METRICS[metric_key][1]
            reply = (f"Ranked by {label} (excluding PS non-working days), {direction} first"
                     f"{scope_note}:\n\n{format_metric_rows(rows, metric_key)}")
            if session is not None:
                def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month,
                           date_range=date_range, limit=500, _metric_key=metric_key, _ascending=ranking_ascending):
                    _rows = queries.metric_ranking_ps_filtered(
                        _metric_key, dept_name, month=month, date_range=date_range,
                        ascending=_ascending, employee_ids=employee_ids, limit=limit, exclude_ps_off=True,
                    )
                    _label = queries.PS_FILTERED_METRICS[_metric_key][1]
                    _direction = "lowest" if _ascending else "highest"
                    return (f"Ranked by {_label} (excluding PS non-working days), {_direction} first"
                            f"{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n"
                            f"{format_metric_rows(_rows, _metric_key)}", _rows)
                session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                             dept_name=dept_name, employee_ids=employee_ids, team_label=team_label,
                                             month=month, date_range=date_range)
            return ChatResponse(reply=reply, rows=rows)
        row = queries.employee_metric_ps_filtered(emp_id, metric_key, month=month, date_range=date_range, exclude_ps_off=True)
        if not row or not row.get("days_counted"):
            return ChatResponse(reply=f"No data found for {emp_name} for that period.")
        # Category E edge case: zero PS-working days in scope -> graceful
        # "no data", not a nonsensical average-of-nothing (NULL) answer.
        if not row.get("ps_working_days"):
            return ChatResponse(
                reply=f"{emp_name} had zero PS-working days{scope_note} — no reliable {row['label']} data to report "
                      f"once PS non-working days are excluded.",
                rows=[row],
            )
        value = row["metric_value"]
        formatted = _fmt(value) if isinstance(value, (int, float)) else value
        off = row.get("ps_off_days") or 0
        excl_note = f" (excluding {off} PS non-working day(s))" if off else " (no PS non-working days to exclude in this period)"
        return ChatResponse(
            reply=f"{row['label']} for {emp_name}{scope_note}{excl_note}: {formatted}",
            rows=[row],
        )

    if intent in _EMP_FIELD_INTENTS or intent == "emp_attendance_summary":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(
                reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                needs_clarification=True, clarification_options=e.candidates,
            )
        if emp_id is None:
            # No named employee - if this is clearly a bulk/all-employees
            # phrasing ("give me score of all the employees of AI Labs
            # department") or there's an active per-employee ranking in
            # session context, answer with a ranking/list for the metric
            # instead of failing with "couldn't find that employee" (that
            # error should only ever fire for a genuinely unresolvable named
            # person, not a plural/bulk request that never named one).
            _metric_key = _EMP_FIELD_TO_METRIC_KEY.get(intent)
            _last = session_store.get_last_list(session) if session is not None else None
            _bulk_request = _BULK_ALL_EMPLOYEES_PATTERN.search(message) is not None
            _ranking_context = _last is not None and _last.get("kind") == "ranking"
            # item #56 fix: same bare department/RM-team overview redirect as
            # emp_overview below - "score of ai labs dept" resolves no
            # employee but a dept_name, so answer the department's overview
            # via build_query() instead of falling through to "couldn't find
            # that employee". Only fires when neither the bulk-list nor
            # active-ranking-context branch above already handled it.
            if dept_name and not (_bulk_request or _ranking_context):
                reply, rows = build_query_overview_reply("department", dept_name, message, period=date_range, session=session)
                return ChatResponse(reply=reply, rows=rows)
            if _metric_key is not None and (_bulk_request or _ranking_context):
                rows = queries.metric_ranking(
                    _metric_key, dept_name, month, ascending=False, employee_ids=employee_ids,
                    limit=limit or 500, reporting_user_id=manager_id if employee_ids is None else None,
                )
                label = queries.METRICS[_metric_key][1]
                reply = f"{label.capitalize()}{scope_note} (full list):\n\n{format_metric_rows(rows, _metric_key)}"
                if session is not None:
                    def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month,
                               date_range=date_range, limit=500, _mk=_metric_key, _rid=manager_id):
                        _rows = queries.metric_ranking(_mk, dept_name, month, ascending=False, employee_ids=employee_ids,
                                                        limit=limit, reporting_user_id=_rid if employee_ids is None else None)
                        _label = queries.METRICS[_mk][1]
                        return (f"{_label.capitalize()}{_scope_note_generic(team_label, dept_name, month, date_range)} "
                                f"(full list):\n\n{format_metric_rows(_rows, _mk)}", _rows)
                    session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                                 dept_name=dept_name, employee_ids=employee_ids, team_label=team_label,
                                                 month=month, date_range=date_range)
                return ChatResponse(reply=reply, rows=rows)
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        detail = queries.employee_detail(emp_id, month)
        if intent in _EMP_FIELD_INTENTS:
            field_label, getter = _EMP_FIELD_INTENTS[intent]
            if not detail:
                return ChatResponse(reply=f"No data found for {emp_name} for that period.")
            # Avoid a redundant double scope note (e.g. "for Deepanshu Saini
            # for Deepanshu Saini's team") when the employee being looked up
            # IS the person whose team scope is already active in the
            # session - in that case the "for {team_label}" portion adds
            # nothing since {emp_name} is already named explicitly.
            emp_scope_note = scope_note
            if team_label and team_label == f"{emp_name}'s team":
                emp_scope_note = _period_note(month, date_range)
            reply = f"{field_label} for {emp_name}{emp_scope_note}: {getter(detail)}"
            if intent in _PS_CAVEAT_METRIC_BY_INTENT:
                reply += _ps_off_caveat(emp_id, month, date_range)
            return ChatResponse(reply=reply, rows=[detail])
        return ChatResponse(reply=format_employee_detail(detail, emp_name), rows=[detail] if detail else [])

    # "is X improving/declining" - an INDIVIDUAL employee's own trend, never
    # the team/dept ranking used by "improving"/"declining". Resolved purely
    # via extract_employee (see _INDIVIDUAL_EMP_INTENTS bypass in
    # handle_message, which keeps this off the manager/admin-confirmation
    # path even when the named employee also happens to be a manager or a
    # universal-access admin). Defaults to a month-over-month comparison,
    # then offers a week-by-week breakdown as a same-session follow-up.
    if intent == "emp_trend":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        trend_month = _first_month(month) or entities.extract_month("")[0]
        row, meta = queries.employee_pace_trend_monthly(emp_id, trend_month)
        reply = format_monthly_pace_trend(row, meta, emp_name)
        reply += "\n\nWant this broken down by week instead?"
        if session is not None:
            session["awaiting_weekly_breakdown"] = True
            session["weekly_breakdown_employee_id"] = emp_id
            session["weekly_breakdown_employee_name"] = emp_name
            session["awaiting_ranking_weekly_breakdown"] = False
            session["ranking_weekly_breakdown_employee_ids"] = None
            session["ranking_weekly_breakdown_label"] = None
        return ChatResponse(reply=reply, rows=[row] if row else [])

    if intent == "emp_trend_2month":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        rows2mo = queries.employee_trend_two_month(emp_id, _first_month(month))
        return ChatResponse(reply=format_trend_2month(rows2mo, emp_name), rows=rows2mo)

    # --- Category D: team/dept improving-declining aggregates ---
    if intent == "team_improving":
        if employee_ids is None:
            return ChatResponse(reply="I need a resolved team to answer that — please ask about 'my team' after identifying yourself, or name a manager.")
        summary = queries.team_delta_summary(employee_ids, _first_month(month))
        if not summary:
            return ChatResponse(reply=f"Not enough reliable data to judge whether {team_label or 'the team'} is improving this month.")
        sign = "+" if summary["avg_delta"] > 0 else ""
        direction = "improving" if summary["avg_delta"] > 0 else "declining" if summary["avg_delta"] < 0 else "flat"
        return ChatResponse(reply=f"{team_label or 'Your team'} is {direction} on average: {sign}{summary['avg_delta']:.0f} pts "
                                   f"(across {summary['n_employees']} employees with reliable data).")

    if intent == "dept_trend":
        ascending = "declin" in message.lower()
        rows = queries.dept_delta_ranking(_first_month(month), ascending=ascending, limit=limit or 5)
        if not rows:
            return ChatResponse(reply="Not enough reliable data across departments this month.")
        label = "declining" if ascending else "improving"
        if len(rows) == 1:
            r = rows[0]
            body = f"{r['dept_name']} — avg change {r['avg_delta']:+.0f} pts ({r['n_employees']} employees)"
        else:
            headers = ["#", "Department", "Avg change (pts)", "Employees"]
            data = [[i, r["dept_name"], f"{r['avg_delta']:+.0f}", r["n_employees"]] for i, r in enumerate(rows, 1)]
            body = _render_table(headers, data)
        return ChatResponse(reply=f"Departments {label} the most:\n\n" + body, rows=rows)

    # --- Category E: department-level aggregates & comparison ---
    if intent == "dept_compare":
        left, right = entities.split_comparison(message)
        d1, _ = entities.extract_department(left or "", fallback_text=fb)
        d2, _ = entities.extract_department(right or "", fallback_text=fb)
        if not d1 or not d2:
            return ChatResponse(reply="I need two department names to compare — e.g. \"compare Accounts vs Billing\".")
        rows = queries.compare_depts(d1, d2, month)
        return ChatResponse(reply=format_compare(rows, [d1, d2], format_dept_summary), rows=[r for r in rows if r])

    if intent in ("dept_best", "dept_worst", "dept_avg"):
        ascending = intent == "dept_worst"
        # Item B: last-60-days default when nothing was named at all (see
        # the _METRIC_INTENTS branch above for the full rationale).
        _dr_month, _dr_date_range = month, None
        if _no_period_named_at_all(message, session):
            _dr_month, _dr_date_range = None, queries.default_period_last_60_days()

        # Item #73: detect an explicitly-named effectiveness/engagement/
        # discipline metric (with the capped-vs-percentage business rule)
        # instead of always hardcoding pace_score - reuses the SAME shared
        # normalizer the build_query() extraction cascade uses
        # (_detect_pct_capped_metrics), not a new parallel mechanism. Falls
        # through to the existing pace_score/dept_ranking() path unchanged
        # whenever no such metric is named (the overwhelmingly common
        # "which department is doing best overall" case), so this is
        # additive, not a rewrite of the existing behaviour.
        _bq_metrics = _detect_pct_capped_metrics(message)
        if _bq_metrics:
            bq_metric = _bq_metrics[0]
            if _dr_date_range is not None:
                bq_period = _dr_date_range
            else:
                _fm = _first_month(_dr_month)
                bq_period = _month_str_to_range(_fm) if _fm else None
            rows = queries.build_query("department", [bq_metric], period=bq_period,
                                        limit=limit, ascending=ascending)
            label = _BUILD_QUERY_METRIC_LABELS.get(bq_metric, bq_metric)
            table = _format_build_query_rows(rows, "department", [bq_metric])
            return ChatResponse(reply=f"Departments ranked by {label}:\n\n{table}", rows=rows)

        metric_key = "pace_score"
        rows = queries.dept_ranking(metric_key, _dr_month, ascending=ascending, limit=limit, date_range=_dr_date_range)
        return ChatResponse(reply=f"Departments ranked by {queries.METRICS[metric_key][1]}:\n\n{format_dept_rows(rows, metric_key)}", rows=rows)

    # RM (reporting-manager) team ranking - "which RM team has the most/
    # least score" - same shape as dept_best/dept_worst just grouped by
    # reporting_manager_name instead of dept_name (queries.rm_ranking()).
    # A different ranking DIMENSION than the department/employee rankings
    # above is itself a signal this is a fresh query, not a continuation of
    # whatever department/employee scope was sticky from a prior turn - see
    # the dept_name-reset carve-out for these two intents further up in
    # this function (search "rm_ranking_best", "rm_ranking_worst").
    if intent in ("rm_ranking_best", "rm_ranking_worst"):
        metric_key = "pace_score"
        ascending = intent == "rm_ranking_worst"
        # Item B: same last-60-days default as dept_best/dept_worst above.
        _rr_month, _rr_date_range = month, None
        if _no_period_named_at_all(message, session):
            _rr_month, _rr_date_range = None, queries.default_period_last_60_days()
        rows = queries.rm_ranking(metric_key, _rr_month, ascending=ascending, limit=limit, date_range=_rr_date_range)
        return ChatResponse(reply=f"Reporting-manager teams ranked by {queries.METRICS[metric_key][1]}:\n\n{format_rm_rows(rows, metric_key)}", rows=rows)

    if intent == "dept_count":
        if not dept_name:
            return ChatResponse(reply="Which department did you mean?")
        summary = queries.dept_summary(dept_name, month)
        n = summary["n_employees"] if summary else 0
        return ChatResponse(reply=f"{dept_name} has {n} employees{_period_note(month, None)}.")

    if intent == "dept_summary":
        if not dept_name:
            return ChatResponse(reply="Which department did you mean?")
        summary = queries.dept_summary(dept_name, month)
        return ChatResponse(reply=format_dept_summary(summary, dept_name), rows=[summary] if summary else [])

    # --- Category G: employee comparison / meeting minutes ---
    if intent == "employee_compare":
        left, right = entities.split_comparison(message)
        # If both sides resolve to department names instead of employees,
        # this is really a department comparison (e.g. "compare Accounts vs
        # Billing" has no "department" keyword for the dept_compare regex to
        # anchor on) — redirect rather than duplicate the comparison logic.
        d1, _ = entities.extract_department(left or "", fallback_text=fb)
        d2, _ = entities.extract_department(right or "", fallback_text=fb)
        if d1 and d2:
            rows = queries.compare_depts(d1, d2, month)
            return ChatResponse(reply=format_compare(rows, [d1, d2], format_dept_summary), rows=[r for r in rows if r])
        try:
            e1, n1 = entities.extract_employee(left or message, fallback_text=fb)
        except entities.Ambiguous:
            e1, n1 = None, None
        try:
            e2, n2 = entities.extract_employee(right or "", fallback_text=fb)
        except entities.Ambiguous:
            e2, n2 = None, None
        if not e1 or not e2:
            return ChatResponse(reply="I need two employee names to compare — e.g. \"compare Aarna Jain vs Abhi jain\".")
        rows = queries.compare_employees(e1, e2, month)
        return ChatResponse(reply=format_compare(rows, [n1, n2], format_employee_detail), rows=[r for r in rows if r])

    if intent == "meeting_min_ranking":
        rows = queries.meeting_minutes_ranking(dept_name, month, employee_ids=employee_ids, limit=limit)
        return ChatResponse(reply=f"Meeting minutes ranking{scope_note}:\n\n{format_meeting_rows(rows)}", rows=rows)

    # --- Category F: attendance thresholds ---
    if intent == "chronic_late":
        rows = queries.chronic_late(dept_name, month, employee_ids=employee_ids, limit=limit)
        return ChatResponse(reply=f"Chronically late employees{scope_note}:\n\n{format_chronic_late(rows, queries.CHRONIC_LATE_THRESHOLD)}", rows=rows)

    if intent == "perfect_attendance":
        rows = queries.perfect_attendance(dept_name, month, employee_ids=employee_ids, limit=limit)
        return ChatResponse(reply=f"Perfect attendance{scope_note}:\n\n{format_perfect_attendance(rows)}", rows=rows)

    # --- Category H: team/manager views ---
    if intent == "team_how_doing":
        if employee_ids is None:
            return ChatResponse(reply="I need a resolved team to answer that — please ask about 'my team' after identifying yourself, or name a manager.")
        summary = queries.team_summary(employee_ids, month, label=team_label)
        return ChatResponse(reply=format_team_summary(summary, team_label or "Your team"), rows=[summary] if summary else [])

    if intent == "team_lowest_scorers":
        if employee_ids is None:
            return ChatResponse(reply="I need a resolved team to answer that — please ask about 'my team' after identifying yourself, or name a manager.")
        rows = queries.metric_ranking("pace_score", None, month, ascending=True, employee_ids=employee_ids, limit=limit)
        return ChatResponse(reply=f"Lowest PACE scores in {team_label or 'your team'}:\n\n{format_metric_rows(rows, 'pace_score')}", rows=rows)

    if intent == "team_compare":
        if employee_ids is None or session is None:
            return ChatResponse(reply="I need to know your team first — please identify yourself so I can resolve 'my team'.")
        left, right = entities.split_comparison(message)
        # Resolve the other person via the SAME email-access pipeline as "my
        # team" / the named-manager path above, so this comparison uses an
        # identical team definition on both sides instead of a shallower
        # reporting_user_id-only lookup.
        other_ids, other_is_universal, other_manager_name, other_candidates = team.resolve_named_person_team(right or message)
        if other_candidates:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(other_candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=other_candidates)
        if not other_ids:
            return ChatResponse(reply="I need another manager's name to compare against — e.g. \"compare my team to Nikhil Kumar's team\".")
        if other_is_universal:
            return ChatResponse(
                reply=f"{other_manager_name} has admin-level access, so their 'team' would mean essentially the "
                      f"whole company — please compare against a specific department or a non-admin manager instead."
            )
        my_summary = queries.team_summary(employee_ids, month, label=team_label)
        other_summary = queries.team_summary(other_ids, month, label=f"{other_manager_name}'s team")
        return ChatResponse(
            reply=format_compare([my_summary, other_summary], [team_label or "Your team", f"{other_manager_name}'s team"], format_team_summary),
            rows=[r for r in (my_summary, other_summary) if r],
        )

    if intent == "new_joiners":
        rows = queries.new_joiners(employee_ids=employee_ids, dept_name=dept_name)
        return ChatResponse(reply=f"New joiners{scope_note.replace(_period_note(month, date_range), '')}:\n\n{format_new_joiners(rows)}", rows=rows)

    if intent in ("improving", "declining"):
        # Item B (SESSION_HANDOFF.md): this ranking is inherently a
        # month-vs-prior-month DELTA (queries.pace_score_trend_ranking()
        # internally computes prev_month = _prev_month(month) and compares
        # the two) - a plain date-range window doesn't map onto that
        # semantics, so "last 60 days" isn't literally applicable here.
        # Judgment call, flagged in SESSION_HANDOFF.md: when NO period was
        # named at all, use the last FULLY COMPLETED calendar month (vs the
        # one before it) instead of the old default of the current,
        # still-in-progress month (vs last month) - this is exactly the
        # same underlying fix or item B everywhere else (avoid comparing
        # against a partial/scarce-data month), applied in the way that
        # actually fits this function's month-over-month mechanics.
        _trend_month = _first_month(month)
        if _no_period_named_at_all(message, session):
            _trend_month = queries._prev_month(f"{datetime.date.today().year:04d}-{datetime.date.today().month:02d}")
        rows, meta = queries.pace_score_trend_ranking(
            dept_name, _trend_month, declining=(intent == "declining"),
            reporting_user_id=manager_id if employee_ids is None else None,
            employee_ids=employee_ids, limit=limit,
        )
        label = "declining" if intent == "declining" else "improving"
        _trend_scope_note = (f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")) + _period_note(_trend_month, None)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=_trend_month, date_range=None, limit=500,
                       _declining=(intent == "declining"), _rid=manager_id, _label=label):
                _rows, _meta = queries.pace_score_trend_ranking(
                    dept_name, _first_month(month), declining=_declining,
                    reporting_user_id=_rid if employee_ids is None else None,
                    employee_ids=employee_ids, limit=limit,
                )
                return f"Who is {_label}{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_trend_rows(_rows, _meta)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=_trend_month, date_range=None)
        return ChatResponse(reply=f"Who is {label}{_trend_scope_note}:\n\n{format_trend_rows(rows, meta)}", rows=rows)

    # --- Category A (new): Leave & absence ---
    if intent in ("leave_emp_check", "call_emp", "visit_emp", "wfh_emp", "d_score_emp",
                  "shift_type_emp", "breakshift_emp", "offline_emp", "meeting_ratio_emp", "meeting_had_emp"):
        # "beside X"/"except X"/... exclusion (new capability): a query that
        # NAMES an employee but only to exclude them from an otherwise
        # org/dept-wide list ("beside muskan who all did visit yesterday")
        # must NOT resolve as an individual lookup about that person - it
        # must resolve straight to the plural day_flag_list path, excluding
        # them. Checked first, before the normal employee resolution below,
        # so the named person is never mistaken for the query's subject.
        if (intent in ("leave_emp_check", "call_emp", "visit_emp", "wfh_emp")
                and _EXCLUDE_PATTERN.search(message)):
            try:
                _excl_id, _excl_name = entities.extract_employee(message, fallback_text=fb)
            except entities.Ambiguous as e:
                return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                     needs_clarification=True, clarification_options=e.candidates)
            flag_key = _detect_day_flag(message)
            if _excl_id is not None and flag_key:
                day_scope_note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")
                day_scope_note += " " + (_period_label_for_range(date_range) if date_range else _period_note(month, None))
                day_scope_note += f" (excluding {_excl_name})"
                rows = queries.day_flag_list(flag_key, dept_name=dept_name, employee_ids=employee_ids,
                                              month=None if date_range else period_month, date_range=date_range,
                                              limit=limit, exclude_employee_id=_excl_id)
                return ChatResponse(reply=format_day_list(rows, flag_key, day_scope_note), rows=rows)
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            # No specific employee named — if this reads as a plural "who"
            # question (e.g. "who was on WFH last month", no name given) and
            # we can detect which DAY_FLAGS flag it's about, reroute to the
            # SAME day_flag_list plumbing used by the explicit day_list
            # intent, just scoped by month instead of date_range when no
            # day/week reference was given either — reuses queries.day_flag_list
            # rather than adding a parallel path. Otherwise, keep the
            # original "couldn't find that employee" error.
            if intent in ("leave_emp_check", "call_emp", "visit_emp", "wfh_emp") and re.search(r"\bwho\b", message, re.I):
                flag_key = _detect_day_flag(message)
                if flag_key:
                    day_scope_note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")
                    day_scope_note += " " + (_period_label_for_range(date_range) if date_range else _period_note(month, None))
                    rows = queries.day_flag_list(flag_key, dept_name=dept_name, employee_ids=employee_ids,
                                                  month=None if date_range else period_month, date_range=date_range, limit=limit)
                    return ChatResponse(reply=format_day_list(rows, flag_key, day_scope_note), rows=rows)
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")

        if intent == "leave_emp_check":
            rows = queries.leave_status_for_employee(emp_id, month=period_month, date_range=date_range)
            if not rows:
                return ChatResponse(reply=f"{emp_name} took no leave{_period_note(month, date_range)}.")
            return ChatResponse(reply=f"{emp_name} took leave on {len(rows)} day(s){_period_note(month, date_range)}:\n\n"
                                       + format_leave_rows(rows), rows=rows)

        if intent == "call_emp":
            row = queries.call_activity_for_employee(emp_id, month=period_month, date_range=date_range)
            if not row:
                return ChatResponse(reply=f"No call data found for {emp_name}{_period_note(month, date_range)}.")
            return ChatResponse(reply=f"{emp_name}{_period_note(month, date_range)}: {row['total_calls']} calls, "
                                       f"{row['total_call_min']} total call minutes.", rows=[row])

        if intent == "visit_emp":
            rows = queries.visit_activity_for_employee(emp_id, month=period_month, date_range=date_range)
            if not rows:
                return ChatResponse(reply=f"{emp_name} made no client visits{_period_note(month, date_range)}.")
            if len(rows) == 1:
                r = rows[0]
                body = f"{r['worked_day']}: visit={r['visit_flag']} type={r['visit_type'] or 'N/A'}"
            else:
                headers = ["Date", "Visit", "Type"]
                data = [[r["worked_day"], r["visit_flag"], r["visit_type"] or "N/A"] for r in rows]
                body = _render_table(headers, data)
            return ChatResponse(reply=f"{emp_name} visited clients on {len(rows)} day(s){_period_note(month, date_range)}:\n\n" + body, rows=rows)

        if intent == "wfh_emp":
            rows = queries.wfh_status_for_employee(emp_id, month=period_month, date_range=date_range)
            if not rows:
                return ChatResponse(reply=f"{emp_name} did not take WFH{_period_note(month, date_range)}.")
            if len(rows) == 1:
                r = rows[0]
                body = f"{r['worked_day']}: {r['wfh_status']}"
            else:
                headers = ["Date", "WFH status"]
                data = [[r["worked_day"], r["wfh_status"]] for r in rows]
                body = _render_table(headers, data)
            return ChatResponse(reply=f"{emp_name} took WFH on {len(rows)} day(s){_period_note(month, date_range)}:\n\n" + body, rows=rows)

        if intent == "d_score_emp":
            rows = queries.d_score_ranking(None, month=period_month, date_range=date_range, limit=100000)
            match = next((r for r in rows if r["employee_id"] == emp_id), None)
            if not match:
                return ChatResponse(reply=f"No d_score data found for {emp_name}{_period_note(month, date_range)} "
                                           f"(d_score is populated for only a subset of employees).")
            return ChatResponse(reply=f"{emp_name}'s avg d_score{_period_note(month, date_range)}: "
                                       f"{_fmt(match['avg_d_score'])} (based on {match['scored_days']} scored days).", rows=[match])

        if intent == "shift_type_emp" or intent == "breakshift_emp":
            rows = queries.shift_type_for_employee(emp_id, month=period_month, date_range=date_range)
            if not rows:
                return ChatResponse(reply=f"No roster data found for {emp_name}{_period_note(month, date_range)}.")
            if len(rows) == 1:
                r = rows[0]
                body = f"{r['worked_day']}: shift={r['shift_type']}, roster_type={r['mct_roster_shift_type']}, break_shift_match={r['breakshift_match_flag']}"
            else:
                headers = ["Date", "Shift", "Roster type", "Break-shift match"]
                data = [[r["worked_day"], r["shift_type"], r["mct_roster_shift_type"], r["breakshift_match_flag"]] for r in rows]
                body = _render_table(headers, data)
            return ChatResponse(reply=f"Roster/shift info for {emp_name}:\n\n" + body, rows=rows)

        if intent == "offline_emp":
            rows = queries.offline_status_for_employee(emp_id, month=period_month, date_range=date_range)
            if not rows:
                return ChatResponse(reply=f"No device/offline data found for {emp_name}{_period_note(month, date_range)}.")
            if len(rows) == 1:
                r = rows[0]
                body = f"{r['worked_day']}: {r['offline_attendance_flag']}, PS installed={r['ps_installed_new']}, PS worked={r['ps_worked_flag_day']}"
            else:
                headers = ["Date", "Offline flag", "PS installed", "PS worked"]
                data = [[r["worked_day"], r["offline_attendance_flag"], r["ps_installed_new"], r["ps_worked_flag_day"]] for r in rows]
                body = _render_table(headers, data)
            return ChatResponse(reply=f"Offline/device status for {emp_name}:\n\n" + body, rows=rows)

        if intent == "meeting_ratio_emp":
            row = queries.meeting_activity_for_employee(emp_id, month=period_month, date_range=date_range)
            if not row:
                return ChatResponse(reply=f"No meeting data found for {emp_name}{_period_note(month, date_range)}.")
            ratio_str = f"{row['meeting_ratio']*100:.0f}%" if row["meeting_ratio"] is not None else "N/A"
            return ChatResponse(reply=f"{emp_name}{_period_note(month, date_range)}: {row['total_meetings']} meetings, "
                                       f"{row['total_meeting_min']} meeting minutes, meeting/productive-time ratio: {ratio_str}.", rows=[row])

        # Item #79 gap-fill (CSV row 51): "meetings (had any)" - a boolean
        # derived operation (meeting_count > 0), DISTINCT from the numeric
        # meeting-count/meeting-minutes metrics above. Reuses the existing
        # DAY_FLAGS["had_meetings"] infra (day_flag_list scoped to this one
        # employee via employee_ids) rather than a new SQL formula - the same
        # infra day_count/day_list already use for the org-wide "who had
        # meetings" phrasing, just scoped down to a single named employee.
        if intent == "meeting_had_emp":
            rows = queries.day_flag_list("had_meetings", employee_ids=[emp_id], month=period_month, date_range=date_range)
            if not rows:
                return ChatResponse(reply=f"No, {emp_name} did not have any meetings{_period_note(month, date_range)}.")
            r = rows[0]
            return ChatResponse(reply=f"Yes, {emp_name} had meetings on {r['matching_days']} day(s){_period_note(month, date_range)}.", rows=rows)

    if intent == "leave_who":
        rows = queries.who_on_leave(dept_name, month=period_month, date_range=date_range, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=period_month, date_range=date_range, limit=500):
                _rows = queries.who_on_leave(dept_name, month=month, date_range=date_range, limit=limit)
                return f"On leave{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_leave_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="day_flag", rerun_list=_rerun, rerun_same=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=period_month, date_range=date_range)
        return ChatResponse(reply=f"On leave{scope_note}:\n\n{format_leave_rows(rows)}", rows=rows)

    if intent == "half_day_ranking":
        rows = queries.half_day_ranking(dept_name, month=period_month, date_range=date_range, limit=limit)
        return ChatResponse(reply=f"Half-day ranking{scope_note}:\n\n{format_count_rows(rows, 'half_days', 'half-days')}", rows=rows)

    if intent == "leave_by_dept":
        rows = queries.leave_counts_by_dept(month=period_month, date_range=date_range)
        return ChatResponse(reply=f"Leave counts by department{_period_note(month, date_range)}:\n\n{format_count_rows(rows, 'leave_days', 'leave days')}", rows=rows)

    if intent == "zero_leave":
        rows = queries.zero_leave_employees(dept_name, month=period_month, date_range=date_range, limit=limit)
        return ChatResponse(reply=f"Zero-leave employees{scope_note}:\n\n{format_count_rows(rows, 'days_worked', 'days worked, 0 leave')}", rows=rows)

    # --- Category B (new): Calls ---
    if intent in ("call_most", "call_fewest", "call_duration"):
        metric = "avg_duration" if intent == "call_duration" else "total_calls"
        ascending = intent == "call_fewest"
        rows = queries.call_ranking(dept_name, metric=metric, month=period_month, date_range=date_range, ascending=ascending, limit=limit)
        label = "call duration" if metric == "avg_duration" else "call count"
        return ChatResponse(reply=f"Ranked by {label}{scope_note}:\n\n{format_count_rows(rows, 'metric_value', label)}", rows=rows)

    # --- Category C (new): Visits ---
    if intent == "visit_ranking":
        rows = queries.visit_ranking(dept_name, month=period_month, date_range=date_range, employee_ids=employee_ids, limit=limit)
        return ChatResponse(reply=f"Visit ranking{scope_note}:\n\n{format_count_rows(rows, 'visit_days', 'visit days')}", rows=rows)

    if intent == "zero_visit":
        rows = queries.zero_visit_employees(dept_name, month=period_month, date_range=date_range, limit=limit)
        return ChatResponse(reply=f"Zero-visit employees{scope_note}:\n\n{format_count_rows(rows, 'days_worked', 'days worked, 0 visits')}", rows=rows)

    # --- Category D (new): WFH ---
    if intent == "wfh_ranking":
        rows = queries.wfh_ranking(dept_name, month=period_month, date_range=date_range, employee_ids=employee_ids, limit=limit)
        return ChatResponse(reply=f"WFH ranking{scope_note}:\n\n{format_count_rows(rows, 'wfh_days', 'WFH days')}", rows=rows)

    if intent == "wfh_by_dept":
        rows = queries.wfh_by_dept(month=period_month, date_range=date_range)
        return ChatResponse(reply=f"WFH by department{_period_note(month, date_range)}:\n\n{format_count_rows(rows, 'wfh_days', 'WFH days')}", rows=rows)

    # --- Category E (new): Tasks/todos ---
    _TASK_INTENT_MAP = {
        "todos_created_ranking": "todos_created", "todos_assigned_ranking": "todos_assigned",
        "tasks_created_ranking": "tasks_created", "tasks_assigned_ranking": "tasks_assigned",
        "ontime_completion_ranking": "ontime_completion_rate", "responsiveness_ranking": "responsiveness_score",
        "extension_adherence_ranking": "extension_adherence_score",
    }
    if intent in _TASK_INTENT_MAP:
        metric_key = _TASK_INTENT_MAP[intent]
        rows = queries.task_metric_ranking(metric_key, dept_name, month=period_month, date_range=date_range, limit=limit)
        label = queries.TASK_METRICS[metric_key][1]
        note = ""
        if metric_key in ("ontime_completion_rate", "responsiveness_score", "extension_adherence_score"):
            note = "\n\n(Note: this metric is only populated for a subset of employees — many will show no data.)"
        return ChatResponse(reply=f"Ranked by {label}{scope_note}:\n\n{format_count_rows(rows, 'metric_value', label)}{note}", rows=rows)

    # --- Category F (new): Meetings ---
    if intent == "meeting_count_ranking":
        rows = queries.meeting_count_ranking(dept_name, month=period_month, date_range=date_range, limit=limit)
        return ChatResponse(reply=f"Meeting count ranking{scope_note}:\n\n{format_count_rows(rows, 'total_meetings', 'meetings')}", rows=rows)

    # --- Category G (new): Quality / d_score ---
    if intent == "d_score_ranking":
        ascending = "worst" in message.lower() or "lowest" in message.lower()
        rows = queries.d_score_ranking(dept_name, month=period_month, date_range=date_range, ascending=ascending, limit=limit)
        note = "\n\n(Note: d_score is populated for only a subset of employees — most will show no data.)"
        return ChatResponse(reply=f"d_score ranking{scope_note}:\n\n{format_count_rows(rows, 'avg_d_score', 'avg d_score')}{note}", rows=rows)

    if intent == "d_score_trend":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        row = queries.d_score_trend(emp_id, _first_month(month))
        if not row:
            return ChatResponse(reply=f"Not enough reliable d_score data (d_score is mostly null; needs "
                                       f"{queries.MIN_DAYS_FOR_DELTA}+ scored days in both this month and last) to judge a trend for {emp_name}.")
        direction = "improved" if row["cur_avg"] > row["prev_avg"] else "declined" if row["cur_avg"] < row["prev_avg"] else "stayed flat"
        return ChatResponse(reply=f"{emp_name}'s d_score has {direction}: {_fmt(row['prev_avg'])} -> {_fmt(row['cur_avg'])}.", rows=[row])

    # --- Category H (new): Roster/shift/OT ---
    if intent == "ot_ranking":
        rows = queries.ot_hours_ranking(dept_name, month=period_month, date_range=date_range, employee_ids=employee_ids, limit=limit)
        return ChatResponse(reply=f"OT hours ranking{scope_note}:\n\n{format_count_rows(rows, 'ot_hours', 'OT hours')}", rows=rows)

    # --- Category I (new): Offline/device status ---
    if intent == "offline_ranking":
        rows = queries.offline_attendance_ranking(dept_name, month=period_month, date_range=date_range, limit=limit)
        return ChatResponse(reply=f"Offline attendance ranking{scope_note}:\n\n{format_count_rows(rows, 'offline_days', 'offline days')}", rows=rows)

    if intent == "ps_install_rate":
        rows = queries.ps_install_rate_by_dept(month=period_month, date_range=date_range)
        if not rows:
            body = "No data found."
        elif len(rows) == 1:
            r = rows[0]
            body = f"{r['dept_name']} — PS installed rate: {_fmt(r['ps_installed_pct'])}%"
        else:
            headers = ["#", "Department", "PS installed rate (%)"]
            data = [[i, r["dept_name"], _fmt(r["ps_installed_pct"])] for i, r in enumerate(rows, 1)]
            body = _render_table(headers, data)
        return ChatResponse(reply=f"PS install rate by department{_period_note(month, date_range)}:\n\n" + body, rows=rows)

    # --- Category J (new): Org info ---
    if intent == "grade_lookup":
        m = re.search(r"\bgrade\s+([a-z0-9\-]+)\b", message, re.I)
        if not m:
            return ChatResponse(reply="Which grade did you mean — e.g. \"employees with grade A\"?")
        grade = m.group(1)
        rows = queries.employees_by_grade(grade, dept_name=dept_name)
        if not rows:
            return ChatResponse(reply=f"No employees found with grade {grade}{scope_note}.")
        if len(rows) == 1:
            r = rows[0]
            body = f"{r['emp_name']} ({r['dept_name']}, {r['designation']})"
        else:
            headers = ["#", "Employee", "Department", "Designation"]
            data = [[i, r["emp_name"], r["dept_name"], r["designation"]] for i, r in enumerate(rows, 1)]
            body = _render_table(headers, data)
        return ChatResponse(reply=f"Employees with grade {grade}{scope_note}:\n\n" + body, rows=rows)

    if intent == "designation_breakdown":
        rows = queries.designation_breakdown(dept_name=dept_name)
        return ChatResponse(reply=f"Designation breakdown{scope_note}:\n\n{format_count_rows(rows, 'n_employees', 'employees', name_field='designation')}", rows=rows)

    if intent == "avg_tenure":
        rows = queries.average_tenure(dept_name=dept_name)
        if not rows:
            return ChatResponse(reply="No tenure data found.")
        if len(rows) == 1:
            r = rows[0]
            body = f"{r['dept_name']} — avg tenure {_fmt(r['avg_tenure_days'], 0)} days ({r['n_employees']} employees)"
        else:
            headers = ["#", "Department", "Avg tenure (days)", "Employees"]
            data = [[i, r["dept_name"], _fmt(r["avg_tenure_days"], 0), r["n_employees"]] for i, r in enumerate(rows, 1)]
            body = _render_table(headers, data)
        return ChatResponse(reply=f"Average tenure{scope_note}:\n\n" + body, rows=rows)

    # --- Category K (new round 2) ---
    # Both rankings now compare CURRENT-MONTH-AVG vs PRIOR-MONTH-AVG PACE
    # score (see queries._score_delta_ranking_monthly) rather than
    # first-vs-last scored day within the period, and both offer a
    # same-session weekly-breakdown follow-up for the same ranked employee
    # set - mirrors the individual emp_trend weekly-offer pattern, using its
    # own "ranking_weekly_breakdown_*" session fields (distinct from
    # emp_trend's "weekly_breakdown_*" fields) so the two flows never
    # collide or leak state into each other.
    if intent == "score_drop_ranking":
        rows, meta = queries.score_drop_ranking(dept_name, employee_ids=employee_ids, month=period_month, date_range=date_range, limit=limit)
        reply = format_score_delta_ranking(rows, meta, f"Biggest PACE score drop{scope_note}")
        reply += "\n\nWant this broken down by week instead?"
        if session is not None:
            session["awaiting_ranking_weekly_breakdown"] = True
            session["ranking_weekly_breakdown_employee_ids"] = [r["employee_id"] for r in rows] or None
            session["ranking_weekly_breakdown_label"] = team_label or dept_name or "that scope"
            session["awaiting_weekly_breakdown"] = False
            session["weekly_breakdown_employee_id"] = None
            session["weekly_breakdown_employee_name"] = None

            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=period_month, date_range=date_range, limit=500):
                _rows, _meta = queries.score_drop_ranking(dept_name, employee_ids=employee_ids, month=month, date_range=date_range, limit=limit)
                return format_score_delta_ranking(_rows, _meta, f"Biggest PACE score drop{_scope_note_generic(team_label, dept_name, month, date_range)} (full list)"), _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=period_month, date_range=date_range)
        return ChatResponse(reply=reply, rows=rows)

    if intent == "score_improvement_alltime":
        rows, meta = queries.score_improvement_alltime(dept_name, employee_ids=employee_ids, month=period_month, limit=limit)
        note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")
        reply = format_score_delta_ranking(rows, meta, f"Most improved{note}")
        reply += "\n\nWant this broken down by week instead?"
        if session is not None:
            session["awaiting_ranking_weekly_breakdown"] = True
            session["ranking_weekly_breakdown_employee_ids"] = [r["employee_id"] for r in rows] or None
            session["ranking_weekly_breakdown_label"] = team_label or dept_name or "that scope"
            session["awaiting_weekly_breakdown"] = False
            session["weekly_breakdown_employee_id"] = None
            session["weekly_breakdown_employee_name"] = None

            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=period_month, date_range=date_range, limit=500):
                _rows, _meta = queries.score_improvement_alltime(dept_name, employee_ids=employee_ids, month=month, limit=limit)
                return format_score_delta_ranking(_rows, _meta, f"Most improved{_scope_note_generic(team_label, dept_name, month, date_range)} (full list)"), _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=period_month, date_range=date_range)
        return ChatResponse(reply=reply, rows=rows)

    if intent == "subscore_delta_ranking":
        # Item #84 (Finding 3): reached via the redirect (see the
        # rule_intent nulling/redirect block earlier in this function) when
        # subscore_trend_emp's own pattern matched but no specific employee
        # is named - i.e. this is actually a company-wide/department-wide
        # sub-metric-decline ranking, not a single-employee trend lookup.
        subscore_key = _detect_subscore_key(message, default="engagement", include_working_hours=True)
        metric_key = {
            "engagement": "engagement_pct", "effectiveness": "effectiveness_pct",
            "discipline": "discipline_pct", "working_hours": "working_pct",
        }[subscore_key]
        # Direction: "declin*"/"drop*"/"fell"/"worse"/"decreased" -> ascending
        # (most-negative delta first); "improv*"/"better"/"increased" ->
        # descending. Defaults to ascending (decline) since that's the
        # wording that actually triggers this redirect in practice
        # (_SUBSCORE_TREND_PATTERNS requires improv*/declin* wording).
        ascending = not re.search(r"\bimprov\w*|better|increas\w*\b", message, re.IGNORECASE)
        rows, meta = queries.subscore_delta_ranking(
            metric_key, dept_name=dept_name, employee_ids=employee_ids,
            month=period_month, date_range=date_range, ascending=ascending, limit=limit,
        )
        direction_label = "decline" if ascending else "improvement"
        reply = format_subscore_delta_ranking(rows, meta, f"Biggest {meta['label']} {direction_label}{scope_note}")
        if session is not None:
            session_store.set_last_list(
                session, kind="ranking", answer_kind="list",
                dept_name=dept_name, employee_ids=[r["employee_id"] for r in rows] or None,
                team_label=team_label, month=period_month, date_range=date_range,
            )
        return ChatResponse(reply=reply, rows=rows)

    if intent == "gainer_loser_ranking":
        filter_sql, filter_footer = _resolve_population_filter(message)
        directions = _gainer_loser_directions(message)
        gainers, losers, excluded_count, meta = queries.gainer_loser_ranking(
            dept_name, employee_ids=employee_ids, filter_sql=filter_sql, limit=limit, directions=directions
        )
        reply = format_gainer_loser_ranking(gainers, losers, meta, scope_note, directions, filter_footer)
        if session is not None:
            session_store.set_last_answer_filters(
                session, label=f"that gainer/loser ranking{scope_note}",
                **_default_filters_from_message(message))
        return ChatResponse(reply=reply, rows=gainers + losers)

    if intent == "emp_overview":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            # item #56 fix: a bare department/RM-team overview phrase ("how
            # is ai labs doing", "how is ai labs dept doing") with no keyword
            # like "department"/"my team" was wrongly caught by
            # _EMP_OVERVIEW_PATTERNS' name-lookup path and failed here with
            # "couldn't find that employee" - dept_name is already resolved
            # above from the same message (entities.extract_department has
            # no keyword requirement), so redirect to a department overview
            # via the new build_query() engine instead of failing.
            if dept_name:
                reply, rows = build_query_overview_reply("department", dept_name, message, period=date_range, session=session)
                return ChatResponse(reply=reply, rows=rows)
            if manager_id:
                reply, rows = build_query_overview_reply("rm", manager_name, message, period=date_range, session=session)
                return ChatResponse(reply=reply, rows=rows)
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        detail = queries.employee_detail(emp_id, month)
        return ChatResponse(reply=format_employee_detail(detail, emp_name), rows=[detail] if detail else [])

    if intent == "subscore_compare_emp":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        row = queries.subscore_compare_for_employee(emp_id, month=period_month, date_range=date_range)
        if not row:
            return ChatResponse(reply=f"No data found for {emp_name}{_period_note(month, date_range)}.")
        vals = {"Engagement": row["engagement"], "Effectiveness": row["effectiveness"], "Discipline": row["discipline"]}
        present = {k: v for k, v in vals.items() if v is not None}
        if not present:
            return ChatResponse(reply=f"No engagement/effectiveness/discipline data found for {emp_name}{_period_note(month, date_range)}.", rows=[row])
        strongest = max(present, key=present.get)
        weakest = min(present, key=present.get)
        parts = ", ".join(f"{k} {_fmt(v)}%" for k, v in vals.items())
        return ChatResponse(
            reply=f"{emp_name}{_period_note(month, date_range)}: {parts}.\n"
                  f"Strongest: {strongest} ({_fmt(present[strongest])}%), Weakest: {weakest} ({_fmt(present[weakest])}%).",
            rows=[row],
        )

    if intent == "subscore_trend_emp":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        subscore_key = _detect_subscore_key(message)
        row = queries.subscore_trend(emp_id, subscore_key, _first_month(month))
        label = queries.SUBSCORES[subscore_key][1]
        if not row:
            return ChatResponse(reply=f"Not enough reliable data ({queries.MIN_DAYS_FOR_DELTA}+ days needed in both this month "
                                       f"and last) to judge a {label.lower()} trend for {emp_name}.")
        direction = "improved" if row["cur_avg"] > row["prev_avg"] else "declined" if row["cur_avg"] < row["prev_avg"] else "stayed flat"
        return ChatResponse(reply=f"{emp_name}'s {label.lower()} has {direction}: {_fmt(row['prev_avg'])}% -> {_fmt(row['cur_avg'])}%.", rows=[row])

    if intent == "status_improving":
        statuses = [s.capitalize() for s in re.findall(r"\b(black|red)\b", message, re.I)] or ["Black", "Red"]
        statuses = list(dict.fromkeys(statuses))
        rows = queries.status_improving_ranking(statuses, _first_month(month), dept_name=dept_name, employee_ids=employee_ids, limit=limit)
        label = "/".join(statuses)
        if not rows:
            return ChatResponse(reply=f"No currently-{label} employees have a reliable improving trend this month{scope_note}.")
        if len(rows) == 1:
            r = rows[0]
            body = f"{r['emp_name']} ({r['dept_name']}) — currently {r['overall_std_pace_status']}, +{_fmt(r['pace_score_delta'])} pts this month"
        else:
            headers = ["#", "Employee", "Department", "Current status", "Change this month (pts)"]
            data = [[i, r["emp_name"], r["dept_name"], r["overall_std_pace_status"], f"+{_fmt(r['pace_score_delta'])}"] for i, r in enumerate(rows, 1)]
            body = _render_table(headers, data)
        return ChatResponse(reply=f"Currently-{label} employees improving month-over-month{scope_note}:\n\n" + body, rows=rows)

    if intent in ("ot_subscore", "wfh_subscore"):
        metric_key = _detect_subscore_key(message, default="engagement", include_working_hours=True)
        ranking_fn = queries.ot_subscore_ranking if intent == "ot_subscore" else queries.wfh_subscore_ranking
        emp_fn = queries.ot_subscore_for_employee if intent == "ot_subscore" else queries.wfh_subscore_for_employee
        context_label = "OT" if intent == "ot_subscore" else "WFH"
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is not None:
            row = emp_fn(emp_id, month=period_month, date_range=date_range)
            if not row or not row.get("sessions_counted"):
                return ChatResponse(reply=f"No {context_label} session data found for {emp_name}{_period_note(month, date_range)}.")
            return ChatResponse(
                reply=f"{emp_name}'s {context_label} sessions{_period_note(month, date_range)} ({row['sessions_counted']} sessions): "
                      f"Engagement {_fmt(row['engagement'])}%, Effectiveness {_fmt(row['effectiveness'])}%, "
                      f"Discipline {_fmt(row['discipline'])}%, Working hours {_fmt(row['working_hours'])}%.",
                rows=[row],
            )
        rows = ranking_fn(metric_key, dept_name, employee_ids=employee_ids, month=period_month, date_range=date_range, limit=limit)
        label = queries.CAPPED_SUBSCORES[metric_key][1]
        return ChatResponse(reply=f"{context_label} {label}{scope_note}:\n\n{format_count_rows(rows, 'metric_value', f'{label} %')}", rows=rows)

    if intent in ("ps_worked_emp", "ps_worked_ranking"):
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is not None:
            row = queries.ps_worked_ratio_for_employee(emp_id, month=period_month, date_range=date_range)
            if not row or not row.get("total_days"):
                return ChatResponse(reply=f"No PS working-status data found for {emp_name}{_period_note(month, date_range)}.")
            pct = (row["days_worked"] / row["total_days"]) * 100
            return ChatResponse(reply=f"{emp_name}{_period_note(month, date_range)}: PS working on {row['days_worked']} of "
                                       f"{row['total_days']} days ({_fmt(pct)}%). Note: this is a day-count ratio, not a "
                                       f"month-over-month trend — see report for why.", rows=[row])
        ascending = "not working" in message.lower() or "fewest" in message.lower()
        rows = queries.ps_worked_ratio_ranking(dept_name, employee_ids=employee_ids, month=period_month, date_range=date_range, ascending=ascending, limit=limit)
        return ChatResponse(reply=f"PS working-days ratio{scope_note}:\n\n{format_count_rows(rows, 'metric_value', 'PS working %')}", rows=rows)

    # --- NEW capability 1: day-specific COUNT / LIST ---
    if intent in ("day_count", "day_list"):
        flag_key = _detect_day_flag(message)
        # A day/week reference (date_range) is the common case; if none was
        # given, fall back to month scope (month defaults to the current
        # month when unmentioned, same as every other intent) instead of
        # erroring — reuses the exact same day_flag_count/day_flag_list
        # functions (both already accept `month` as an alternative to
        # `date_range`), so this is still the one query plumbing path, not a
        # parallel one.
        if flag_key is None:
            return ChatResponse(
                reply="I couldn't tell which attendance/leave/WFH/visit/etc. flag and which day you meant — "
                      "try e.g. \"how many employees were on WFH yesterday\" or \"who was on leave last week\"."
            )
        day_scope_note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")
        day_scope_note += " " + (_period_label_for_range(date_range) if date_range else _period_note(month, None))
        scope_month = None if date_range else period_month

        def _day_note(team_label, dept_name, month, date_range):
            note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else "")
            note += " " + (_period_label_for_range(date_range) if date_range else _period_note(month, None))
            return note

        if intent == "day_count":
            result = queries.day_flag_count(flag_key, dept_name=dept_name, employee_ids=employee_ids, month=scope_month, date_range=date_range)
            if session is not None:
                def _rerun_list(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=scope_month, date_range=date_range, limit=500, _flag_key=flag_key):
                    _rows = queries.day_flag_list(_flag_key, dept_name=dept_name, employee_ids=employee_ids, month=month, date_range=date_range, limit=limit)
                    return format_day_list(_rows, _flag_key, _day_note(team_label, dept_name, month, date_range) + " (full list)"), _rows

                def _rerun_same(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=scope_month, date_range=date_range, limit=500, _flag_key=flag_key):
                    _result = queries.day_flag_count(_flag_key, dept_name=dept_name, employee_ids=employee_ids, month=month, date_range=date_range)
                    return format_day_count(_result, _flag_key, _day_note(team_label, dept_name, month, date_range)), [_result]

                session_store.set_last_list(session, kind="day_flag", rerun_list=_rerun_list, rerun_same=_rerun_same,
                                             answer_kind="count", dept_name=dept_name, employee_ids=employee_ids,
                                             team_label=team_label, month=scope_month, date_range=date_range)
            return ChatResponse(reply=format_day_count(result, flag_key, day_scope_note), rows=[result])
        rows = queries.day_flag_list(flag_key, dept_name=dept_name, employee_ids=employee_ids, month=scope_month, date_range=date_range, limit=limit)
        if session is not None:
            def _rerun_list(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=scope_month, date_range=date_range, limit=500, _flag_key=flag_key):
                _rows = queries.day_flag_list(_flag_key, dept_name=dept_name, employee_ids=employee_ids, month=month, date_range=date_range, limit=limit)
                return format_day_list(_rows, _flag_key, _day_note(team_label, dept_name, month, date_range) + " (full list)"), _rows
            session_store.set_last_list(session, kind="day_flag", rerun_list=_rerun_list, rerun_same=_rerun_list,
                                         answer_kind="list", dept_name=dept_name, employee_ids=employee_ids,
                                         team_label=team_label, month=scope_month, date_range=date_range)
        return ChatResponse(reply=format_day_list(rows, flag_key, day_scope_note), rows=rows)

    # --- NEW capability 2: status-category filters ---
    if intent in ("status_list", "status_count", "status_distribution", "status_transitions"):
        # item #59 fix: no color word named in the message used to silently
        # default to ["Red", "Black"] here, narrowing an unqualified request
        # ("list of all the employees") to only Red/Black-status employees
        # and truncating the real headcount. `statuses=None` now means NO
        # status filter at all (every current status) - see
        # queries.status_list()/status_count()'s own docstrings. Scoped only
        # to status_list/status_count: status_transitions computes its own
        # from_status/to_status independently just below and never reads
        # this `statuses` variable, and status_distribution doesn't use it
        # either, so neither is affected by this default change - a message
        # that genuinely needs two named statuses for a transition still
        # requires them, this default never substitutes for that.
        statuses = [s.capitalize() for s in re.findall(r"\b(black|red|amber|green)\b", message, re.I)]
        statuses = list(dict.fromkeys(statuses)) or None
        st_scope_note = f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else " company-wide")

        def _status_note(team_label, dept_name):
            return f" for {team_label}" if team_label else (f" in {dept_name}" if dept_name else " company-wide")

        if intent == "status_list":
            rows = queries.status_list(statuses, dept_name=dept_name, employee_ids=employee_ids, limit=limit)
            if session is not None:
                def _rerun_list(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=None, date_range=None, limit=500, _statuses=statuses):
                    _rows = queries.status_list(_statuses, dept_name=dept_name, employee_ids=employee_ids, limit=limit)
                    return format_status_list(_rows, _statuses, _status_note(team_label, dept_name) + " (full list)"), _rows
                session_store.set_last_list(session, kind="status", rerun_list=_rerun_list, rerun_same=_rerun_list,
                                             answer_kind="list", dept_name=dept_name, employee_ids=employee_ids,
                                             team_label=team_label, statuses=statuses)
            return ChatResponse(reply=format_status_list(rows, statuses, st_scope_note), rows=rows)

        if intent == "status_count":
            n = queries.status_count(statuses, dept_name=dept_name, employee_ids=employee_ids)
            label = "/".join(statuses) if statuses else "all-status"
            if session is not None:
                def _rerun_list(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=None, date_range=None, limit=500, _statuses=statuses):
                    _rows = queries.status_list(_statuses, dept_name=dept_name, employee_ids=employee_ids, limit=limit)
                    return format_status_list(_rows, _statuses, _status_note(team_label, dept_name) + " (full list)"), _rows

                def _rerun_same(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=None, date_range=None, limit=500, _statuses=statuses):
                    _n = queries.status_count(_statuses, dept_name=dept_name, employee_ids=employee_ids)
                    _label = "/".join(_statuses) if _statuses else "all-status"
                    return f"{_n} employee(s) are currently {_label}{_status_note(team_label, dept_name)}.", [{"n": _n}]

                session_store.set_last_list(session, kind="status", rerun_list=_rerun_list, rerun_same=_rerun_same,
                                             answer_kind="count", dept_name=dept_name, employee_ids=employee_ids,
                                             team_label=team_label, statuses=statuses)
            return ChatResponse(reply=f"{n} employee(s) are currently {label}{st_scope_note}.", rows=[{"n": n}])

        if intent == "status_distribution":
            rows = queries.status_distribution_by_dept(limit=limit or 50)
            return ChatResponse(reply=f"PACE status distribution by department:\n\n{format_status_distribution(rows)}", rows=rows)

        if intent == "status_transitions":
            ml = message.lower()
            from_status = to_status = None
            m = re.search(r"from\s+(black|red|amber|green)\s+to\s+(black|red|amber|green)", ml)
            if m:
                from_status, to_status = m.group(1).capitalize(), m.group(2).capitalize()
            else:
                # "currently X but was Y last month" / "is X but was Y" -
                # current status comes first, prior status is the one tied
                # to "last month"/"was".
                m2 = re.search(r"(currently\s+|is\s+)(black|red|amber|green)\b.*\bwas\s+(black|red|amber|green)\b", ml)
                if m2:
                    to_status, from_status = m2.group(2).capitalize(), m2.group(3).capitalize()
            trend_month = _first_month(month) or entities.extract_month("")[0]
            rows = queries.status_transitions(trend_month, from_status=from_status, to_status=to_status,
                                                dept_name=dept_name, employee_ids=employee_ids, limit=limit)
            return ChatResponse(reply=format_status_transitions(rows, from_status, to_status), rows=rows)

    if intent == "status_emp":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        rows = queries.status_list(["Black", "Red", "Amber", "Green", "NJ"], dept_name=None, employee_ids=[emp_id], limit=1)
        if not rows:
            return ChatResponse(reply=f"No status data found for {emp_name}.")
        return ChatResponse(reply=f"{emp_name}'s current PACE status: {rows[0]['overall_std_pace_status']}.", rows=rows)

    # --- NEW capability 3: full multi-month trend history ---
    if intent == "full_trend_emp":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
            # No employee named - this intent is checked before
            # full_trend_dept, so a phrasing like "IT-Development's average
            # pace score trend" (no literal word "department") lands here
            # first; fall back to a department-scoped trend if one resolves.
            if dept_name:
                rows = queries.dept_full_monthly_trend(dept_name)
                return ChatResponse(reply=format_full_trend(rows, dept_name), rows=rows)
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code.")
        metric_key = _detect_full_trend_metric(message)
        if metric_key in queries.COUNT_METRICS:
            rows = queries.employee_monthly_count_trend(emp_id, metric_key)
            label = queries.COUNT_METRICS[metric_key][2]
            return ChatResponse(reply=format_full_trend(rows, emp_name, label), rows=rows)
        if metric_key != "pace_score" and metric_key not in queries.METRICS:
            metric_key = "pace_score"
        label = "PACE score" if metric_key == "pace_score" else queries.METRICS[metric_key][1]
        rows = queries.employee_full_monthly_trend(emp_id, metric_key=metric_key)
        return ChatResponse(reply=format_full_trend(rows, emp_name, label), rows=rows)

    if intent == "full_trend_dept":
        if not dept_name:
            return ChatResponse(reply="Which department did you mean?")
        rows = queries.dept_full_monthly_trend(dept_name)
        return ChatResponse(reply=format_full_trend(rows, dept_name), rows=rows)

    if intent == "full_trend_team":
        if employee_ids is None:
            return ChatResponse(reply="I need a resolved team to answer that — please ask about 'my team' after identifying yourself, or name a manager.")
        rows = queries.team_full_monthly_trend(employee_ids)
        return ChatResponse(reply=format_full_trend(rows, team_label or "your team"), rows=rows)

    # --- gap-category-E synonym intents that need a non-metric_ranking function ---
    if intent == "fewest_wfh":
        rows = queries.wfh_ranking(dept_name, month=period_month, date_range=date_range, employee_ids=employee_ids, limit=limit)
        rows = sorted(rows, key=lambda r: r["wfh_days"])[: (limit or queries.LIMIT)]
        return ChatResponse(reply=f"Fewest WFH days{scope_note}:\n\n{format_count_rows(rows, 'wfh_days', 'WFH days')}", rows=rows)

    return ChatResponse(reply=intents.FALLBACK_MESSAGE)


def _resolve_vague_list_followup(last_list, message, raw_message, session):
    """Re-run the query behind session['last_list'] (see
    session_store.set_last_list) in response to a vague follow-up ("list
    them", "who are they", "give me the list") - expanded to a full list,
    using the SAME filters that produced the original answer, UNLESS the
    current message explicitly re-scopes (a named department/status/time
    period), in which case that explicit piece overrides just that one
    filter (never silently blended, never ignored - see category F)."""
    dept_name, dept_candidates = entities.extract_department(message, fallback_text=raw_message)
    if dept_candidates:
        return ChatResponse(
            reply=f"I found multiple matching departments: {', '.join(dept_candidates)}. Which one did you mean?",
            needs_clarification=True, clarification_options=dept_candidates,
        )
    date_start, date_end, date_range_mentioned = entities.extract_date_range(message)
    new_date_range = (date_start, date_end) if date_range_mentioned else None
    months_list, month_mentioned = ([], False) if date_range_mentioned else entities.extract_months(message)
    new_month = (months_list if len(months_list) > 1 else (months_list[0] if months_list else None))

    # Explicit scope-broadening override ("in the whole company", "overall",
    # "company-wide", ...) always wins over whatever scope is currently
    # sticky, regardless of how that stickiness was set (named department,
    # "my team", fuzzy/typo-corrected match, ...) - drops to company-wide
    # (no department, no employee_ids/team_label) outright.
    _scope_override = _SCOPE_OVERRIDE_PATTERN.search(message) is not None

    eff_dept = None if _scope_override else (dept_name if dept_name else last_list["dept_name"])
    # A newly-named department switches the scope away from whatever team/
    # "my team" scoping produced the original answer - it's a different
    # scope entirely, not a refinement of it.
    eff_employee_ids = None if _scope_override else (last_list["employee_ids"] if dept_name is None else None)
    eff_team_label = None if _scope_override else (last_list["team_label"] if dept_name is None else None)
    if new_date_range is not None:
        eff_month, eff_date_range = None, new_date_range
    elif month_mentioned:
        eff_month, eff_date_range = new_month, None
    else:
        eff_month, eff_date_range = last_list["month"], last_list["date_range"]

    # Explicit status switch on a RANKING answer ("show me the black ones
    # instead") - re-route to a status list instead of expanding the
    # ranking, since that's what's actually being asked for.
    status_words = [s.capitalize() for s in re.findall(r"\b(black|red|amber|green)\b", message, re.I)]
    if status_words and last_list["kind"] == "ranking":
        rows = queries.status_list(status_words, dept_name=eff_dept, employee_ids=eff_employee_ids, limit=500)
        note = f" for {eff_team_label}" if eff_team_label else (f" in {eff_dept}" if eff_dept else " company-wide")
        reply = format_status_list(rows, status_words, note)
        session_store.set_last_list(session, kind="status", rerun_list=None, rerun_same=None, answer_kind="list",
                                     dept_name=eff_dept, employee_ids=eff_employee_ids, team_label=eff_team_label,
                                     statuses=status_words)
        return ChatResponse(reply=reply, rows=rows)

    # A bare re-scope-only follow-up ("what about last month") that does NOT
    # also ask for names/a list keeps the ORIGINAL answer's shape (e.g.
    # still a bare count) rather than being force-expanded into a list.
    use_same_shape = (
        (_VAGUE_RESCOPE_PATTERN.search(message) is not None or _scope_override)
        and _VAGUE_LIST_EXPAND_PATTERN.search(message) is None
        and last_list.get("rerun_same") is not None
    )
    rerun = last_list.get("rerun_same") if use_same_shape else last_list.get("rerun_list")
    if rerun is None:
        rerun = last_list.get("rerun_same") or last_list.get("rerun_list")
    if rerun is None:
        return ChatResponse(
            reply="I'm not able to expand that previous answer into a list right now — could you re-ask your "
                  "question with the specifics (department/time period/status)?"
        )

    reply, rows = rerun(dept_name=eff_dept, employee_ids=eff_employee_ids, team_label=eff_team_label,
                         month=eff_month, date_range=eff_date_range)
    return ChatResponse(reply=reply, rows=rows)


def handle_message(message: str, session_id: str = "default") -> ChatResponse:
    session = session_store.get_session(session_id)

    # --- Step 1: resume an identity request in progress ---
    if session["awaiting_identity"]:
        email, name_or_candidates, code = team.resolve_employee_email(message)
        if email:
            session["email"] = email
            session["emp_name"] = name_or_candidates
            session["emp_code"] = code
            session["awaiting_identity"] = False
            pending = session.pop("pending_message", None)
            session.pop("pending_message_raw", None)  # re-derived fresh below, this stale copy isn't needed
            if pending:
                return handle_message(pending, session_id)
            return ChatResponse(reply=f"Thanks, {name_or_candidates} — you're all set. What would you like to know?")
        if name_or_candidates:  # ambiguous match, candidates list
            return ChatResponse(
                reply=f"I found multiple employees matching that: {', '.join(name_or_candidates)}. "
                      f"Please give me the exact name or employee code.",
                needs_clarification=True,
                clarification_options=name_or_candidates,
            )
        return ChatResponse(
            reply="I couldn't find an employee matching that. Please try your exact full name or employee code."
        )

    # --- Step 2: resume an admin-access confirmation in progress ---
    if session["awaiting_admin_confirmation"]:
        pending = session.pop("pending_message", None)
        pending_raw = session.pop("pending_message_raw", None)
        session["awaiting_admin_confirmation"] = False
        if _CONFIRM_PATTERN.search(message):
            if pending:
                intent = intents.match_intent(pending)
                dept_name, _ = entities.extract_department(pending, fallback_text=pending_raw)
                p_start, p_end, p_mentioned = entities.extract_date_range(pending)
                p_date_range = (p_start, p_end) if p_mentioned else None
                _p_months, _ = ([], False) if p_mentioned else entities.extract_months(pending)
                month = _p_months if len(_p_months) > 1 else (_p_months[0] if _p_months else None)
                return answer_intent(intent, dept_name, month, None, None, employee_ids=None, team_label="the full company", message=pending, session=session, date_range=p_date_range, raw_message=pending_raw)
            return ChatResponse(reply="Okay — go ahead and ask your question again for the full company view.")
        # not a clear confirmation: fall through and treat this message as a fresh query
        return handle_message(message, session_id)

    # --- Step 3: resume a weekly-breakdown offer in progress ---
    # Cleared unconditionally below (whether or not this message turns out to
    # be the affirmative follow-up), so a later, unrelated message can never
    # accidentally reuse this stale state.
    if session["awaiting_weekly_breakdown"] or session["awaiting_ranking_weekly_breakdown"]:
        # Only one of these two is ever actually pending at a time (each
        # handler that sets one also clears the other - see emp_trend and
        # score_drop_ranking/score_improvement_alltime in answer_intent()),
        # but both are popped/cleared here unconditionally regardless of
        # which one fires, so stale state from either flow can never leak
        # into a later, unrelated message.
        was_awaiting_individual = session["awaiting_weekly_breakdown"]
        emp_id = session.pop("weekly_breakdown_employee_id", None)
        emp_name = session.pop("weekly_breakdown_employee_name", None)
        ranking_employee_ids = session.pop("ranking_weekly_breakdown_employee_ids", None)
        ranking_label = session.pop("ranking_weekly_breakdown_label", None)
        session["awaiting_weekly_breakdown"] = False
        session["awaiting_ranking_weekly_breakdown"] = False
        if was_awaiting_individual:
            if emp_id is not None and _WEEKLY_FOLLOWUP_PATTERN.search(message):
                rows = queries.employee_weekly_pace_trend(emp_id)
                # employee_weekly_pace_trend only restricts to Standard-shift
                # rows - it does NOT filter on PS status or visit status at
                # all (see queries.py docstring) - tracked here (not None,
                # not "working"/"no") so a later filter meta-follow-up
                # ("did you remove PS-not-working days from this?") answers
                # accurately instead of assuming the project-wide defaults.
                session_store.set_last_answer_filters(
                    session, label=f"{emp_name}'s week-by-week PACE trend",
                    ps_status=None, visit_status=None, shift_type="standard",
                )
                return ChatResponse(reply=format_weekly_trend(rows, emp_name), rows=rows)
        else:
            if ranking_employee_ids and _WEEKLY_FOLLOWUP_PATTERN.search(message):
                rows = queries.ranking_weekly_pace_trend(ranking_employee_ids)
                return ChatResponse(reply=format_ranking_weekly_trend(rows, ranking_label or "that scope"), rows=rows)
        # not a recognizable "yes"/"show weekly" follow-up: treat this message
        # as a fresh, unrelated query instead.
        return handle_message(message, session_id)

    # Offline typo correction (spellcheck.py) - only applied here, in the
    # normal query flow. Deliberately NOT applied during identity resolution
    # (awaiting_identity above) or the admin-confirmation pending-message
    # replay - names/codes and yes/no confirmations must stay exact as typed.
    raw_message = message
    message = spellcheck.correct_typos(message)

    # --- Bug 1 fix: vague "list the thing I was just shown" follow-up
    # resolution - deterministic, checked BEFORE LLM/rule-based intent
    # classification so it can never be second-guessed by either matcher,
    # and resolved against session["last_list"] (the most recent list/
    # count/ranking-producing answer), NOT the unrelated whole-session
    # sticky department/employee/time-period context (item #26/#29) and NOT
    # the pronoun "last discussed employee" state (item #30) - see
    # session_store.set_last_list for what does/doesn't set this. ---
    last_list = session_store.get_last_list(session)
    if last_list is not None and (
        _VAGUE_LIST_EXPAND_PATTERN.search(message) is not None
        or _VAGUE_RESCOPE_PATTERN.search(message) is not None
        # Explicit scope-broadening override ("in the whole company",
        # "overall", ...) with no other recognizable ranking/metric keyword
        # of its own - can only be a re-scope of whatever was last asked,
        # same as "what about ..." above. A message that ALSO carries its
        # own real query keywords is left to the normal intent pipeline
        # below (dept_best/rm_ranking/etc. are already company-wide by
        # definition and don't need this path at all).
        or (_SCOPE_OVERRIDE_PATTERN.search(message) is not None
            and intents.match_intent(message) is None)
    ):
        return _resolve_vague_list_followup(last_list, message, raw_message, session)
    if last_list is None and _VAGUE_LIST_EXPAND_STRICT.search(message) is not None:
        # These phrasings ("list them", "who are they", "show me their
        # names", ...) can NEVER be a meaningful standalone/fresh query -
        # there is nothing to guess at, so ask for clarification rather
        # than letting the LLM/rule matcher hallucinate a list from
        # unrelated context (category E safety requirement).
        return ChatResponse(
            reply="I don't have a specific list from our conversation to expand — could you tell me what you'd "
                  "like the names/list for (e.g. a department, status, or time period)?",
            needs_clarification=True,
        )

    # --- Filter/methodology meta-follow-up ("in this have u removed ps not
    # working days?") - checked BEFORE intent classification so it can never
    # be misrouted to an unrelated fresh-query intent (real repro: this
    # question right after an individual's weekly PACE trend was getting
    # routed to a completely unrelated company-wide ranking). See
    # _handle_filter_meta_followup above and
    # session_store.set_last_answer_filters/get_last_answer_filters. ---
    _filter_meta_response = _handle_filter_meta_followup(message, session)
    if _filter_meta_response is not None:
        return _filter_meta_response

    # --- Item #84: two DELIBERATELY-DEFERRED ambiguities (per this round's
    # task brief - a business-decision call, not something to guess at).
    # Both checked BEFORE intent classification, same "deterministic safety
    # check wins" precedent as the filter-meta-followup check above, so
    # neither can ever be silently answered by a guessed reading.
    #
    # (1) "which employees are driving <department>'s performance" is
    # genuinely ambiguous between at least 3 readings (top individual
    # scorers / biggest month-over-month improvers / biggest positive
    # deviation from the company average) - see SESSION_HANDOFF.md item
    # #83 section 5. Live-confirmed this round: without this check, the
    # "driving performance" clause was silently DROPPED (the rest of the
    # question, e.g. a department ranking, still answered) rather than
    # flagged - which is not a fabricated number, but also isn't the
    # controlled clarification the task requires. Checked on the raw
    # message so it fires regardless of what else the question also asks.
    _DRIVING_PERFORMANCE_PATTERN = re.compile(
        r"\bemployees?\b[^.?!]{0,60}\bdriving\b|\bdriving\b[^.?!]{0,60}\bperformance\b",
        re.IGNORECASE)
    if _DRIVING_PERFORMANCE_PATTERN.search(message):
        return ChatResponse(
            reply=(
                "\"Which employees are driving that performance\" could mean a few different things — "
                "which would you like?\n"
                "1. The top individual scorers in that scope right now\n"
                "2. The employees with the biggest month-over-month improvement\n"
                "3. The employees contributing the most above the company average\n\n"
                "Let me know which one, and I can pull that up (or ask me any other part of your "
                "question separately in the meantime)."
            ),
            needs_clarification=True,
            clarification_options=["Top scorers", "Biggest improvers", "Biggest above-average contributors"],
        )

    # (2) "highest AND lowest" in one question (e.g. "who has the highest
    # and lowest PACE among WFH employees") - the exact row-count semantics
    # (top-1+bottom-1 of the full filtered population vs. of whatever N is
    # shown) is a NEW operation type not yet wired up (see SESSION_HANDOFF.md
    # item #83 section 5) - live-confirmed this round: without this check,
    # the question silently answered with just a single-direction ranking
    # table, ignoring the "and lowest"/"and highest" half entirely. A
    # controlled fallback here beats guessing which semantics to apply.
    _BOTH_ENDS_PATTERN = re.compile(
        r"\bhighest\b[^.?!]{0,40}\band\b[^.?!]{0,10}\blowest\b"
        r"|\blowest\b[^.?!]{0,40}\band\b[^.?!]{0,10}\bhighest\b"
        r"|\bbest\b[^.?!]{0,40}\band\b[^.?!]{0,10}\bworst\b"
        r"|\bworst\b[^.?!]{0,40}\band\b[^.?!]{0,10}\bbest\b",
        re.IGNORECASE)
    if _BOTH_ENDS_PATTERN.search(message):
        return ChatResponse(
            reply=(
                "I can look up the highest or lowest separately — which would you like, or both "
                "(as two separate answers)?"
            ),
            needs_clarification=True,
            clarification_options=["Highest only", "Lowest only", "Both, separately"],
        )

    # --- Bare superlative direction follow-up ("least", "most", "highest",
    # "lowest", ...) right after a ranking (item #63) - checked BEFORE intent
    # classification (both rule-based AND the fuzzy fallback inside
    # intents.match_intent) so a bare direction word can never be misrouted
    # by the fuzzy matcher's single-contained-word collision (see the
    # comment on _BARE_DIRECTION_LOW/_BARE_DIRECTION_HIGH above). ---
    _bare_direction_response = _handle_bare_direction_followup(message, session)
    if _bare_direction_response is not None:
        return _bare_direction_response

    # --- List-population pronoun follow-up ("their pace score" / "there
    # pace score" right after a department/RM-team roster or ranking, item
    # #67) - checked BEFORE intent classification, same reasoning as the two
    # checks immediately above: a plural population reference must never be
    # misrouted by the fuzzy matcher or the LLM into an unrelated fresh
    # query. See _handle_list_pronoun_metric_followup above. ---
    _list_pronoun_response = _handle_list_pronoun_metric_followup(message, session)
    if _list_pronoun_response is not None:
        return _list_pronoun_response

    # --- LLM-first intent classification (Gemini), with rule-based fallback
    # and safety cross-check ---
    # The rule-based matcher (intents.match_intent) is ALWAYS computed too,
    # both as the fallback when Gemini is unavailable/fails, AND as an
    # independent second opinion used purely for the opposite-direction
    # safety check below. Gemini's output only ever selects an intent NAME
    # from the exact same fixed intent set already used by the rule-based
    # matcher (validated in llm_nlu.classify) - it never invents a new code
    # path, never touches the DB, and never produces the final answer; the
    # rest of this function (dept/employee/month extraction, answer_intent(),
    # queries.py) runs completely unchanged regardless of which matcher
    # picked the intent.
    rule_intent = intents.match_intent(message)
    llm_result = llm_nlu.classify(raw_message)

    # Item #76 (Phase 3, Part B): pace_score_best/pace_score_worst's own
    # regex patterns (app/intents.py) are broad "top N employees"/"most/
    # least score"-shaped matches that fire even when the message names a
    # DIFFERENT metric entirely - live-verified this round: "top 5 employees
    # by engagement last week" matched pace_score_best, which (per
    # _METRIC_INTENTS above) always ranks by plain pace_score regardless of
    # what was actually asked - the exact silent-wrong-metric failure mode
    # this project must eliminate, just discovered in an OLD intent rather
    # than the new cascade this time. Not this intent's FUNCTION being
    # touched (per this round's constraints) - only a redirect, same
    # established "narrow deterministic override" pattern as
    # _NEW_VOCAB_OVERRIDE_PATTERN below: whenever the raw message names a
    # specific non-pace_score metric, null the match so classify()/the
    # extraction cascade's ranking support (which correctly detects the
    # named metric) gets the turn instead.
    _PACE_SCORE_BEST_WRONG_METRIC_PATTERN = re.compile(
        r"\b(engagement|effectiveness|discipline|working\s*hours?|worked\s*hours?|capped|"
        r"late[- ]?comings?|early[- ]?leav\w*|deficient|productive|whatsapp|"
        r"ai\s*(tool|min)|tools?\s*(and|&)\s*mail)\b", re.IGNORECASE)
    if (rule_intent in ("pace_score_best", "pace_score_worst")
            and _PACE_SCORE_BEST_WRONG_METRIC_PATTERN.search(message)):
        rule_intent = None

    # Item #76 (Phase 3, Part B): the same silent-wrong-population failure
    # mode, found in a second old-intent family this round. _EMP_FIELD_INTENTS
    # (emp_pace_score/emp_late_comings/emp_engagement/etc.) all resolve
    # through a single plain whole-month lookup with NO ps/visit/shift/WFH
    # filter support at all - live-verified: "late comings for Aryan Gupta on
    # OT days in August" matched emp_late_comings and silently returned his
    # WHOLE-MONTH late-coming count (2), completely ignoring "on OT days".
    # Same redirect-not-modify fix as above: whenever the raw message
    # actually names an explicit population filter that build_query() DOES
    # support (reusing the exact same deterministic detector
    # _detect_build_query_filters() already uses for the rule-based
    # build_query() callers), null the match so classify()/the extraction
    # cascade - which correctly applies the named filter - gets the turn.
    if rule_intent in _EMP_FIELD_INTENTS and _detect_build_query_filters(message):
        rule_intent = None

    # Item #76 (Phase 3, Part B): a third instance of the same failure class,
    # this time a DIMENSION mismatch rather than a metric/filter one. A
    # family of old intents (declining/improving/most_disciplined/
    # least_disciplined/engagement_high|low/effectiveness_high|low/
    # most_late_comings/most_early_leavings/most_deficient_hours/
    # highest_working_pct) always rank/report at EMPLOYEE grain - live-
    # verified: "which department has the biggest pace score drop this
    # month" matched "declining" (which has no department-vs-department
    # concept at all) and silently answered "no employees had enough data",
    # ignoring that a DEPARTMENT-level ranking was asked for; "which
    # manager's team has the lowest discipline this month" matched
    # "least_disciplined" and returned an employee list instead of a
    # per-manager-team ranking. dept_best/dept_worst/rm_ranking_best/
    # rm_ranking_worst already solve exactly this for the "score"/pct-capped
    # wording they explicitly cover (item #73) - this extends the same
    # "dimension-explicit phrasing wins" precedent as a redirect (not a
    # rewrite) for every OTHER employee-level-only intent: whenever the
    # message explicitly names "department"/"dept" or "RM/manager/reporting
    # manager team" as the ranking scope, null the employee-level match so
    # classify()/the extraction cascade's ranking support (dimension=
    # department|rm) - or, failing that, the clearly-labeled AI-generated
    # sql_fallback - answers the dimension actually asked for instead of a
    # silently-substituted employee-level one.
    _EMPLOYEE_LEVEL_ONLY_RANKING_INTENTS = {
        "declining", "improving", "most_disciplined", "least_disciplined",
        "engagement_high", "engagement_low", "effectiveness_high", "effectiveness_low",
        "most_late_comings", "most_early_leavings", "most_deficient_hours", "highest_working_pct",
    }
    _DIMENSION_SCOPE_OVERRIDE_PATTERN = re.compile(
        r"\bwhich\s+(?:dept|department)\b|\b(?:dept|department)\b[^.?!]{0,20}\b(?:most|least|highest|lowest|best|worst|top|bottom|biggest)\b"
        r"|\bwhich\s+(?:rm|manager|reporting manager)(?:'s)?\s*team\b"
        r"|\b(?:rm|manager|reporting manager)(?:'s)?\s*team\b[^.?!]{0,20}\b(?:most|least|highest|lowest|best|worst|top|bottom|biggest)\b",
        re.IGNORECASE,
    )
    if (rule_intent in _EMPLOYEE_LEVEL_ONLY_RANKING_INTENTS
            and _DIMENSION_SCOPE_OVERRIDE_PATTERN.search(message)):
        rule_intent = None

    # Item #79 follow-up 2 (this round): the 9798ddd redirect below nulled
    # rule_intent to None, on the theory that "the cascade" would then pick
    # up average_metric instead. Live-verified this round that this DIDN'T
    # actually fix anything: "average meeting minutes for Rudhi" (Rudhi is
    # a real, unambiguous, single-match employee) still returned the same
    # company-wide top-10 ranking table. Root cause: intents.match_intent()
    # returns the FIRST pattern match only (app/intents.py, _INTENTS is a
    # flat ordered list) - it was never re-invoked here, so nulling
    # rule_intent to None does NOT retroactively try average_metric's own
    # pattern. Once rule_intent is None, control falls to
    # `elif llm_result is not None: intent = llm_intent` a few dozen lines
    # below - and llm_nlu.py's own few-shot examples explicitly map
    # "meeting minutes ranking"-shaped phrasing to intent="meeting_min_ranking"
    # (see FEW_SHOT list), so Gemini's independent guess for this exact
    # phrasing is ALSO "meeting_min_ranking" essentially every time,
    # silently reproducing the identical bug via the LLM path instead of the
    # rule path. The actual fix: since we already know (by construction -
    # the regex below is checked first) that the message contains the
    # literal "meeting minutes" wording that _AVG_METRIC_WORD/
    # _AVERAGE_METRIC_PATTERNS (app/intents.py) recognizes, deterministically
    # route to "average_metric" directly instead of nulling to None - this
    # guarantees the correctly-working rule-based handler above (with its
    # existing extract_employee()/extract_department()/extract_manager()
    # resolution, including proper entities.Ambiguous clarification
    # handling - see its handler above) gets the turn regardless of what
    # either match_intent()'s first-match-wins or the LLM's own classify()
    # guess would otherwise have picked, since rule_intent-is-not-None wins
    # outright over the LLM unconditionally (see "General precedence flip"
    # below). Other meeting_min_ranking phrasings that DON'T literally say
    # "meeting minutes" (e.g. "time in meetings") aren't covered by
    # BUILD_QUERY_METRICS at all yet, so those still fall back to nulling to
    # None (unchanged prior behavior - out of this round's confirmed-repro
    # scope, see SESSION_HANDOFF.md item #79 follow-up 2).
    if rule_intent == "meeting_min_ranking" and re.search(r"\b(avg|average|mean)\b", message, re.IGNORECASE):
        if re.search(r"\bmeeting\s*minutes?\b", message, re.IGNORECASE):
            rule_intent = "average_metric"
        else:
            rule_intent = None

    # Item #82: the SAME collision class as the meeting_min_ranking block
    # immediately above, found in a different old-intent family this round.
    # _EMP_ENGAGEMENT_PATTERNS/_EMP_DISCIPLINE_PATTERNS/_EMP_EFFECTIVENESS_
    # PATTERNS/_EMP_WORKING_PCT_PATTERNS (app/intents.py) each include a
    # broad r"\b<metric> (%|percent|percentage) (of|for)\b" pattern meant
    # for single-employee lookups ("engagement percentage for Rudhi") - but
    # that pattern is only a SUBSTRING match, so it also fires on "average
    # engagement percentage for the whole company" (the substring
    # "engagement percentage for" is still present), and because these
    # emp_* intents are registered earlier in intents.py's _INTENTS list
    # than average_metric (checked first, first-match-wins), they steal the
    # message before average_metric ever gets a turn. The handler then tries
    # to resolve "the whole company" as an employee name and fails ("I
    # couldn't find that employee") instead of ever computing a company-wide
    # average - the exact silent-wrong-intent failure mode already fixed
    # once above for meeting_min_ranking. Confirmed via grep across all of
    # app/intents.py that these four are the only _EMP_*_PATTERNS with this
    # exact "<metric> (%|percent|percentage) (of|for)" shape colliding with
    # an _AVG_METRIC_WORD entry (emp_late_comings/emp_early_leavings/
    # emp_deficient_hours use "of|for" without the %/percent/percentage
    # wording and are not part of _AVG_METRIC_WORD's own metric-word list,
    # so they don't collide the same way and are left untouched).
    # Same fix, same rationale: since we already know (by construction) the
    # message contains "avg"/"average"/"mean" wording that _AVG_METRIC_WORD/
    # _AVERAGE_METRIC_PATTERNS recognizes for exactly these four bare metric
    # words (engagement/effectiveness/discipline/working hours are all
    # listed in _AVG_METRIC_WORD, app/intents.py), deterministically
    # redirect to average_metric rather than nulling to None - this
    # guarantees the correctly-working rule-based handler (with its own
    # employee -> dept -> RM -> company-wide scope resolution) gets the
    # turn. A genuine single-employee query with no avg/average/mean wording
    # ("engagement percentage for Rudhi") never matches this condition and
    # is completely unaffected - none of these four intents or their
    # underlying functions are otherwise touched.
    if rule_intent in (
        "emp_engagement", "emp_discipline", "emp_effectiveness", "emp_working_pct",
    ) and re.search(r"\b(avg|average|mean)\b", message, re.IGNORECASE):
        rule_intent = "average_metric"

    # Item #84 (Finding 3): the SAME collision class as the two blocks
    # immediately above, in a third old-intent family. _SUBSCORE_TREND_
    # PATTERNS (intents.py, intent "subscore_trend_emp") matches ANY message
    # containing an area word (engagement/effectiveness/discipline) followed
    # later by "improv*"/"declin*" wording, ANYWHERE in the message, with no
    # requirement that a specific employee actually be named - live-
    # confirmed this round: "which employees had the biggest engagement
    # decline vs last month" and "who are the employees whose PACE score has
    # declined the most compared with the previous month" both matched this
    # single-employee-only intent, whose handler then tried (and failed) to
    # resolve "employees"/"the employees whose PACE score" as ONE employee's
    # name and returned "I couldn't find that employee" instead of ever
    # attempting the company-/department-wide ranking the question actually
    # asked for. Same redirect-not-rewrite fix, same rationale: reuse the
    # SAME entities.extract_employee() resolution the handler itself would
    # use, and only redirect when it genuinely finds no specific named
    # employee in the message (a multi-match still lets subscore_trend_emp's
    # own clarification fire, unchanged - _extract_employee_ctx below would
    # raise Ambiguous the same way). Redirects to the new
    # "subscore_delta_ranking" intent (handled above, in the main dispatch),
    # which reuses queries.subscore_delta_ranking() - a new, generalized
    # month-over-month sub-metric delta ranking (previously this only
    # existed for whole-PACE-score deltas via score_drop_ranking/
    # score_improvement_alltime) rather than guessing at an unsupported
    # shape.
    if rule_intent == "subscore_trend_emp":
        try:
            _std_emp_id, _ = entities.extract_employee(message, fallback_text=raw_message)
        except entities.Ambiguous:
            _std_emp_id = "ambiguous"  # let subscore_trend_emp's own clarification fire, unchanged
        if not _std_emp_id:
            rule_intent = "subscore_delta_ranking"

    # Item #84 (Finding 3 follow-up, discovered via this round's own live-
    # test instruction): "who are the employees whose PACE score has
    # declined the most compared with the previous month" does NOT match
    # _SUBSCORE_TREND_PATTERNS (no engagement/effectiveness/discipline word
    # present) - live-confirmed this round it instead reaches match_intent()'s
    # FUZZY fallback (_fuzzy_match_intent), which matches "full_trend_emp"
    # against that intent's own few-shot examples ("pace score trend", "score
    # trend over time", "month on month score") purely on string similarity,
    # with - same class of bug - no requirement that a specific employee be
    # named. full_trend_emp's handler then fails the identical way
    # ("I couldn't find that employee"). Same redirect-not-rewrite fix: when
    # no employee (and no department) is resolvable, this is really a
    # company-/department-wide PACE-SCORE trend ranking (not a sub-metric
    # one - no engagement/effectiveness/discipline word was named, so
    # subscore_delta_ranking above doesn't apply) - redirect to the
    # EXISTING score_drop_ranking/score_improvement_alltime intents
    # (item #74/#75's already-verified precomputed pace_score_delta
    # ranking), picking direction from the same declin*/drop*/fell vs
    # improv*/increas* wording used elsewhere in this cascade.
    if rule_intent == "full_trend_emp":
        try:
            _fte_emp_id, _ = entities.extract_employee(message, fallback_text=raw_message)
        except entities.Ambiguous:
            _fte_emp_id = "ambiguous"
        _fte_dept_name, _fte_dept_candidates = entities.extract_department(message, fallback_text=raw_message)
        if not _fte_emp_id and not _fte_dept_name and not _fte_dept_candidates:
            if re.search(r"\b(declin\w*|drop\w*|fell|decreas\w*|worse)\b", message, re.IGNORECASE):
                rule_intent = "score_drop_ranking"
            elif re.search(r"\b(improv\w*|increas\w*|better)\b", message, re.IGNORECASE):
                rule_intent = "score_improvement_alltime"

    # Item #72 (see the fuller override comment below): live testing found
    # this goes deeper than classify() alone - some of these phrasings ALSO
    # explicit-regex-match an old intent's pattern directly (e.g. "day level
    # pace score for Accounts department..." matches emp_pace_score's
    # r"\bpace score (of|for)\b", which normally correctly wins outright
    # against everything else - see the "General precedence flip" comment
    # below). None of the ~123 existing intents' patterns were written with
    # this new vocabulary in mind, so a rule_intent match here is never
    # actually correct for these specific markers - null it out too (same
    # pattern-trust rationale, just applied one step earlier in the
    # pipeline) so the cascade can reach extract_build_query().
    # Item #73: dept_best/dept_worst are exempted from this nulling - unlike
    # the ~123 other existing intents, these two were EXTENDED this round
    # (new explicit patterns in intents.py) specifically to understand this
    # new vocabulary (capped/percentage-qualified effectiveness/engagement/
    # discipline) and resolve the real metric themselves, deterministically,
    # via the shared _detect_pct_capped_metrics() normalizer in their
    # handler below - nulling them here would incorrectly divert a
    # correctly-matched "which department has the best/worst X" ranking
    # question to the LLM extraction cascade or sql_fallback instead of the
    # now-metric-aware rule-based handler.
    # Item #79: "average_metric"/"meeting_had_emp" are exempted for the same
    # reason - both are the CORRECT deterministic handlers for the new
    # engagement_minutes/meeting_minutes/tasks_*/todos_*/had-any-meetings
    # vocabulary this round added (average_metric via the new
    # BUILD_QUERY_METRICS entries + extended _AVG_METRIC_WORD; meeting_had_emp
    # via the new DAY_FLAGS-reuse boolean intent) - nulling them here would
    # send an already-correctly-resolved match to the LLM cascade instead.
    if (rule_intent is not None
            and rule_intent not in ("dept_best", "dept_worst", "average_metric", "meeting_had_emp")
            and _NEW_VOCAB_OVERRIDE_PATTERN.search(message)):
        rule_intent = None

    # --- New-vocabulary deterministic override (item #72) ------------------
    # classify()'s system prompt (llm_nlu.py) already instructs the LLM to
    # return "none" for these phrasings (capped_* metrics, scoped/period-
    # qualified pace status, day-level/event-level pace score, precomputed
    # dept score) so extract_build_query()'s more precise cascade step gets a
    # chance - but that's a probabilistic prompt-engineering guardrail, and
    # live testing (item #72) found classify() still occasionally guesses an
    # old percentage/status/plain-pace_score intent for these anyway (e.g.
    # "day level pace score for Accounts department over the last 2 weeks"
    # -> an old dept intent using the keyword-only _detect_build_query_metrics
    # detector, which has no entry for pace_score_day_level at all and
    # silently substitutes plain pace_score - exactly the silent-wrong-metric
    # failure mode this round is required to eliminate). This regex-based
    # override is DETERMINISTIC (same pattern-trust rationale as the
    # existing _ps_override/_gainer_loser_override/etc. above): only fires
    # when rule_intent is None (never overrides a real rule-based match,
    # same safety invariant as everywhere else in this function), and simply
    # discards llm_result so the cascade proceeds to the `intent is None`
    # branch, giving extract_build_query() the chance the prompt guardrail
    # alone couldn't reliably guarantee.
    if rule_intent is None and llm_result is not None and _NEW_VOCAB_OVERRIDE_PATTERN.search(message):
        llm_result = None

    # Pronoun override (see _PRONOUN_PATTERN above): a message referring to a
    # person via "he"/"she"/etc. that the rule-based matcher already resolved
    # to an individual-employee-scoped intent always wins over the LLM's
    # intent choice, since Gemini has no session context to know the pronoun
    # refers to whoever was last individually discussed and otherwise tends
    # to guess a generic org-wide ranking intent instead. This is checked
    # BEFORE the llm_result branch below so it short-circuits that logic
    # entirely (including the opposite-direction clarification check, which
    # would otherwise fire spuriously e.g. LLM="improving" vs rule="declining"
    # style conflicts caused purely by the LLM's missing context, not a real
    # ambiguity in the user's own wording).
    _pronoun_override = (
        _PRONOUN_PATTERN.search(message) is not None
        and rule_intent in _INDIVIDUAL_EMP_INTENTS
    )

    # PS-exclusion override (Part 3): these three intents (ps_exclude_metric/
    # ps_ratio_info/ps_explain) are brand new and Gemini has no few-shot
    # examples for them, so it tends to misclassify a PS-worded question as
    # whatever generic metric it superficially resembles (e.g. reading "how
    # many days..." as a deficient-hours count). The rule-based regexes for
    # this family are narrow/explicit and rarely false-positive, so trust
    # them over the LLM whenever they fire, same rationale as the pronoun
    # override above.
    _ps_override = rule_intent in ("ps_exclude_metric", "ps_ratio_info", "ps_explain")

    # PS-not-installed override: same rationale as _ps_override above - this
    # DAY_FLAGS flag is brand new and Gemini tends to pull "ps not installed"
    # wording toward the older, more familiar ps_install_rate intent (a
    # department PERCENTAGE, not this per-day distinct-employee COUNT/LIST)
    # even with a few-shot example added. The rule-based day_count/day_list
    # regexes for this flag are narrow/explicit, so trust them over the LLM
    # whenever they fire.
    _ps_not_installed_override = (
        rule_intent in ("day_count", "day_list") and _detect_day_flag(message) == "ps_not_installed"
    )

    # Gainer/loser direction override: Gemini has no few-shot examples yet
    # for the direction-only phrasings ("best/worst performers", "who
    # dropped/improved the most", "losser" typo) added for Item 1, and tends
    # to misclassify them as a generic metric-ranking intent instead. The
    # rule-based _GAINER_LOSER_PATTERNS regexes are narrow/explicit and
    # rarely false-positive, so trust them over the LLM whenever they fire -
    # same rationale as the PS overrides above.
    _gainer_loser_override = rule_intent == "gainer_loser_ranking"

    # --- General precedence flip (this round) ---------------------------
    # Historically this block preferred llm_intent whenever Gemini/OpenAI
    # responded, with a small set of hand-built deterministic overrides
    # (_pronoun_override / _ps_override / _ps_not_installed_override /
    # _gainer_loser_override) bolted on top to force specific known-bad
    # misclassification patterns back to the rule-based answer. Each of
    # those overrides was really the same underlying lesson: the narrow,
    # explicit regex matcher is usually MORE trustworthy than the LLM once
    # it has actually fired, because it can't be fooled by phrasing outside
    # its few-shot coverage the way the LLM can. This round generalizes
    # that lesson: whenever the rule-based matcher finds ANY intent at all
    # (rule_intent is not None), it wins outright. The LLM is now consulted
    # to resolve intent ONLY when the rule-based matcher found nothing
    # (rule_intent is None) - i.e. it fills gaps rather than second-guessing
    # matches. The four old overrides are kept in place below (not deleted)
    # as explicit, self-documenting special cases of this same rule, purely
    # as a readability/safety-net aid - they are now redundant with the
    # general rule (each fires only when rule_intent already matched one of
    # those specific intents, which now always wins anyway) - the four
    # `_..._override` booleans above are now DEAD/unused variables (no
    # longer referenced in the branch below); they are left in place,
    # uncalled, purely as documentation of the specific failure patterns the
    # general rule now subsumes, and can be deleted in a future cleanup pass
    # once this round's live regression testing has stood for a while. See
    # SESSION_HANDOFF.md Part 1
    # of this round for the regression-risk discussion (LLM correctly
    # overriding a WRONG rule match, or handling genuinely novel phrasing
    # the rule-based matcher used to incorrectly claim, are both now LOST
    # whenever rule_intent is non-None but wrong - flagged explicitly).
    # Low-confidence guess kept around ONLY for the Part-3 "did you mean X?"
    # cascade below - never used to answer directly.
    llm_low_confidence_guess = None
    if rule_intent is not None:
        intent = rule_intent
    elif llm_result is not None:
        llm_intent = llm_result["intent"]
        llm_confidence = llm_result.get("confidence") or 0.0
        if llm_intent == "none" or llm_confidence < _LLM_LOW_CONFIDENCE_THRESHOLD:
            # Rule-based matcher found nothing AND the LLM either found
            # nothing ("none") or is not confident enough to trust outright.
            # Part 3: don't guess and don't immediately show the generic
            # fallback either - try the SQL-generation fallback path first
            # (handled by the `intent is None` branch further below), and
            # keep the LLM's own guess (if it made one) around only as a
            # "did you mean X?" suggestion for if/when SQL-fallback itself
            # fails to produce anything sensible.
            if llm_intent != "none":
                llm_low_confidence_guess = llm_intent
            intent = None
        else:
            intent = llm_intent
            # Entity-hint augmentation: splice any employee/department name Gemini
            # extracted into the text that the EXISTING extract_employee()/
            # extract_department() functions parse. This lets those unchanged,
            # safety-checked resolvers (exact match -> fuzzy match, same as
            # always) pick up a name Gemini normalized/understood but that the
            # rule-based regex text-matching might otherwise miss (e.g. a casual
            # phrasing or minor misspelling) - Gemini never resolves the name
            # itself, it only proposes text that flows through the same
            # resolution/safety pipeline as before.
            llm_entities = llm_result.get("entities", {})
            hint_bits = []
            for key in ("employee", "department", "manager"):
                val = llm_entities.get(key)
                if val and val.lower() not in message.lower():
                    hint_bits.append(str(val))
            if hint_bits:
                message = message + " " + " ".join(hint_bits)
    else:
        intent = None

    # Opposite-direction safety cross-check (kept fully intact, per explicit
    # instruction - this is a deliberately SEPARATE safety mechanism, not an
    # artifact of the old precedence order). Previously this only ran on the
    # "LLM won" branch; it now also has to be checked when rule_intent won,
    # since rule_intent winning unconditionally could otherwise silently
    # suppress a case where the LLM confidently flagged the OPPOSITE
    # direction from what the rule-based matcher matched - that disagreement
    # is exactly the signal this check exists to catch, regardless of which
    # side "wins" for the final answer.
    if llm_result is not None and rule_intent is not None:
        llm_intent = llm_result["intent"]
        opposite_of_llm = intents._OPPOSITE_INTENTS.get(llm_intent)
        if opposite_of_llm is not None and rule_intent == opposite_of_llm:
            return ChatResponse(
                reply=(
                    "I'm not fully sure which direction you mean — did you want "
                    f"\"{llm_intent.replace('_', ' ')}\" or \"{opposite_of_llm.replace('_', ' ')}\"? "
                    "Please rephrase more specifically (e.g. use \"best\"/\"worst\" or "
                    "\"improving\"/\"declining\" explicitly)."
                ),
                needs_clarification=True,
                clarification_options=[llm_intent, opposite_of_llm],
            )

    if intent == "month_compare":
        # Month-vs-month comparison has its own dedicated resolution (see
        # _handle_month_compare), mirroring day_compare's short-circuit
        # immediately below for the same reason: the generic dept-extraction/
        # self-referential/named-manager machinery doesn't know about
        # two-month comparisons.
        return _handle_month_compare(message, raw_message, session)

    if intent == "day_compare":
        # Day-vs-day / metric comparison has its own dedicated dept/employee
        # resolution (see _handle_day_compare) - deliberately short-circuits
        # here, BEFORE the generic dept-extraction/self-referential/
        # named-manager machinery below, since none of that machinery knows
        # about two-date comparisons and this intent's population is always
        # "the full department/company on a fixed day," never a "my team"
        # scope in this round.
        return _handle_day_compare(message, raw_message, session)

    if intent == "roster_list":
        # item #66: generic "give me complete list of <dept/team> employees"
        # phrasing - route deterministically through the SAME
        # build_query_overview_reply() engine "how is X doing" uses, so the
        # employee count/roster this returns is guaranteed to share the
        # exact same population definition (period/PS/visit/shift filters)
        # as that summary, instead of either the wrong status_list intent
        # (item #66's original bug) or the non-deterministic sql_fallback
        # LLM path (which has no such consistency guarantee and costs real
        # spend). Resolution order mirrors the intent-is-None build_query
        # fallback below exactly (employee -> department -> RM).
        # Same "team" wording -> manager-resolved-first order as item #64's
        # average_metric (see its comment above) - several managers in this
        # dataset are ALSO individual employees (e.g. Nikhil Kumar), so
        # without this, "complete list of <name>'s team employees" would
        # resolve to that person's own single-row list instead of their
        # team's roster.
        _rl_wants_team = re.search(r"\bteam\b", message, re.IGNORECASE) is not None
        _rl_emp_id = _rl_emp_name = None
        _rl_mgr_id = _rl_mgr_name = None
        if _rl_wants_team:
            try:
                _rl_mgr_id, _rl_mgr_name = entities.extract_manager(message, fallback_text=raw_message)
            except entities.Ambiguous as e:
                return ChatResponse(reply=f"Multiple managers match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                     needs_clarification=True, clarification_options=e.candidates)
            if not _rl_mgr_id:
                try:
                    _rl_emp_id, _rl_emp_name = entities.extract_employee(message, fallback_text=raw_message)
                except entities.Ambiguous:
                    pass  # a bare name-fragment collision on "team" wording - not a real employee lookup, ignore
        else:
            try:
                _rl_emp_id, _rl_emp_name = entities.extract_employee(message, fallback_text=raw_message)
            except entities.Ambiguous as e:
                return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                     needs_clarification=True, clarification_options=e.candidates)
            if not _rl_emp_id:
                try:
                    _rl_mgr_id, _rl_mgr_name = entities.extract_manager(message, fallback_text=raw_message)
                except entities.Ambiguous:
                    _rl_mgr_id, _rl_mgr_name = None, None
        _rl_dept_name, _rl_dept_candidates = entities.extract_department(message, fallback_text=raw_message)
        if _rl_dept_candidates:
            return ChatResponse(
                reply=f"I found multiple matching departments: {', '.join(_rl_dept_candidates)}. Which one did you mean?",
                needs_clarification=True, clarification_options=_rl_dept_candidates,
            )
        _rl_date_start, _rl_date_end, _rl_date_mentioned = entities.extract_date_range(message)
        _rl_period = (_rl_date_start, _rl_date_end) if _rl_date_mentioned else None
        if _rl_emp_id:
            reply, rows = build_query_overview_reply("employee", _rl_emp_id, message, period=_rl_period, session=session, display_name=_rl_emp_name)
            return ChatResponse(reply=reply, rows=rows)
        if _rl_dept_name:
            reply, rows = build_query_overview_reply("department", _rl_dept_name, message, period=_rl_period, session=session)
            return ChatResponse(reply=reply, rows=rows)
        if _rl_mgr_id:
            # Same universal-access admin-confirmation gate every other
            # named-manager "team" query in this file goes through (see the
            # team.resolve_named_person_team() call further below in this
            # function) - this branch returns early (before that shared
            # gate), so it must apply the same check itself rather than
            # silently bypassing it (a real gap found during item #66 live
            # testing: without this, "list of <universal-access manager>'s
            # team employees" would answer with a real company-wide-sized
            # roster instead of the required admin-confirmation prompt).
            if _rl_wants_team:
                _rl_ids, _rl_is_universal, _rl_resolved_name, _rl_candidates = team.resolve_named_person_team(_rl_mgr_name)
                if _rl_candidates:
                    return ChatResponse(
                        reply=f"Multiple employees match '{_rl_mgr_name}': {', '.join(_rl_candidates)}. Which one did you mean?",
                        needs_clarification=True, clarification_options=_rl_candidates,
                    )
                if _rl_is_universal:
                    session["awaiting_admin_confirmation"] = True
                    session["pending_message"] = message
                    session["pending_message_raw"] = raw_message
                    return ChatResponse(
                        reply=f"{_rl_mgr_name} has admin-level access, so their 'team' would mean essentially the "
                              f"whole company. Please specify a department instead, or explicitly confirm "
                              f"(\"yes\" / \"full company\") if you really want a company-wide view.",
                        needs_clarification=True,
                    )
            reply, rows = build_query_overview_reply("rm", _rl_mgr_name, message, period=_rl_period, session=session)
            return ChatResponse(reply=reply, rows=rows)
        reply, rows = build_query_overview_reply("company", None, message, period=_rl_period, session=session)
        return ChatResponse(reply=reply, rows=rows)

    if intent is None:
        # Neither the rule-based matcher nor a confident LLM classification
        # matched an existing intent (rule_intent is None, and either the LLM
        # also found nothing or its self-reported confidence was below
        # _LLM_LOW_CONFIDENCE_THRESHOLD - see the branch above). Part 3 cascade:
        # try the SQL-generation fallback path FIRST, automatically, before
        # showing any clarification/generic-fallback message - the goal is a
        # natural direct answer, not an extra confirm-dialog round-trip. This
        # is a distinct, explicitly-authorized code path (see sql_fallback.py)
        # that drafts read-only SQL with GPT-5 mini and executes it through
        # the same existing DB connection, already labeled AI-generated/
        # unverified in its own reply text.
        #
        # Item #70 (this round): a NEW extraction-LLM cascade step runs FIRST,
        # ahead of sql_fallback - it tries to parse the message into a
        # build_query() call (dimension/metrics/filters/period), which is
        # cheaper, more deterministic, and reuses already-verified SQL versus
        # free-form SQL generation. Only engages when it can extract and
        # validate something real; any failure/hallucination falls straight
        # through to sql_fallback below unaffected, so raw-SQL stays the
        # final last-resort for genuine matrix outliers (tasks/todos, calls/
        # meetings, d_score, etc. - things build_query() cannot express).
        try:
            extraction_result = _extraction_llm_reply(raw_message, message, session)
        except Exception:
            logging.getLogger("pace_chatbot.main").exception("_extraction_llm_reply() raised unexpectedly")
            extraction_result = None
        if extraction_result is not None:
            _ext_reply, _ext_rows = extraction_result
            return ChatResponse(reply=_ext_reply, rows=_ext_rows)

        try:
            fallback_result = sql_fallback.answer(raw_message)
        except Exception:
            logging.getLogger("pace_chatbot.main").exception("sql_fallback.answer() raised unexpectedly")
            fallback_result = None
        if fallback_result is not None:
            return ChatResponse(reply=fallback_result["reply"], rows=fallback_result.get("rows", []))
        # SQL-fallback itself failed to produce anything sensible (empty
        # result, a SQL execution error, the safety check rejected the
        # generated query, or it raised) - ONLY NOW fall through to a
        # clarification. If the LLM had a low-confidence guess, offer it as
        # a "did you mean X?" suggestion instead of the bare generic message.
        if llm_low_confidence_guess is not None:
            guess_label = llm_low_confidence_guess.replace("_", " ")
            return ChatResponse(
                reply=(
                    f"I couldn't confidently answer that. Did you mean something like a \"{guess_label}\" "
                    "question? Could you rephrase or be more specific?"
                ),
                needs_clarification=True,
                clarification_options=[llm_low_confidence_guess],
            )
        # ADDITIVE fallback (build_query() engine): only reached when NONE of
        # the ~87 hand-verified rule-based intents matched, the LLM
        # classification also found nothing usable, AND the SQL-generation
        # fallback above produced nothing - i.e. this is strictly lower
        # priority than every existing intent/path, by construction (all of
        # them return before reaching here). Covers bare dimension+overview
        # phrasing with no existing pattern coverage, e.g. a named employee/
        # department/RM with no recognized metric/ranking keyword at all.
        try:
            _bq_emp_id, _bq_emp_name = entities.extract_employee(message, fallback_text=raw_message)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        _bq_dept_name, _bq_dept_candidates = entities.extract_department(message, fallback_text=raw_message)
        try:
            _bq_mgr_id, _bq_mgr_name = entities.extract_manager(message, fallback_text=raw_message)
        except entities.Ambiguous:
            _bq_mgr_id, _bq_mgr_name = None, None
        _bq_date_start, _bq_date_end, _bq_date_mentioned = entities.extract_date_range(message)
        _bq_period = (_bq_date_start, _bq_date_end) if _bq_date_mentioned else None
        if _bq_emp_id:
            reply, rows = build_query_overview_reply("employee", _bq_emp_id, message, period=_bq_period, session=session, display_name=_bq_emp_name)
            return ChatResponse(reply=reply, rows=rows)
        if _bq_dept_name and not _bq_dept_candidates:
            reply, rows = build_query_overview_reply("department", _bq_dept_name, message, period=_bq_period, session=session)
            return ChatResponse(reply=reply, rows=rows)
        if _bq_mgr_id:
            reply, rows = build_query_overview_reply("rm", _bq_mgr_name, message, period=_bq_period, session=session)
            return ChatResponse(reply=reply, rows=rows)
        return ChatResponse(reply=intents.FALLBACK_MESSAGE)

    # Pass the RAW (pre-spellcheck) text as a fallback: dictionary spellcheck
    # can occasionally corrupt a genuinely truncated department/name fragment
    # into an unrelated short word (e.g. "offi" -> "off"), which would
    # otherwise silently kill a match the raw text could still resolve.
    dept_name, dept_candidates = entities.extract_department(message, fallback_text=raw_message)
    if dept_candidates and intent not in ("dept_compare", "employee_compare", "team_compare"):
        return ChatResponse(
            reply=f"I found multiple matching departments: {', '.join(dept_candidates)}. Which one did you mean?",
            needs_clarification=True,
            clarification_options=dept_candidates,
        )

    # Response-SHAPE continuity fix: a message that names ONLY a department
    # (no metric/ranking keyword of its own) defaults to intent
    # "dept_summary" - a department-level AGGREGATE. But if the session's
    # last list-producing answer was a per-EMPLOYEE ranking (kind=="ranking"
    # in session_store's last_list, e.g. the "who has the least score"
    # conversation), a bare department mention right after it reads as
    # ADDING SCOPE to that ongoing ranking ("...in AI Labs") - not a request
    # to switch to a completely different response shape (a company-wide-
    # style department summary). Only fires when a department was actually
    # NAMED this turn (dept_name is not None here, before any sticky-context
    # fallback runs below) and a rerun_list callable is available to
    # actually re-scope the prior ranking to it.
    if intent == "dept_summary" and dept_name is not None and session is not None:
        _last = session_store.get_last_list(session)
        if _last is not None and _last.get("kind") == "ranking" and _last.get("rerun_list") is not None:
            reply, rows = _last["rerun_list"](dept_name=dept_name)
            session_store.push_context(session, dept_name=dept_name)
            return ChatResponse(reply=reply, rows=rows)

    date_start, date_end, date_range_mentioned = entities.extract_date_range(message)
    date_range = (date_start, date_end) if date_range_mentioned else None
    # If a day/week reference was given, don't also default month to "current
    # month" - the new query functions take date_range OR month, not both.
    # Multi-month support ("wfh in june, july, august"): extract_months()
    # collects EVERY month named (extract_month() only ever returned the
    # first). `month` stays a plain scalar 'YYYY-MM' string (or None) in
    # the single/no-mention case - identical to the old extract_month()
    # result - and only becomes a list when 2+ distinct months are named,
    # so every existing single-month call site below is unaffected; the
    # query-layer functions (_period_filter and the legacy VIEW-based
    # ranking functions in queries.py) accept both transparently via
    # queries._month_param(). A few trend/delta functions (this-month-vs-
    # prior-month comparisons) don't have defined multi-month semantics and
    # explicitly collapse back to the first month via _first_month() at
    # their call sites below.
    _months_list, month_mentioned = (([], False) if date_range_mentioned else entities.extract_months(message))
    month = _months_list if len(_months_list) > 1 else (_months_list[0] if _months_list else None)

    # "total [visit/wfh/leave] taken by X" with NO month/date named at all
    # must NOT silently narrow to the current month (extract_months() above
    # defaults `month` to the current month with month_mentioned=False when
    # nothing was said) - "total" with no period implies the whole available
    # data window, so reroute to the same month-by-month breakdown used by
    # the "[metric] month wise" phrasing (full_trend_emp / _detect_full_trend_metric,
    # which already resolves "wfh"/"visit"/"leave" from the message text and
    # queries the full history with no month filter at all). Only fires when
    # the user actually wrote "total" AND named neither a month nor a
    # date/week reference - an explicit period ("total visits in July")
    # still scopes normally via the existing visit_emp/wfh_emp/leave_emp_check
    # path below, untouched.
    if (intent in ("visit_emp", "wfh_emp", "leave_emp_check")
            and not month_mentioned and not date_range_mentioned
            and re.search(r"\btotal\b", message, re.I)):
        intent = "full_trend_emp"

    # Explicit-month/date-range override for full_trend_emp (extends the
    # _PRONOUN_PATTERN "deterministic override wins regardless of which
    # classifier proposed what" pattern to this case): a month-by-month
    # BREAKDOWN only makes sense when no single period was named. If the
    # message explicitly names a month or date range, "total visits by X in
    # July" can only mean the single July value, never a table spanning
    # every month - so full_trend_emp is never the right final intent here,
    # no matter whether the rule-based matcher (via the "total" reroute
    # right above) or Gemini (which has its own few-shot bias toward
    # full_trend_emp for "breakdown"/"trend" phrasing, independent of the
    # rule-based path and this file's month/date extraction) is the one
    # that proposed it. Force it back to the matching single-value emp
    # intent for that period, using the same metric-detection function
    # full_trend_emp itself uses so the metric being asked about doesn't
    # change - only whether it's rendered as one value or a full table.
    if intent == "full_trend_emp" and (month_mentioned or date_range_mentioned):
        _trend_metric_key = _detect_full_trend_metric(message)
        _single_intent = _FULL_TREND_METRIC_TO_SINGLE_INTENT.get(_trend_metric_key)
        if _single_intent is not None:
            intent = _single_intent

    # --- Conversational context carry-forward (feature) ---
    # Remember exactly what THIS message explicitly named, before any
    # fallback fills gaps in - this is what gets recorded into the rolling
    # history below, so the history only ever reflects real mentions, never
    # propagated/inherited guesses (which would otherwise let stale context
    # live forever).
    _explicit_dept_this_turn = dept_name
    _explicit_month_this_turn = month if month_mentioned else None
    _explicit_date_range_this_turn = date_range if date_range_mentioned else None

    # Department fallback: only for intents that are NOT individual-employee
    # lookups (an individual lookup resolves its own employee directly and
    # must never inherit a department scope meant for an unrelated ranking
    # query - e.g. "pace score of Aryan Gupta" right after "who's in red in
    # Founders Office" should NOT scope Aryan's own lookup to that dept).
    if (dept_name is None and intent not in _INDIVIDUAL_EMP_INTENTS and intent not in _DUAL_PURPOSE_EMP_INTENTS
            and intent not in _DEPT_LEVEL_RANKING_INTENTS):
        dept_name = session_store.get_recent_context(session, "dept_name")

    # Time-period fallback: only when the current message named neither a
    # month nor a date range, and the intent's semantics aren't a fixed
    # current-vs-prior comparison (see _PERIOD_CONTEXT_BLACKLIST above).
    if not month_mentioned and not date_range_mentioned and intent not in _PERIOD_CONTEXT_BLACKLIST:
        _ctx_date_range = session_store.get_recent_context(session, "date_range")
        if _ctx_date_range is not None:
            date_range = _ctx_date_range
        else:
            _ctx_month = session_store.get_recent_context(session, "month")
            if _ctx_month is not None:
                month = _ctx_month

    session_store.push_context(
        session, dept_name=_explicit_dept_this_turn,
        month=_explicit_month_this_turn, date_range=_explicit_date_range_this_turn,
    )

    # --- Self-referential "my team" path ---
    # Score-drop/score-improvement questions with NO department named are
    # treated as implicitly self-referential too, so "whose score dropped
    # the most?" (no dept mentioned) scopes to the asker's own access -
    # same identity flow, same team.resolve_team()/admin-confirmation check
    # as an explicit "my team" question - rather than silently running
    # unscoped across the whole company.
    _implicit_self_ref = intent in ("score_drop_ranking", "score_improvement_alltime") and not dept_name
    _is_explicit_self_ref = entities.is_self_referential(message)
    if _is_explicit_self_ref or _implicit_self_ref:
        if session["email"] is None:
            session["awaiting_identity"] = True
            session["pending_message"] = message
            session["pending_message_raw"] = raw_message
            return ChatResponse(
                reply="To look up your team, please tell me your full name or employee code.",
                needs_clarification=True,
            )

        employee_ids, is_universal = team.resolve_team(session["email"])
        if is_universal:
            # Admin (universal-access) users asking "my team"/"my
            # <something>" specifically (the literal self-referential
            # phrasing, "my"/"our") get the full company DIRECTLY, no
            # confirmation prompt - this was a confirmed business-rule
            # change. The named-manager path ("X's team" where X happens to
            # be a universal-access admin, handled further below) and the
            # _implicit_self_ref case (no "my"/"our" wording at all, just no
            # department named) deliberately keep the confirmation prompt -
            # the business rule is scoped specifically to the asker's own
            # literal "my team", not other self-referential-adjacent paths.
            if _is_explicit_self_ref:
                return answer_intent(intent, None, month, None, None, employee_ids=employee_ids,
                                      team_label="the full company", message=message, session=session,
                                      date_range=date_range, raw_message=raw_message)
            session["awaiting_admin_confirmation"] = True
            session["pending_message"] = message
            session["pending_message_raw"] = raw_message
            return ChatResponse(
                reply="You have admin-level access, so 'my team' would mean essentially the whole company. "
                      "Please specify a department or manager instead, or explicitly confirm "
                      "(\"yes\" / \"full company\") if you really want a company-wide view.",
                needs_clarification=True,
            )

        return answer_intent(intent, None, month, None, None, employee_ids=employee_ids, team_label=f"{session['emp_name']}'s team", message=message, session=session, date_range=date_range, raw_message=raw_message)

    # --- Individual-employee bypass ---
    # Genuinely single-employee intents (e.g. "is X improving", "pace score
    # of X", "was X on leave") must resolve the employee directly via
    # entities.extract_employee() inside answer_intent() and must NEVER go
    # through extract_manager()/resolve_named_person_team() below - that
    # team-resolution path (and its universal-access admin-confirmation
    # check) exists for genuine team/group-scope queries ("my team", "[name]'s
    # team", department-less rankings), not for a lookup about one named
    # person. Without this bypass, asking about an individual who ALSO
    # happens to manage people (or who happens to be a universal-access admin
    # email) incorrectly triggered "this person has admin access, please
    # specify a department" instead of just answering about that person.
    _bypass_team_routing = intent in _INDIVIDUAL_EMP_INTENTS
    if not _bypass_team_routing and intent in _DUAL_PURPOSE_EMP_INTENTS:
        # ot_subscore/wfh_subscore/ps_worked_ranking cover BOTH an individual
        # lookup and a team/dept ranking under one intent name - only bypass
        # when the message actually names a specific employee.
        try:
            _probe_emp_id, _probe_emp_name = entities.extract_employee(message, fallback_text=raw_message)
            _bypass_team_routing = _probe_emp_id is not None
        except entities.Ambiguous:
            # A name WAS given, just ambiguous - still an individual-shaped
            # query, so bypass and let answer_intent's own extract_employee
            # call raise (and report) the same Ambiguous error to the user.
            _bypass_team_routing = True
    if _bypass_team_routing:
        return answer_intent(intent, dept_name, month, None, None, message=message, session=session, date_range=date_range, raw_message=raw_message)

    try:
        # Same raw-text fallback already used for extract_department: spellcheck
        # can corrupt a real proper name into an unrelated English word (e.g.
        # "Yadav" -> "Adam"), which would otherwise silently fail to find the
        # named manager and fall through to an unscoped answer instead.
        manager_id, manager_name = entities.extract_manager(message, fallback_text=raw_message)
    except entities.Ambiguous as e:
        return ChatResponse(
            reply=f"Multiple managers match that name: {', '.join(e.candidates)}. Which one did you mean?",
            needs_clarification=True,
            clarification_options=e.candidates,
        )

    # --- Named-manager "team" path: resolve via the SAME email-access
    # pipeline as "my team", so "Nikhil Kumar's team" and Nikhil Kumar
    # himself asking "my team" always agree. ---
    if manager_name:
        employee_ids, is_universal, resolved_name, candidates = team.resolve_named_person_team(manager_name)
        if candidates:
            return ChatResponse(
                reply=f"Multiple employees match '{manager_name}': {', '.join(candidates)}. Which one did you mean?",
                needs_clarification=True,
                clarification_options=candidates,
            )
        if employee_ids is None:
            # Couldn't resolve this manager to an employee/email via the team
            # pipeline (e.g. no email_access row) - fall back to the old
            # reporting_user_id-based path rather than dropping the query.
            return answer_intent(intent, dept_name, month, manager_id, manager_name, message=message, session=session, date_range=date_range, raw_message=raw_message)
        if is_universal:
            session["awaiting_admin_confirmation"] = True
            session["pending_message"] = message
            session["pending_message_raw"] = raw_message
            return ChatResponse(
                reply=f"{manager_name} has admin-level access, so their 'team' would mean essentially the whole "
                      f"company. Please specify a department instead, or explicitly confirm "
                      f"(\"yes\" / \"full company\") if you really want a company-wide view.",
                needs_clarification=True,
            )
        return answer_intent(
            intent, None, month, None, None,
            employee_ids=employee_ids, team_label=f"{manager_name}'s team", message=message, session=session,
            date_range=date_range, raw_message=raw_message,
        )

    return answer_intent(intent, dept_name, month, manager_id, manager_name, message=message, session=session, date_range=date_range, raw_message=raw_message)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return handle_message(req.message, req.session_id)


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/dashboard.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")
