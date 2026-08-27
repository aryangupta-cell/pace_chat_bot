import datetime
import logging
import re

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import entities, queries, intents, team, session_store, spellcheck, llm_nlu

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
    r"|give me the (full |whole )?list\b"
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
    r")",
    re.IGNORECASE,
)

_VAGUE_LIST_EXPAND_PATTERN = re.compile(
    r"(" + _VAGUE_LIST_EXPAND_STRICT.pattern + "|" + _VAGUE_LIST_EXPAND_LOOSE.pattern + r")",
    re.IGNORECASE,
)

# A narrower "re-scope only" follow-up ("what about last month", "what about
# next week") - no explicit ask for names/a list, just a change of time
# period/department applied to the SAME prior answer, kept in its ORIGINAL
# answer shape (e.g. still a bare count) rather than force-expanded to a
# list. Deliberately gated to short messages starting with this phrase, to
# minimize collision with unrelated fresh queries.
_VAGUE_RESCOPE_PATTERN = re.compile(r"^\s*what about\b", re.IGNORECASE)


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
        ("offline", [r"\boffline\b", r"ps not installed", r"ps installed"]),
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
    "defaulter": "were marked defaulter", "offline": "were offline / PS not installed",
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
        "were marked defaulter": "was marked defaulter", "were offline / PS not installed": "was offline / PS not installed",
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
    label = "/".join(statuses)
    if not rows:
        return f"No one is currently {label}{scope_note}."
    if len(rows) == 1:
        r = rows[0]
        return f"{r['emp_name']} ({r['dept_name']}) — {r['overall_std_pace_status']}"
    headers = ["#", "Employee", "Department", "Status"]
    data = [[i, r["emp_name"], r["dept_name"], r["overall_std_pace_status"]] for i, r in enumerate(rows, 1)]
    return f"Employees currently {label}{scope_note}:\n\n" + _render_table(headers, data)


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
    "emp_attendance_summary", "emp_trend", "emp_trend_2month", "emp_overview",
    "subscore_compare_emp", "subscore_trend_emp", "d_score_trend", "d_score_emp",
    "leave_emp_check", "call_emp", "visit_emp", "wfh_emp",
    "shift_type_emp", "breakshift_emp", "offline_emp", "meeting_ratio_emp",
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
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Best attendance{scope_note}:\n\n{format_attendance_rows(rows)}", rows=rows)

    if intent == "attendance_worst":
        rows = queries.attendance_ranking(dept_name, month, worst=True, employee_ids=employee_ids, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.attendance_ranking(dept_name, month, worst=True, employee_ids=employee_ids, limit=limit)
                return f"Worst attendance{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_attendance_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Worst attendance{scope_note}:\n\n{format_attendance_rows(rows)}", rows=rows)

    if intent == "productive_high":
        rows = queries.productive_time_ranking(dept_name, month, lowest=False, employee_ids=employee_ids, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.productive_time_ranking(dept_name, month, lowest=False, employee_ids=employee_ids, limit=limit)
                return f"Most productive time{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_productive_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Most productive time{scope_note}:\n\n{format_productive_rows(rows)}", rows=rows)

    if intent == "productive_low":
        rows = queries.productive_time_ranking(dept_name, month, lowest=True, employee_ids=employee_ids, limit=limit)
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500):
                _rows = queries.productive_time_ranking(dept_name, month, lowest=True, employee_ids=employee_ids, limit=limit)
                return f"Least productive time{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_productive_rows(_rows)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Least productive time{scope_note}:\n\n{format_productive_rows(rows)}", rows=rows)

    # --- Category B/C: generic metric rankings ---
    if intent in _METRIC_INTENTS:
        metric_key, ascending = _METRIC_INTENTS[intent]
        rows = queries.metric_ranking(
            metric_key, dept_name, month, ascending=ascending, employee_ids=employee_ids,
            limit=limit, reporting_user_id=manager_id if employee_ids is None else None,
        )
        label = queries.METRICS[metric_key][1]
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500,
                       _metric_key=metric_key, _ascending=ascending, _label=label, _rid=manager_id):
                _rows = queries.metric_ranking(_metric_key, dept_name, month, ascending=_ascending, employee_ids=employee_ids,
                                                limit=limit, reporting_user_id=_rid if employee_ids is None else None)
                return f"Ranked by {_label}{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_metric_rows(_rows, _metric_key)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Ranked by {label}{scope_note}:\n\n{format_metric_rows(rows, metric_key)}", rows=rows)

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
            return ChatResponse(reply="I couldn't find that employee — please give me their exact full name or employee code, or a department for a ranking.")
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
        metric_key = "pace_score"
        ascending = intent == "dept_worst"
        rows = queries.dept_ranking(metric_key, month, ascending=ascending, limit=limit)
        return ChatResponse(reply=f"Departments ranked by {queries.METRICS[metric_key][1]}:\n\n{format_dept_rows(rows, metric_key)}", rows=rows)

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
        rows, meta = queries.pace_score_trend_ranking(
            dept_name, _first_month(month), declining=(intent == "declining"),
            reporting_user_id=manager_id if employee_ids is None else None,
            employee_ids=employee_ids, limit=limit,
        )
        label = "declining" if intent == "declining" else "improving"
        if session is not None:
            def _rerun(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range, limit=500,
                       _declining=(intent == "declining"), _rid=manager_id, _label=label):
                _rows, _meta = queries.pace_score_trend_ranking(
                    dept_name, _first_month(month), declining=_declining,
                    reporting_user_id=_rid if employee_ids is None else None,
                    employee_ids=employee_ids, limit=limit,
                )
                return f"Who is {_label}{_scope_note_generic(team_label, dept_name, month, date_range)} (full list):\n\n{format_trend_rows(_rows, _meta)}", _rows
            session_store.set_last_list(session, kind="ranking", rerun_list=_rerun, answer_kind="list",
                                         dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=month, date_range=date_range)
        return ChatResponse(reply=f"Who is {label}{scope_note}:\n\n{format_trend_rows(rows, meta)}", rows=rows)

    # --- Category A (new): Leave & absence ---
    if intent in ("leave_emp_check", "call_emp", "visit_emp", "wfh_emp", "d_score_emp",
                  "shift_type_emp", "breakshift_emp", "offline_emp", "meeting_ratio_emp"):
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

    if intent == "emp_overview":
        try:
            emp_id, emp_name = _extract_employee_ctx(message, fb, session)
        except entities.Ambiguous as e:
            return ChatResponse(reply=f"Multiple employees match that name: {', '.join(e.candidates)}. Which one did you mean?",
                                 needs_clarification=True, clarification_options=e.candidates)
        if emp_id is None:
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
                      "try e.g. \"how many employees were on WFH yesterday\" or \"who was on leave today\"."
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
        statuses = [s.capitalize() for s in re.findall(r"\b(black|red|amber|green)\b", message, re.I)]
        statuses = list(dict.fromkeys(statuses)) or ["Red", "Black"]
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
            label = "/".join(statuses)
            if session is not None:
                def _rerun_list(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=None, date_range=None, limit=500, _statuses=statuses):
                    _rows = queries.status_list(_statuses, dept_name=dept_name, employee_ids=employee_ids, limit=limit)
                    return format_status_list(_rows, _statuses, _status_note(team_label, dept_name) + " (full list)"), _rows

                def _rerun_same(dept_name=dept_name, employee_ids=employee_ids, team_label=team_label, month=None, date_range=None, limit=500, _statuses=statuses):
                    _n = queries.status_count(_statuses, dept_name=dept_name, employee_ids=employee_ids)
                    _label = "/".join(_statuses)
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

    eff_dept = dept_name if dept_name else last_list["dept_name"]
    # A newly-named department switches the scope away from whatever team/
    # "my team" scoping produced the original answer - it's a different
    # scope entirely, not a refinement of it.
    eff_employee_ids = last_list["employee_ids"] if dept_name is None else None
    eff_team_label = last_list["team_label"] if dept_name is None else None
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
        _VAGUE_RESCOPE_PATTERN.search(message) is not None
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

    if _pronoun_override:
        intent = rule_intent
    elif _ps_override:
        intent = rule_intent
    elif llm_result is not None:
        llm_intent = llm_result["intent"]
        # Safety cross-check: for opposite-direction-sensitive intents, if
        # the independent rule-based matcher ALSO confidently landed on the
        # exact opposite intent, don't guess - ask the user to clarify
        # instead of silently picking one direction.
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
        intent = rule_intent

    if intent is None:
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
    if dept_name is None and intent not in _INDIVIDUAL_EMP_INTENTS and intent not in _DUAL_PURPOSE_EMP_INTENTS:
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
    if entities.is_self_referential(message) or _implicit_self_ref:
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
