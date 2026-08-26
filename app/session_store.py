"""In-memory per-session state. No persistence, no auth - the app has none
today and this is scoped to add just enough state to identify "who is
asking" for a session's lifetime, consistent with the existing single-shared-
access prototype design. Resets on server restart.
"""

_SESSIONS = {}


def get_session(session_id):
    return _SESSIONS.setdefault(session_id, {
        "email": None,
        "emp_name": None,
        "emp_code": None,
        "awaiting_identity": False,
        "awaiting_admin_confirmation": False,
        "pending_message": None,
        "pending_message_raw": None,
        # Set right after answering an individual "is X improving"-style
        # monthly trend question, so a same-session "yes"/"show weekly"
        # follow-up can be answered inline without re-naming the employee.
        # Cleared unconditionally on the very next message, whether or not
        # that message was actually the affirmative follow-up, so a later
        # unrelated query never accidentally reuses stale state.
        "awaiting_weekly_breakdown": False,
        "weekly_breakdown_employee_id": None,
        "weekly_breakdown_employee_name": None,
        # Same pattern as the individual awaiting_weekly_breakdown fields
        # above, but for the RANKING follow-up (score_drop_ranking /
        # score_improvement_alltime) offering a weekly breakdown of the
        # same ranked employee set instead of one named employee. Kept as
        # distinct fields (not reusing the individual ones) so the two
        # flows can never collide or leak state into each other; both sets
        # are cleared unconditionally on the very next message in
        # main.handle_message, regardless of which one (if either) was
        # actually pending.
        "awaiting_ranking_weekly_breakdown": False,
        "ranking_weekly_breakdown_employee_ids": None,
        "ranking_weekly_breakdown_label": None,
        # Conversational context carry-forward: the single most-recently
        # EXPLICITLY-named department/employee/time-period, kept STICKY for
        # the entire session (not a rolling window) - so a follow-up that
        # omits one of these can fall back to whatever was last explicitly
        # in scope, no matter how many messages back it was mentioned (e.g.
        # "who's in red in Founders Office" -> six unrelated questions later
        # -> "who all are in black?" should still stay scoped to Founders
        # Office). Overwritten whenever a NEW explicit value appears; never
        # expires on its own within a session. A brand-new session_id starts
        # with all of these None, so there is no cross-session leakage - see
        # push_context()/get_recent_context() below. Only ever used as a
        # FALLBACK when the current message's own extraction finds nothing;
        # never overrides an explicit value in the current message.
        "sticky_context": {
            "dept_name": None,
            "employee_id": None,
            "employee_name": None,
            "month": None,
            "date_range": None,
        },
        # The single most-recent LIST-PRODUCING answer (a day-flag count/
        # list, a status count/list, or a ranking) - kept SEPARATE from
        # sticky_context above (which is about filling in MISSING scope for
        # a NEW question) and from the pronoun-resolution "last discussed
        # employee" state. This exists to answer a vague follow-up like
        # "list them" / "who are they" / "show me their names" by
        # re-running the SAME query that produced the most recent list-type
        # answer, expanded to a full list, using the SAME filters that
        # produced it - not some older/stickier tracked value and not an
        # unrelated individual-employee lookup that happened in between.
        # Deliberately only ever set by call sites that produce a genuine
        # list/count/ranking answer (see main.py's set_last_list() call
        # sites) - single-employee lookups must NEVER overwrite this.
        "last_list": None,
    })


def push_context(session, dept_name=None, employee_id=None, employee_name=None, month=None, date_range=None):
    """Record what was EXPLICITLY named in this turn (pass None for anything
    not mentioned this turn - do not pass through an already-inherited
    value, so this only reflects real mentions, not propagated guesses).
    Each non-None field OVERWRITES the sticky value for that field and it
    then persists for the rest of the session (until overwritten again),
    rather than aging out after a fixed number of turns."""
    ctx = session.setdefault("sticky_context", {
        "dept_name": None, "employee_id": None, "employee_name": None,
        "month": None, "date_range": None,
    })
    if dept_name is not None:
        ctx["dept_name"] = dept_name
    if employee_id is not None:
        ctx["employee_id"] = employee_id
        ctx["employee_name"] = employee_name
    if month is not None:
        ctx["month"] = month
    if date_range is not None:
        ctx["date_range"] = date_range


def set_last_list(session, kind, rerun_list=None, rerun_same=None, answer_kind="count",
                   dept_name=None, employee_ids=None, team_label=None, month=None,
                   date_range=None, statuses=None):
    """Record the most recent LIST-PRODUCING answer, for resolving vague
    follow-ups ("list them", "who are they", "show me their names") against
    the CORRECT prior answer instead of stale sticky_context or an
    unrelated individual-employee lookup.

    `kind` - a rough category tag ('day_flag' | 'status' | 'ranking'),
    mostly useful for status<->ranking switching on an explicit re-scope
    ("show me the black ones instead").
    `rerun_list` - a zero-default-arg callable(dept_name=, employee_ids=,
    team_label=, month=, date_range=, limit=) -> (reply_text, rows) that
    re-runs the SAME underlying query EXPANDED to a full list/names (used
    for "list them"/"show me their names"-style follow-ups).
    `rerun_same` - same signature, but re-runs the query in its ORIGINAL
    answer shape (e.g. still a bare count) - used for a follow-up that only
    re-scopes (e.g. "what about last month") without asking for names.
    `answer_kind` - 'count' or 'list', whichever shape the ORIGINAL answer
    was, so a pure re-scope follow-up can preserve it.
    The remaining kwargs are the filters that produced the original answer,
    stored so a follow-up's explicit re-scoping (a named department/status/
    time period) can override just that one piece rather than starting
    over."""
    session["last_list"] = {
        "kind": kind, "rerun_list": rerun_list, "rerun_same": rerun_same,
        "answer_kind": answer_kind, "dept_name": dept_name, "employee_ids": employee_ids,
        "team_label": team_label, "month": month, "date_range": date_range,
        "statuses": statuses,
    }


def get_last_list(session):
    return session.get("last_list")


def clear_last_list(session):
    session["last_list"] = None


def get_recent_context(session, field):
    """Returns the sticky value currently held for `field` ('dept_name',
    'employee_id', 'employee_name', 'month', or 'date_range'), or None if
    nothing has been explicitly mentioned yet this session. Name kept as
    `get_recent_context` for call-site compatibility, though the value is
    no longer window-limited - it's whatever was last explicitly set,
    persisted for the whole session."""
    ctx = session.get("sticky_context") or {}
    return ctx.get(field)
