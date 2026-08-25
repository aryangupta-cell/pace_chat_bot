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
        # Conversational context carry-forward: a rolling short history (most
        # recent last) of what department/employee/time-period was EXPLICITLY
        # named in recent turns, so a follow-up that omits one of these can
        # fall back to what was recently in scope (e.g. "who's in red in
        # Founders Office" -> "who all are in black?" should stay scoped to
        # Founders Office). Deliberately a small rolling list, not a single
        # flag, per the "last 3-5 messages" requirement - see
        # push_context()/get_recent_context() below. Only ever used as a
        # FALLBACK when the current message's own extraction finds nothing;
        # never overrides an explicit value in the current message.
        "context_history": [],
    })


# How many recent turns' explicit mentions to remember for fallback lookups.
CONTEXT_HISTORY_LEN = 5


def push_context(session, dept_name=None, employee_id=None, employee_name=None, month=None, date_range=None):
    """Record what was EXPLICITLY named in this turn (pass None for anything
    not mentioned this turn - do not pass through an already-inherited
    value, so the history reflects real mentions, not propagated guesses).
    Keeps only the last CONTEXT_HISTORY_LEN entries."""
    if dept_name is None and employee_id is None and month is None and date_range is None:
        return
    history = session.setdefault("context_history", [])
    history.append({
        "dept_name": dept_name,
        "employee_id": employee_id,
        "employee_name": employee_name,
        "month": month,
        "date_range": date_range,
    })
    del history[:-CONTEXT_HISTORY_LEN]


def get_recent_context(session, field):
    """Walks the last CONTEXT_HISTORY_LEN turns, most recent first, and
    returns the first non-None value found for `field` ('dept_name',
    'employee_id', 'employee_name', 'month', or 'date_range'). Returns None
    if nothing recent set that field."""
    history = session.get("context_history") or []
    for entry in reversed(history):
        val = entry.get(field)
        if val is not None:
            return val
    return None
