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


def get_recent_context(session, field):
    """Returns the sticky value currently held for `field` ('dept_name',
    'employee_id', 'employee_name', 'month', or 'date_range'), or None if
    nothing has been explicitly mentioned yet this session. Name kept as
    `get_recent_context` for call-site compatibility, though the value is
    no longer window-limited - it's whatever was last explicitly set,
    persisted for the whole session."""
    ctx = session.get("sticky_context") or {}
    return ctx.get(field)
