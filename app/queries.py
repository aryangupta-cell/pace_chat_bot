import datetime

from .db import run_query
from .entities import VIEW

LIMIT = 10

# Minimum number of Standard-shift days required in BOTH the current and
# prior month before an employee's delta is considered reliable enough to
# rank. Judgment call: most months have ~20-27 working days here, so 10 is
# roughly "at least half the month" - low enough to include employees with
# some leave/absence, high enough to exclude 1-2 day flukes (the kind of
# case that produced 40+ point deltas purely from a tiny sample).
MIN_DAYS_FOR_DELTA = 10


def _prev_month(month_str):
    year, month = (int(x) for x in month_str.split("-"))
    first_of_month = datetime.date(year, month, 1)
    prev_month_end = first_of_month - datetime.timedelta(days=1)
    return f"{prev_month_end.year:04d}-{prev_month_end.month:02d}"


def _is_partial_month(month_str):
    """True if month_str is the live current calendar month (still filling in)."""
    now = datetime.date.today()
    return month_str == f"{now.year:04d}-{now.month:02d}"


def _month_param(month):
    """Normalizes a `month` value for use with SQL '= any(%(month)s)':
    None stays None, a scalar 'YYYY-MM' string becomes a 1-element list,
    and an already-list/tuple value (the multi-month case — e.g.
    entities.extract_months() found "june, july, august" in one query)
    passes through as a list unchanged. This is what lets every ranking/
    aggregate function below transparently SUM across multiple named
    months (via `to_char(worked_day,'YYYY-MM') = any(%(month)s)`) while a
    single month keeps behaving exactly as a scalar equality check did
    before — Postgres's `= any(array[x])` is equivalent to `= x`."""
    if month is None:
        return None
    if isinstance(month, (list, tuple)):
        return list(month) if month else None
    return [month]


def attendance_ranking(dept_name, month, worst=False, employee_ids=None, limit=None):
    order = "desc" if worst else "asc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(coalesce(lc_flag_per_day,0)) as total_lc,
               sum(coalesce(el_flag_per_day,0)) as total_el,
               sum(coalesce(dh_flag_per_day,0)) as total_dh,
               sum(coalesce(defaulter_count_per_day,0)) as total_defaulter,
               (sum(coalesce(lc_flag_per_day,0)) + sum(coalesce(el_flag_per_day,0))
                + sum(coalesce(dh_flag_per_day,0))) as flag_sum
        from {VIEW}
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        order by flag_sum {order}, total_defaulter {order}
        limit {lim}
    """
    return run_query(sql, {"dept_name": dept_name, "month": _month_param(month), "employee_ids": employee_ids})


def productive_time_ranking(dept_name, month, lowest=False, employee_ids=None, limit=None):
    order = "asc" if lowest else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(coalesce(productive_and_meeting_min,0)) as total_productive_min
        from {VIEW}
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        order by total_productive_min {order}
        limit {lim}
    """
    return run_query(sql, {"dept_name": dept_name, "month": _month_param(month), "employee_ids": employee_ids})


def pace_score_trend_ranking(dept_name, month, declining=False, reporting_user_id=None, employee_ids=None, limit=None):
    """Ranks employees by pace_score_delta (this month's avg vs prior month's
    avg new_pace_score_7_3_event_level), excluding anyone with fewer than
    MIN_DAYS_FOR_DELTA Standard-shift days of data in EITHER month being
    compared - a 1-2 day sample produced wildly misleading deltas (40+ points)
    in the underlying data, so those are dropped rather than shown as
    "reliable" trend signal. Returns (rows, meta) where meta carries the
    prior-month string and whether the current month is still partial (live,
    not yet closed out) so the caller can add a caution note.
    """
    order = "asc" if declining else "desc"
    prev_month = _prev_month(month) if month else None
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) as days_current_month,
               sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) as days_prev_month,
               max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_prev_month end) as pace_score_prev_month,
               max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) as pace_score_delta
        from {VIEW}
        where to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(reporting_user_id)s is null or reporting_user_id = %(reporting_user_id)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        having sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) >= %(min_days)s
           and sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) >= %(min_days)s
           and max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) is not null
        order by pace_score_delta {order}
        limit {lim}
    """
    rows = run_query(sql, {
        "dept_name": dept_name,
        "month": month,
        "prev_month": prev_month,
        "reporting_user_id": reporting_user_id,
        "employee_ids": employee_ids,
        "min_days": MIN_DAYS_FOR_DELTA,
    })
    meta = {
        "prev_month": prev_month,
        "partial_month": _is_partial_month(month) if month else False,
        "min_days": MIN_DAYS_FOR_DELTA,
    }
    return rows, meta


def team_attendance_ranking(reporting_user_id, month, worst=False):
    order = "desc" if worst else "asc"
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(coalesce(lc_flag_per_day,0)) as total_lc,
               sum(coalesce(el_flag_per_day,0)) as total_el,
               sum(coalesce(dh_flag_per_day,0)) as total_dh,
               sum(coalesce(defaulter_count_per_day,0)) as total_defaulter,
               (sum(coalesce(lc_flag_per_day,0)) + sum(coalesce(el_flag_per_day,0))
                + sum(coalesce(dh_flag_per_day,0))) as flag_sum
        from {VIEW}
        where reporting_user_id = %(reporting_user_id)s
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
        group by employee_id, emp_name, dept_name
        order by flag_sum {order}, total_defaulter {order}
        limit {LIMIT}
    """
    return run_query(sql, {"reporting_user_id": reporting_user_id, "month": _month_param(month)})


# ---------------------------------------------------------------------------
# Generic metric ranking (Category B/C/G) — one reusable function driving all
# "best/worst/most/least/top N/bottom N by <metric>" question types instead of
# a separate hand-written query per metric.
# ---------------------------------------------------------------------------

# key -> (sql aggregate expression, human label). "avg" metrics are percentage/
# score style fields (meaningful as a per-employee average across their days);
# "sum" metrics are count/minute style fields (meaningful as a total).
METRICS = {
    "pace_score": ("avg(overall_pace_score)", "avg PACE score"),
    "engagement": ("avg(engagement_pct)", "avg engagement %"),
    "effectiveness": ("avg(effectiveness_pct)", "avg effectiveness %"),
    "discipline": ("avg(discipline_pct)", "avg discipline %"),
    "working_pct": ("avg(working_pct)", "avg working hours %"),
    "late_comings": ("sum(coalesce(lc_flag_per_day,0))", "late-comings"),
    "early_leavings": ("sum(coalesce(el_flag_per_day,0))", "early leavings"),
    "deficient_hours_days": ("sum(coalesce(dh_flag_per_day,0))", "deficient-hour days"),
    "defaulter_days": ("sum(coalesce(defaulter_count_per_day,0))", "defaulter days"),
    "whatsapp_min": ("sum(coalesce(whatsapp_min,0))", "WhatsApp minutes"),
    "ai_min": ("sum(coalesce(ai_min,0))", "AI tool minutes"),
    "tools_and_mails_min": ("sum(coalesce(tools_and_mails_min,0))", "tools & mail minutes"),
    "productive_min": ("sum(coalesce(productive_and_meeting_min,0))", "productive minutes"),
}


def metric_ranking(metric_key, dept_name, month, ascending=False, employee_ids=None, limit=None, reporting_user_id=None):
    """Generic best/worst (or top-N/bottom-N via `limit`) ranking by any key
    in METRICS, optionally scoped by dept_name, employee_ids, and/or a
    specific manager's reporting_user_id."""
    expr, _ = METRICS[metric_key]
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               {expr} as metric_value,
               count(*) as days_counted
        from {VIEW}
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
          and (%(reporting_user_id)s is null or reporting_user_id = %(reporting_user_id)s)
        group by employee_id, emp_name, dept_name
        order by metric_value {order} nulls last
        limit {lim}
    """
    return run_query(sql, {
        "dept_name": dept_name, "month": _month_param(month), "employee_ids": employee_ids,
        "reporting_user_id": reporting_user_id,
    })


# ---------------------------------------------------------------------------
# Category A — single employee detail lookups
# ---------------------------------------------------------------------------

def employee_detail(employee_id, month):
    """One aggregated row of everything needed for the Category A per-
    employee questions (PACE score, attendance, productivity, usage, %s)."""
    sql = f"""
        select employee_id, emp_name, dept_name, reporting_manager_name, designation,
               count(*) as days_worked,
               avg(overall_pace_score) as avg_pace_score,
               sum(coalesce(lc_flag_per_day,0)) as total_lc,
               sum(coalesce(el_flag_per_day,0)) as total_el,
               sum(coalesce(dh_flag_per_day,0)) as total_dh,
               sum(coalesce(defaulter_count_per_day,0)) as total_defaulter,
               sum(coalesce(productive_and_meeting_min,0)) as total_productive_min,
               sum(coalesce(whatsapp_min,0)) as total_whatsapp_min,
               sum(coalesce(ai_min,0)) as total_ai_min,
               sum(coalesce(tools_and_mails_min,0)) as total_tools_min,
               avg(discipline_pct) as avg_discipline_pct,
               avg(engagement_pct) as avg_engagement_pct,
               avg(effectiveness_pct) as avg_effectiveness_pct,
               avg(working_pct) as avg_working_pct,
               max(pace_score_delta) as pace_score_delta,
               max(pace_score_prev_month) as pace_score_prev_month
        from {VIEW}
        where employee_id = %(employee_id)s
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
        group by employee_id, emp_name, dept_name, reporting_manager_name, designation
    """
    rows = run_query(sql, {"employee_id": employee_id, "month": _month_param(month)})
    return rows[0] if rows else None


def employee_pace_trend_monthly(employee_id, month):
    """Single-employee month-over-month PACE score trend (this month's avg
    new_pace_score_7_3_event_level vs prior month's), using the SAME
    precomputed pace_score_delta/pace_score_prev_month columns and
    MIN_DAYS_FOR_DELTA reliability gate as pace_score_trend_ranking - just
    scoped to one employee_id instead of ranking a team/dept. This is the
    query individual "is X improving" questions should use (NOT the
    team-scoped ranking function), so a single employee's own trend never
    depends on team/admin-access resolution.

    Returns (row_or_None, meta). row is None if this employee doesn't have
    at least MIN_DAYS_FOR_DELTA days of data in BOTH months."""
    prev_month = _prev_month(month) if month else None
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) as days_current_month,
               sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) as days_prev_month,
               max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_prev_month end) as pace_score_prev_month,
               max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) as pace_score_delta
        from {VIEW}
        where employee_id = %(employee_id)s
          and to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
        group by employee_id, emp_name, dept_name
    """
    rows = run_query(sql, {"employee_id": employee_id, "month": month, "prev_month": prev_month})
    row = rows[0] if rows else None
    meta = {
        "prev_month": prev_month,
        "partial_month": _is_partial_month(month) if month else False,
        "min_days": MIN_DAYS_FOR_DELTA,
    }
    if (not row or row["days_current_month"] < MIN_DAYS_FOR_DELTA
            or row["days_prev_month"] < MIN_DAYS_FOR_DELTA or row["pace_score_delta"] is None):
        return None, meta
    return row, meta


def employee_weekly_pace_trend(employee_id, num_weeks=6):
    """Week-by-week (ISO Monday-Sunday, same week convention as
    entities.extract_date_range's 'this week'/'last week') breakdown of
    new_pace_score_7_3_event_level for one employee, most recent num_weeks
    that have at least 2 scored days - a week-over-week delta is included
    for every week after the first returned. Falls back to public.pace_1
    directly since the capped sub-metrics are day-level and not in
    pace_chatbot_view (same rationale as the Category K functions above).

    BUG FIX: previously averaged new_pace_score_7_3_event_level (day-level
    precomputed score) directly across the week. This does not match the
    real ETL's period-score methodology (see employee_full_monthly_trend's
    docstring) - fixed to the same 3-step aggregation (avg the 4 capped
    sub-metrics across Standard-shift rows in the week, apply the score
    formula once)."""
    sql = """
        select date_trunc('week', worked_day)::date as week_start,
               (date_trunc('week', worked_day)::date + interval '6 days')::date as week_end,
               least(100, round(((avg(capped_engagement) * avg(capped_effectiveness) * avg(capped_working_hours) * 7)
                    + (avg(capped_discipline) * 3)) * 10)) as avg_score,
               count(*) as scored_days
        from public.pace_1
        where employee_id = %(employee_id)s
          and shift_type = 'Standard'
          and capped_engagement is not null and capped_effectiveness is not null
          and capped_discipline is not null and capped_working_hours is not null
        group by 1
        having count(*) >= 2
        order by 1 desc
        limit %(num_weeks)s
    """
    rows = run_query(sql, {"employee_id": employee_id, "num_weeks": num_weeks})
    rows = list(reversed(rows))  # chronological order, oldest first
    for i, r in enumerate(rows):
        if i == 0 or r["avg_score"] is None or rows[i - 1]["avg_score"] is None:
            r["delta"] = None
        else:
            r["delta"] = r["avg_score"] - rows[i - 1]["avg_score"]
    return rows


# ---------------------------------------------------------------------------
# Category D — two-month lookback / team & department delta aggregation
# ---------------------------------------------------------------------------

def employee_trend_two_month(employee_id, month):
    """This month's pace_score_delta plus the same for the prior month, i.e.
    a 2-month-back comparison for one employee."""
    prev_month = _prev_month(month)
    sql = f"""
        select to_char(worked_day,'YYYY-MM') as mo,
               max(pace_score_delta) as delta,
               max(pace_score_prev_month) as prev_avg,
               count(*) as days
        from {VIEW}
        where employee_id = %(employee_id)s
          and to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
        group by 1
        order by 1
    """
    return run_query(sql, {"employee_id": employee_id, "month": month, "prev_month": prev_month})


def team_delta_summary(employee_ids, month):
    """Aggregate pace_score_delta across a set of employee_ids (a team),
    excluding anyone below MIN_DAYS_FOR_DELTA in either month — same
    reliability rule as pace_score_trend_ranking."""
    prev_month = _prev_month(month)
    sql = f"""
        select employee_id, emp_name,
               sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) as days_current_month,
               sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) as days_prev_month,
               max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) as pace_score_delta
        from {VIEW}
        where employee_id = any(%(employee_ids)s)
          and to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
        group by employee_id, emp_name
        having sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) >= %(min_days)s
           and sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) >= %(min_days)s
           and max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) is not null
    """
    rows = run_query(sql, {
        "employee_ids": employee_ids, "month": month, "prev_month": prev_month, "min_days": MIN_DAYS_FOR_DELTA,
    })
    if not rows:
        return None
    avg_delta = sum(r["pace_score_delta"] for r in rows) / len(rows)
    return {"avg_delta": avg_delta, "n_employees": len(rows), "prev_month": prev_month}


def dept_delta_ranking(month, ascending=False, limit=None):
    """Which department is improving/declining the most — dept-level average
    pace_score_delta, same reliability filter as the employee version."""
    prev_month = _prev_month(month)
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        with per_emp as (
            select employee_id, dept_name,
                   sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) as days_current_month,
                   sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) as days_prev_month,
                   max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) as delta
            from {VIEW}
            where to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
            group by employee_id, dept_name
        )
        select dept_name, avg(delta) as avg_delta, count(*) as n_employees
        from per_emp
        where days_current_month >= %(min_days)s and days_prev_month >= %(min_days)s and delta is not null
        group by dept_name
        order by avg_delta {order} nulls last
        limit {lim}
    """
    return run_query(sql, {
        "month": month, "prev_month": prev_month, "min_days": MIN_DAYS_FOR_DELTA,
    })


# ---------------------------------------------------------------------------
# Category E — department-level aggregates & comparison
# ---------------------------------------------------------------------------

def dept_summary(dept_name, month):
    sql = f"""
        select dept_name,
               count(distinct employee_id) as n_employees,
               avg(overall_pace_score) as avg_pace_score,
               avg(engagement_pct) as avg_engagement_pct,
               avg(effectiveness_pct) as avg_effectiveness_pct,
               avg(discipline_pct) as avg_discipline_pct,
               sum(coalesce(lc_flag_per_day,0)) as total_lc,
               sum(coalesce(el_flag_per_day,0)) as total_el,
               sum(coalesce(dh_flag_per_day,0)) as total_dh,
               sum(coalesce(productive_and_meeting_min,0)) as total_productive_min
        from {VIEW}
        where dept_name = %(dept_name)s
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
        group by dept_name
    """
    rows = run_query(sql, {"dept_name": dept_name, "month": _month_param(month)})
    return rows[0] if rows else None


def dept_ranking(metric_key, month, ascending=False, limit=None):
    """Best/worst department by a METRICS key, averaged per-department."""
    expr, _ = METRICS[metric_key]
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select dept_name, count(distinct employee_id) as n_employees, {expr} as metric_value
        from {VIEW}
        where (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
        group by dept_name
        order by metric_value {order} nulls last
        limit {lim}
    """
    return run_query(sql, {"month": _month_param(month)})


def compare_depts(dept_a, dept_b, month):
    return [dept_summary(dept_a, month), dept_summary(dept_b, month)]


# ---------------------------------------------------------------------------
# Category G — employee-vs-employee comparison (reused by dept comparison's
# sibling in Category E and by "compare my team to another manager's team")
# ---------------------------------------------------------------------------

def compare_employees(employee_id_a, employee_id_b, month):
    return [employee_detail(employee_id_a, month), employee_detail(employee_id_b, month)]


def team_summary(employee_ids, month, label=None):
    if not employee_ids:
        return None
    sql = f"""
        select count(distinct employee_id) as n_employees,
               avg(overall_pace_score) as avg_pace_score,
               avg(engagement_pct) as avg_engagement_pct,
               avg(effectiveness_pct) as avg_effectiveness_pct,
               avg(discipline_pct) as avg_discipline_pct,
               sum(coalesce(lc_flag_per_day,0)) as total_lc,
               sum(coalesce(el_flag_per_day,0)) as total_el,
               sum(coalesce(dh_flag_per_day,0)) as total_dh
        from {VIEW}
        where employee_id = any(%(employee_ids)s)
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
    """
    rows = run_query(sql, {"employee_ids": employee_ids, "month": _month_param(month)})
    row = rows[0] if rows else None
    if row is not None:
        row["label"] = label
    return row


# ---------------------------------------------------------------------------
# Category F — attendance thresholds / meeting minutes (pace_1 fallback)
# ---------------------------------------------------------------------------

# Judgment call: "chronically late" = 3 or more late-coming-flagged days in
# the queried month. Chosen to mirror MIN_DAYS_FOR_DELTA's spirit (a small,
# explicit, documented threshold) rather than reusing the ETL's per-day Red/
# Black status, which is a composite score band, not a raw LC count.
CHRONIC_LATE_THRESHOLD = 3


def chronic_late(dept_name, month, employee_ids=None, threshold=CHRONIC_LATE_THRESHOLD, limit=None):
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, sum(coalesce(lc_flag_per_day,0)) as total_lc
        from {VIEW}
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        having sum(coalesce(lc_flag_per_day,0)) >= %(threshold)s
        order by total_lc desc
        limit {lim}
    """
    return run_query(sql, {"dept_name": dept_name, "month": _month_param(month), "employee_ids": employee_ids, "threshold": threshold})


def perfect_attendance(dept_name, month, employee_ids=None, limit=None):
    """0 defaulter days in the month, among employees with worked-day rows."""
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, count(*) as days_worked,
               sum(coalesce(defaulter_count_per_day,0)) as total_defaulter
        from {VIEW}
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        having sum(coalesce(defaulter_count_per_day,0)) = 0
        order by days_worked desc
        limit {lim}
    """
    return run_query(sql, {"dept_name": dept_name, "month": _month_param(month), "employee_ids": employee_ids})


def meeting_minutes_ranking(dept_name, month, employee_ids=None, ascending=False, limit=None):
    """meeting_in_min isn't in pace_chatbot_view, so this falls back to
    pace_1 directly (read-only, same pattern team.py already uses)."""
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, sum(coalesce(meeting_in_min,0)) as total_meeting_min
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        order by total_meeting_min {order} nulls last
        limit {lim}
    """
    return run_query(sql, {"dept_name": dept_name, "month": _month_param(month), "employee_ids": employee_ids})


# ---------------------------------------------------------------------------
# Category H — team view: new joiners
# ---------------------------------------------------------------------------

# Judgment call: reuses the production ETL's own tenure framing referenced in
# this project's context — "New Joiner" = doj within the last 20 days,
# "Potential New Joiner" = 21-40 days — rather than inventing new thresholds.
NEW_JOINER_DAYS = 20
POTENTIAL_NEW_JOINER_DAYS = 40


# ---------------------------------------------------------------------------
# Time-period helper: many new categories below need day/week granularity in
# addition to the existing month-string filtering. Rather than duplicate the
# `to_char(worked_day,'YYYY-MM') = %(month)s` pattern with a parallel
# `worked_day BETWEEN start AND end` pattern in every new function, this
# helper builds a single SQL fragment (both null-safe, so exactly one of
# month/date_range is expected to be set by the caller - entities.extract_date_range
# is checked first, falling back to entities.extract_month) plus its params.
# ---------------------------------------------------------------------------

def _period_filter(month, date_range):
    """date_range: (start_date, end_date) or None. `month`: None, a scalar
    'YYYY-MM' string, or a list of them (multi-month "june, july, august"
    style query — see _month_param) — normalized via _month_param and
    matched with `= any(...)`, so a single month behaves exactly as before.
    Returns (sql_fragment, params)."""
    start, end = date_range if date_range else (None, None)
    frag = (
        "(%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s)) "
        "and (%(date_start)s is null or worked_day between %(date_start)s and %(date_end)s)"
    )
    params = {"month": _month_param(month) if not date_range else None, "date_start": start, "date_end": end}
    return frag, params


# ---------------------------------------------------------------------------
# Category A (new) — Leave & absence. pace_1 fallback: applied_leave_type,
# applied_leave_status, final_half_day_flag, half_day_count_per_employee are
# not in pace_chatbot_view.
# ---------------------------------------------------------------------------

def leave_status_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, dept_name, worked_day, applied_leave_type, applied_leave_status, final_half_day_flag
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        order by worked_day desc
    """
    params["employee_id"] = employee_id
    return run_query(sql, params)


def who_on_leave(dept_name, month=None, date_range=None, limit=None):
    frag, params = _period_filter(month, date_range)
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, worked_day, applied_leave_type, applied_leave_status
        from public.pace_1
        where applied_leave_type is not null
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and {frag}
        order by worked_day desc
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


def half_day_ranking(dept_name, month=None, date_range=None, limit=None):
    frag, params = _period_filter(month, date_range)
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, max(coalesce(half_day_count_per_employee,0)) as half_days
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and {frag}
        group by employee_id, emp_name, dept_name
        order by half_days desc
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


def leave_counts_by_dept(month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select dept_name, count(*) as leave_days
        from public.pace_1
        where applied_leave_type is not null and {frag}
        group by dept_name
        order by leave_days desc
    """
    return run_query(sql, params)


def zero_leave_employees(dept_name, month=None, date_range=None, limit=None):
    frag, params = _period_filter(month, date_range)
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, count(*) as days_worked
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s) and {frag}
        group by employee_id, emp_name, dept_name
        having sum(case when applied_leave_type is not null then 1 else 0 end) = 0
        order by days_worked desc
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category B (new) — Calls. Distinct from meeting_count/meeting_in_min.
# ---------------------------------------------------------------------------

def call_activity_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name,
               sum(coalesce(total_calls,0)) as total_calls,
               sum(coalesce(call_duration_min,0)) as total_call_min,
               count(*) as days_counted
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        group by employee_id, emp_name
    """
    params["employee_id"] = employee_id
    rows = run_query(sql, params)
    return rows[0] if rows else None


def call_ranking(dept_name, metric="total_calls", month=None, date_range=None, ascending=False, limit=None):
    """metric: 'total_calls' or 'avg_duration' (avg call_duration_min per day)."""
    frag, params = _period_filter(month, date_range)
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    expr = "sum(coalesce(total_calls,0))" if metric == "total_calls" else "avg(call_duration_min)"
    sql = f"""
        select employee_id, emp_name, dept_name, {expr} as metric_value
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s) and {frag}
        group by employee_id, emp_name, dept_name
        order by metric_value {order} nulls last
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category C (new) — Visits.
# ---------------------------------------------------------------------------

def visit_activity_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, worked_day, visit_flag, visit_type
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        order by worked_day desc
    """
    params["employee_id"] = employee_id
    return run_query(sql, params)


def visit_ranking(dept_name, month=None, date_range=None, employee_ids=None, limit=None):
    frag, params = _period_filter(month, date_range)
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when visit_flag = 'Yes' then 1 else 0 end) as visit_days
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s)) and {frag}
        group by employee_id, emp_name, dept_name
        order by visit_days desc
        limit {lim}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    return run_query(sql, params)


def zero_visit_employees(dept_name, month=None, date_range=None, limit=None):
    frag, params = _period_filter(month, date_range)
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, count(*) as days_worked
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s) and {frag}
        group by employee_id, emp_name, dept_name
        having sum(case when visit_flag = 'Yes' then 1 else 0 end) = 0
        order by days_worked desc
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category D (new) — WFH.
# ---------------------------------------------------------------------------

def wfh_status_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, worked_day, wfh_status
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        order by worked_day desc
    """
    params["employee_id"] = employee_id
    return run_query(sql, params)


def wfh_ranking(dept_name, month=None, date_range=None, employee_ids=None, limit=None):
    frag, params = _period_filter(month, date_range)
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when wfh_status = 'Work From Home' then 1 else 0 end) as wfh_days
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s)) and {frag}
        group by employee_id, emp_name, dept_name
        order by wfh_days desc
        limit {lim}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    return run_query(sql, params)


def wfh_by_dept(month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select dept_name,
               sum(case when wfh_status = 'Work From Home' then 1 else 0 end) as wfh_days,
               count(*) as total_days
        from public.pace_1
        where {frag}
        group by dept_name
        order by wfh_days desc
    """
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category E (new) — Tasks/todos & task-quality scores.
# JUDGMENT CALL / DATA-COMPLETENESS: ontime_completion_rate, responsiveness_score,
# and extension_adherence_score are ~69% NULL in pace_1 (re-verified during this
# build; d_score similarly ~84% NULL, see Category G) - populated only for a
# task-management-eligible employee subset. Rankings/averages on these
# therefore silently exclude most employees; callers should surface that.
# ---------------------------------------------------------------------------

TASK_METRICS = {
    "todos_created": ("sum(coalesce(todos_created,0))", "todos created"),
    "todos_assigned": ("sum(coalesce(todos_assigned,0))", "todos assigned"),
    "tasks_created": ("sum(coalesce(tasks_created,0))", "tasks created"),
    "tasks_assigned": ("sum(coalesce(tasks_assigned,0))", "tasks assigned"),
    "ontime_completion_rate": ("avg(ontime_completion_rate)", "avg on-time completion rate"),
    "responsiveness_score": ("avg(responsiveness_score)", "avg responsiveness score"),
    "extension_adherence_score": ("avg(extension_adherence_score)", "avg extension adherence score"),
}


_TASK_SCORE_METRICS = {"ontime_completion_rate", "responsiveness_score", "extension_adherence_score"}


def task_metric_ranking(metric_key, dept_name, month=None, date_range=None, ascending=False, limit=None):
    frag, params = _period_filter(month, date_range)
    expr, _ = TASK_METRICS[metric_key]
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    # Score-style metrics (ontime_completion_rate/responsiveness_score/
    # extension_adherence_score) are ~69% NULL (task-management-eligible
    # subset only) - exclude rows with no data at all rather than showing a
    # page of "None" results, same pattern as d_score_ranking's `having`.
    having = f"having count({metric_key}) > 0" if metric_key in _TASK_SCORE_METRICS else ""
    sql = f"""
        select employee_id, emp_name, dept_name, {expr} as metric_value, count(*) as days_counted
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s) and {frag}
        group by employee_id, emp_name, dept_name
        {having}
        order by metric_value {order} nulls last
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category F (new/extended) — Meetings, incl. meeting-to-productive ratio.
# JUDGMENT CALL: "meeting ratio" = meeting_in_min / productive_and_meeting_min
# (the latter already includes meeting time in this schema's definition, so
# this reads as "what share of counted productive time was spent in
# meetings"), guarded against divide-by-zero.
# ---------------------------------------------------------------------------

def meeting_activity_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name,
               sum(coalesce(meeting_count,0)) as total_meetings,
               sum(coalesce(meeting_in_min,0)) as total_meeting_min,
               sum(coalesce(productive_and_meeting_min,0)) as total_productive_min
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        group by employee_id, emp_name
    """
    params["employee_id"] = employee_id
    rows = run_query(sql, params)
    if not rows:
        return None
    row = rows[0]
    row["meeting_ratio"] = (row["total_meeting_min"] / row["total_productive_min"]) if row["total_productive_min"] else None
    return row


def meeting_count_ranking(dept_name, month=None, date_range=None, ascending=False, limit=None):
    frag, params = _period_filter(month, date_range)
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, sum(coalesce(meeting_count,0)) as total_meetings
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s) and {frag}
        group by employee_id, emp_name, dept_name
        order by total_meetings {order} nulls last
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category G (new) — Quality / d_score.
# DATA-COMPLETENESS: d_score is ~84% NULL in pace_1 (re-verified during this
# build). No prior-period d_score column exists the way pace_score_delta does
# for overall_pace_score, so "has d_score improved" is approximated here as a
# simple this-month-avg vs prior-month-avg comparison (same MIN_DAYS_FOR_DELTA
# reliability threshold reused), NOT a precomputed delta column.
# ---------------------------------------------------------------------------

def d_score_ranking(dept_name, month=None, date_range=None, ascending=False, limit=None):
    frag, params = _period_filter(month, date_range)
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, avg(d_score) as avg_d_score, count(d_score) as scored_days
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s) and {frag}
        group by employee_id, emp_name, dept_name
        having count(d_score) > 0
        order by avg_d_score {order} nulls last
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


def d_score_trend(employee_id, month):
    """This month's avg d_score vs prior month's, gated by MIN_DAYS_FOR_DELTA
    scored (non-null) days in each month - not raw worked days, since d_score
    itself is mostly null."""
    prev_month = _prev_month(month)
    sql = f"""
        select
            avg(case when to_char(worked_day,'YYYY-MM') = %(month)s then d_score end) as cur_avg,
            count(case when to_char(worked_day,'YYYY-MM') = %(month)s and d_score is not null then 1 end) as cur_n,
            avg(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then d_score end) as prev_avg,
            count(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s and d_score is not null then 1 end) as prev_n
        from public.pace_1
        where employee_id = %(employee_id)s
          and to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
    """
    rows = run_query(sql, {"employee_id": employee_id, "month": month, "prev_month": prev_month})
    row = rows[0] if rows else None
    if not row or row["cur_n"] < MIN_DAYS_FOR_DELTA or row["prev_n"] < MIN_DAYS_FOR_DELTA:
        return None
    return row


# ---------------------------------------------------------------------------
# Category H (new) — Roster/shift/OT.
# mct_roster_crosses_midnight does NOT exist in pace_1's columns (verified via
# information_schema during this build) - skipped, not guessed at.
# JUDGMENT CALL: "OT hours" = sum(worked_hours) on days where shift_type = 'Overtime'.
# ---------------------------------------------------------------------------

def shift_type_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, worked_day, shift_type, mct_roster_shift_type,
               mct_roster_shift_raw, mct_roster_shift_start, mct_roster_shift_end, breakshift_match_flag
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        order by worked_day desc
    """
    params["employee_id"] = employee_id
    return run_query(sql, params)


def ot_hours_ranking(dept_name, month=None, date_range=None, employee_ids=None, ascending=False, limit=None):
    frag, params = _period_filter(month, date_range)
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(coalesce(worked_hours,0)) as ot_hours,
               count(*) as ot_days
        from public.pace_1
        where shift_type = 'Overtime (OT)'
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s)) and {frag}
        group by employee_id, emp_name, dept_name
        order by ot_hours {order}
        limit {lim}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category I (new) — Offline/device status.
# ---------------------------------------------------------------------------

def offline_status_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, worked_day, offline_attendance_flag, ps_installed_new, ps_worked_flag_day
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        order by worked_day desc
    """
    params["employee_id"] = employee_id
    return run_query(sql, params)


def offline_attendance_ranking(dept_name, month=None, date_range=None, limit=None):
    frag, params = _period_filter(month, date_range)
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when offline_attendance_flag = 'Offline Attendance' then 1 else 0 end) as offline_days
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s) and {frag}
        group by employee_id, emp_name, dept_name
        order by offline_days desc
        limit {lim}
    """
    params["dept_name"] = dept_name
    return run_query(sql, params)


def ps_install_rate_by_dept(month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select dept_name,
               avg(coalesce(ps_installed_new,0)::numeric) * 100 as ps_installed_pct,
               count(*) as days_counted
        from public.pace_1
        where {frag}
        group by dept_name
        order by ps_installed_pct asc
    """
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category J (new) — Org info: grade, designation, tenure.
# ---------------------------------------------------------------------------

def employees_by_grade(grade, dept_name=None):
    sql = f"""
        select distinct employee_id, emp_name, dept_name, grade, designation, doj
        from {VIEW}
        where grade = %(grade)s
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
        order by emp_name
    """
    return run_query(sql, {"grade": grade, "dept_name": dept_name})


def designation_breakdown(dept_name=None):
    sql = f"""
        select designation, count(distinct employee_id) as n_employees
        from {VIEW}
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
        group by designation
        order by n_employees desc
    """
    return run_query(sql, {"dept_name": dept_name})


def average_tenure(dept_name=None, as_of=None):
    as_of = as_of or datetime.date.today()
    sql = f"""
        select dept_name, avg(%(as_of)s::date - doj) as avg_tenure_days, count(distinct employee_id) as n_employees
        from {VIEW}
        where doj is not null
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
        group by dept_name
        order by dept_name
    """
    rows = run_query(sql, {"as_of": as_of, "dept_name": dept_name})
    return rows


# ---------------------------------------------------------------------------
# Category K (new round 2) — score-delta ranking (period-scoped "biggest
# drop" and all-time "improved the most"), sub-score cross-compare/trend,
# status-filtered improvers, OT/WFH-specific sub-scores, PS-working ratio.
# new_pace_score_7_3_event_level is day-level and NOT in pace_chatbot_view,
# so this whole section falls back to pace_1 directly (read-only, same
# pattern as the rest of the pace_1 fallback functions above).
# ---------------------------------------------------------------------------

def _current_month():
    now = datetime.date.today()
    return f"{now.year:04d}-{now.month:02d}"


def _score_delta_ranking_monthly(dept_name, employee_ids, month, ascending, limit):
    """Shared by score_drop_ranking and score_improvement_alltime: ranks
    employees by CURRENT-MONTH-AVG vs PRIOR-MONTH-AVG new_pace_score_7_3_event_level
    (the precomputed pace_score_delta/pace_score_prev_month columns), same
    query pattern and MIN_DAYS_FOR_DELTA reliability gate as
    pace_score_trend_ranking/employee_pace_trend_monthly - NOT a first-
    scored-day vs last-scored-day comparison within the period (that was the
    old, misleading methodology: a single bad/good day at either edge of the
    scope could swing the "delta" by 40+ points regardless of the rest of
    the month). Returns (rows, meta) where meta carries the prior-month
    string and whether the current month is still partial, so the caller can
    add the same caution note used elsewhere."""
    prev_month = _prev_month(month)
    lim = limit or LIMIT
    order = "asc" if ascending else "desc"
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) as days_current_month,
               sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) as days_prev_month,
               max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_prev_month end) as pace_score_prev_month,
               max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) as pace_score_delta
        from {VIEW}
        where to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        having sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) >= %(min_days)s
           and sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) >= %(min_days)s
           and max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) is not null
        order by pace_score_delta {order}
        limit {lim}
    """
    rows = run_query(sql, {
        "dept_name": dept_name,
        "employee_ids": employee_ids,
        "month": month,
        "prev_month": prev_month,
        "min_days": MIN_DAYS_FOR_DELTA,
    })
    meta = {
        "prev_month": prev_month,
        "partial_month": _is_partial_month(month),
        "min_days": MIN_DAYS_FOR_DELTA,
    }
    return rows, meta


def score_drop_ranking(dept_name=None, employee_ids=None, month=None, date_range=None, limit=None):
    """Biggest PACE-score drop this month vs prior month — current-month avg
    vs prior-month avg new_pace_score_7_3_event_level, most negative first.
    `date_range` is accepted for call-site compatibility but a month-over-
    month comparison needs one specific month, so a day/week-granularity
    date_range is resolved down to the calendar month its start date falls
    in (falling back to the current month if neither is given)."""
    resolved_month = month
    if not resolved_month and date_range and date_range[0]:
        start = date_range[0]
        resolved_month = f"{start.year:04d}-{start.month:02d}"
    resolved_month = resolved_month or _current_month()
    return _score_delta_ranking_monthly(dept_name, employee_ids, resolved_month, ascending=True, limit=limit)


def score_improvement_alltime(dept_name=None, employee_ids=None, month=None, limit=None):
    """Most improved this month vs prior month — current-month avg vs
    prior-month avg new_pace_score_7_3_event_level, most positive first.
    Previously compared first-scored-day vs most-recent-scored-day over the
    ENTIRE history on record; changed (mirroring score_drop_ranking's fix)
    to the same reliable month-over-month methodology, defaulting to the
    current month when none is given."""
    resolved_month = month or _current_month()
    return _score_delta_ranking_monthly(dept_name, employee_ids, resolved_month, ascending=False, limit=limit)


def ranking_weekly_pace_trend(employee_ids, num_weeks=4):
    """Week-by-week (ISO Monday-Sunday) new_pace_score_7_3_event_level for a
    SET of employees — the weekly-breakdown counterpart of
    employee_weekly_pace_trend, but for a whole ranking/scope (e.g. the same
    department ranked by score_drop_ranking/score_improvement_alltime)
    instead of one named employee. Returns one row per employee per week,
    most recent `num_weeks` weeks that have at least 2 scored days for that
    employee, with a week-over-week delta per employee (reset at each
    employee's first returned week, same convention as
    employee_weekly_pace_trend).

    BUG FIX: see employee_weekly_pace_trend's docstring - same fix, the 4
    capped sub-metrics are averaged across the week first and the score
    formula applied once, instead of averaging the precomputed day-level
    new_pace_score_7_3_event_level."""
    if not employee_ids:
        return []
    sql = """
        select employee_id, emp_name,
               date_trunc('week', worked_day)::date as week_start,
               (date_trunc('week', worked_day)::date + interval '6 days')::date as week_end,
               least(100, round(((avg(capped_engagement) * avg(capped_effectiveness) * avg(capped_working_hours) * 7)
                    + (avg(capped_discipline) * 3)) * 10)) as avg_score,
               count(*) as scored_days
        from public.pace_1
        where employee_id = any(%(employee_ids)s)
          and shift_type = 'Standard'
          and capped_engagement is not null and capped_effectiveness is not null
          and capped_discipline is not null and capped_working_hours is not null
        group by employee_id, emp_name, 3, 4
        having count(*) >= 2
        order by employee_id, week_start
    """
    rows = run_query(sql, {"employee_ids": employee_ids})
    by_emp = {}
    for r in rows:
        by_emp.setdefault(r["employee_id"], []).append(r)
    result = []
    for emp_rows in by_emp.values():
        for i, r in enumerate(emp_rows):
            if i == 0 or emp_rows[i - 1]["avg_score"] is None:
                r["delta"] = None
            else:
                r["delta"] = r["avg_score"] - emp_rows[i - 1]["avg_score"]
        result.extend(emp_rows[-num_weeks:])
    return result


# --- Sub-score (engagement/effectiveness/discipline) cross-compare & trend --

SUBSCORES = {
    "engagement": ("engagement_pct", "Engagement"),
    "effectiveness": ("effectiveness_pct", "Effectiveness"),
    "discipline": ("discipline_pct", "Discipline"),
}


def subscore_compare_for_employee(employee_id, month=None, date_range=None):
    """avg engagement_pct/effectiveness_pct/discipline_pct for one employee in
    scope, for cross-comparison (weakest/strongest of the three)."""
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, dept_name,
               avg(engagement_pct) as engagement, avg(effectiveness_pct) as effectiveness,
               avg(discipline_pct) as discipline, count(*) as days_counted
        from {VIEW}
        where employee_id = %(employee_id)s and {frag}
        group by employee_id, emp_name, dept_name
    """
    params["employee_id"] = employee_id
    rows = run_query(sql, params)
    return rows[0] if rows else None


def subscore_trend(employee_id, subscore_key, month):
    """This month's avg vs prior month's avg for one of engagement_pct/
    effectiveness_pct/discipline_pct — same query-time comparison pattern and
    MIN_DAYS_FOR_DELTA reliability gate as d_score_trend, since (unlike
    overall_pace_score) there's no precomputed delta column per sub-score."""
    col, _ = SUBSCORES[subscore_key]
    prev_month = _prev_month(month)
    sql = f"""
        select
            avg(case when to_char(worked_day,'YYYY-MM') = %(month)s then {col} end) as cur_avg,
            count(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 end) as cur_n,
            avg(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then {col} end) as prev_avg,
            count(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 end) as prev_n
        from {VIEW}
        where employee_id = %(employee_id)s
          and to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
    """
    rows = run_query(sql, {"employee_id": employee_id, "month": month, "prev_month": prev_month})
    row = rows[0] if rows else None
    if not row or row["cur_n"] < MIN_DAYS_FOR_DELTA or row["prev_n"] < MIN_DAYS_FOR_DELTA:
        return None
    return row


# --- Status-filtered improvers (currently Black/Red, improving MoM) ---------

def status_improving_ranking(statuses, month, dept_name=None, employee_ids=None, limit=None):
    """Employees whose MOST RECENT worked_day's overall_std_pace_status is in
    `statuses` (e.g. ['Black','Red']), ranked by pace_score_delta this month
    (most improving first), same MIN_DAYS_FOR_DELTA reliability gate as
    pace_score_trend_ranking. "Currently" = latest available worked_day on
    record for that employee, not necessarily within `month`."""
    prev_month = _prev_month(month)
    lim = limit or LIMIT
    sql = f"""
        with latest as (
            select distinct on (employee_id) employee_id, overall_std_pace_status
            from {VIEW}
            order by employee_id, worked_day desc
        ),
        deltas as (
            select employee_id, emp_name, dept_name,
                   sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) as days_current_month,
                   sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) as days_prev_month,
                   max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) as pace_score_delta
            from {VIEW}
            where to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
              and (%(dept_name)s is null or dept_name = %(dept_name)s)
              and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
            group by employee_id, emp_name, dept_name
            having sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) >= %(min_days)s
               and sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) >= %(min_days)s
               and max(case when to_char(worked_day,'YYYY-MM') = %(month)s then pace_score_delta end) > 0
        )
        select d.*, l.overall_std_pace_status
        from deltas d join latest l on l.employee_id = d.employee_id
        where l.overall_std_pace_status = any(%(statuses)s)
        order by d.pace_score_delta desc
        limit {lim}
    """
    return run_query(sql, {
        "month": month, "prev_month": prev_month, "dept_name": dept_name, "employee_ids": employee_ids,
        "statuses": statuses, "min_days": MIN_DAYS_FOR_DELTA,
    })


# --- OT-specific / WFH-specific engagement/effectiveness/discipline --------
# capped_engagement/capped_effectiveness/capped_discipline are confirmed
# ROW-LEVEL (session-level): an Overtime row and a Standard row for the same
# employee-day carry their OWN distinct values, so filtering shift_type=
# 'Overtime (OT)' or wfh_status='Work From Home' isolates those sessions
# exactly, not approximately. Values are 0-1 scale here (unlike the view's
# 0-100 engagement_pct/etc.), displayed as-is *100 for a consistent %.

CAPPED_SUBSCORES = {
    "engagement": ("capped_engagement", "Engagement"),
    "effectiveness": ("capped_effectiveness", "Effectiveness"),
    "discipline": ("capped_discipline", "Discipline"),
    "working_hours": ("capped_working_hours", "Working hours"),
}


def _session_filter_subscore_ranking(session_where, metric_key, dept_name=None, employee_ids=None,
                                      month=None, date_range=None, ascending=False, limit=None):
    col, _ = CAPPED_SUBSCORES[metric_key]
    frag, params = _period_filter(month, date_range)
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name, avg({col}) * 100 as metric_value, count(*) as sessions_counted
        from public.pace_1
        where {session_where}
          and {col} is not null
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
          and {frag}
        group by employee_id, emp_name, dept_name
        order by metric_value {order} nulls last
        limit {lim}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    return run_query(sql, params)


def ot_subscore_ranking(metric_key, dept_name=None, employee_ids=None, month=None, date_range=None, ascending=False, limit=None):
    return _session_filter_subscore_ranking(
        "shift_type = 'Overtime (OT)'", metric_key, dept_name, employee_ids, month, date_range, ascending, limit
    )


def wfh_subscore_ranking(metric_key, dept_name=None, employee_ids=None, month=None, date_range=None, ascending=False, limit=None):
    return _session_filter_subscore_ranking(
        "wfh_status = 'Work From Home'", metric_key, dept_name, employee_ids, month, date_range, ascending, limit
    )


def _session_filter_subscore_for_employee(session_where, employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name,
               avg(capped_engagement) * 100 as engagement, avg(capped_effectiveness) * 100 as effectiveness,
               avg(capped_discipline) * 100 as discipline, avg(capped_working_hours) * 100 as working_hours,
               count(*) as sessions_counted
        from public.pace_1
        where {session_where} and employee_id = %(employee_id)s and {frag}
        group by employee_id, emp_name
    """
    params["employee_id"] = employee_id
    rows = run_query(sql, params)
    return rows[0] if rows else None


def ot_subscore_for_employee(employee_id, month=None, date_range=None):
    return _session_filter_subscore_for_employee("shift_type = 'Overtime (OT)'", employee_id, month, date_range)


def wfh_subscore_for_employee(employee_id, month=None, date_range=None):
    return _session_filter_subscore_for_employee("wfh_status = 'Work From Home'", employee_id, month, date_range)


# --- PS working/not-working — ratio, not a trend (see Part 1 investigation:
# ps_worked_flag_day varies meaningfully day-to-day for most employees, but
# there's no prior-period baseline column to build a true delta on, so this
# is framed as a simple days-worked / total-days ratio over the period,
# consistent with the finding documented in the project report). ---------

def ps_worked_ratio_for_employee(employee_id, month=None, date_range=None):
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name,
               sum(case when ps_worked_flag_day = 1 then 1 else 0 end) as days_worked,
               count(*) as total_days
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
        group by employee_id, emp_name
    """
    params["employee_id"] = employee_id
    rows = run_query(sql, params)
    return rows[0] if rows else None


def ps_worked_ratio_ranking(dept_name=None, employee_ids=None, month=None, date_range=None, ascending=True, limit=None):
    """Ranked by PS-working ratio (days worked / total days) — ascending=True
    (default) surfaces the WORST/least-working first, since that's the more
    actionable direction for this metric."""
    frag, params = _period_filter(month, date_range)
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when ps_worked_flag_day = 1 then 1 else 0 end) as days_worked,
               count(*) as total_days,
               (sum(case when ps_worked_flag_day = 1 then 1 else 0 end)::numeric / count(*)) * 100 as metric_value
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
          and {frag}
        group by employee_id, emp_name, dept_name
        order by metric_value {order} nulls last
        limit {lim}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    return run_query(sql, params)


def new_joiners(employee_ids=None, dept_name=None, as_of=None):
    as_of = as_of or datetime.date.today()
    sql = f"""
        select distinct employee_id, emp_name, dept_name, doj,
               (%(as_of)s::date - doj) as tenure_days
        from {VIEW}
        where doj is not null
          and (%(as_of)s::date - doj) <= %(potential_days)s
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        order by doj desc
    """
    rows = run_query(sql, {
        "as_of": as_of, "dept_name": dept_name, "employee_ids": employee_ids,
        "potential_days": POTENTIAL_NEW_JOINER_DAYS,
    })
    for r in rows:
        r["status"] = "New Joiner" if r["tenure_days"] <= NEW_JOINER_DAYS else "Potential New Joiner"
    return rows


# ---------------------------------------------------------------------------
# NEW capability 1 — generic day/period-scoped flag COUNT and LIST helper.
# One shared pattern parameterized by a boolean SQL condition, instead of a
# one-off count/list function per category (attendance, leave, WFH, visits,
# late-coming, early-leaving, OT, deficient-hours, PS-offline, calls,
# meetings, tasks). Falls back to public.pace_1 (day-level; matches every
# other new-capability fallback already in this file) so a single day
# (yesterday/today/explicit date) as well as week/month periods all work via
# the existing _period_filter helper.
# ---------------------------------------------------------------------------

# key -> (sql boolean condition on public.pace_1, human label for "who/how
# many were X"). Each condition is evaluated per worked_day row; COUNT/LIST
# below count/list DISTINCT employees with at least one True row in period.
DAY_FLAGS = {
    "attendance": ("punch_in_ts is not null", "punched attendance"),
    "absent": ("punch_in_ts is null", "absent (no punch-in)"),
    "leave": ("applied_leave_type is not null", "on leave"),
    "wfh": ("wfh_status = 'Work From Home'", "on WFH"),
    "visit": ("visit_flag = 'Yes'", "on a client visit"),
    "late": ("coalesce(lc_flag_per_day,0) > 0", "came late"),
    "early_leave": ("coalesce(el_flag_per_day,0) > 0", "left early"),
    "overtime": ("shift_type = 'Overtime (OT)'", "did overtime"),
    "deficient_hours": ("coalesce(dh_flag_per_day,0) > 0", "marked deficient hours"),
    "defaulter": ("coalesce(defaulter_count_per_day,0) > 0", "marked as a defaulter"),
    "offline": ("offline_attendance_flag = 'Offline Attendance'", "offline / PS not installed"),
    "ps_worked": ("ps_worked_flag_day = 1", "had PS working"),
    "zero_productive": ("coalesce(productive_and_meeting_min,0) = 0", "had zero productive minutes"),
    "called_clients": ("coalesce(total_calls,0) > 0", "made calls"),
    "had_meetings": ("coalesce(meeting_count,0) > 0", "had meetings"),
    "completed_tasks": ("coalesce(tasks_created,0) > 0 or coalesce(tasks_assigned,0) > 0", "had task activity"),
}


def day_flag_count(flag_key, dept_name=None, employee_ids=None, month=None, date_range=None):
    """Number of DISTINCT employees matching DAY_FLAGS[flag_key] at least
    once in the given period (typically a single day via date_range, but
    also works for a week/month like every other period-scoped function
    here). Returns {"n": int, "total": int} — total is the distinct
    employee count with ANY row at all in scope, useful for "X of Y" framing."""
    condition, _ = DAY_FLAGS[flag_key]
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select
            count(distinct case when {condition} then employee_id end) as n,
            count(distinct employee_id) as total
        from public.pace_1
        where (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
          and {frag}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    rows = run_query(sql, params)
    return rows[0] if rows else {"n": 0, "total": 0}


def day_flag_list(flag_key, dept_name=None, employee_ids=None, month=None, date_range=None, limit=None):
    """List of employees (name/dept/day(s)) matching DAY_FLAGS[flag_key] at
    least once in the given period — the LIST counterpart of day_flag_count."""
    condition, _ = DAY_FLAGS[flag_key]
    frag, params = _period_filter(month, date_range)
    lim = limit or 200  # list intents want the full roster, not top-10
    sql = f"""
        select employee_id, emp_name, dept_name, count(*) as matching_days,
               min(worked_day) as first_day, max(worked_day) as last_day
        from public.pace_1
        where {condition}
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
          and {frag}
        group by employee_id, emp_name, dept_name
        order by emp_name
        limit {lim}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    rows = run_query(sql, params)
    return rows


# ---------------------------------------------------------------------------
# NEW capability 2 — status-category filters (Black/Red/Amber/Green), based
# on overall_std_pace_status (verified: values are Amber/Black/Green/NJ/Red,
# aliased in pace_chatbot_view from new_pace_status_overall_last_60_days_7_3;
# thresholds Black <50, Red 50-64, Amber 65-79, Green 80-100 are already
# baked into that upstream column, not recomputed here).
# ---------------------------------------------------------------------------

def _latest_status_cte():
    """CTE: each employee's MOST RECENT worked_day row status in scope."""
    return """
        latest as (
            select distinct on (employee_id) employee_id, emp_name, dept_name, overall_std_pace_status, worked_day
            from {VIEW}
            where (%(dept_name)s is null or dept_name = %(dept_name)s)
              and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
            order by employee_id, worked_day desc
        )
    """.replace("{VIEW}", VIEW)


def status_list(statuses, dept_name=None, employee_ids=None, limit=None):
    """Employees whose CURRENT (latest worked_day) overall_std_pace_status is
    in `statuses` (e.g. ['Red'])."""
    lim = limit or 200
    sql = f"""
        with {_latest_status_cte()}
        select employee_id, emp_name, dept_name, overall_std_pace_status
        from latest
        where overall_std_pace_status = any(%(statuses)s)
        order by emp_name
        limit {lim}
    """
    return run_query(sql, {"dept_name": dept_name, "employee_ids": employee_ids, "statuses": statuses})


def status_count(statuses, dept_name=None, employee_ids=None):
    sql = f"""
        with {_latest_status_cte()}
        select count(*) as n
        from latest
        where overall_std_pace_status = any(%(statuses)s)
    """
    rows = run_query(sql, {"dept_name": dept_name, "employee_ids": employee_ids, "statuses": statuses})
    return rows[0]["n"] if rows else 0


def status_distribution_by_dept(limit=None):
    """Per-department count (and %) of employees in each status bucket,
    based on each employee's CURRENT (latest worked_day) status."""
    lim = limit or 50
    sql = f"""
        with {_latest_status_cte().replace("%(dept_name)s is null or dept_name = %(dept_name)s", "true").replace("%(employee_ids)s is null or employee_id = any(%(employee_ids)s)", "true")}
        select dept_name,
               count(*) as n_employees,
               count(*) filter (where overall_std_pace_status = 'Black') as black_n,
               count(*) filter (where overall_std_pace_status = 'Red') as red_n,
               count(*) filter (where overall_std_pace_status = 'Amber') as amber_n,
               count(*) filter (where overall_std_pace_status = 'Green') as green_n,
               round(100.0 * count(*) filter (where overall_std_pace_status = 'Black') / nullif(count(*),0), 1) as black_pct,
               round(100.0 * count(*) filter (where overall_std_pace_status = 'Red') / nullif(count(*),0), 1) as red_pct,
               round(100.0 * count(*) filter (where overall_std_pace_status = 'Amber') / nullif(count(*),0), 1) as amber_pct,
               round(100.0 * count(*) filter (where overall_std_pace_status = 'Green') / nullif(count(*),0), 1) as green_pct
        from latest
        group by dept_name
        order by (count(*) filter (where overall_std_pace_status = 'Red')
                   + count(*) filter (where overall_std_pace_status = 'Black')) desc
        limit {lim}
    """
    return run_query(sql, {})


def _month_avg_status_cte(month):
    """Bucket an employee's AVERAGE overall_pace_score in a given month into
    the same Black/Red/Amber/Green thresholds the upstream status column
    uses (Black <50, Red 50-64, Amber 65-79, Green >=80) — used for
    status-transition queries where we need a PRIOR month's status and no
    precomputed prior-month status column exists (only pace_score_prev_month,
    a numeric average, does)."""
    return f"""
        select employee_id, emp_name, dept_name, avg(overall_pace_score) as avg_score,
               count(*) as days_counted
        from {VIEW}
        where to_char(worked_day,'YYYY-MM') = %(month)s
        group by employee_id, emp_name, dept_name
        having count(*) >= %(min_days)s
    """


def _bucket_status(avg_score):
    if avg_score is None:
        return None
    if avg_score < 50:
        return "Black"
    if avg_score < 65:
        return "Red"
    if avg_score < 80:
        return "Amber"
    return "Green"


def status_transitions(month, from_status=None, to_status=None, dept_name=None, employee_ids=None, limit=None):
    """Employees whose bucketed status (from monthly-avg overall_pace_score,
    same MIN_DAYS_FOR_DELTA reliability gate used elsewhere) changed between
    the prior month and `month`. Buckets are computed in Python (not SQL)
    from each month's avg score via _bucket_status, since only the CURRENT
    month has a precomputed status column upstream — the prior month must be
    derived the same way pace_score_prev_month itself is derived (monthly
    average), then bucketed with the same thresholds."""
    prev_month = _prev_month(month)
    cur_sql = _month_avg_status_cte(month)
    prev_sql = _month_avg_status_cte(prev_month)
    cur_rows = run_query(cur_sql, {"month": month, "min_days": MIN_DAYS_FOR_DELTA})
    prev_rows = run_query(prev_sql, {"month": prev_month, "min_days": MIN_DAYS_FOR_DELTA})
    prev_by_id = {r["employee_id"]: r for r in prev_rows}
    out = []
    for r in cur_rows:
        prev = prev_by_id.get(r["employee_id"])
        if not prev:
            continue
        if dept_name and r["dept_name"] != dept_name:
            continue
        if employee_ids is not None and r["employee_id"] not in employee_ids:
            continue
        cur_bucket = _bucket_status(r["avg_score"])
        prev_bucket = _bucket_status(prev["avg_score"])
        if cur_bucket == prev_bucket:
            continue
        if from_status and prev_bucket != from_status:
            continue
        if to_status and cur_bucket != to_status:
            continue
        out.append({
            "employee_id": r["employee_id"], "emp_name": r["emp_name"], "dept_name": r["dept_name"],
            "prev_status": prev_bucket, "cur_status": cur_bucket,
            "prev_avg": prev["avg_score"], "cur_avg": r["avg_score"],
        })
    out.sort(key=lambda x: x["emp_name"] or "")
    lim = limit or 200
    return out[:lim]


# ---------------------------------------------------------------------------
# NEW capability 3 — full multi-month trend history (every month's avg PACE
# score, not just current-vs-prior), reusing the same monthly-average
# methodology (new_pace_score_7_3_event_level via the view's
# overall_pace_score) grouped by calendar month, across the full data window.
# ---------------------------------------------------------------------------

def employee_full_monthly_trend(employee_id, metric_key="pace_score"):
    """One row per calendar month this employee has data for (chronological),
    avg of the given METRICS key (default overall PACE score). Unlike
    employee_pace_trend_monthly, this returns EVERY month on record, not
    just current + prior, and has no MIN_DAYS_FOR_DELTA gate (each row shows
    its own days_counted so the caller can judge reliability visually).

    BUG FIX (see SESSION_HANDOFF.md section 2/5): for metric_key="pace_score"
    this MUST NOT use pace_chatbot_view's overall_pace_score column, because
    that column is aliased from last_60_days_new_pace_score_7_3 - a rolling
    60-*worked*-day window score that is effectively FROZEN per employee
    (confirmed live: Rudhi's rolling avg was 90.0/90.0/90.0/90.0 across 4
    distinct calendar months). Falls back to public.pace_1 directly for the
    pace_score metric.

    SECOND BUG FIX (later round): grouping day-level
    new_pace_score_7_3_event_level by month and averaging it is ALSO wrong -
    it does not match the real ETL's overall_new_pace_score_7_3, which
    averages the 4 CAPPED sub-metrics (capped_engagement/effectiveness/
    discipline/working_hours) across the period FIRST and applies the score
    formula ONCE to those averages (Jensen's inequality: averaging a product
    of averages != averaging per-row products). Confirmed live against
    Looker's real overall_new_pace_score_7_3 for Aryan Gupta (99/96/96/87)
    which the old per-day-average method got close-but-wrong (96/95/95/87).
    Fixed to the correct 3-step aggregation: filter Standard-shift rows in
    the period -> AVG each of the 4 capped sub-metrics -> apply the score
    formula once per period. Other METRICS keys (engagement/effectiveness/
    discipline/working_pct/etc.) are unaffected by this fix and keep using
    pace_chatbot_view as before."""
    if metric_key == "pace_score":
        sql = """
            with per_month as (
                select to_char(worked_day,'YYYY-MM') as mo,
                       avg(capped_engagement) as avg_e,
                       avg(capped_effectiveness) as avg_ef,
                       avg(capped_discipline) as avg_d,
                       avg(capped_working_hours) as avg_w,
                       count(*) as days_counted
                from public.pace_1
                where employee_id = %(employee_id)s and shift_type = 'Standard'
                  and capped_engagement is not null and capped_effectiveness is not null
                  and capped_discipline is not null and capped_working_hours is not null
                group by 1
            )
            select mo,
                   least(100, round(((avg_e * avg_ef * avg_w * 7) + (avg_d * 3)) * 10)) as metric_value,
                   days_counted
            from per_month
            order by mo
        """
        return run_query(sql, {"employee_id": employee_id})
    expr, _ = METRICS[metric_key]
    sql = f"""
        select to_char(worked_day,'YYYY-MM') as mo, {expr} as metric_value, count(*) as days_counted
        from {VIEW}
        where employee_id = %(employee_id)s
        group by 1
        order by 1
    """
    return run_query(sql, {"employee_id": employee_id})


def dept_full_monthly_trend(dept_name, metric_key="pace_score"):
    """See employee_full_monthly_trend's docstring for the 3-step-aggregation
    bug fix rationale (avg the 4 capped sub-metrics per period, apply the
    score formula once) - same fix applied here, grouped by month."""
    if metric_key == "pace_score":
        sql = """
            with per_month as (
                select to_char(worked_day,'YYYY-MM') as mo,
                       avg(capped_engagement) as avg_e,
                       avg(capped_effectiveness) as avg_ef,
                       avg(capped_discipline) as avg_d,
                       avg(capped_working_hours) as avg_w,
                       count(distinct employee_id) as n_employees
                from public.pace_1
                where dept_name = %(dept_name)s and shift_type = 'Standard'
                  and capped_engagement is not null and capped_effectiveness is not null
                  and capped_discipline is not null and capped_working_hours is not null
                group by 1
            )
            select mo,
                   least(100, round(((avg_e * avg_ef * avg_w * 7) + (avg_d * 3)) * 10)) as metric_value,
                   n_employees
            from per_month
            order by mo
        """
        return run_query(sql, {"dept_name": dept_name})
    expr, _ = METRICS[metric_key]
    sql = f"""
        select to_char(worked_day,'YYYY-MM') as mo, {expr} as metric_value, count(distinct employee_id) as n_employees
        from {VIEW}
        where dept_name = %(dept_name)s
        group by 1
        order by 1
    """
    return run_query(sql, {"dept_name": dept_name})


def team_full_monthly_trend(employee_ids, metric_key="pace_score"):
    """See employee_full_monthly_trend's docstring for the 3-step-aggregation
    bug fix rationale - same fix applied here, grouped by month."""
    if metric_key == "pace_score":
        sql = """
            with per_month as (
                select to_char(worked_day,'YYYY-MM') as mo,
                       avg(capped_engagement) as avg_e,
                       avg(capped_effectiveness) as avg_ef,
                       avg(capped_discipline) as avg_d,
                       avg(capped_working_hours) as avg_w,
                       count(distinct employee_id) as n_employees
                from public.pace_1
                where employee_id = any(%(employee_ids)s) and shift_type = 'Standard'
                  and capped_engagement is not null and capped_effectiveness is not null
                  and capped_discipline is not null and capped_working_hours is not null
                group by 1
            )
            select mo,
                   least(100, round(((avg_e * avg_ef * avg_w * 7) + (avg_d * 3)) * 10)) as metric_value,
                   n_employees
            from per_month
            order by mo
        """
        return run_query(sql, {"employee_ids": employee_ids})
    expr, _ = METRICS[metric_key]
    sql = f"""
        select to_char(worked_day,'YYYY-MM') as mo, {expr} as metric_value, count(distinct employee_id) as n_employees
        from {VIEW}
        where employee_id = any(%(employee_ids)s)
        group by 1
        order by 1
    """
    return run_query(sql, {"employee_ids": employee_ids})
