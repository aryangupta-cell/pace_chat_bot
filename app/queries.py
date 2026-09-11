import datetime

from .db import run_query
from .entities import VIEW, last_4_weeks_periods

LIMIT = 10

# Minimum number of Standard-shift days required in BOTH the current and
# prior month before an employee's delta is considered reliable enough to
# rank. Judgment call: most months have ~20-27 working days here, so 10 is
# roughly "at least half the month" - low enough to include employees with
# some leave/absence, high enough to exclude 1-2 day flukes (the kind of
# case that produced 40+ point deltas purely from a tiny sample).
MIN_DAYS_FOR_DELTA = 10


# ---------------------------------------------------------------------------
# Centralized PACE score formula (item #75, Phase 2 of the item #74 audit).
# Single source of truth for the "filter to Standard-shift rows -> AVG each
# of the 4 capped_* sub-metrics separately -> apply the score formula ONCE
# to those averages" calculation. Before this, the exact same SQL fragment
# was hand-copied across ~11 call sites (metric_ranking, dept_ranking,
# rm_ranking, employee_weekly_pace_trend, ranking_weekly_pace_trend,
# _month_avg_status_cte, employee_full_monthly_trend, dept_full_monthly_
# trend, team_full_monthly_trend, _gainer_loser_cte, PS_FILTERED_METRICS,
# build_query) - all textually consistent at the time (see item #74), but
# with nothing enforcing that beyond code comments referencing each other.
# This is a PURE REFACTOR: every call site below now references one of
# these two constants instead of its own hand-copied literal, and the
# golden-validation harness (scripts/golden_validate_pace_score.py,
# SESSION_HANDOFF.md item #75) proves the computed values are unchanged.
#
# NEVER average a precomputed per-row/per-day score across a period instead
# of using this formula - that is the Jensen's-inequality bug fixed
# throughout this file's history (see employee_full_monthly_trend's
# docstring for the concrete example).
# ---------------------------------------------------------------------------

# Form 1: direct SQL aggregate expression - each avg(capped_*) is computed
# in-place inside a GROUP BY query. Use this when no CTE has already
# pre-aggregated the 4 capped sub-metrics into avg_e/avg_ef/avg_d/avg_w
# columns.
PACE_SCORE_AGG_SQL = (
    "least(100, round(((avg(capped_engagement) * avg(capped_effectiveness) * "
    "avg(capped_working_hours) * 7) + (avg(capped_discipline) * 3)) * 10))"
)

# Form 2: mathematically identical formula, but reading from already-
# aggregated avg_e/avg_ef/avg_d/avg_w columns (e.g. produced by a prior CTE)
# instead of re-computing avg() inline. Callers using this form MUST alias
# their averaged columns exactly avg_e (capped_engagement), avg_ef (capped_
# effectiveness), avg_d (capped_discipline), avg_w (capped_working_hours).
PACE_SCORE_FROM_AVGS_SQL = (
    "least(100, round(((avg_e * avg_ef * avg_w * 7) + (avg_d * 3)) * 10))"
)

# Qualifying-row rule the PACE formula is defined over, whenever the 4
# capped_* columns are read directly from public.pace_1 (documentation
# only - existing call sites already spell this out inline in their own
# WHERE clauses and are NOT refactored to use this constant, to keep this
# round's diff to the formula/status expressions only; provided for any
# NEW call site, e.g. the golden-validation harness, to reuse).
PACE_SCORE_QUALIFYING_ROWS_SQL = (
    "shift_type = 'Standard' "
    "and capped_engagement is not null and capped_effectiveness is not null "
    "and capped_discipline is not null and capped_working_hours is not null"
)


def pace_status_sql(score_expr):
    """Centralized status-banding SQL expression (Black <50 / Red 50-64 /
    Amber 65-79 / Green >=80), applicable to ANY computed score expression -
    not just the stored 60-day column. Thin, discoverable wrapper around
    _PACE_STATUS_CASE_SQL (defined later in this file; already the single
    source of truth for the thresholds - this just gives it a public name
    alongside the score-formula constants above, needed so a future
    arbitrary-period PACE status feature has one obvious function to call
    for both the score AND its status, not just the score)."""
    return _PACE_STATUS_CASE_SQL.format(score=score_expr)


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


def metric_ranking(metric_key, dept_name, month, ascending=False, employee_ids=None, limit=None,
                    reporting_user_id=None, date_range=None):
    """Generic best/worst (or top-N/bottom-N via `limit`) ranking by any key
    in METRICS, optionally scoped by dept_name, employee_ids, and/or a
    specific manager's reporting_user_id.

    BUG FIX (this round, Part 4b): for metric_key="pace_score" with a
    SPECIFIC month filter given, ranking on pace_chatbot_view's
    overall_pace_score (a rolling 60-*worked*-day window score, aliased from
    last_60_days_new_pace_score_7_3) can under-represent that specific past
    month, same root cause as employee_full_monthly_trend's/
    _month_avg_status_cte's pre-existing fix. When a single specific month is
    named, switch to the same capped-average-first recompute directly from
    pace_1 (average the 4 capped sub-metrics for that month, then apply the
    score formula once) instead of the view's rolling-window column. With NO
    month filter (org-wide "who has the best pace score right now") or a
    multi-month list, the rolling-window semantics are still the intended
    "current standing" answer, so this branch only fires for exactly one
    named month - the view-based query below is otherwise unchanged.

    `date_range` (item B, SESSION_HANDOFF.md): new, additive alternative to
    `month` — a (start_date, end_date) tuple, used by callers that resolved
    "no period named at all" to the last-60-days default
    (queries.default_period_last_60_days()) instead of the old current-month
    default. Mutually exclusive with `month` (mirrors every other date_range/
    month dual-mode function in this file, e.g. metric_ranking_ps_filtered).
    For metric_key="pace_score" with a date_range given, uses the SAME
    capped-average-first-then-formula-once recompute as the single-month
    branch above (still avoiding the Jensen's-inequality bug), just filtered
    by `worked_day between` instead of `to_char(...) = month`."""
    if date_range is not None:
        start, end = date_range
        order = "asc" if ascending else "desc"
        lim = limit or LIMIT
        if metric_key == "pace_score":
            sql = f"""
                select employee_id, emp_name, dept_name,
                       {PACE_SCORE_AGG_SQL} as metric_value,
                       count(*) as days_counted
                from public.pace_1
                where worked_day between %(date_start)s and %(date_end)s and shift_type = 'Standard'
                  and capped_engagement is not null and capped_effectiveness is not null
                  and capped_discipline is not null and capped_working_hours is not null
                  and (%(dept_name)s is null or dept_name = %(dept_name)s)
                  and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
                  and (%(reporting_user_id)s is null or reporting_user_id = %(reporting_user_id)s)
                group by employee_id, emp_name, dept_name
                order by metric_value {order} nulls last
                limit {lim}
            """
            return run_query(sql, {
                "date_start": start, "date_end": end, "dept_name": dept_name, "employee_ids": employee_ids,
                "reporting_user_id": reporting_user_id,
            })
        expr, _ = METRICS[metric_key]
        sql = f"""
            select employee_id, emp_name, dept_name,
                   {expr} as metric_value,
                   count(*) as days_counted
            from {VIEW}
            where (%(dept_name)s is null or dept_name = %(dept_name)s)
              and worked_day between %(date_start)s and %(date_end)s
              and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
              and (%(reporting_user_id)s is null or reporting_user_id = %(reporting_user_id)s)
            group by employee_id, emp_name, dept_name
            order by metric_value {order} nulls last
            limit {lim}
        """
        return run_query(sql, {
            "dept_name": dept_name, "date_start": start, "date_end": end, "employee_ids": employee_ids,
            "reporting_user_id": reporting_user_id,
        })
    month_list = _month_param(month)
    if metric_key == "pace_score" and month_list is not None and len(month_list) == 1:
        order = "asc" if ascending else "desc"
        lim = limit or LIMIT
        sql = f"""
            select employee_id, emp_name, dept_name,
                   {PACE_SCORE_AGG_SQL} as metric_value,
                   count(*) as days_counted
            from public.pace_1
            where to_char(worked_day,'YYYY-MM') = %(month)s and shift_type = 'Standard'
              and capped_engagement is not null and capped_effectiveness is not null
              and capped_discipline is not null and capped_working_hours is not null
              and (%(dept_name)s is null or dept_name = %(dept_name)s)
              and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
              and (%(reporting_user_id)s is null or reporting_user_id = %(reporting_user_id)s)
            group by employee_id, emp_name, dept_name
            order by metric_value {order} nulls last
            limit {lim}
        """
        return run_query(sql, {
            "month": month_list[0], "dept_name": dept_name, "employee_ids": employee_ids,
            "reporting_user_id": reporting_user_id,
        })
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
        "dept_name": dept_name, "month": month_list, "employee_ids": employee_ids,
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
    sql = f"""
        select date_trunc('week', worked_day)::date as week_start,
               (date_trunc('week', worked_day)::date + interval '6 days')::date as week_end,
               {PACE_SCORE_AGG_SQL} as avg_score,
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


def dept_ranking(metric_key, month, ascending=False, limit=None, date_range=None):
    """Best/worst department by a METRICS key, averaged per-department.

    BUG FIX (see SESSION_HANDOFF.md): for metric_key="pace_score" with a
    SPECIFIC month filter given, this previously averaged the view's
    pre-computed overall_pace_score (a rolling 60-*worked*-day window score)
    per department - the same Jensen's-inequality bug already fixed
    elsewhere (employee_full_monthly_trend, metric_ranking, status_
    transitions): the real ETL formula multiplies several averaged terms
    together, so averaging the pre-computed daily/rolling score is NOT the
    same as averaging the 4 capped_* ingredients first and applying the
    score formula once. Fixed to the same capped-average-first recompute,
    grouped by department, using the identical single-named-month branch
    condition metric_ranking() uses (a rolling-window "current standing"
    answer is still correct with no month filter or a multi-month list).

    `date_range` (item B): same additive last-60-days-default support as
    metric_ranking() — see its docstring."""
    if date_range is not None:
        start, end = date_range
        order = "asc" if ascending else "desc"
        lim = limit or LIMIT
        if metric_key == "pace_score":
            sql = f"""
                select dept_name, count(distinct employee_id) as n_employees,
                       {PACE_SCORE_AGG_SQL} as metric_value
                from public.pace_1
                where worked_day between %(date_start)s and %(date_end)s and shift_type = 'Standard'
                  and capped_engagement is not null and capped_effectiveness is not null
                  and capped_discipline is not null and capped_working_hours is not null
                group by dept_name
                order by metric_value {order} nulls last
                limit {lim}
            """
            return run_query(sql, {"date_start": start, "date_end": end})
        expr, _ = METRICS[metric_key]
        sql = f"""
            select dept_name, count(distinct employee_id) as n_employees, {expr} as metric_value
            from {VIEW}
            where worked_day between %(date_start)s and %(date_end)s
            group by dept_name
            order by metric_value {order} nulls last
            limit {lim}
        """
        return run_query(sql, {"date_start": start, "date_end": end})
    month_list = _month_param(month)
    if metric_key == "pace_score" and month_list is not None and len(month_list) == 1:
        order = "asc" if ascending else "desc"
        lim = limit or LIMIT
        sql = f"""
            select dept_name, count(distinct employee_id) as n_employees,
                   {PACE_SCORE_AGG_SQL} as metric_value
            from public.pace_1
            where to_char(worked_day,'YYYY-MM') = %(month)s and shift_type = 'Standard'
              and capped_engagement is not null and capped_effectiveness is not null
              and capped_discipline is not null and capped_working_hours is not null
            group by dept_name
            order by metric_value {order} nulls last
            limit {lim}
        """
        return run_query(sql, {"month": month_list[0]})
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
    return run_query(sql, {"month": month_list})


def compare_depts(dept_a, dept_b, month):
    return [dept_summary(dept_a, month), dept_summary(dept_b, month)]


def rm_ranking(metric_key, month, ascending=False, limit=None, date_range=None):
    """Best/worst reporting-manager team by a METRICS key, averaged per-RM
    team - same shape/pattern as dept_ranking() above, just grouped by
    reporting_manager_name instead of dept_name (new intent: 'which RM team
    has the most/least score').

    BUG FIX: this was modeled on dept_ranking() and inherited its same
    Jensen's-inequality bug for metric_key="pace_score" (see dept_ranking's
    docstring) - fixed with the identical capped-average-first branch,
    grouped by reporting_manager_name instead of dept_name.

    `date_range` (item B): same additive last-60-days-default support as
    metric_ranking()/dept_ranking() — see metric_ranking()'s docstring."""
    if date_range is not None:
        start, end = date_range
        order = "asc" if ascending else "desc"
        lim = limit or LIMIT
        if metric_key == "pace_score":
            sql = f"""
                select reporting_manager_name, count(distinct employee_id) as n_employees,
                       {PACE_SCORE_AGG_SQL} as metric_value
                from public.pace_1
                where worked_day between %(date_start)s and %(date_end)s and shift_type = 'Standard'
                  and capped_engagement is not null and capped_effectiveness is not null
                  and capped_discipline is not null and capped_working_hours is not null
                  and reporting_manager_name is not null
                group by reporting_manager_name
                order by metric_value {order} nulls last
                limit {lim}
            """
            return run_query(sql, {"date_start": start, "date_end": end})
        expr, _ = METRICS[metric_key]
        sql = f"""
            select reporting_manager_name, count(distinct employee_id) as n_employees, {expr} as metric_value
            from {VIEW}
            where worked_day between %(date_start)s and %(date_end)s
              and reporting_manager_name is not null
            group by reporting_manager_name
            order by metric_value {order} nulls last
            limit {lim}
        """
        return run_query(sql, {"date_start": start, "date_end": end})
    month_list = _month_param(month)
    if metric_key == "pace_score" and month_list is not None and len(month_list) == 1:
        order = "asc" if ascending else "desc"
        lim = limit or LIMIT
        sql = f"""
            select reporting_manager_name, count(distinct employee_id) as n_employees,
                   {PACE_SCORE_AGG_SQL} as metric_value
            from public.pace_1
            where to_char(worked_day,'YYYY-MM') = %(month)s and shift_type = 'Standard'
              and capped_engagement is not null and capped_effectiveness is not null
              and capped_discipline is not null and capped_working_hours is not null
              and reporting_manager_name is not null
            group by reporting_manager_name
            order by metric_value {order} nulls last
            limit {lim}
        """
        return run_query(sql, {"month": month_list[0]})
    expr, _ = METRICS[metric_key]
    order = "asc" if ascending else "desc"
    lim = limit or LIMIT
    sql = f"""
        select reporting_manager_name, count(distinct employee_id) as n_employees, {expr} as metric_value
        from {VIEW}
        where (%(month)s is null or to_char(worked_day,'YYYY-MM') = any(%(month)s))
          and reporting_manager_name is not null
        group by reporting_manager_name
        order by metric_value {order} nulls last
        limit {lim}
    """
    return run_query(sql, {"month": month_list})


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
    # Bug fix: previously returned EVERY day in scope regardless of whether
    # the employee was actually on leave that day (a full daily dump), even
    # though the caller ("did X take leave" / "how many times did X take
    # leave") only cares about the days leave was actually taken. Filter to
    # rows matching DAY_FLAGS["leave"]'s condition so only actual leave days
    # come back — same fix applied to visit_activity_for_employee and
    # wfh_status_for_employee below, which had the identical defect.
    condition, _ = DAY_FLAGS["leave"]
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, dept_name, worked_day, applied_leave_type, applied_leave_status, final_half_day_flag
        from public.pace_1
        where employee_id = %(employee_id)s and {condition} and {frag}
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
    # See leave_status_for_employee above: filter to actual visit days only.
    condition, _ = DAY_FLAGS["visit"]
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, worked_day, visit_flag, visit_type
        from public.pace_1
        where employee_id = %(employee_id)s and {condition} and {frag}
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
    # See leave_status_for_employee above: filter to actual WFH days only.
    condition, _ = DAY_FLAGS["wfh"]
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select employee_id, emp_name, worked_day, wfh_status
        from public.pace_1
        where employee_id = %(employee_id)s and {condition} and {frag}
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


# Item #84 (Finding 3 follow-through): generalizes _score_delta_ranking_monthly
# to an arbitrary sub-metric column (engagement_pct/effectiveness_pct/
# discipline_pct/working_pct), not just the precomputed pace_score_delta
# column. score_drop_ranking/score_improvement_alltime above can't be reused
# directly - there is no precomputed *_delta column for these sub-metrics -
# so this averages the raw column per employee per month (same shape as
# BUILD_QUERY_METRICS' own "avg(<col>)" treatment) and computes the delta in
# SQL, applying the SAME MIN_DAYS_FOR_DELTA reliability gate as the
# pace_score version so a 1-2 day sample can't produce a misleading swing.
SUBSCORE_DELTA_COLUMNS = {
    "engagement_pct": ("engagement_pct", "engagement %"),
    "effectiveness_pct": ("effectiveness_pct", "effectiveness %"),
    "discipline_pct": ("discipline_pct", "discipline %"),
    "working_pct": ("working_pct", "working hours %"),
}


def subscore_delta_ranking(metric_key, dept_name=None, employee_ids=None, month=None,
                            date_range=None, ascending=True, limit=None):
    """Ranks employees by CURRENT-MONTH-AVG vs PRIOR-MONTH-AVG of one of the
    4 pct sub-metrics (engagement/effectiveness/discipline/working hours),
    most-declined first by default (ascending=True). `metric_key` must be a
    key of SUBSCORE_DELTA_COLUMNS - callers validate this before calling.
    `date_range` (like score_drop_ranking) is resolved down to the calendar
    month its start date falls in when no explicit `month` is given."""
    if metric_key not in SUBSCORE_DELTA_COLUMNS:
        raise ValueError(f"subscore_delta_ranking: unknown metric_key {metric_key!r}")
    col, label = SUBSCORE_DELTA_COLUMNS[metric_key]
    resolved_month = month
    if not resolved_month and date_range and date_range[0]:
        start = date_range[0]
        resolved_month = f"{start.year:04d}-{start.month:02d}"
    resolved_month = resolved_month or _current_month()
    prev_month = _prev_month(resolved_month)
    lim = limit or LIMIT
    order = "asc" if ascending else "desc"
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) as days_current_month,
               sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) as days_prev_month,
               avg(case when to_char(worked_day,'YYYY-MM') = %(month)s then {col} end) as cur_avg,
               avg(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then {col} end) as prev_avg,
               avg(case when to_char(worked_day,'YYYY-MM') = %(month)s then {col} end)
                 - avg(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then {col} end) as delta
        from {VIEW}
        where to_char(worked_day,'YYYY-MM') in (%(month)s, %(prev_month)s)
          and (%(dept_name)s is null or dept_name = %(dept_name)s)
          and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
        group by employee_id, emp_name, dept_name
        having sum(case when to_char(worked_day,'YYYY-MM') = %(month)s then 1 else 0 end) >= %(min_days)s
           and sum(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then 1 else 0 end) >= %(min_days)s
           and avg(case when to_char(worked_day,'YYYY-MM') = %(month)s then {col} end) is not null
           and avg(case when to_char(worked_day,'YYYY-MM') = %(prev_month)s then {col} end) is not null
        order by delta {order}
        limit {lim}
    """
    rows = run_query(sql, {
        "dept_name": dept_name,
        "employee_ids": employee_ids,
        "month": resolved_month,
        "prev_month": prev_month,
        "min_days": MIN_DAYS_FOR_DELTA,
    })
    meta = {
        "label": label,
        "prev_month": prev_month,
        "month": resolved_month,
        "partial_month": _is_partial_month(resolved_month),
        "min_days": MIN_DAYS_FOR_DELTA,
    }
    return rows, meta


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
    sql = f"""
        select employee_id, emp_name,
               date_trunc('week', worked_day)::date as week_start,
               (date_trunc('week', worked_day)::date + interval '6 days')::date as week_end,
               {PACE_SCORE_AGG_SQL} as avg_score,
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


# ---------------------------------------------------------------------------
# Top-10 gainer/loser ranking: last 4 COMPLETE calendar weeks (Mon-Sun) vs
# the 4 complete calendar weeks immediately before that. Uses the SAME
# capped-average-first 3-step methodology as employee_full_monthly_trend/
# ranking_weekly_pace_trend (avg the 4 capped sub-metrics across each
# period's Standard days FIRST, then apply the score formula ONCE per
# period) - NOT overall_new_pace_score_7_3/last_60_days_new_pace_score_7_3
# (different fixed rolling windows) and NOT an average of precomputed
# daily scores (the Jensen's-inequality bug fixed elsewhere in this
# project - see employee_full_monthly_trend's history).
# ---------------------------------------------------------------------------

def _gainer_loser_cte(dept_name, employee_ids, filter_sql):
    """Shared CTE chain: per-employee current/prior-period capped-average
    scores, filtered identically in BOTH periods by dept/employee_ids and by
    `filter_sql` - the full population filter (shift_type + visit_flag +
    ps_worked_flag_day, per the default-population rule), which restricts
    WHICH rows are included - it never changes the window itself. `filter_sql`
    is required (callers must always pass a population filter; main.py's
    _resolve_population_filter always supplies one, applying the confirmed
    default of shift_type='Standard' AND visit_flag='No' AND
    ps_worked_flag_day=1 when the user didn't ask for anything else)."""
    pop_filter = f"and {filter_sql}" if filter_sql else "and shift_type = 'Standard'"
    return f"""
        per_period as (
            select employee_id, emp_name, dept_name,
                   case when worked_day between %(cur_start)s and %(cur_end)s then 'cur'
                        when worked_day between %(prior_start)s and %(prior_end)s then 'prior'
                        else null end as period,
                   capped_engagement, capped_effectiveness, capped_discipline, capped_working_hours
            from public.pace_1
            where worked_day between %(prior_start)s and %(cur_end)s
              and capped_engagement is not null and capped_effectiveness is not null
              and capped_discipline is not null and capped_working_hours is not null
              and (%(dept_name)s is null or dept_name = %(dept_name)s)
              and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
              {pop_filter}
        ),
        agg as (
            select employee_id, emp_name, dept_name, period,
                   avg(capped_engagement) as avg_e, avg(capped_effectiveness) as avg_ef,
                   avg(capped_discipline) as avg_d, avg(capped_working_hours) as avg_w,
                   count(*) as n
            from per_period
            where period is not null
            group by employee_id, emp_name, dept_name, period
        ),
        scored as (
            select employee_id, emp_name, dept_name, period,
                   {PACE_SCORE_FROM_AVGS_SQL} as score,
                   n
            from agg
        ),
        pivoted as (
            select employee_id, emp_name, dept_name,
                   max(case when period = 'cur' then score end) as cur_score,
                   max(case when period = 'prior' then score end) as prior_score,
                   max(case when period = 'cur' then n end) as cur_n,
                   max(case when period = 'prior' then n end) as prior_n
            from scored
            group by employee_id, emp_name, dept_name
        )
    """


def gainer_loser_ranking(dept_name=None, employee_ids=None, filter_sql=None, limit=None, directions=("gainers", "losers")):
    """Top gainers and/or top losers ranked by score CHANGE (current
    4-complete-calendar-weeks period minus the prior 4-complete-calendar-weeks
    period). Employees need >=2 days of data in BOTH periods (under whatever
    population filter is in effect) to qualify. `directions` controls which
    of ('gainers', 'losers') are actually computed/returned - a losers-only
    question should never compute or show gainers, and vice versa.
    Returns (gainers, losers, excluded_count, meta) - gainers/losers is an
    empty list (not None) for a direction not requested."""
    cur_start, cur_end, prior_start, prior_end = last_4_weeks_periods()
    lim = limit or LIMIT
    params = {
        "cur_start": cur_start, "cur_end": cur_end,
        "prior_start": prior_start, "prior_end": prior_end,
        "dept_name": dept_name, "employee_ids": employee_ids,
    }
    cte = _gainer_loser_cte(dept_name, employee_ids, filter_sql)

    # Population count (for the exclusion note): everyone who shows up in
    # either period under the same dept/employee_ids/filter scope, vs. those
    # who actually qualify (>=2 Standard days in BOTH periods).
    pop_sql = f"""
        with {cte}
        select
            count(*) as total_seen,
            count(*) filter (where cur_n >= 2 and prior_n >= 2) as qualified
        from pivoted
    """
    pop_row = run_query(pop_sql, params)
    total_seen = pop_row[0]["total_seen"] if pop_row else 0
    qualified_count = pop_row[0]["qualified"] if pop_row else 0
    excluded_count = total_seen - qualified_count

    def _ranked(ascending):
        order = "asc" if ascending else "desc"
        sql = f"""
            with {cte},
            qualified as (
                select employee_id, emp_name, dept_name, cur_score, prior_score,
                       (cur_score - prior_score) as score_change
                from pivoted
                where cur_n >= 2 and prior_n >= 2
            )
            select employee_id, emp_name, dept_name, cur_score, prior_score, score_change
            from qualified
            order by score_change {order}
            limit {lim}
        """
        return run_query(sql, params)

    losers = _ranked(ascending=True) if "losers" in directions else []
    gainers = _ranked(ascending=False) if "gainers" in directions else []
    meta = {
        "cur_start": cur_start, "cur_end": cur_end,
        "prior_start": prior_start, "prior_end": prior_end,
        "excluded_count": excluded_count, "qualified_count": qualified_count,
    }
    return gainers, losers, excluded_count, meta


# ---------------------------------------------------------------------------
# Day-vs-day / metric comparison ("was 2 Sept or 7 Sept better", "was Aryan
# better on Sept 4 vs Sept 7?"). Single-day SNAPSHOT population aggregates,
# queried directly against pace_1 (shift_type='Standard', worked_day=<date>)
# rather than pace_chatbot_view's monthly CTEs - those are for period
# averages, this is a single fixed day, so there is no Jensen's-inequality
# capped-average correction needed here (that correction only applies when
# averaging a MULTI-day period for one employee; a plain AVG() of a
# population on one fixed day has no such issue). The default/explicit
# metric mapping below is a CONFIRMED business rule, not a guess.
# ---------------------------------------------------------------------------

DAY_COMPARE_METRICS = {
    "pace": ("new_pace_score_7_3_event_level", "PACE score"),
    "engagement": ("engagement_pct", "Engagement"),
    "discipline": ("discipline_pct", "Discipline"),
    "working": ("working_pct", "Working %"),
    "effectiveness": ("effectiveness_pct", "Effectiveness"),
}
DEFAULT_DAY_COMPARE_METRIC = "pace"


def day_compare(date1, date2, dept_name=None, employee_id=None, metric_keys=None, filter_sql=None):
    """Population-average comparison of one or more metrics between two
    fixed days. `dept_name` holds the department FIXED (both days, same
    department) - this is the "which day was better for department X" case,
    distinct from a department-vs-department comparison on one day (that's
    the existing dept_compare intent, untouched). `employee_id` scopes to
    one employee instead of a population average. `filter_sql` is the full
    population filter (shift_type + visit_flag + ps_worked_flag_day, per the
    default-population rule) - required; main.py's _resolve_population_filter
    always supplies one, applying shift_type='Standard' AND visit_flag='No'
    AND ps_worked_flag_day=1 by default. Returns a list of dicts, one per
    requested metric: {metric_key, label, val1, val2, n1, n2}."""
    keys = metric_keys or [DEFAULT_DAY_COMPARE_METRIC]
    pop_filter = f"and {filter_sql}" if filter_sql else "and shift_type = 'Standard'"
    results = []
    for key in keys:
        col, label = DAY_COMPARE_METRICS.get(key, DAY_COMPARE_METRICS[DEFAULT_DAY_COMPARE_METRIC])
        sql = f"""
            select worked_day, avg({col}) as val, count(*) as n
            from public.pace_1
            where worked_day = any(%(dates)s)
              and {col} is not null
              and (%(dept_name)s is null or dept_name = %(dept_name)s)
              and (%(employee_id)s is null or employee_id = %(employee_id)s)
              {pop_filter}
            group by worked_day
        """
        params = {"dates": [date1, date2], "dept_name": dept_name, "employee_id": employee_id}
        rows = run_query(sql, params)
        by_day = {r["worked_day"]: r for r in rows}
        r1 = by_day.get(date1)
        r2 = by_day.get(date2)
        results.append({
            "metric_key": key,
            "label": label,
            "val1": r1["val"] if r1 else None,
            "val2": r2["val"] if r2 else None,
            "n1": r1["n"] if r1 else 0,
            "n2": r2["n"] if r2 else 0,
        })
    return results


def day_compare_ranking(date1, date2, dept_name=None, employee_ids=None, metric_key=None,
                         filter_sql=None, ascending=True, limit=None):
    """Item #87 (bug C): per-EMPLOYEE two-date delta ranking - "compare the
    employees' engagement between 8 Sept and 10 Sept, who decreased the
    most?" needs a per-employee breakdown ranked by change, not
    day_compare()'s single population-average delta. Reuses day_compare()'s
    own single-day/no-Jensen's-inequality reasoning (a plain per-day value,
    no multi-day averaging) and DAY_COMPARE_METRICS for the metric mapping,
    plus the exact 2-period-pivot-then-join shape gainer_loser_ranking()
    already uses for month-over-month deltas - just pivoted on two fixed
    dates instead of two 4-week periods, and keyed by employee_id instead of
    a population aggregate.

    Only employees with a non-null value on BOTH dates (under the same
    population filter day_compare() uses) qualify - same "needs data on
    both sides" precedent as gainer_loser_ranking()'s cur_n>=2/prior_n>=2
    gate, just for single days instead of 4-week windows.

    Returns (rows, label) - rows are {employee_id, emp_name, dept_name,
    val1, val2, delta}, ordered by delta ascending (biggest decrease first)
    or descending (biggest increase first) per `ascending`; label is the
    metric's display label (for the reply header)."""
    key = metric_key or DEFAULT_DAY_COMPARE_METRIC
    col, label = DAY_COMPARE_METRICS.get(key, DAY_COMPARE_METRICS[DEFAULT_DAY_COMPARE_METRIC])
    pop_filter = f"and {filter_sql}" if filter_sql else "and shift_type = 'Standard'"
    lim = limit or LIMIT
    order = "asc" if ascending else "desc"
    sql = f"""
        with d1 as (
            select employee_id, emp_name, dept_name, {col} as val1
            from public.pace_1
            where worked_day = %(date1)s
              and {col} is not null
              and (%(dept_name)s is null or dept_name = %(dept_name)s)
              and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
              {pop_filter}
        ),
        d2 as (
            select employee_id, {col} as val2
            from public.pace_1
            where worked_day = %(date2)s
              and {col} is not null
              and (%(dept_name)s is null or dept_name = %(dept_name)s)
              and (%(employee_ids)s is null or employee_id = any(%(employee_ids)s))
              {pop_filter}
        )
        select d1.employee_id, d1.emp_name, d1.dept_name, d1.val1, d2.val2,
               (d2.val2 - d1.val1) as delta
        from d1
        join d2 on d1.employee_id = d2.employee_id
        order by delta {order}
        limit {lim}
    """
    params = {"date1": date1, "date2": date2, "dept_name": dept_name, "employee_ids": employee_ids}
    rows = run_query(sql, params)
    return rows, label


def month_compare(month1, month2, dept_name=None, employee_id=None, metric_keys=None, filter_sql=None):
    """Company-wide (or dept/employee-scoped) average comparison of one or
    more metrics between two full calendar months ('YYYY-MM' strings) -
    the month-granularity sibling of day_compare() above. Reuses
    DAY_COMPARE_METRICS for metric labels/defaults and the exact same
    tie-handling contract (result shape: {metric_key, label, val1, val2,
    n1, n2}), so main.py's existing format_day_compare() formats this
    output verbatim - no parallel response formatter was written.

    For metric_key == DEFAULT_DAY_COMPARE_METRIC ("pace"), this uses the
    SAME capped-average-first-then-formula-once methodology as
    employee_full_monthly_trend()/dept_ranking()/rm_ranking()/build_query()
    (Jensen's-inequality fix, see SESSION_HANDOFF.md items #52/#55): the 4
    capped_* ingredients are averaged across EVERY applicable row for that
    month at the requested scope (company-wide by default, or narrowed by
    dept_name/employee_id) FIRST, then the score formula is applied ONCE
    per month - never averaging a per-row precomputed score. This is a
    genuinely different aggregation shape from day_compare()'s single-day
    pace metric (which needs no such correction - a single calendar day
    has no per-employee multi-day averaging to get wrong), but the exact
    same fix already proven correct for month-level aggregates elsewhere
    in this file.

    For every other metric_key, this mirrors day_compare()'s own simple
    avg(<pct column>) treatment for non-pace metrics, for consistency with
    the sibling feature rather than introducing a second methodology.

    `filter_sql` is the full population filter, same convention as
    day_compare()/gainer_loser_ranking() - required; main.py's
    _resolve_population_filter always supplies one (shift_type='Standard'
    AND visit_flag='No' AND ps_worked_flag_day=1 by default)."""
    keys = metric_keys or [DEFAULT_DAY_COMPARE_METRIC]
    pop_filter = f"and {filter_sql}" if filter_sql else "and shift_type = 'Standard'"
    months = [month1, month2]
    results = []
    for key in keys:
        if key == "pace":
            sql = f"""
                select to_char(worked_day,'YYYY-MM') as mo,
                       avg(capped_engagement) as avg_e,
                       avg(capped_effectiveness) as avg_ef,
                       avg(capped_discipline) as avg_d,
                       avg(capped_working_hours) as avg_w,
                       count(*) as n
                from public.pace_1
                where to_char(worked_day,'YYYY-MM') = any(%(months)s)
                  and capped_engagement is not null and capped_effectiveness is not null
                  and capped_discipline is not null and capped_working_hours is not null
                  and (%(dept_name)s is null or dept_name = %(dept_name)s)
                  and (%(employee_id)s is null or employee_id = %(employee_id)s)
                  {pop_filter}
                group by 1
            """
            params = {"months": months, "dept_name": dept_name, "employee_id": employee_id}
            rows = run_query(sql, params)
            by_mo = {r["mo"]: r for r in rows}

            def _score(r):
                if not r or r["avg_e"] is None:
                    return None, 0
                val = min(100, round(
                    ((float(r["avg_e"]) * float(r["avg_ef"]) * float(r["avg_w"]) * 7)
                     + (float(r["avg_d"]) * 3)) * 10
                ))
                return val, r["n"]

            v1, n1 = _score(by_mo.get(month1))
            v2, n2 = _score(by_mo.get(month2))
            label = DAY_COMPARE_METRICS["pace"][1]
        else:
            col, label = DAY_COMPARE_METRICS.get(key, DAY_COMPARE_METRICS[DEFAULT_DAY_COMPARE_METRIC])
            sql = f"""
                select to_char(worked_day,'YYYY-MM') as mo, avg({col}) as val, count(*) as n
                from public.pace_1
                where to_char(worked_day,'YYYY-MM') = any(%(months)s)
                  and {col} is not null
                  and (%(dept_name)s is null or dept_name = %(dept_name)s)
                  and (%(employee_id)s is null or employee_id = %(employee_id)s)
                  {pop_filter}
                group by 1
            """
            params = {"months": months, "dept_name": dept_name, "employee_id": employee_id}
            rows = run_query(sql, params)
            by_mo = {r["mo"]: r for r in rows}
            r1 = by_mo.get(month1)
            r2 = by_mo.get(month2)
            v1 = r1["val"] if r1 else None
            n1 = r1["n"] if r1 else 0
            v2 = r2["val"] if r2 else None
            n2 = r2["n"] if r2 else 0

        results.append({
            "metric_key": key,
            "label": label,
            "val1": v1,
            "val2": v2,
            "n1": n1,
            "n2": n2,
        })
    return results


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
    "offline": ("offline_attendance_flag = 'Offline Attendance'", "marked offline attendance"),
    "ps_not_installed": ("ps_installed_new = 0", "did not have PS installed"),
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


def day_flag_list(flag_key, dept_name=None, employee_ids=None, month=None, date_range=None, limit=None,
                   exclude_employee_id=None):
    """List of employees (name/dept/day(s)) matching DAY_FLAGS[flag_key] at
    least once in the given period — the LIST counterpart of day_flag_count.

    `exclude_employee_id`, if given, drops that one employee from the result
    (the "beside X"/"except X"/"excluding X" list-exclusion capability) —
    everyone else in scope is still listed normally."""
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
          and (%(exclude_employee_id)s is null or employee_id != %(exclude_employee_id)s)
          and {frag}
        group by employee_id, emp_name, dept_name
        order by emp_name
        limit {lim}
    """
    params["dept_name"] = dept_name
    params["employee_ids"] = employee_ids
    params["exclude_employee_id"] = exclude_employee_id
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
    in `statuses` (e.g. ['Red']). `statuses=None` (or empty) means NO status
    filter at all - i.e. every current status - NOT the old hardcoded
    ["Red","Black"] default (item #59 fix: that default silently narrowed a
    bare "list of all employees"-style request to only Red/Black employees,
    truncating the real headcount)."""
    lim = limit or 200
    sql = f"""
        with {_latest_status_cte()}
        select employee_id, emp_name, dept_name, overall_std_pace_status
        from latest
        where (%(statuses)s is null or overall_std_pace_status = any(%(statuses)s))
        order by emp_name
        limit {lim}
    """
    return run_query(sql, {"dept_name": dept_name, "employee_ids": employee_ids, "statuses": statuses or None})


def status_count(statuses, dept_name=None, employee_ids=None):
    """`statuses=None` (or empty) means NO status filter - counts everyone in
    scope, not just the old hardcoded ["Red","Black"] default - see
    status_list() docstring (item #59)."""
    sql = f"""
        with {_latest_status_cte()}
        select count(*) as n
        from latest
        where (%(statuses)s is null or overall_std_pace_status = any(%(statuses)s))
    """
    rows = run_query(sql, {"dept_name": dept_name, "employee_ids": employee_ids, "statuses": statuses or None})
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
    """Bucket an employee's PACE score for a given SPECIFIC PAST month into
    the same Black/Red/Amber/Green thresholds the upstream status column
    uses (Black <50, Red 50-64, Amber 65-79, Green >=80).

    BUG FIX (this round, per SESSION_HANDOFF.md Part 4b): previously this
    averaged pace_chatbot_view's overall_pace_score column (aliased from
    last_60_days_new_pace_score_7_3, a rolling 60-*worked*-day window score),
    which can under-represent a specific past month since it's a rolling
    figure computed as of "now," not as of that month. Switched to the same
    capped-average-first recompute already used correctly elsewhere in this
    codebase (employee_full_monthly_trend's pace_score branch): average the 4
    capped sub-metrics for the SPECIFIC requested month directly from
    pace_1, then apply the score formula once. Only the score computation
    changed - the MIN_DAYS_FOR_DELTA reliability gate (`having count(*) >=
    %(min_days)s`) and dept_name/emp_name grouping are unchanged."""
    return f"""
        select employee_id, emp_name, dept_name,
               {PACE_SCORE_AGG_SQL} as avg_score,
               count(*) as days_counted
        from public.pace_1
        where to_char(worked_day,'YYYY-MM') = %(month)s and shift_type = 'Standard'
          and capped_engagement is not null and capped_effectiveness is not null
          and capped_discipline is not null and capped_working_hours is not null
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
        sql = f"""
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
                   {PACE_SCORE_FROM_AVGS_SQL} as metric_value,
                   days_counted
            from per_month
            order by mo
        """
        return run_query(sql, {"employee_id": employee_id})
    if metric_key in _CAPPED_SUBSCORE_COL:
        # Same Jensen's-inequality correctness fix as pace_score above:
        # engagement_pct/effectiveness_pct/discipline_pct in pace_1 are NOT
        # a simple linear rescaling of capped_engagement/capped_effectiveness/
        # capped_discipline (confirmed live: e.g. effectiveness_pct=50.00 pairs
        # with capped_effectiveness=0.53, not 0.50 - the capping step is
        # nonlinear). Averaging the pre-computed daily *_pct columns directly
        # would NOT match averaging the capped column first, so this mirrors
        # pace_chatbot_view/employee_full_monthly_trend's established fix:
        # average the capped column per month, then scale to a percentage.
        col = _CAPPED_SUBSCORE_COL[metric_key]
        sql = f"""
            select to_char(worked_day,'YYYY-MM') as mo,
                   avg({col}) * 100 as metric_value,
                   count(*) as days_counted
            from public.pace_1
            where employee_id = %(employee_id)s and shift_type = 'Standard' and {col} is not null
            group by 1
            order by 1
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


# Sub-score percentage metrics that need the capped-column-average
# methodology (see employee_full_monthly_trend's docstring) rather than a
# plain avg() of the precomputed pace_chatbot_view *_pct column.
_CAPPED_SUBSCORE_COL = {
    "engagement": "capped_engagement",
    "effectiveness": "capped_effectiveness",
    "discipline": "capped_discipline",
}


# ---------------------------------------------------------------------------
# Month-wise COUNT breakdowns (WFH/visit/leave/late/early/deficient-hours/OT)
# for a single employee - the count-based sibling of
# employee_full_monthly_trend's avg-based metrics. Straightforward COUNT(*)/
# SUM() grouped by calendar month, no formula subtlety (unlike the
# percentage sub-scores above).
# ---------------------------------------------------------------------------

COUNT_METRICS = {
    "wfh": ("public.pace_1", "sum(case when wfh_status = 'Work From Home' then 1 else 0 end)", "WFH days"),
    "visit": ("public.pace_1", "sum(case when visit_flag = 'Yes' then 1 else 0 end)", "visit days"),
    "leave": ("public.pace_1", "sum(case when applied_leave_type is not null then 1 else 0 end)", "leave days"),
    "late_comings": (VIEW, "sum(coalesce(lc_flag_per_day,0))", "late-coming days"),
    "early_leavings": (VIEW, "sum(coalesce(el_flag_per_day,0))", "early-leaving days"),
    "deficient_hours": (VIEW, "sum(coalesce(dh_flag_per_day,0))", "deficient-hour days"),
    "ot_days": (None, None, "OT days"),
    "ot_hours": (None, None, "OT hours"),
}


def employee_monthly_count_trend(employee_id, metric_key):
    """One row per calendar month, a COUNT/SUM-based metric (not an average)
    for a single employee - e.g. "WFH days month wise". OT days/hours are a
    special case (filtered to shift_type = 'Overtime (OT)', same convention
    as ot_hours_ranking) rather than a plain table/expr pair."""
    if metric_key in ("ot_days", "ot_hours"):
        expr = "count(*)" if metric_key == "ot_days" else "sum(coalesce(worked_hours,0))"
        sql = f"""
            select to_char(worked_day,'YYYY-MM') as mo, {expr} as metric_value, count(*) as days_counted
            from public.pace_1
            where employee_id = %(employee_id)s and shift_type = 'Overtime (OT)'
            group by 1
            order by 1
        """
        return run_query(sql, {"employee_id": employee_id})
    table, expr, _ = COUNT_METRICS[metric_key]
    sql = f"""
        select to_char(worked_day,'YYYY-MM') as mo, {expr} as metric_value, count(*) as days_counted
        from {table}
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
        sql = f"""
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
                   {PACE_SCORE_FROM_AVGS_SQL} as metric_value,
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
        sql = f"""
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
                   {PACE_SCORE_FROM_AVGS_SQL} as metric_value,
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


# ---------------------------------------------------------------------------
# PS (ps_worked_flag_day) exclusion support.
#
# `ps_worked_flag_day` only exists on public.pace_1 (NOT on
# pace_chatbot_view, confirmed against the live schema), so every function
# below queries pace_1 directly rather than VIEW. When PS wasn't working that
# day, the day's engagement/effectiveness/productivity numbers reflect no
# real usage capture and are unreliable — these functions let a query
# explicitly drop those rows (ps_worked_flag_day = 0) before aggregating, or
# report on the PS-off days themselves.
#
# Threshold used for the PROACTIVE data-quality caveat (Category C): flagged
# only when PS-off days are >= 25% of the days counted in the period AND
# there are at least 3 days in the period at all (so a single-day "yesterday"
# query, where a PS-off day is either 0% or 100% of the period and adds no
# real information, never triggers a noisy caveat). This is a judgment call,
# not a value pulled from the data.
# ---------------------------------------------------------------------------

PS_OFF_CAVEAT_MIN_DAYS = 3
PS_OFF_CAVEAT_RATIO = 0.25

PS_FILTERED_METRICS = {
    "pace_score": (PACE_SCORE_AGG_SQL, "PACE score"),
    "engagement": ("avg(engagement_pct)", "engagement %"),
    "effectiveness": ("avg(effectiveness_pct)", "effectiveness %"),
    "discipline": ("avg(discipline_pct)", "discipline %"),
    "working_hours": ("avg(working_pct)", "working hours %"),
    "productive_min": ("sum(coalesce(productive_and_meeting_min,0))", "productive minutes"),
    "whatsapp_min": ("sum(coalesce(whatsapp_min,0))", "WhatsApp minutes"),
    "ai_min": ("sum(coalesce(ai_min,0))", "AI tool usage minutes"),
    "tools_and_mails_min": ("sum(coalesce(tools_and_mails_min,0))", "tools & mail minutes"),
}


def employee_metric_ps_filtered(employee_id, metric_key, month=None, date_range=None, exclude_ps_off=True):
    """avg/sum of one PS_FILTERED_METRICS metric for one employee, optionally
    excluding ps_worked_flag_day=0 rows first. Also returns the day counts so
    call sites can render "no data" gracefully (Category E) instead of a bare
    NULL when every day in scope is PS-off."""
    expr, label = PS_FILTERED_METRICS[metric_key]
    frag, params = _period_filter(month, date_range)
    ps_frag = "and ps_worked_flag_day = 1" if exclude_ps_off else ""
    sql = f"""
        select {expr} as metric_value,
               count(*) as days_counted,
               sum(case when ps_worked_flag_day = 1 then 1 else 0 end) as ps_working_days,
               sum(case when ps_worked_flag_day = 0 or ps_worked_flag_day is null then 1 else 0 end) as ps_off_days
        from public.pace_1
        where employee_id = %(employee_id)s and {frag} {ps_frag}
    """
    params["employee_id"] = employee_id
    rows = run_query(sql, params)
    row = rows[0] if rows else None
    if row is not None:
        row["label"] = label
    return row


def metric_ranking_ps_filtered(metric_key, dept_name, month=None, date_range=None, ascending=False,
                                employee_ids=None, limit=None, exclude_ps_off=True):
    """Same shape as metric_ranking(), but for a "remove/exclude PS
    non-working rows, then rank by X" follow-up with NO named employee (a
    ranking request, not a single-employee lookup) - queries pace_1 directly
    (ps_worked_flag_day isn't on pace_chatbot_view, same reason as the other
    PS-filtered functions above) and drops ps_worked_flag_day=0 rows before
    aggregating each employee's metric."""
    expr, label = PS_FILTERED_METRICS[metric_key]
    frag, params = _period_filter(month, date_range)
    filters = [frag]
    if dept_name:
        filters.append("dept_name = %(dept_name)s")
        params["dept_name"] = dept_name
    if employee_ids is not None:
        filters.append("employee_id = any(%(employee_ids)s)")
        params["employee_ids"] = employee_ids
    if exclude_ps_off:
        filters.append("ps_worked_flag_day = 1")
    where_clause = " and ".join(filters)
    order = "asc" if ascending else "desc"
    sql = f"""
        select employee_id, emp_name, dept_name,
               {expr} as metric_value,
               count(*) as days_counted
        from public.pace_1
        where {where_clause}
        group by employee_id, emp_name, dept_name
        order by metric_value {order} nulls last
        limit %(limit)s
    """
    params["limit"] = limit or LIMIT
    rows = run_query(sql, params)
    for r in rows:
        r["label"] = label
    return rows


def ps_working_ratio(employee_id, month=None, date_range=None):
    """Informational PS-working-day count/ratio for one employee (Category B)."""
    frag, params = _period_filter(month, date_range)
    sql = f"""
        select count(*) as total_days,
               sum(case when ps_worked_flag_day = 1 then 1 else 0 end) as ps_working_days,
               sum(case when ps_worked_flag_day = 0 or ps_worked_flag_day is null then 1 else 0 end) as ps_off_days
        from public.pace_1
        where employee_id = %(employee_id)s and {frag}
    """
    params["employee_id"] = employee_id
    rows = run_query(sql, params)
    return rows[0] if rows else None


def ps_off_ranking(dept_name, month=None, date_range=None, employee_ids=None, ascending=False, limit=None):
    """Ranking of employees by PS non-working day count (Category B: "who has
    the most/fewest PS non-working days")."""
    frag, params = _period_filter(month, date_range)
    filters = [frag]
    if dept_name:
        filters.append("dept_name = %(dept_name)s")
        params["dept_name"] = dept_name
    if employee_ids is not None:
        filters.append("employee_id = any(%(employee_ids)s)")
        params["employee_ids"] = employee_ids
    where_clause = " and ".join(filters)
    order = "asc" if ascending else "desc"
    sql = f"""
        select employee_id, emp_name, dept_name,
               sum(case when ps_worked_flag_day = 0 or ps_worked_flag_day is null then 1 else 0 end) as ps_off_days,
               count(*) as total_days
        from public.pace_1
        where {where_clause}
        group by employee_id, emp_name, dept_name
        order by ps_off_days {order} nulls last
        limit %(limit)s
    """
    params["limit"] = limit or 10
    return run_query(sql, params)


# ---------------------------------------------------------------------------
# Category N (new, additive) — build_query(): a general parametrized query
# engine for the ~87 hand-verified intents' fallback path. NOT wired into any
# existing intent - see app/main.py's routing, which only reaches this after
# rule-based matching AND the LLM-classification/SQL-fallback cascade both
# fail to find a real intent (or for bare dimension+overview phrasing with no
# existing coverage, e.g. the item #56 "how is ai labs doing" bug). Every one
# of the 87 existing intents keeps using its own hand-written function,
# unchanged, with priority over this - per the project's standing rule
# against discarding prior hand-verified work (see SESSION_HANDOFF.md).
#
# Always queries public.pace_1 directly (never pace_chatbot_view): every
# filter this engine supports by default (ps_worked_flag_day, visit_flag,
# shift_type) lives only on pace_1, same reason metric_ranking_ps_filtered()
# and the other PS-filtered functions above do the same thing.
# ---------------------------------------------------------------------------

BUILD_QUERY_DIMENSIONS = {
    "employee": ("employee_id, emp_name, dept_name", ["employee_id", "emp_name", "dept_name"]),
    "rm": ("reporting_manager_name", ["reporting_manager_name"]),
    "department": ("dept_name", ["dept_name"]),
    # "company": no GROUP BY at all - every matching row collapses into one
    # aggregate row, used by the new average_metric intent for a "whole
    # company" scope (no department/RM/employee named at all). New,
    # additive - every existing caller still passes "employee"/"rm"/
    # "department" and is unaffected.
    "company": (None, []),
}

# metric key -> (sql aggregate expression against pace_1, human label).
# "pace_score" is handled separately below (capped-average-first-then-
# formula-once, same pattern as dept_ranking()/rm_ranking() - see their
# docstrings for the Jensen's-inequality bug this avoids).
BUILD_QUERY_METRICS = {
    "engagement_pct": ("avg(engagement_pct)", "avg engagement %"),
    "effectiveness_pct": ("avg(effectiveness_pct)", "avg effectiveness %"),
    "discipline_pct": ("avg(discipline_pct)", "avg discipline %"),
    "working_pct": ("avg(working_pct)", "avg working hours %"),
    "LC": ("sum(coalesce(lc_flag_per_day,0))", "late-comings"),
    "EL": ("sum(coalesce(el_flag_per_day,0))", "early leavings"),
    "DH": ("sum(coalesce(dh_flag_per_day,0))", "deficient-hour days"),
    "working_hours": ("sum(coalesce(worked_hours,0))", "total working hours"),
    # Added for the new average_metric intent - raw productive-minutes was
    # previously only reachable via dedicated functions, not this general
    # engine. Averaged per employee-day, same treatment as the *_pct metrics
    # above (not summed like the count metrics).
    "productive_minutes": ("avg(coalesce(productive_and_meeting_min,0))", "avg productive minutes"),

    # --- Item #70 gap-fill (this round) -----------------------------------
    # Raw capped ingredients, exposed as directly queryable metrics in their
    # own right (previously ONLY reachable buried inside the pace_score
    # formula below, per item #70 finding #3). Averaged per row in scope,
    # same treatment as the other *_pct metrics - these are still the
    # INTERNAL capped 0-1(ish) values, not the uncapped user-facing
    # percentages (engagement_pct etc.) - deliberately kept as a distinct
    # metric family, not a rename, per PROJECT_BACKUP_2026-09-09.md's
    # documented "capped_* is internal only" note.
    "capped_engagement": ("avg(capped_engagement)", "avg capped engagement (internal)"),
    "capped_effectiveness": ("avg(capped_effectiveness)", "avg capped effectiveness (internal)"),
    "capped_discipline": ("avg(capped_discipline)", "avg capped discipline (internal)"),

    # pace_score_day_level: new_pace_score_7_3_event_level, per item #70
    # finding #2. Distinct from "pace_score" below (which is the
    # capped-average-first-then-formula-once PERIOD aggregate). This is the
    # per-row EVENT-LEVEL score column averaged over whatever period/
    # dimension is requested - for a single-day period this is exactly the
    # employee_day_summary() day-level score; for a longer period it's an
    # average of daily event-level scores (a genuinely different number from
    # "pace_score", not a duplicate - judgment call, see SESSION_HANDOFF.md).
    "pace_score_day_level": ("avg(new_pace_score_7_3_event_level)", "avg day-level (event) pace score"),

    # dept_score_60_days_precomputed: the ETL-precomputed dept_score_60_days_7_3
    # column, per item #70 finding #5. Deliberately named distinctly from
    # "pace_score" (the existing LIVE-RECOMPUTED capped-average score) to
    # avoid the exact naming-collision bug class documented in
    # PROJECT_BACKUP_2026-09-09.md (Active-vs-Inactive PACE score collision).
    # Only meaningful for dimension="department" - it is a department-grain
    # precomputed column, NOT filtered by the period param (it's already a
    # fixed rolling-60-day ETL figure), so period is ignored for this metric
    # specifically (documented, not a bug).
    "dept_score_60_days_precomputed": ("avg(dept_score_60_days_7_3)", "precomputed dept score (last 60 days, ETL)"),

    # --- Item #79 gap-fill (rows 39/40/49) --------------------------------
    # Per item #77's read-only audit (SESSION_HANDOFF.md). engagement_minutes
    # was completely unwired anywhere in app/*.py - modeled on its raw-minutes
    # sibling productive_minutes above (averaged per employee-day, not summed).
    "engagement_minutes": ("avg(coalesce(engagement_minutes,0))", "avg engagement minutes"),
    # meeting_minutes: reuses the exact sum(coalesce(meeting_in_min,0))
    # expression already shipped in meeting_minutes_ranking()/
    # meeting_activity_for_employee() - not a new formula. Key deliberately
    # named "meeting_minutes" (not the raw column name "meeting_in_min") to
    # avoid confusion with the pre-existing meeting_min_ranking intent name,
    # same naming-collision-avoidance precedent as dept_score_60_days_precomputed.
    "meeting_minutes": ("sum(coalesce(meeting_in_min,0))", "total meeting minutes"),
    # meeting_count: the raw meeting-count column, distinct from meeting_minutes
    # (a duration). Not explicitly in item #77's plan but needed so "how many
    # meetings did X have" (a numeric-count question) is answerable and clearly
    # distinguishable from the new boolean "had any meetings" day-flag path
    # (per the coordinator's row-51 requirement) rather than falling through
    # unrecognized. Same sum(coalesce(...,0)) treatment as meeting_minutes.
    "meeting_count": ("sum(coalesce(meeting_count,0))", "total meetings"),
    # tasks_created / tasks_assigned: byte-identical sum(coalesce(...,0))
    # expressions copied from TASK_METRICS, kept as two SEPARATE keys per
    # item #77's recommendation - no combined "task_activity" magnitude
    # metric invented (no existing precedent for that combination anywhere
    # in the codebase). The boolean "had any task activity" framing stays
    # exclusively on the existing DAY_FLAGS["completed_tasks"] path.
    "tasks_created": ("sum(coalesce(tasks_created,0))", "tasks created"),
    "tasks_assigned": ("sum(coalesce(tasks_assigned,0))", "tasks assigned"),
    # todos_created / todos_assigned: same gap pattern as tasks_created/
    # tasks_assigned (CSV rows 42/43), verified present as raw columns but
    # unwired into BUILD_QUERY_METRICS - same sum(coalesce(...,0)) treatment.
    "todos_created": ("sum(coalesce(todos_created,0))", "todos created"),
    "todos_assigned": ("sum(coalesce(todos_assigned,0))", "todos assigned"),
}

# pace_status banding thresholds - MUST mirror _bucket_status() above
# (Black <50, Red 50-64, Amber 65-79, Green >=80) so this stays in sync with
# the existing status_list()/status_count() family instead of drifting.
_PACE_STATUS_CASE_SQL = (
    "case when {score} is null then null "
    "when {score} < 50 then 'Black' "
    "when {score} < 65 then 'Red' "
    "when {score} < 80 then 'Amber' "
    "else 'Green' end"
)

BUILD_QUERY_DEFAULT_PERIOD_DAYS = 60


def default_period_last_60_days():
    """Shared "no period named at all" default — last 60 days (today
    inclusive). Item B (SESSION_HANDOFF.md): the matrix-wide default was
    changed from "current (in-progress) calendar month" to this, since a
    partial current month frequently produced misleading "not enough data"
    answers even though 60 real days of usable history exists. Was already
    used by build_query() (item #57) under the old private name
    `_build_query_default_period` — renamed to a public, shared helper so
    other ranking functions (metric_ranking/dept_ranking/rm_ranking) can
    reuse the exact same window instead of re-deriving it."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=BUILD_QUERY_DEFAULT_PERIOD_DAYS - 1)
    return start, end


# Backward-compatible alias (old private name) — kept in case any other
# in-repo caller still references it directly.
_build_query_default_period = default_period_last_60_days


def build_query(dimension, metrics, filters=None, period=None, name_filter=None, limit=None, scope=None,
                 ascending=False, latest_n_days=None):
    """General parametrized engine: SELECT <metrics> GROUP BY <dimension> FROM
    public.pace_1 WHERE <filters> AND <period>.

    dimension: "employee" | "rm" | "department" - what to group by.
    metrics: list of BUILD_QUERY_METRICS keys, plus "pace_score" (special-
        cased below to reuse the exact capped-average-first-then-formula-once
        pattern already verified in dept_ranking()/rm_ranking()).
    ascending: item #73 addition - False (default, unchanged behaviour) sorts
        the ranking metric descending ("best"/"most"); True sorts ascending
        ("worst"/"least"), same convention as dept_ranking()/rm_ranking()'s
        own `ascending` param. Only affects multi-row rankings (no
        name_filter) - a single name_filter query already collapses to <= 1
        row regardless of sort direction.
    filters: dict, any of:
        ps_status: "working" (default, ps_worked_flag_day=1) | "not_working" (=0) | "any" (no filter)
        visit_status: "no" (default, visit_flag='No') | "yes" (visit_flag='Yes') | "any"
        work_mode: None (default - no filter) | "wfh" (wfh_status='Work From Home') | "office"
        shift_type: "Standard" (default) | any other pace_1 shift_type value | "any"
    period: (start_date, end_date) tuple, or None -> defaults to the last 60
        days (today inclusive). Mutually exclusive with `latest_n_days` -
        ignored (not applied at all) whenever `latest_n_days` is given.
    latest_n_days: item #76 (Phase 3) addition - "N qualifying rows" mode,
        distinct from `period`'s fixed calendar-date-range mode. When set,
        `period` is ignored entirely and NO date-range clause is applied to
        the qualifying population at all - instead, ALL other filters
        (ps_status/visit_status/work_mode/shift_type/name_filter/scope/the
        capped-not-null clause for pace_score/pace_status) are applied FIRST
        to define the qualifying population, THEN the latest N rows *per
        employee* (ordered by worked_day desc) are taken from that
        population via a row_number() CTE, and only THEN aggregated. This is
        the deliberate fix for the "last 10 WFH days" business rule: taking
        a fixed calendar window first and filtering afterward would silently
        under-count whenever the qualifying condition (e.g. WFH) doesn't
        hold on every day of that window - this mode never does that.
        Partitioning by employee_id (rather than by the requested dimension)
        is intentional so the semantics compose correctly regardless of
        dimension: a department/RM/company-level "last N qualifying days"
        query is simply each employee's own latest N qualifying days,
        aggregated together - never one shared calendar window applied
        uniformly to every employee regardless of when THEIR qualifying days
        actually fell.
    name_filter: optional exact dept_name / reporting_manager_name / employee_id
        to scope to one group (e.g. a single department's overview).
    limit: max rows returned (default LIMIT for a ranking-style call; a
        single-name_filter call naturally returns <= 1 row regardless).
    scope: optional (scope_dimension, scope_name) tuple - an ADDITIONAL
        filter on a DIFFERENT dimension than the one being grouped by, e.g.
        dimension="employee" with scope=("department", "AI Labs") to list
        every employee IN that department (item #59: response-shape
        switching from a department summary to its per-employee list reuses
        this, same rerun_list mechanism as the older ranking functions).

    Returns a list of dict rows, one per group, each with the dimension's
    key column(s), `n_employees`, and one column per requested metric
    (pace_score as `pace_score`, others under their BUILD_QUERY_METRICS key).
    """
    if dimension not in BUILD_QUERY_DIMENSIONS:
        raise ValueError(f"build_query: unknown dimension {dimension!r}")
    group_cols, select_cols = BUILD_QUERY_DIMENSIONS[dimension]

    filters = dict(filters or {})
    ps_status = filters.get("ps_status", "working")
    visit_status = filters.get("visit_status", "no")
    work_mode = filters.get("work_mode")  # None means "no filter" - matches the confirmed default
    shift_type = filters.get("shift_type", "Standard")

    where = []
    params = {}
    if ps_status == "working":
        where.append("ps_worked_flag_day = 1")
    elif ps_status == "not_working":
        where.append("(ps_worked_flag_day = 0 or ps_worked_flag_day is null)")
    # ps_status == "any" -> no clause

    if visit_status == "no":
        where.append("visit_flag = 'No'")
    elif visit_status == "yes":
        where.append("visit_flag = 'Yes'")
    # visit_status == "any" -> no clause

    if work_mode == "wfh":
        where.append("wfh_status = 'Work From Home'")
    elif work_mode == "office":
        where.append("(wfh_status is null or wfh_status <> 'Work From Home')")
    # work_mode is None -> no clause (confirmed default: no work-mode filter)

    if shift_type and shift_type != "any":
        where.append("shift_type = %(shift_type)s")
        params["shift_type"] = shift_type

    if latest_n_days is None:
        # Fixed calendar-window mode (unchanged from before this round).
        if period is None:
            period = _build_query_default_period()
        start, end = period
        where.append("worked_day between %(date_start)s and %(date_end)s")
        params["date_start"] = start
        params["date_end"] = end
    # else: latest_n_days mode - NO date clause here at all; the row-count
    # window is applied later via the row_number() CTE, after every other
    # filter below has already defined the qualifying population.

    if name_filter:
        col = {"employee": "employee_id", "rm": "reporting_manager_name", "department": "dept_name"}[dimension]
        where.append(f"{col} = %(name_filter)s")
        params["name_filter"] = name_filter

    if dimension in ("rm", "department"):
        col = "reporting_manager_name" if dimension == "rm" else "dept_name"
        where.append(f"{col} is not null")

    if scope:
        scope_dim, scope_name = scope
        scope_col = {"employee": "employee_id", "rm": "reporting_manager_name", "department": "dept_name"}[scope_dim]
        where.append(f"{scope_col} = %(scope_name)s")
        params["scope_name"] = scope_name

    where_clause = " and ".join(where) if where else "true"

    select_exprs = list(select_cols)
    want_pace_score = "pace_score" in metrics
    # pace_status (item #70 finding #1): Black/Red/Amber/Green banding of the
    # SAME live-recomputed capped-average pace_score this engine already
    # computes for the requested dimension/period/filters - NOT the older
    # status_list()/status_count() family's "latest single worked_day" CTE
    # (a structurally different, non-composable code path). This is a
    # deliberate, documented difference: pace_status here bands the PERIOD
    # aggregate score for whatever scope was requested, consistent with how
    # every other build_query() metric behaves.
    want_pace_status = "pace_status" in metrics
    # dept_status_60_days_derived (item #70 finding #4): dept_status_60_days
    # does not exist as a real column anywhere - this DERIVES a department-
    # level status banding from dept_score_60_days_7_3 using the identical
    # thresholds _bucket_status()/pace_status use, since no real column
    # exists. Only meaningful for dimension="department".
    want_dept_status_derived = "dept_status_60_days_derived" in metrics
    metric_exprs = []
    for m in metrics:
        if m in ("pace_score", "pace_status", "dept_status_60_days_derived"):
            continue
        if m not in BUILD_QUERY_METRICS:
            raise ValueError(f"build_query: unknown metric {m!r}")
        expr, _ = BUILD_QUERY_METRICS[m]
        metric_exprs.append(f'{expr} as "{m}"')

    if want_dept_status_derived:
        score_expr = "avg(dept_score_60_days_7_3)"
        metric_exprs.append(
            f'{pace_status_sql(score_expr)} as "dept_status_60_days_derived"'
        )

    if want_pace_score or want_pace_status:
        # Same capped-average-first-then-formula-once pattern as
        # dept_ranking()/rm_ranking() for metric_key="pace_score" - averaging
        # the 4 capped_* ingredients per group and applying the score formula
        # ONCE, instead of averaging the view's pre-computed per-row score
        # (the Jensen's-inequality bug fixed in commit c1604cb). Item #75:
        # now sourced from the single centralized PACE_SCORE_AGG_SQL constant
        # instead of its own hand-copied literal.
        pace_score_expr = PACE_SCORE_AGG_SQL
        if want_pace_score:
            metric_exprs.append(f'{pace_score_expr} as "pace_score"')
        if want_pace_status:
            metric_exprs.append(
                f'{pace_status_sql(pace_score_expr)} as "pace_status"'
            )
        where.append(
            "capped_engagement is not null and capped_effectiveness is not null "
            "and capped_discipline is not null and capped_working_hours is not null"
        )
        where_clause = " and ".join(where)

    lim = limit or LIMIT
    # "company" dimension has no group-by column(s) at all (select_cols is
    # empty, group_cols is None) - every matching row collapses into ONE
    # aggregate row, so the GROUP BY clause is omitted entirely rather than
    # grouping by nothing. Every other dimension keeps the original
    # behaviour unchanged.
    select_parts = [c for c in ([", ".join(select_cols)] if select_cols else []) if c] + \
        ["count(distinct employee_id) as n_employees"] + metric_exprs
    group_by_clause = f"\n        group by {group_cols}" if group_cols else ""
    if want_pace_score:
        order_col = '"pace_score"'
    elif want_dept_status_derived:
        order_col = '"dept_status_60_days_derived"'
    elif metrics:
        order_col = f'"{metrics[0]}"'
    else:
        order_col = "n_employees"
    order_dir = "asc" if ascending else "desc"
    params["limit"] = lim

    if latest_n_days is None:
        sql = f"""
            select {", ".join(select_parts)}
            from public.pace_1
            where {where_clause}{group_by_clause}
            order by {order_col} {order_dir} nulls last
            limit %(limit)s
        """
        return run_query(sql, params)

    # latest_n_days mode (item #76): `where_clause` above already reflects
    # every non-date filter (ps/visit/work_mode/shift_type/name_filter/
    # scope/capped-not-null) - the qualifying population. Rank each
    # employee's own rows within that population by worked_day desc, keep
    # only the latest N per employee, THEN aggregate - never the reverse
    # order. `from filtered` (not `from public.pace_1`) is the only
    # difference in the outer query's FROM/WHERE-rn shape vs the plain mode
    # above; every select/group-by/order-by expression is identical since
    # `filtered` carries every pace_1 column through unchanged (`select *`).
    params["latest_n_days"] = int(latest_n_days)
    sql = f"""
        with filtered as (
            select *,
                   row_number() over (partition by employee_id order by worked_day desc) as rn
            from public.pace_1
            where {where_clause}
        )
        select {", ".join(select_parts)}
        from filtered
        where rn <= %(latest_n_days)s{group_by_clause}
        order by {order_col} {order_dir} nulls last
        limit %(limit)s
    """
    return run_query(sql, params)


def build_query_day_flags(employee_id, date):
    """Item #70 finding #6: single-day LC/EL/DH yes/no flags through the
    build_query() engine, reusing build_query()'s own SQL (not duplicating
    employee_day_summary()'s query) - for a single-day period the existing
    sum(coalesce(lc_flag_per_day,0)) etc. metrics collapse to exactly 0 or 1
    for one employee, so this is a thin convenience wrapper: call
    build_query(dimension="employee", period=(date,date), name_filter=...)
    and cast the 3 count metrics to booleans. Returns a dict with
    employee_id/emp_name/dept_name/lc/el/dh (booleans), or None if the
    employee has no Standard-shift row that day (same "no row" contract as
    employee_day_summary())."""
    rows = build_query(
        dimension="employee",
        metrics=["LC", "EL", "DH"],
        filters={"shift_type": "Standard", "ps_status": "any", "visit_status": "any"},
        period=(date, date),
        name_filter=employee_id,
        limit=1,
    )
    if not rows:
        return None
    row = rows[0]
    return {
        "employee_id": row.get("employee_id"),
        "emp_name": row.get("emp_name"),
        "dept_name": row.get("dept_name"),
        "lc": bool(row.get("LC")),
        "el": bool(row.get("EL")),
        "dh": bool(row.get("DH")),
    }


# ---------------------------------------------------------------------------
# Category N+1 (new, additive) — employee_day_summary(): a single-employee,
# single-day SNAPSHOT, deliberately a different output shape from every
# ranking/aggregate function above (no GROUP BY, no population average - one
# row for one person on one day). Queries public.pace_1 directly (not
# pace_chatbot_view) for two reasons: (1) it needs new_pace_score_7_3_event_level,
# the day-level event score, which IS selected on pace_1 but the view only
# exposes rolling/monthly aggregates of it; (2) shift_type='Standard' is
# applied explicitly here (same "exactly one row per employee/day" grain
# guarantee documented in PROJECT_BACKUP_2026-09-09.md §2) rather than relying
# on the view's baked-in filter, keeping this function self-contained and
# consistent with the other pace_1-direct functions above (day_compare(),
# build_query()).
# ---------------------------------------------------------------------------

def employee_day_summary(employee_id, date):
    """Returns a single dict (or None if the employee has no Standard-shift
    row for that day - e.g. leave/absent/OT-only day) with:
    employee_id, emp_name, dept_name, reporting_manager_name, worked_day,
    lc (bool), el (bool), dh (bool), pace_score (day-level event score,
    new_pace_score_7_3_event_level - NOT the rolling 60-day or capped-average
    aggregate used elsewhere, per the confirmed spec for this feature)."""
    sql = """
        select employee_id, emp_name, dept_name, reporting_manager_name,
               worked_day,
               coalesce(lc_flag_per_day, 0) > 0 as lc,
               coalesce(el_flag_per_day, 0) > 0 as el,
               coalesce(dh_flag_per_day, 0) > 0 as dh,
               new_pace_score_7_3_event_level as pace_score
        from public.pace_1
        where employee_id = %(employee_id)s
          and worked_day = %(date)s
          and shift_type = 'Standard'
        limit 1
    """
    rows = run_query(sql, {"employee_id": employee_id, "date": date})
    return rows[0] if rows else None
