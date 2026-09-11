"""Golden-validation harness for the centralized PACE score formula (item
#75, Phase 2 of the item #74 audit).

PURPOSE
-------
Independently reconstructs each employee's "current" PACE score using the
SAME centralized formula app/queries.py now uses everywhere
(PACE_SCORE_AGG_SQL / PACE_SCORE_QUALIFYING_ROWS_SQL), applied over their
most recent 60 QUALIFYING rows (shift_type='Standard', all 4 capped_*
columns non-null) - NOT a fixed 60-CALENDAR-day window. Compares the
result against the stored `last_60_days_new_pace_score_7_3` column (read
directly from public.pace_1) for an exact match.

WHY "last 60 qualifying rows" and not "last 60 calendar days":
Live testing against production (see SESSION_HANDOFF.md item #75) showed
that for STABLE employees (flat month-to-month trend) a plain 60-calendar-
day window (build_query()'s own default_period_last_60_days()) already
matches the stored column exactly. For VOLATILE employees (large recent
swings, e.g. a steep decline into a low-day-count partial current month)
a 60-calendar-day window can be off by a few points, because
build_query()'s default filters ALSO restrict to ps_worked_flag_day=1 AND
visit_flag='No' (see queries.py:~2783-2791) - filters the "clean" formula
(as used by employee_full_monthly_trend/dept_ranking/rm_ranking) does NOT
apply. This script queries pace_1 directly with the clean qualifying-row
filter only (matching PACE_SCORE_QUALIFYING_ROWS_SQL) and windows by ROW
COUNT (most recent 60 qualifying rows), which is the closest true
reconstruction of a rolling "last 60 [qualifying] days" window without
guessing at the upstream ETL's exact calendar-day boundary/refresh timing.
If this script is run with real DB credentials and still shows a
mismatch for some employees, that is itself useful signal - re-read this
docstring's "STOP AND INVESTIGATE" section below before assuming the
centralized formula is wrong.

USAGE
-----
Requires PACE_DB_HOST / PACE_DB_PORT / PACE_DB_NAME / PACE_DB_USER /
PACE_DB_PASSWORD in the environment (same convention as app/db.py). Not
runnable in a sandbox with no DB credentials (see SESSION_HANDOFF.md items
#71-75's recurring "no local DB credentials" note) - in that situation,
validate instead via the live chat endpoint per the methodology recorded
in SESSION_HANDOFF.md item #75's live-testing transcript.

    python scripts/golden_validate_pace_score.py [employee_id ...]

With no arguments, validates a small built-in sample. All exploratory
queries below are pure SELECTs (no writes) - the DB user is NOT read-only
at the grant level, so this script deliberately never issues INSERT/
UPDATE/DELETE/DDL.

STOP AND INVESTIGATE if a mismatch appears
-------------------------------------------
Per the item #75 task's own instruction: do NOT paper over a mismatch.
Things to check, in order:
  1. Wrong window - is the true ETL window 60 calendar days, 60 worked-
     day rows, or something else (e.g. anchored to the ETL's last refresh
     time rather than "now")? Compare against a plain calendar-day
     reconstruction too (this script prints both).
  2. Wrong filter - does the true formula apply ps_worked_flag_day/
     visit_flag restrictions after all? Compare the "clean" vs
     "default-filtered" reconstructions (this script prints both).
  3. Wrong rounding - least(100, round(...)) rounds ONCE at the end;
     verify no intermediate rounding crept in.
  4. Wrong row eligibility - confirm shift_type='Standard' and all 4
     capped_* columns non-null is really the full eligibility rule (not,
     e.g., also requiring ps_worked_flag_day=1 for THIS specific column).
"""
import os
import sys

import psycopg2
import psycopg2.extras

# Import the single source of truth for the formula/status SQL fragments -
# this script MUST use the exact same constants app/queries.py uses, never
# a hand-retyped copy, or a future formula change could silently desync
# the "golden" reconstruction from the real thing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.queries import (  # noqa: E402
    PACE_SCORE_FROM_AVGS_SQL,
    PACE_SCORE_QUALIFYING_ROWS_SQL,
    pace_status_sql,
)

DB_CONFIG = {
    "host": os.environ.get("PACE_DB_HOST"),
    "port": os.environ.get("PACE_DB_PORT", "5432"),
    "dbname": os.environ.get("PACE_DB_NAME"),
    "user": os.environ.get("PACE_DB_USER"),
    "password": os.environ.get("PACE_DB_PASSWORD"),
}

# A handful of real employee_ids spanning different current statuses,
# picked during item #75's live-production investigation (SESSION_HANDOFF
# .md). Override via argv for any other employee_id.
DEFAULT_SAMPLE_EMPLOYEE_IDS = [
    36014,  # Rudhi - Green, stable trend
    36173,  # Tanu Mehra - Green, stable trend
    32133,  # Hardik Khandelwal - Black, volatile/declining trend (known
            # edge case - see docstring above)
]


def _connect():
    missing = [k for k, v in DB_CONFIG.items() if not v]
    if missing:
        raise SystemExit(
            f"Missing DB config: {missing}. Set PACE_DB_HOST/PORT/NAME/USER/PASSWORD "
            "in the environment before running this script (see SESSION_HANDOFF.md)."
        )
    return psycopg2.connect(**DB_CONFIG)


def golden_validate(conn, employee_id):
    """Returns a dict with the stored column, two independent
    reconstructions (row-count window and calendar-day window, both using
    the CLEAN qualifying-row filter - no ps_worked_flag_day/visit_flag),
    and a third reconstruction using build_query()'s own DEFAULT filters
    (ps_worked_flag_day=1 AND visit_flag='No') over the calendar window,
    for direct comparison against what build_query()'s pace_score metric
    itself would return live."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Everything below is a read-only SELECT - wrap in a transaction
        # that is always rolled back, never committed (the DB user is not
        # read-only at the grant level, so this is a deliberate safety
        # habit, not a functional requirement for SELECTs).
        cur.execute("BEGIN")
        try:
            # 1. Stored column - read directly, single most recent row.
            cur.execute(
                """
                select last_60_days_new_pace_score_7_3 as stored_score,
                       new_pace_status_overall_last_60_days_7_3 as stored_status,
                       worked_day
                from public.pace_1
                where employee_id = %(employee_id)s
                order by worked_day desc
                limit 1
                """,
                {"employee_id": employee_id},
            )
            latest = cur.fetchone()

            # 2. Row-count window: last 60 qualifying rows (clean filter,
            # no ps/visit restriction), avg the 4 capped columns, apply the
            # centralized formula ONCE.
            cur.execute(
                f"""
                with last60 as (
                    select capped_engagement, capped_effectiveness,
                           capped_discipline, capped_working_hours, worked_day
                    from public.pace_1
                    where employee_id = %(employee_id)s and {PACE_SCORE_QUALIFYING_ROWS_SQL}
                    order by worked_day desc
                    limit 60
                ),
                agg as (
                    select avg(capped_engagement) as avg_e, avg(capped_effectiveness) as avg_ef,
                           avg(capped_discipline) as avg_d, avg(capped_working_hours) as avg_w,
                           count(*) as n, min(worked_day) as window_start, max(worked_day) as window_end
                    from last60
                )
                select {PACE_SCORE_FROM_AVGS_SQL} as recon_score,
                       {pace_status_sql(PACE_SCORE_FROM_AVGS_SQL)} as recon_status,
                       n, window_start, window_end
                from agg
                """,
                {"employee_id": employee_id},
            )
            row_window = cur.fetchone()

            # 3. Calendar window (today-59..today), clean filter (no
            # ps/visit restriction) - what employee_full_monthly_trend-
            # style functions would give if extended to an arbitrary
            # calendar range.
            cur.execute(
                f"""
                with in_range as (
                    select capped_engagement, capped_effectiveness,
                           capped_discipline, capped_working_hours
                    from public.pace_1
                    where employee_id = %(employee_id)s and {PACE_SCORE_QUALIFYING_ROWS_SQL}
                      and worked_day between current_date - interval '59 days' and current_date
                ),
                agg as (
                    select avg(capped_engagement) as avg_e, avg(capped_effectiveness) as avg_ef,
                           avg(capped_discipline) as avg_d, avg(capped_working_hours) as avg_w,
                           count(*) as n
                    from in_range
                )
                select {PACE_SCORE_FROM_AVGS_SQL} as recon_score,
                       {pace_status_sql(PACE_SCORE_FROM_AVGS_SQL)} as recon_status,
                       n
                from agg
                """,
                {"employee_id": employee_id},
            )
            calendar_window = cur.fetchone()

            # 4. Calendar window WITH build_query()'s own default filters
            # (ps_worked_flag_day=1 AND visit_flag='No') added, to compare
            # directly against what build_query(dimension="employee",
            # metrics=["pace_score"], name_filter=<name>) returns live with
            # no filter overrides.
            cur.execute(
                f"""
                with in_range as (
                    select capped_engagement, capped_effectiveness,
                           capped_discipline, capped_working_hours
                    from public.pace_1
                    where employee_id = %(employee_id)s and {PACE_SCORE_QUALIFYING_ROWS_SQL}
                      and worked_day between current_date - interval '59 days' and current_date
                      and ps_worked_flag_day = 1 and visit_flag = 'No'
                ),
                agg as (
                    select avg(capped_engagement) as avg_e, avg(capped_effectiveness) as avg_ef,
                           avg(capped_discipline) as avg_d, avg(capped_working_hours) as avg_w,
                           count(*) as n
                    from in_range
                )
                select {PACE_SCORE_FROM_AVGS_SQL} as recon_score,
                       {pace_status_sql(PACE_SCORE_FROM_AVGS_SQL)} as recon_status,
                       n
                from agg
                """,
                {"employee_id": employee_id},
            )
            calendar_window_bq_filtered = cur.fetchone()
        finally:
            cur.execute("ROLLBACK")

    return {
        "employee_id": employee_id,
        "stored": latest,
        "row_window_60": row_window,
        "calendar_window_60_clean": calendar_window,
        "calendar_window_60_build_query_default_filters": calendar_window_bq_filtered,
    }


def main():
    ids = [int(a) for a in sys.argv[1:]] or DEFAULT_SAMPLE_EMPLOYEE_IDS
    conn = _connect()
    try:
        for emp_id in ids:
            result = golden_validate(conn, emp_id)
            stored = result["stored"] or {}
            rw = result["row_window_60"] or {}
            cw = result["calendar_window_60_clean"] or {}
            cwf = result["calendar_window_60_build_query_default_filters"] or {}
            print(f"\n=== employee_id {emp_id} ===")
            print(f"  stored:                        score={stored.get('stored_score')} "
                  f"status={stored.get('stored_status')} (latest worked_day={stored.get('worked_day')})")
            print(f"  reconstructed (last 60 rows):  score={rw.get('recon_score')} "
                  f"status={rw.get('recon_status')} n={rw.get('n')} "
                  f"window=[{rw.get('window_start')}..{rw.get('window_end')}]")
            print(f"  reconstructed (60 cal days):   score={cw.get('recon_score')} "
                  f"status={cw.get('recon_status')} n={cw.get('n')}")
            print(f"  reconstructed (60 cal days,")
            print(f"    build_query default filters): score={cwf.get('recon_score')} "
                  f"status={cwf.get('recon_status')} n={cwf.get('n')}")
            match = stored.get("stored_score") is not None and (
                float(stored["stored_score"]) == float(rw.get("recon_score") or -1)
            )
            print(f"  EXACT MATCH (row-window vs stored): {match}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
