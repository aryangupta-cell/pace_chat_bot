import os
import psycopg2
import psycopg2.extras


def get_conn():
    return psycopg2.connect(
        host=os.environ["PACE_DB_HOST"],
        port=os.environ.get("PACE_DB_PORT", "5432"),
        dbname=os.environ["PACE_DB_NAME"],
        user=os.environ["PACE_DB_USER"],
        password=os.environ["PACE_DB_PASSWORD"],
    )


def run_query(sql, params=None):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {})
            return cur.fetchall()
    finally:
        conn.close()


def run_query_rollback_only(sql, params=None):
    """Executes `sql` in its own explicit transaction (autocommit OFF) and
    ALWAYS rolls back afterward — never commits — regardless of success or
    failure. Results are fetched and returned before the rollback, so a
    genuine read-only SELECT works normally; the only effect of this
    function vs. run_query() is that ANY write the query performed (e.g. an
    accidental INSERT/UPDATE/DELETE that slipped past an app-level
    SELECT-only check) is undone before it can persist.

    This is intentionally a SEPARATE function from run_query() rather than a
    flag on it: run_query() is used throughout the app for normal,
    already-tested queries and must keep committing as before. This function
    exists ONLY for the SQL-generation fallback path (app/sql_fallback.py),
    which executes LLM-generated SQL that has not been human-reviewed and
    therefore needs this extra transaction-level safety net. It is NOT a
    substitute for the existing app-level "reject non-SELECT" check in that
    module — it is an additional layer on top of it.
    """
    conn = get_conn()
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {})
            try:
                rows = cur.fetchall()
            except psycopg2.ProgrammingError:
                # e.g. the statement produced no result set at all
                rows = []
            return rows
    finally:
        try:
            conn.rollback()
        finally:
            conn.close()
