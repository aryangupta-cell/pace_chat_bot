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
