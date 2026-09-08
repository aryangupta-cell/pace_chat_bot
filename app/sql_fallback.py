"""SQL-generation fallback path (Part 3 of the GPT-5-mini migration task,
explicitly authorized by the user — previously this doc's own header said
"NOT wired up anywhere yet"; this module is what wires it up).

ONLY triggered from main.py when NEITHER the rule-based matcher
(intents.match_intent) NOR the LLM intent classifier (llm_nlu.classify) can
match a question to an existing queries.py function/intent (i.e. intent is
None). Existing queries.py functions remain the PRIMARY path for anything
they already cover — this module never overrides an already-matched intent.

SAFETY MODEL:
  - GPT-5 mini is given ONLY the condensed schema/rules reference doc
    (pace_chatbot_llm_reference.md) as context — never the raw ETL source,
    never table DDL beyond what that doc documents.
  - The generated SQL is executed via db.run_query_rollback_only(), which
    runs against the SAME DB (user aryangupta_ds) but in its own explicit
    transaction that is ALWAYS rolled back after fetching results — never
    committed. This is an additional safety layer scoped only to this
    fallback path: even if a write statement slipped past _is_safe_select
    below, it would be undone before it could persist. All other queries in
    this app still go through db.run_query(), which commits normally and is
    unchanged.
  - IMPORTANT, discovered during this round: aryangupta_ds is NOT actually a
    database-privilege-enforced read-only user — `has_database_privilege`
    checks and a live CREATE TABLE test (rolled back, nothing persisted)
    showed it holds INSERT/UPDATE/DELETE/TRUNCATE/REFERENCES/TRIGGER on 25
    public-schema tables and can CREATE new objects. There is NO database-
    level backstop. The `_is_safe_select` check below is therefore not
    "defense in depth" on top of a read-only grant — it is the ONLY real
    safety barrier for this path, and must never be bypassed or weakened.
  - `_is_safe_select` rejects anything that is not a single, standalone
    SELECT statement: multiple statements (any semicolon before the very
    end), any DML/DDL keyword (INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/
    CREATE/GRANT/REVOKE/EXECUTE/CALL/COPY/VACUUM/SET/RESET/etc.), and
    anything that isn't SELECT/WITH as the leading statement, case-
    insensitively.
  - Every fallback answer is clearly labeled AI-generated/unverified in the
    reply text (visually distinct from tested-function responses).
  - Every fallback attempt (accepted or rejected) is logged via
    usage_log.log_fallback_query for later review/promotion into real
    queries.py functions.
"""

import json
import logging
import os
import re
import time

from . import db
from .usage_log import log_usage, log_fallback_query

logger = logging.getLogger("pace_chatbot.sql_fallback")

OPENAI_MODEL = "gpt-5-mini"
_TIMEOUT_SECONDS = 30.0
_MAX_ROWS = 200

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFERENCE_DOC_PATH = os.path.join(_REPO_ROOT, "pace_chatbot_llm_reference.md")

UNVERIFIED_LABEL = (
    "\n\n_This answer was generated dynamically by AI and hasn't been manually "
    "verified — treat with appropriate caution. If it looks wrong, please "
    "flag it so it can be checked._"
)

_DISALLOWED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"EXECUTE|CALL|COPY|VACUUM|SET|RESET|MERGE|REPLACE|LOCK|LISTEN|NOTIFY|"
    r"DO|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|INTO)\b",
    re.IGNORECASE,
)


def _load_reference_doc():
    try:
        with open(_REFERENCE_DOC_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        logger.exception("Could not load %s", _REFERENCE_DOC_PATH)
        return ""


_REFERENCE_DOC = _load_reference_doc()

_STABLE_INSTRUCTIONS = f"""You are a careful, read-only PostgreSQL analyst for an internal HR/
attendance analytics system called PACE. A user has asked a question that no
existing pre-built function in this app can answer, so you must write a NEW
SQL query yourself, using ONLY the schema and rules documented below.

HARD RULES (violating these makes your answer unusable — follow them exactly):
1. Output EXACTLY ONE PostgreSQL statement, and it MUST be a single SELECT
   (a WITH ... SELECT CTE is fine). No semicolons except optionally one at
   the very end. No INSERT/UPDATE/DELETE/DROP/ALTER/CREATE or any other
   write/DDL statement, ever, under any circumstance.
2. Only reference tables/views/columns that are documented below
   (public.pace_chatbot_view, and public.pace_1 for the extra columns
   listed). Never invent a column or table name.
3. Follow the aggregation rules in the reference doc exactly (e.g. the
   capped-average-before-formula rule for PACE score/sub-score period
   aggregates) — do not average an already-computed daily score across days.
4. Always add a LIMIT clause (200 or fewer rows) unless the question is
   obviously a single-row aggregate (a count, an average, etc.).
5. If the question cannot be answered with a safe, single read-only SELECT
   against the documented schema, respond with SQL: null and explain why in
   `explanation` instead of guessing at an unsafe or unanswerable query.

Respond with JSON: {{"sql": "<the SELECT statement, or null>", "explanation": "<one sentence, plain English, no SQL jargon, describing what the query does or why it can't be answered>"}}

=== SCHEMA + RULES REFERENCE DOC ===
{_REFERENCE_DOC}
=== END REFERENCE DOC ===
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": ["string", "null"]},
        "explanation": {"type": "string"},
    },
    "required": ["sql", "explanation"],
    "additionalProperties": False,
}


def _get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
        return openai.OpenAI(api_key=api_key)
    except Exception:
        logger.exception("Failed to construct OpenAI client for sql_fallback")
        return None


def _is_safe_select(sql):
    """Returns (ok: bool, reason: str|None). The ONLY real safety barrier on
    this path — see module docstring. Rejects anything that is not a single
    standalone SELECT/WITH statement, case-insensitively."""
    if not sql or not sql.strip():
        return False, "empty SQL"
    s = sql.strip()
    # strip a single trailing semicolon (and any trailing whitespace after it)
    s_body = s[:-1].strip() if s.endswith(";") else s
    if ";" in s_body:
        return False, "multiple statements (semicolon found before the end)"
    if not re.match(r"^\s*(WITH|SELECT)\b", s_body, re.IGNORECASE):
        return False, "does not start with SELECT or WITH"
    m = _DISALLOWED_KEYWORDS.search(s_body)
    if m:
        return False, f"disallowed keyword found: {m.group(0).upper()}"
    return True, None


def generate_sql(question, timeout=_TIMEOUT_SECONDS):
    """Calls GPT-5 mini to draft a read-only SQL query for `question`.
    Returns (sql_or_none, explanation) — sql_or_none is None if the model
    declined or the call failed."""
    client = _get_openai_client()
    if client is None:
        return None, "LLM unavailable"

    t0 = time.time()
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            # STABLE first: system rules + full reference doc (identical on
            # every call) so OpenAI's automatic prompt caching can hit.
            instructions=_STABLE_INSTRUCTIONS,
            # VARIABLE last: only this question changes call to call.
            input=f"User question: \"{question}\"\n\nRespond with the JSON described above.",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pace_sql_fallback",
                    "schema": _RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
            reasoning={"effort": "low"},
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("sql_fallback OpenAI call failed: %s", e)
        return None, f"LLM call failed: {e}"

    latency = time.time() - t0
    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    itd = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(itd, "cached_tokens", None) if itd is not None else None
    log_usage(
        path="sql_fallback", provider="openai", model=getattr(resp, "model", OPENAI_MODEL),
        input_tokens=input_tokens, output_tokens=output_tokens,
        cached_tokens=cached_tokens, latency=latency,
    )
    logger.info(
        "sql_fallback generate: %.2fs input_tokens=%s output_tokens=%s cached_tokens=%s",
        latency, input_tokens, output_tokens, cached_tokens,
    )

    try:
        data = json.loads(resp.output_text)
    except Exception:
        return None, "LLM returned unparseable output"

    return data.get("sql"), data.get("explanation", "")


def answer(question):
    """Full fallback pipeline: generate SQL -> safety-check -> execute ->
    format a labeled response. Returns a dict {reply, rows} or None if the
    fallback genuinely could not produce anything (caller should show the
    normal FALLBACK_MESSAGE in that case)."""
    sql, explanation = generate_sql(question)

    if not sql:
        log_fallback_query(question, sql, accepted=False, reject_reason=explanation or "model declined")
        return None

    ok, reason = _is_safe_select(sql)
    if not ok:
        logger.warning("sql_fallback REJECTED unsafe SQL (%s): %r", reason, sql)
        log_fallback_query(question, sql, accepted=False, reject_reason=reason)
        return None

    # Belt-and-suspenders row cap even if the model forgot LIMIT.
    exec_sql = sql
    if not re.search(r"\bLIMIT\s+\d+\b", exec_sql, re.IGNORECASE):
        exec_sql = exec_sql.rstrip().rstrip(";") + f" LIMIT {_MAX_ROWS}"

    try:
        rows = db.run_query_rollback_only(exec_sql)
    except Exception as e:
        logger.warning("sql_fallback query execution failed: %s | sql=%r", e, exec_sql)
        log_fallback_query(question, sql, accepted=True, reject_reason=None, error=str(e))
        return None

    log_fallback_query(question, sql, accepted=True, row_count=len(rows))

    if not rows:
        reply = f"I generated a query for this, but it returned no matching data.{UNVERIFIED_LABEL}"
        return {"reply": reply, "rows": []}

    reply_lines = [explanation or "Here's what I found:", ""]
    if len(rows) == 1 and len(rows[0]) <= 4:
        row = rows[0]
        reply_lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
    else:
        cols = list(rows[0].keys())
        reply_lines.append(" | ".join(cols))
        for r in rows[: min(len(rows), _MAX_ROWS)]:
            reply_lines.append(" | ".join(str(r[c]) for c in cols))

    reply = "\n".join(reply_lines) + UNVERIFIED_LABEL
    return {"reply": reply, "rows": rows}
