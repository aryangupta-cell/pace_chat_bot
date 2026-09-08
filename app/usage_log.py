"""Append-only, DB-independent usage/cost logging (Part 4 of the GPT-5-mini
migration). Deliberately NOT a database table — the task explicitly
authorized only a plain log file, no new DB objects.

Writes one JSON line per LLM call to LOG_DIR/llm_usage.log, tagged by `path`
("classify" for the intent-classification path, "sql_fallback" for the
SQL-generation fallback path) and `provider` ("openai"/"gemini"), so real
per-request token counts accumulate from actual traffic and can be reviewed/
aggregated later instead of relying on estimates.

LOG_DIR defaults to a local `logs/` directory next to the app package;
overridable via the LLM_LOG_DIR env var (useful on hosts with a different
writable path, e.g. Render's ephemeral disk still allows local writes, they
just don't persist across deploys — acceptable for this purpose per the
task's own log-file authorization).
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger("pace_chatbot.usage_log")

_LOG_DIR = os.environ.get("LLM_LOG_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_USAGE_LOG_PATH = os.path.join(_LOG_DIR, "llm_usage.log")
_FALLBACK_LOG_PATH = os.path.join(_LOG_DIR, "sql_fallback.log")

_lock = threading.Lock()


def _ensure_dir():
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        return True
    except Exception:
        logger.exception("Could not create log dir %s", _LOG_DIR)
        return False


def _append_json_line(path, record):
    if not _ensure_dir():
        return
    line = json.dumps(record, default=str)
    try:
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        logger.exception("Failed to write log line to %s", path)


def log_usage(path, provider, model, input_tokens, output_tokens, cached_tokens=None, latency=None):
    """path: "classify" | "sql_fallback" — tags which pipeline stage made the call."""
    record = {
        "ts": time.time(),
        "path": path,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "latency_s": round(latency, 3) if latency is not None else None,
    }
    _append_json_line(_USAGE_LOG_PATH, record)


def log_fallback_query(question, generated_sql, accepted, reject_reason=None, row_count=None, error=None):
    """Logs every fallback-path attempt (accepted or rejected) so these can
    be reviewed and potentially promoted into real queries.py functions
    later, per the task's explicit requirement."""
    record = {
        "ts": time.time(),
        "question": question,
        "generated_sql": generated_sql,
        "accepted": accepted,
        "reject_reason": reject_reason,
        "row_count": row_count,
        "error": error,
    }
    _append_json_line(_FALLBACK_LOG_PATH, record)
