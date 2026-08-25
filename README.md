# PACE Chatbot — Phase 1 (local prototype)

Rule-based (keyword-matching, no LLM) chatbot for 3 question types, reading
from `public.pace_chatbot_view` (Postgres, read-only at runtime).

## Supported questions (Phase 1)

1. Best attendance — e.g. "who has the best attendance in Accounts this month"
2. Worst attendance — e.g. "who has the worst attendance in Control Tower in July"
3. Most/least productive time — e.g. "who is most productive in IT-Development this month"

Department and month are both optional. If a department mention is ambiguous
(matches more than one real department, e.g. "Sales" matching 5 different
Sales-* departments), the bot lists the options and asks you to pick one
rather than guessing. Same behavior for manager-name mentions that match more
than one manager. Anything that doesn't match one of the 3 intents gets a
fixed fallback message listing what's supported.

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set the DB connection as environment variables (read-only DB user):
   ```bash
   export PACE_DB_HOST="elginternal.cezhpbp2czml.ap-south-1.rds.amazonaws.com"
   export PACE_DB_PORT="5432"
   export PACE_DB_NAME="elginternal"
   export PACE_DB_USER="aryangupta_ds"
   export PACE_DB_PASSWORD="<the DB password>"
   ```
   (On Windows PowerShell use `$env:PACE_DB_HOST = "..."` etc. instead.)

3. Start the server:
   ```bash
   python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8010
   ```

4. Open http://127.0.0.1:8010 in a browser and start chatting, or hit the
   API directly:
   ```bash
   curl -s -X POST http://127.0.0.1:8010/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"who has the worst attendance in Control Tower this month"}'
   ```

## Notes / known limitations (v1)

- No authentication — single shared access, as intended for this prototype stage.
- "Who is improving/declining" is **not implemented** — the investigation
  found `overall_pace_score` doesn't currently vary month-to-month in the
  underlying data, so `pace_score_delta` is 0 for every row. Revisit once
  that upstream data issue is resolved.
- Manager/"my team" filtering resolves `reporting_manager_name` → the
  underlying `reporting_user_id` and queries by ID, per the plan — it currently
  only supports the best/worst attendance intents by manager, not productive-time.
- Entity caches (department list, manager list) are loaded once per process
  via `lru_cache` — restart the server if the underlying view's distinct
  values change (e.g. a new department is added).
