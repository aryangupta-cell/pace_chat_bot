# PROJECT_HANDOFF.md — PACE Chatbot

## QUICK START

- **PROJECT**: PACE Chatbot — a natural-language chat interface over employee attendance/productivity/"PACE score" performance data, sitting alongside a Looker Studio dashboard. FastAPI + PostgreSQL backend, rule-based + LLM-assisted NLU, deployed on Render.
- **CURRENT STATUS**: "Change 2" (the generalized semantic-extraction / composable query layer, items #70-73) is implemented, extended through several rounds (items #74-90), validated against a full 50+20 question matrix (item #91), and given a final hardening pass (item #92). **Verdict as of the latest commit: READY WITH KNOWN LIMITATIONS** (item #92's own stated verdict, carried forward — not upgraded or downgraded by this document).
- **LATEST COMMIT**: `bbaaf45` — "Item #92 (final docs): WFH filter fix, sticky-context leak fix, cosmetic fix, full regression pass, architectural review, and final verdict". Branch `master`, synced to `origin/master` and `origin/main` (repo convention: every round pushes `git push origin master:main` to keep both in sync). Working tree clean except one pre-existing, intentionally-untracked file, `PROJECT_BACKUP_2026-09-09.md` (an earlier project snapshot doc, not part of any round's diff).
- **READ FIRST**: this file, then `SESSION_HANDOFF.md` in full (2078 lines — it is the chronological, item-numbered ground truth for everything described here; items #52 through #92 cover this project's most recent and most relevant history, and items #70-92 specifically are "Change 2" and its hardening). Then read `app/main.py`, `app/queries.py`, `app/intents.py`, `app/llm_nlu.py`, `app/entities.py`, `app/session_store.py` directly — SESSION_HANDOFF.md itself warns it "may drift out of date" relative to the real code.
- **CURRENT ARCHITECTURE (one paragraph)**: a user message first passes through spellcheck, then a cascade: (1) a handful of narrow, deterministic pre-classification interceptors (filter-meta follow-ups, bare-direction follow-ups, driving-performance/both-ends-ranking ambiguity guards); (2) the ~123-intent rule-based regex matcher (`intents.match_intent()`), which wins unconditionally whenever it returns a non-`None` intent (the "precedence flip", item #52); (3) if that finds nothing, an LLM intent classifier (`llm_nlu.classify()`, GPT-5-mini by default, Gemini as a swappable alternate provider); (4) if that also finds nothing usable, a second, separate LLM call (`llm_nlu.extract_build_query()`) extracts a structured `{dimension, metrics, filters, period, limit, operation, ...}` object, validated and re-resolved deterministically against real entities, then executed via one general parametrized SQL engine, `queries.build_query()`; (5) if even that fails, a last-resort free-form AI-generated-SQL fallback (`sql_fallback.py`, SELECT-only, rollback-only, always labeled "AI-generated/unverified"). Conversational state is tracked across turns via two additive mechanisms: `sticky_context` (single-slot department/employee/period, whole-session-sticky) and the newer `query_context` (structured last-operation/last-dimension/last-result-ids state, added in item #84 specifically so ranking answers produced by the new cascade leave behind enough state for pronoun follow-ups like "their weakest area" to resolve correctly).
- **KNOWN LIMITATIONS (brief — full detail in section 13 below)**:
  - Company-wide aggregate questions with no named employee/department mostly misroute (5 of 7 failed in item #91's Category E).
  - WFH/work-mode filters were dropped silently in the *rule-based* ranking path for one shape (`metric_ranking()`'s "latest N qualifying rows" mode) — fixed for the main case in item #92, but the qualifying-row-count interaction remains unfixed.
  - No general multi-metric or multi-clause query composition beyond the one hand-built "driving performance" 2-step case (item #86 Decision 1) — asking for 2+ metrics at once silently collapses to one.
  - `sql_fallback.py`'s raw-SQL path can leak internal column names/mechanism description into its (clearly labeled) replies.
  - `metric_ranking()` vs `build_query()` tie-order differs on ties (no shared secondary sort key).
  - Several narrow, single-instance routing/phrasing misses (see section 13).
- **NEXT TASK**: none is mandated by this document — the next session should pick from section 16 based on what the user actually asks for, starting with the WFH-filter-vs-qualifying-row-count interaction or the company-wide aggregate routing gap if given a free choice, since item #91 flagged the latter as the single biggest, most consistent gap.
- **IMPORTANT**: fix forward only. Do not revert or rewrite items #52-92's working architecture. Do not add phrase-specific/hardcoded patches — extend the existing generalized/deterministic mechanisms (`_detect_build_query_filters()`, `PACE_SCORE_AGG_SQL`, `query_context`, etc.) the way every prior round did. Verify against live production or real data, never assume a fix works from code review alone (this codebase has a documented history of "diagnosed wrong, shipped a fix that didn't work" — item #89's first attempt — precisely because it wasn't live-retested before being reported done).

---

## 1. Project Overview

**What it does**: PACE Chatbot answers natural-language questions about employee attendance, productivity, and "PACE score" performance (an internal composite performance metric), for a company whose data otherwise lives in a Looker Studio dashboard. It is meant to sit alongside that dashboard (there is a `/dashboard` route embedding the real Looker Studio iframe with a floating chat overlay) and answer ad hoc questions a fixed dashboard can't — "who is improving?", "which department has the lowest PACE score?", "what is Employee X's weakest area?", multi-turn follow-ups like "what about their weakest area?", etc.

**Tech stack**: Python, FastAPI (backend + static file serving), vanilla HTML/JS frontend (`static/index.html`, `static/dashboard.html`, `static/chat-widget.js` — shared chat logic used by both pages), PostgreSQL 15 (AWS RDS, read-write-grants-but-treated-as-read-only), `psycopg2` for DB access, `rapidfuzz` for word-boundary-safe fuzzy name matching, `pyspellchecker` for offline typo correction, and two swappable LLM providers (OpenAI GPT-5-mini, the current default; Google Gemini Flash-Lite as an alternate) via a provider-abstraction layer.

**Repo / app structure** (repo root, `C:\Users\user\Claude Dashboard\pace chatbot`):
- `app/main.py` (~6,300 lines) — `handle_message()`/`answer_intent()`, the central orchestrator and ~123-intent dispatcher, plus every reply-formatting function, every pre-classification interceptor, and `_extraction_llm_reply()` (the "Change 2" cascade step).
- `app/queries.py` (~3,300 lines) — every SQL query function, including the generalized `build_query()` engine, the centralized PACE-score-formula SQL constants, and dozens of older, hand-built, still-in-use query functions.
- `app/intents.py` (~1,400 lines) — `match_intent(text)`: an ordered list of `(intent_name, [regex patterns])` tuples (~123 unique intent names as of item #69's own audit) plus a fuzzy fallback (`_fuzzy_match_intent()`).
- `app/llm_nlu.py` (~1,000 lines) — `classify()` (intent-name classification) and `extract_build_query()` (structured semantic extraction for `build_query()`), both provider-abstracted (OpenAI/Gemini), both with hard-coded prompt-caching structure and a strict try/except safety contract (any LLM failure returns `None`/falls through, never raises).
- `app/entities.py` (~940 lines) — all NL entity extraction: `extract_employee()`, `extract_department()`, `extract_manager()`, `extract_month()`/`extract_months()`, `extract_date_range()`, `extract_two_dates()`, `extract_two_months()`, `extract_single_date()`, `extract_limit()`, `split_comparison()`, `last_4_weeks_periods()`, etc.
- `app/session_store.py` (~326 lines) — in-memory, per-session-id conversational state: `sticky_context` (single-slot dept/employee/period), `last_list` (last list/ranking answer, for vague "list them" follow-ups), `query_context` (structured last-operation/last-dimension/last-result-ids state, item #84), `comparison_entities` (2-slot entity tracker for "compare them" follow-ups, item #84).
- `app/team.py` — "my team"/named-manager team resolution via `pace_1.email_access` (exact-match-only, deliberately not fuzzy).
- `app/sql_fallback.py` — the last-resort free-form AI-generated-SQL path (SELECT-only regex guard, rollback-only DB connection, always-labeled-unverified replies).
- `app/db.py` — `get_conn()`/`run_query()`; `run_query_rollback_only()` used only by `sql_fallback.py`.
- `app/spellcheck.py`, `app/usage_log.py` — offline typo correction, LLM usage logging.
- `SESSION_HANDOFF.md` — the full chronological project history (items #1-92+), the primary source of truth this document is built from.
- `PROJECT_BACKUP_2026-09-09.md` — an earlier snapshot doc, present as an intentionally-untracked file in the working tree; useful as a secondary cross-reference (its §7 column-mapping table is reproduced/adapted in section 6 below), but SESSION_HANDOFF.md items #70+ supersede it for anything after that point.
- `scripts/golden_validate_pace_score.py` — a golden-validation harness (item #75) for cross-checking `build_query()`'s live-recomputed PACE score formula against the stored `last_60_days_new_pace_score_7_3` column; not runnable without DB credentials in a given sandbox, but ready for a session that has them.

**FastAPI/backend architecture**: a single FastAPI app (`app/main.py`) with a `POST /api/chat` endpoint (`{message, session_id}` → `{reply, rows, needs_clarification, clarification_options}`), `GET /api/health`, and static routes for `/` and `/dashboard` (both now serve the same dashboard+chat-overlay page, per item #37). A global `@app.exception_handler(Exception)` (item #13/#16) guarantees any unhandled exception still returns valid JSON, never a raw error page.

**PostgreSQL's role**: all real data lives in `public.pace_1` (a VIEW over the true underlying ETL table `pace_phase_1_table`, populated by an external Python ETL pipeline not visible to this app) and `public.pace_chatbot_view` (a purpose-built VIEW, the ONLY database object this project has ever written to, always via explicit case-by-case authorization — currently on its 4th authorized iteration). Grain: `pace_1` is session-level for days with both Standard and Overtime punches; `pace_chatbot_view` and most query functions restrict to `shift_type='Standard'` to get a clean 1-row-per-employee-day grain, at the cost of undercounting OT-mixed days and dropping the 2.36% of employee-days that are OT-only. Several newer functions (including `build_query()`) query `pace_1` directly instead of the view, because they need columns (`ps_worked_flag_day`, `visit_flag`, `capped_*`) the view does not expose.

**Safe deployment info**: production URL `https://pace-chat-bot.onrender.com` (chat) and `https://pace-chat-bot.onrender.com/dashboard` (identical page, second URL kept for compatibility). GitHub repo: `aryangupta-cell/pace_chat_bot`. Render auto-redeploys on push to the connected branch. **No secrets are stored in this document or in SESSION_HANDOFF.md's current state** — DB and API credentials are referenced only as "see local secrets, not committed" / environment variable names (`PACE_DB_HOST`, `PACE_DB_PORT`, `PACE_DB_NAME`, `PACE_DB_USER`, `PACE_DB_PASSWORD`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `LLM_PROVIDER`). Note: SESSION_HANDOFF.md itself documents (item #32/#33) that the real DB password was committed in plaintext in this file's own early git history (3 commits, before redaction) and has **not** been remediated (rotation recommended, not done) — this is a known, previously-flagged, still-open item, not new information.

**Key entry points**: `app/main.py::handle_message(message, session_id)` is the true entry point every request flows through. `python -m uvicorn app.main:app --host 127.0.0.1 --port 8010` runs it locally (needs the `PACE_DB_*` env vars set).

**How a natural-language question flows through the system, end to end** (current architecture, post-item-#92):
1. `handle_message()` resumes any pending `awaiting_identity`/`awaiting_admin_confirmation`/`awaiting_weekly_breakdown` session state first.
2. Spellcheck correction runs (`raw_message` is saved before correction, for the `fallback_text=` retry pattern used throughout `entities.py`).
3. A handful of narrow, deterministic **pre-classification interceptors** run, in a fixed order, each falling through cleanly (not guessing) when its specific shape doesn't match: filter/methodology meta-follow-ups ("did you exclude X in this?", item #61), bare-direction ranking follow-ups ("least"/"most" alone, item #63), list-pronoun-metric follow-ups ("there"/"their"/"them" after a list, item #67), the driving-performance-ambiguity guard and both-ends-ranking-ambiguity guard (item #84).
4. `rule_intent = intents.match_intent(message)` — the ~123-entry regex list, then a fuzzy fallback. A `_NEW_VOCAB_OVERRIDE_PATTERN`/`_FUZZY_NEW_VOCAB_DISQUALIFY_PATTERN` pair (item #72) can null this back to `None` for specific new-vocabulary phrase shapes that would otherwise be falsely intercepted by an old intent.
5. If `rule_intent` is non-`None`, it **wins unconditionally** (the item #52 "precedence flip") — the LLM classifier is never even consulted.
6. If `rule_intent is None`, `llm_nlu.classify()` (GPT-5-mini or Gemini) is tried.
7. If that also returns nothing usable, `llm_nlu.extract_build_query()` is tried — a **separate** structured-extraction LLM call (not an intent name) whose output is validated field-by-field against real, live data structures (`queries.BUILD_QUERY_DIMENSIONS`, `queries.BUILD_QUERY_METRICS`, real entity-resolution functions) before ever reaching `queries.build_query()`. This is `app/main.py::_extraction_llm_reply()`.
8. If that also fails, `sql_fallback.answer()` (free-form AI-generated SQL, SELECT-only, rollback-only, always labeled unverified) is tried.
9. If even that fails, a final bottom-of-cascade bare employee/department/RM-name lookup via `build_query_overview_reply()` runs, and if nothing resolves, the generic `FALLBACK_MESSAGE` (item #87's updated, accurate capability list) is returned.
10. Whatever path answered, session state (`sticky_context`, `last_list`, `query_context`, `comparison_entities` as applicable) is updated so the next turn's follow-up can resolve against it.

---

## 2. Original Requirement ("Change 2")

**Why Change 2 was needed**: by item #69/#70 (roughly the midpoint of this project's later history), the chatbot had grown to ~123 hand-built, hand-triggered intents (see `app/intents.py`'s `_INTENTS` list) plus a general-purpose fallback engine, `build_query()` (introduced earlier, item #57), that could already answer many {dimension} × {metric} × {filter} × {period} combinations — but only for phrasings that either matched an existing regex or that a bare employee/department/RM-name fallback could resolve. Real CEO-style testing kept surfacing new phrasings and new metric/filter/period combinations (`capped_*` internal ingredients, day-level vs. period-aggregate PACE score, precomputed vs. live-recomputed department scores, arbitrary periods, new filter combinations) that the *old fixed-intent-only approach* could not generalize to — every new phrasing needed its own hand-written regex, and every new metric/filter/period shape needed its own hand-written query function. This was explicitly identified as unsustainable (item #70's own audit table lists 6 concrete metric gaps found this way).

**The problems that existed before Change 2**:
- **Fixed/rule-based intents alone** could express only phrasings someone had already thought to write a regex for — a combinatorially large space (3 dimensions × ~8+ metrics × several filter combinations × arbitrary periods = "thousands of theoretical combinations, of which maybe 50-80 concrete ones are hand-covered today", item #56's own estimate) that could never be fully covered one regex at a time.
- **No generalized semantic extraction** existed — there was no mechanism that could take an arbitrary new phrasing ("show me the internal capped effectiveness ingredient value company wide for the last 10 days") and turn it into a structured query without someone first writing a matching intent.
- **Query construction** was scattered across ~87 individual, purpose-built query functions in `queries.py`, each hand-coding its own dimension/metric/filter/period logic — real correctness bugs (the Jensen's-inequality averaging bug, items #55/#59/#60/#75) had to be found and fixed independently in each one, because there was no single shared formula.
- **Conversational state** (item #83's own audit) was single-slot (`sticky_context`) — a ranking answer left behind no memory of *which* employees/departments were just shown, so "their weakest area" or "the employee with the lowest X" right after a ranking had nothing correct to resolve against.
- **Fallback behavior** before Change 2 was either a generic "I don't understand" message or the free-form `sql_fallback` SQL-generation path (which works, but is explicitly the *least* trustworthy path — unverified, labeled, last-resort by design) — there was no reliable, deterministic middle tier between "matches an existing intent" and "let an LLM write raw SQL".

**What Change 2 built** (items #70-73, the core implementation; #74-90 the extension/hardening rounds): a new cascade step, `llm_nlu.extract_build_query()` + `app/main.py::_extraction_llm_reply()`, that performs **generalized semantic extraction** — a single LLM call turns an arbitrary new phrasing into a structured `{dimension, dimension_name, metrics, filters, period_phrase, limit, operation, ...}` object, which is then deterministically validated (every field checked against real, live data — `BUILD_QUERY_DIMENSIONS.keys()`, `BUILD_QUERY_METRICS.keys()`, real entity-resolution via `entities.py`, real date-phrase re-parsing) before being handed to `queries.build_query()`, the one general **query construction** engine that already existed but was extended (item #70/#71) to cover the metric gaps found. This is architecturally distinct from, and strictly lower-priority than, the ~123 existing fixed/rule-based intents — none of them were replaced or rewritten; the new cascade step only ever engages when both the rule-based matcher and the intent-name LLM classifier find nothing.

---

## 3. Change History (chronological)

This section summarizes the major changes; SESSION_HANDOFF.md's own item numbers are given so a fresh session can find the full detail (live test transcripts, exact commit diffs, judgment calls) for any of these.

**Change 1** (pre-"Change 2" era, items #1-51, briefly): the original rule-based-only chatbot build (~300+ hand-covered question types by item #51), a deliberate later reversal to add an LLM-first NLU layer (Gemini, then GPT-5-mini, item #28/#44), deployment to Render (items #32-34), and a long series of bug-fix rounds (sticky context, pronoun resolution, fallback_text/spellcheck-corruption bug class, the Jensen's-inequality PACE-score-averaging bug found and fixed across multiple functions, WFH/leave/visit filter bugs, etc.). By item #51's investigation, the codebase had ~123 intents and one general fallback engine (`build_query()`, item #57) but no generalized semantic-extraction layer.

**Change 2 — core implementation**:
- **Item #70** (investigation only): audited `build_query()` for metric gaps (found 6: `pace_status`, `pace_score_day_level`, raw `capped_*` ingredients, a nonexistent `dept_status_60_days` column, `dept_score_60_days_precomputed`, and confirmed `LC`/`EL`/`DH` counts already worked) and designed the new extraction-LLM cascade step.
- **Item #71** (implementation): filled all 6 metric gaps in `build_query()`/`BUILD_QUERY_METRICS`, built `llm_nlu.extract_build_query()` and `app/main.py::_extraction_llm_reply()`, wired the new cascade step in between `classify()` and `sql_fallback.answer()`. Commit `08f1a97`. Live-tested successfully for several new-vocabulary phrasings, but flagged (not fixed) that old-intent shadowing sometimes prevented the new step from being reached at all.
- **Item #72** (bug fix): root-caused *why* item #71's two live failures happened — three independent upstream layers (explicit regex, fuzzy-fallback, `classify()`'s own closed-vocabulary guess) were all capable of intercepting new-vocabulary questions before the new cascade step ever ran. Fixed with 3 separate, layer-specific mechanisms (`_NEW_VOCAB_OVERRIDE_PATTERN`, `_FUZZY_NEW_VOCAB_DISQUALIFY_PATTERN`, `classify()` prompt guardrails) plus a deterministic empty-metrics repair (`_repair_new_vocab_metric()`). Commits `df0ccdb`→`feb2e3b` (5 commits).
- **Item #73** (business rule): implemented the capped-vs-percentage business rule (bare "X" → `X_pct`; "capped X" → `capped_X`; "capped X %" → `X_pct`, deliberately non-obvious) as one shared normalizer (`_detect_pct_capped_metrics()`) called from 4 places, plus fixed `dept_best`/`dept_worst`'s hardcoded `pace_score`-only metric. Commit `3eb3c0f`.

**Change 2 — extension/hardening rounds**:
- **Item #74** (audit): Phase 1 audit confirming the real PACE score formula and its correct arbitrary-period recomputation method (capped-average-first, then apply the formula once — see section 5), a full 60-row terminology-CSV cross-check, and a concrete Phase 2/3 plan (centralize the formula, build a golden-validation harness, extend `build_query()` rather than build a parallel engine).
- **Item #75** (refactor): centralized the PACE score formula/status-banding SQL into `PACE_SCORE_AGG_SQL`/`PACE_SCORE_FROM_AVGS_SQL`/`pace_status_sql()` (11 call sites repointed, pure refactor, byte-identical SQL). Built `scripts/golden_validate_pace_score.py`. Live golden-validation against 5 real employees found `build_query()`'s `pace_score` metric is **not** a reliable proxy for the stored 60-day column for employees with meaningful PS-off/visit days (a real, flagged-not-fixed gap). Commit `6276f14`.
- **Item #76** (implementation, ran in parallel with #77's investigation): "Phase 3" — arbitrary-period PACE score/status support (`latest_n_days`/`latest_n_days_offset` qualifying-row mechanism) and composable semantic extraction hardening. Commits `09f8d90`/`8f8e199`/`dafbf48` and follow-ups.
- **Item #77** (investigation only): read-only re-pass confirming the exact nature of 3 remaining metric gaps (`engagement_minutes` unwired entirely; `meeting_in_min` has an existing formula just not in `BUILD_QUERY_METRICS`; `tasks_created`/`tasks_assigned` genuinely ambiguous between 2+ interpretations) plus 2 live, confirmed "average X" collision bugs. Produced a ready-to-execute Phase 4 plan (not yet confirmed executed in this document's read of SESSION_HANDOFF.md — check `app/queries.py`'s `BUILD_QUERY_METRICS` directly for `engagement_minutes`/`meeting_minutes` keys to confirm current status).
- **Item #78** (implementation): Part A arbitrary-period PACE score/status; Part B composable semantic extraction layer hardening.
- **Items #79/#80**: wired `engagement_minutes`/`meeting_in_min`/`tasks`/`todos` into `BUILD_QUERY_METRICS`, fixed several redirect/routing bugs (`meeting_min_ranking` misrouting), added `meeting_count`/`meeting_had_emp`.
- **Item #81**: final-phase comprehensive test pass + one bug fix (raw `employee_id` leaking as a label) + code review + project closeout for items #70-80.
- **Item #82**: fixed the `emp_engagement`/`emp_discipline`/`emp_effectiveness`/`emp_working_pct` vs `average_metric` collision (root-caused earlier, finally fixed here) + first real DB-backed golden validation with temporary credentials.
- **Item #83** (Phase 1 audit only): root-caused **15 new CEO-testing failures (A-O)** by reading `extract_build_query()`/`_extraction_llm_reply()`/`session_store.py` in full. Found the extraction schema had **no `limit` field**, **no explicit singular-vs-ranking/"both ends" field**, **no multi-clause/multi-step representation at all**, and that conversational state was single-slot and the ranking branch never wrote to it. Produced a concrete Phase 2 design (`operation`/`limit`/`output_grain` schema fields, a minimal 2-step `secondary` composition mechanism, a structured `query_context` object) and flagged 2 genuine business-decision ambiguities for the user rather than guessing ("which employees are driving performance" — 3 readings; "highest AND lowest" row-count semantics).
- **Item #84** (Phase 2 implementation): broadened `entities.extract_limit()` (now handles "5 lowest" / "lowest 5" / "give me 5 ..." not just "top/bottom N"), extended the extraction schema with `limit`/`operation` fields, fixed 3 real old-intent-shadowing bugs (`subscore_trend_emp`, `full_trend_emp` fuzzy-match, an `_area_match` plural gap), built the new `query_context`/`comparison_entities` conversational-state mechanisms, and — since the 2 business-decision ambiguities from #83 were still unresolved — wired them to **fail safely with an explicit controlled clarification** rather than silently guessing. 10 commits (`8119997`→`94a7c01`).
- **Item #85** (correction): #84's `full_trend_emp`→`score_drop_ranking` redirect landed on the wrong destination (still hit the "my team" identity gate for a clearly company-wide plural question). Fixed with a new sibling intent, `pace_delta_ranking_cw`, that bypasses that gate. Commits `5026e8f`/`a282466`/`46ee958`.
- **Item #86** (business decisions resolved + K/L/M/N/O chain): the user resolved both #83 ambiguities explicitly — "driving performance" = top individual PACE scorers WITHIN the department (not month-over-month improvement, not deviation-from-average); "highest AND lowest" = a new `rank_both_ends` operation, exactly the top-1 and bottom-1 of the SAME filtered population. Implemented both, plus fully debugged and fixed the full 5-turn K/L/M/N/O conversational chain (department → employee count → employee ranking within it → that employee's weakest area → whether that's also the department's weakest area overall) across 4 commits/sub-rounds, including a real root-cause fix for a dept-sticky-fallback-shadows-employee-pronoun bug. Commits `544e115`, `a3a43b9`, `23377d4`, `5aa6eaf`, `af0d173`.
- **Item #87**: 4 more real bugs from CEO-style testing, fixed in one round: singular department question returning a full ranking instead of 1 row; employee-level two-date comparison collapsing to a company aggregate; single-month lookup wrongly returning a multi-month trend; stale capability-message text. Commit `638b28f`.
- **Item #88**: new business rule — "who is making progress in PACE" (no explicit period) now defaults to a **latest-20-vs-previous-20 qualifying-Standard-shift-rows** comparison, not a calendar-month comparison. Found and fixed 2 more bugs along the way (an unrelated intent, `day_list`, was hijacking date-containing progress questions; an explicit two-month pair was being compared against the wrong predecessor month). Commits `aabe4a3`, `8718692`.
- **Item #89** (correction, important lesson — see section 12): first attempt (`23f7873`) to fix a "No data found" bug for sparse-attendance employees' weakest-area lookup **diagnosed the wrong root cause** (blamed the calendar date-window) and was never live-retested before being reported fixed. The real root cause, found in the corrected round, was the default qualifying-population filters (`ps_worked_flag_day=1` etc.) excluding the employee's rows entirely. Fixed correctly in commit `ad01603` (see item #90 below for the parallel single-intent-shadowing fix).
- **Item #90**: `3cd2dc0` (additive `subscore_compare_emp` fallback) + `ad01603` (the actual qualifying-population-filter-relaxation fix, described above as part of #89's correction).
- **Item #91** (VALIDATION ROUND): the first full 50-question matrix (A-H categories) + 20 unseen questions + full K/L/M/N/O chain re-run + architectural review, run entirely live against production. Found and fixed 2 real bugs (plural "top/bottom N departments" silently returning employee-grain rows; the group-pronoun weakest-area branch missing #89/90's qualifying-population-filter relaxation). Documented, but deliberately did NOT fix, several further gaps (see sections 11/13). Commits `e7520a3`, `2bdae1d`, plus documentation commits `15cc6bd`/`d5259fc`/`f00b6b1`.
- **Item #92** (FINAL HARDENING): fixed both of item #91's two named highest-priority bugs — the WFH-filter silent drop in the rule-based ranking path, and the sticky-department-context leak into an unrelated fresh question — plus one cosmetic bug (a literal `None` leaking into reply text). Commits `024cfd9`, `11cf38a`, `ac3cbc8`, documentation commit `bbaaf45` (**current HEAD**).

---

## 4. Current Architecture

**The full generalized query pipeline**, in order (see section 1's "end to end" list for the numbered version):

```
message
  → spellcheck correction (raw_message preserved)
  → pre-classification interceptors (filter-meta, bare-direction, list-pronoun-metric,
    driving-performance-ambiguity guard, both-ends-ranking-ambiguity guard)
  → rule_intent = intents.match_intent(message)      [~123 regex intents + fuzzy fallback]
  → if rule_intent: WINS UNCONDITIONALLY (item #52's precedence flip)
  → else: llm_result = llm_nlu.classify(raw_message)  [intent-name classification]
  → if neither: extracted = llm_nlu.extract_build_query(raw_message, context_hint)
       → validate dimension against queries.BUILD_QUERY_DIMENSIONS
       → validate/repair metrics against queries.BUILD_QUERY_METRICS.keys() | {pace_score, pace_status, ...}
       → re-parse period_phrase via entities.extract_date_range/extract_month/extract_months
       → re-resolve dimension_name via entities.extract_employee/extract_department/extract_manager
         (NEVER trust the LLM's own name transcription)
       → app/main.py::_extraction_llm_reply() calls queries.build_query(...)
  → if still nothing: sql_fallback.answer(raw_message)  [free-form AI-generated SQL, labeled]
  → if still nothing: bottom-of-cascade bare name lookup via build_query_overview_reply()
  → if still nothing: intents.FALLBACK_MESSAGE
```

**Structured fields used** (real names, confirmed in code):

- `_BQ_EXTRACTION_SCHEMA` (`app/llm_nlu.py`) — the JSON schema `extract_build_query()` returns: `dimension`, `dimension_name`, `metrics` (list), `filters`, `period_phrase`, `unrecognized_metric_phrase`, plus (added item #84) `limit` (nullable int) and `operation` (enum `value|rank_top|rank_bottom|strongest_weakest|trend`), plus (added item #86) the `rank_both_ends` operation.
- `queries.BUILD_QUERY_DIMENSIONS` — `{"employee": (...), "rm": (...), "department": (...), "company": (None, [])}` (the `"company"` dimension, added item #64, does a `GROUP BY`-free single-aggregate-row query).
- `queries.BUILD_QUERY_METRICS` — a dict of `{key: (sql_expr, human_label)}` tuples: `engagement_pct`/`effectiveness_pct`/`discipline_pct`/`working_pct`, `LC`/`EL`/`DH`, `working_hours`, `productive_minutes`, `pace_score` (the capped-average-first formula, special-cased), `pace_status`, `pace_score_day_level`, `capped_engagement`/`capped_effectiveness`/`capped_discipline` (raw internal ingredients, labeled "(internal)"), `dept_status_60_days_derived`, `dept_score_60_days_precomputed` (department-only), `days_counted`, plus (per item #77's plan, confirm current presence in code) `engagement_minutes`/`meeting_minutes`/`tasks_created`/`tasks_assigned`.
- `queries.build_query(dimension, metrics, filters=None, period=None, name_filter=None, limit=None, scope=None, ascending=False, latest_n_days=None, latest_n_days_offset=0, employee_ids=None, reporting_user_id=None)` — the one general query engine. `filters` dict keys: `ps_status`, `visit_status`, `work_mode`, `shift_type` (each accepting `"any"` to drop that clause). `scope=(dimension, name)` filters on a DIFFERENT column than the one being grouped by (e.g. group by employee, filter to one department). `latest_n_days`/`latest_n_days_offset` (item #76/#78/#88) implement the "latest N qualifying Standard-shift rows" window mode, as opposed to a plain calendar-date `period` tuple.
- `session_store.query_context` — `{last_operation, last_dimension, last_result_ids, ascending, metric, period_phrase}`, written by BOTH the extraction cascade's ranking branch and the rule-based `metric_ranking()`/`dept_best`/`dept_worst` handlers (item #86 extended this to the rule-based side too).
- `session_store.comparison_entities` — `{first, second}`, each `None` or `{type, id, name}`, kept in sync automatically inside `push_context()` via a `_push_comparison_entity()` helper — used for "compare them"/"both of them" 2-entity follow-ups.
- `session_store.sticky_context` — `dept_name`, `employee_id`, `employee_name`, `month`, `date_range`, `day_compare_dates`, `month_compare_months`, plus (item #92) a SEPARATE `employee_dept_name` slot introduced specifically to fix the sticky-context leak (see section 8).
- `session_store.last_list` — `{kind, rerun_list, rerun_same, answer_kind, dept_name, employee_ids, team_label, month, date_range, statuses, rerun_opposite, ascending}` — the older, single-slot "last list/ranking answer" tracker, still used by ~20+ rule-based ranking call sites for vague "list them"/"what about the bottom 5" follow-ups.

**Multi-step planning**: deliberately narrow, NOT a general planner (explicit user constraint, item #83/#86). Exactly one hand-built 2-step composition exists: `_handle_driving_performance()` — step 1 resolves the department-level winner/loser (`limit=1`), step 2 ranks employees WITHIN that department by PACE score, reusing `build_query()`'s own `scope=("department", name)` mechanism for both steps. This does NOT generalize to arbitrary "and also show me X" second clauses (see section 13's multi-metric/multi-clause gap).

---

## 5. PACE Business Logic

**Employee/department/company PACE**: PACE score is computed per employee-day (or, for a department, aggregated across an employee population) from 4 "capped" sub-metric ingredients — `capped_engagement`, `capped_effectiveness`, `capped_discipline`, `capped_working_hours` — internal, non-linear transforms of the user-facing percentages (`engagement_pct`, `effectiveness_pct`, `discipline_pct`, `working_pct`; confirmed NOT a simple rescale — e.g. `effectiveness_pct=50.00` pairs with `capped_effectiveness=0.53`, not 0.50).

**The PACE formula** (confirmed directly from the real production ETL source, `pace.py`, pasted into this project's chat history early on):
```
PACE score = LEAST(100, ROUND(
    ((avg_capped_engagement * avg_capped_effectiveness * avg_capped_working_hours * 7)
     + (avg_capped_discipline * 3)) * 10
))
```
This is a "7:3" model — 7 parts engagement×effectiveness×working-hours, 3 parts discipline.

**The correct arbitrary-period recomputation method** (critical — do NOT average a per-row precomputed score across a period, that is the Jensen's-inequality bug this project fixed repeatedly, items #55/#59/#60/#75):
1. Filter to `shift_type='Standard'` rows in the target period/scope, from `public.pace_1` directly (never `pace_chatbot_view` for this).
2. `AVG()` each of the 4 capped sub-metrics SEPARATELY across that row set.
3. Apply the score formula ONCE to those 4 averages.

This is codified as `PACE_SCORE_AGG_SQL` (direct-aggregate form) and `PACE_SCORE_FROM_AVGS_SQL` (pre-aggregated form) in `app/queries.py`, centralized in item #75 and referenced from 11+ call sites (`metric_ranking()`, `dept_ranking()`, `rm_ranking()`, `build_query()`, `employee_full_monthly_trend()`, `_gainer_loser_cte()`, and others).

**`pace_score_day_level`** (`new_pace_score_7_3_event_level`) is a DIFFERENT, deliberately distinct metric: the day-level/event-level score, not a period aggregate. Do not conflate it with the capped-average-first period aggregate above.

**PACE status** bands: Black `<50` / Red `50-64` / Amber `65-79` / Green `>=80`, on the (banded) score — centralized as `_PACE_STATUS_CASE_SQL`/`pace_status_sql()`.

**Last-60-days semantics**: the STORED column `last_60_days_new_pace_score_7_3` is a rolling 60-WORKED-day window (not 60 calendar days), computed by the external ETL, NOT by this app. `build_query()`'s own default `period` (when none is named) uses `default_period_last_60_days()`, a fixed 60-CALENDAR-day window ending today — confirmed (item #75's golden validation) that these two windows do **not** always produce identical numbers, especially for employees with meaningful PS-off/visit-flagged days in the window (`build_query()`'s `pace_score`/`pace_status` metrics apply the default qualifying-population filters, `ps_worked_flag_day=1 AND visit_flag='No'`, which the stored ETL column's own "qualifying" definition does not restrict to).

**"Qualifying Standard rows"** = rows where `shift_type='Standard'` AND all 4 `capped_*` columns are non-null (documented as `PACE_SCORE_QUALIFYING_ROWS_SQL`, a documentation-only constant — existing call sites keep their own inline copy of this filter clause; only the score/status EXPRESSIONS were centralized in item #75, not this filter).

**"Latest-N-qualifying-row" mode** (`build_query(latest_n_days=N, latest_n_days_offset=0)`, items #76/#78/#88): a COMPLETELY DIFFERENT concept from either calendar days or working days — it takes the most recent N rows (by `row_number()` per employee, filters applied BEFORE selecting the N rows) regardless of how many calendar days that spans. `latest_n_days_offset` shifts the window (`rn > offset AND rn <= offset+n`) so a "previous window" (e.g. rows 21-40 back) can be computed with a second call. **Do not conflate calendar days, working days, qualifying Standard rows, and latest-N-qualifying-rows — this codebase treats all four as distinct concepts with distinct code paths.**

**The 4 PACE component percentages**: Engagement %, Effectiveness %, Discipline %, Working %/Working Hours % — the user-facing, uncapped versions of the 4 capped ingredients (`engagement_pct`, `effectiveness_pct`, `discipline_pct`, `working_pct`).

**Strongest/weakest area** (employee-level): comparison across the 4 `*_pct` (or, per query path, `capped_*`) columns for ONE employee — implemented in `subscore_compare_for_employee`/the `_area_match` branch of `_extraction_llm_reply()`. **Department-level weakest/strongest area**: confirmed (item #83 section 5, item #86 Decision 2) to be the AVERAGE of the 4 pct sub-metrics ACROSS ALL EMPLOYEES in the department for the period — not the widest within-department spread, not the sub-metric most employees are individually weakest in. This is the existing, confirmed, in-use definition; any future change to it should be a deliberate decision, not a silent reinterpretation.

**Ranking semantics**: `ascending`/`descending` params throughout; `_ASCENDING_WORDS`/`_RANKING_WORDS` regexes (raw-message-derived) remain the AUTHORITATIVE signal for ranking direction/shape, cross-checked against (but never fully overridden by) the extraction LLM's own `operation` field guess — the "narrow deterministic check wins over probabilistic LLM guess" pattern used throughout Change 2.

**Comparison semantics**: 2-entity comparisons ("compare them"/"both of them") resolve against `session_store.comparison_entities` (a 2-slot, most-recently-shifted tracker, populated automatically for every explicit department/employee mention across ALL intents — same "last explicitly named, no topic-relatedness check" risk profile as `sticky_context`, an accepted tradeoff, not a new one).

**Improvement/progress semantics**: "who is making progress in PACE" (and equivalents — "who is improving in PACE?", "who improved the most?", "who are the biggest PACE improvers?") DEFAULT to a **latest-20-vs-previous-20 qualifying-Standard-shift-rows** comparison (item #88's new business rule) — NOT a calendar-month comparison. An EXPLICIT period named in the message (a month, a month pair, or sticky month/date_range context) still uses the original calendar-month `pace_score_trend_ranking()` path unchanged. `queries.pace_score_progress_ranking()` implements the new default, reusing the `latest_n_days`/`latest_n_days_offset` mechanism via two `build_query()` calls (not a new query engine).

---

## 6. PACE Column/Metric Mappings

Reproduced/adapted from `PROJECT_BACKUP_2026-09-09.md` §7 (its own header notes it is "exactly as produced earlier this session, reproduced here in full — not summarized") and cross-checked against `app/queries.py`'s current code. Where SESSION_HANDOFF.md items #77/#79/#80 mention additions that could not be independently confirmed present in the exact current `BUILD_QUERY_METRICS` dict during this document's read, they are marked "confirm in code" — grep `app/queries.py` for the key name before relying on it.

| NL term(s) | Column(s) | Where in code |
|---|---|---|
| "PACE score", "score" (bare) | `overall_pace_score` (view alias of `last_60_days_new_pace_score_7_3`) | `METRICS["pace_score"]`, `_EMP_PACE_SCORE_PATTERNS` |
| "PACE status", "black/red/amber/green category" | `overall_std_pace_status` (view alias of `new_pace_status_overall_last_60_days_7_3`) | `status_list`/`status_count`/`_PACE_STATUS_CASE_SQL` |
| trend/delta ("improving/declining", month-wise) | `pace_score_prev_month`/`pace_score_delta` (view, capped-average-first, item #31) | `employee_pace_trend_monthly`, `_EMP_TREND_PATTERNS`/`_IMPROVING_PATTERNS`/`_DECLINING_PATTERNS` |
| day-level PACE | `new_pace_score_7_3_event_level` (`pace_1` only) | `employee_weekly_pace_trend`, `day_compare()`, `BUILD_QUERY_METRICS["pace_score_day_level"]` |
| "who dropped/improved the most" | `pace_score_delta` ranking | `score_drop_ranking`/`score_improvement_alltime`/`pace_delta_ranking_cw` |
| department PACE score | `dept_pace_score` (view, sources the Active `dept_score_60_days_7_3`, fixed item #52) | view definition |
| engagement/effectiveness/discipline/working (LC/EL/DH — see below for attendance) | `engagement_pct`/`effectiveness_pct`/`discipline_pct`/`working_pct` (daily, uncapped); `capped_engagement`/`capped_effectiveness`/`capped_discipline`/`capped_working_hours` (period averages, internal) | `METRICS[...]`, `BUILD_QUERY_METRICS[...]`, `_detect_pct_capped_metrics()` (item #73's normalizer) |
| "weakest/strongest area" | comparison across the 4 pct/capped columns | `subscore_compare_for_employee`, `_area_match` |
| "defaulter" | `defaulter_count_per_day` (1 if LC/EL/DH fired) | `METRICS["defaulter_days"]`, `DAY_FLAGS["defaulter"]` |
| attendance/punch-in | `punch_in_ts is not null` | `DAY_FLAGS["attendance"]` |
| "WFH", "work from home", "remote" | `wfh_status = 'Work From Home'` (`pace_1` only) | `DAY_FLAGS["wfh"]`, `build_query()`'s `work_mode` filter |
| "leave", "took leave" | `applied_leave_type is not null` (`pace_1` only) | `DAY_FLAGS["leave"]` |
| "visit", "client visit" | `visit_flag = 'Yes'`/`'No'` (`pace_1` only) | `DAY_FLAGS["visit"]`, `build_query()`'s `visit_status` filter |
| "on OT", "overtime" | `shift_type = 'Overtime (OT)'` | `DAY_FLAGS["overtime"]`, `build_query()`'s `shift_type` filter |
| "PS install rate" (dept %) | dept-level aggregate of `ps_installed_new` | `ps_install_rate_by_dept()` |
| "PS not installed" | `ps_installed_new = 0` (`pace_1` only) | `DAY_FLAGS["ps_not_installed"]` |
| "offline attendance" | `offline_attendance_flag = 'Offline Attendance'` (`pace_1` only) | `DAY_FLAGS["offline"]` (label corrected item #49 — no longer conflated with PS-installed) |
| "PS working" | `ps_worked_flag_day = 1` (`pace_1` only) | `DAY_FLAGS["ps_worked"]`, `ps_working_ratio()`, `build_query()`'s `ps_status` filter |
| "productive time/minutes" | `productive_and_meeting_min` | `METRICS["productive_min"]`, `BUILD_QUERY_METRICS["productive_minutes"]` |
| "WhatsApp minutes" | `whatsapp_min` | `METRICS["whatsapp_min"]` |
| "tools and mails" | `tools_and_mails_min` | `METRICS["tools_and_mails_min"]` |
| "AI usage/minutes" | `ai_min` | `METRICS["ai_min"]` |
| "engagement minutes" | `engagement_minutes` | per item #77/#79, wired into `BUILD_QUERY_METRICS["engagement_minutes"]` — **confirm in code** |
| "meeting minutes" | `meeting_in_min` (`pace_1` only) | `meeting_minutes_ranking()`, `BUILD_QUERY_METRICS["meeting_minutes"]` per item #79 — **confirm in code** |
| "meeting count" | `meeting_count` (`pace_1`) | `meeting_count_ranking`, `DAY_FLAGS["had_meetings"]` |
| "todos created"/"todos assigned" | `todos_created`/`todos_assigned` | `TASK_METRICS[...]` |
| "tasks created"/"tasks assigned" | `tasks_created`/`tasks_assigned` | `TASK_METRICS[...]`, `BUILD_QUERY_METRICS[...]` per item #79 — **confirm in code**; kept as 2 SEPARATE metrics, no combined "task_activity" magnitude (item #77's explicit recommendation) |
| "task activity" (day flag / boolean, distinct from the above) | `tasks_created>0 or tasks_assigned>0` | `DAY_FLAGS["completed_tasks"]` |
| "on-time completion rate"/"responsiveness score"/"extension adherence" | `ontime_completion_rate`/`responsiveness_score`/`extension_adherence_score` (~69% NULL each) | `TASK_METRICS[...]` |
| "calls made" | `total_calls` | `DAY_FLAGS["called_clients"]` |
| "d score"/"dscore" | `d_score` (84% NULL) | `d_score_emp`/`d_score_ranking`/`d_score_trend` |
| "department" | `dept_name`/`dept_id` | `entities.extract_department()` |
| "manager" | `reporting_manager_name`/`reporting_user_id` | `entities.extract_manager()` — **NOT** used for "my team" resolution |
| "grade" | `grade` | `grade_lookup` |
| "designation" | `designation` | `designation_breakdown` |
| "tenure"/"new joiner" | derived from `doj` (`NEW_JOINER_DAYS=20`, `POTENTIAL_NEW_JOINER_DAYS=40`) | `avg_tenure`/`new_joiners` |
| "shift type" | `shift_type` | `shift_type_emp` |
| "email access"/"my team" | `email_access` (comma-delimited, exact-match-only reverse lookup) | `team.py`, NOT `entities.extract_manager()` |

---

## 7. Default Filters

`queries.build_query()`'s default qualifying-population filters (used whenever a filter is not explicitly overridden in the message):
- `shift_type = 'Standard'`
- `visit_flag = 'No'`
- `ps_worked_flag_day = 1`

**WFH is unfiltered by default** — a `work_mode` filter (`ps_status`, `visit_status`, `shift_type`, `work_mode` are the 4 real filter dict keys `build_query()` accepts) is applied ONLY when the message explicitly requests it (e.g. "... among WFH employees", "... working from home"). Passing `"any"` for `ps_status`/`visit_status`/`shift_type` drops that clause entirely (used by the qualifying-population-filter-relaxation retry, see section 8/12).

**How explicit filters override defaults**: `dict.setdefault` semantics — an explicit override already present in the `filters` dict (e.g. "...during OT" → `shift_type='Overtime (OT)'`) is NEVER clobbered by a later relaxation retry; the retry only fills in filters the user did not explicitly name.

**The WFH hardening fix (item #92)**: item #91 found "top 5 WFH employees by PACE score" silently returned the SAME rows as the unfiltered top-5 — the WFH filter was correctly wired into `build_query()`/`_handle_rank_both_ends()` (which is why "highest and lowest PACE among WFH employees" always worked), but the RULE-BASED `_METRIC_INTENTS` dispatch (`pace_score_best`/`pace_score_worst`/etc.) calls a completely SEPARATE, older query function, `queries.metric_ranking()`, which had **no work-mode filtering parameter at all**. Fixed by adding an optional `filters` param to `metric_ranking()` (applied as `wfh_status = 'Work From Home'` directly for `pace_score` branches querying `pace_1`; applied as an `employee_id IN (SELECT ... FROM pace_1 WHERE wfh_status=...)` subquery for the generic-metric branch querying `pace_chatbot_view`, which doesn't expose `wfh_status`). Also added to `queries.pace_score_progress_ranking()`. Both `app/main.py` call sites now call the shared `_detect_build_query_filters(raw_message)` detector and thread the result through, including their `rerun`/`rerun_opposite` follow-up closures (so a bare-direction follow-up preserves the filter). **NOT fixed**: the interaction between the WFH filter and `metric_ranking()`'s "latest N qualifying rows" mode — `metric_ranking()` does not support that mode at all (only a fixed calendar window), so a phrasing like "top 5 pace score for employees over their latest 20 WFH days" applies the WFH filter correctly but `days_counted` comes back 1-3, not up to 20 (see section 13).

---

## 8. Conversational Context

**Two additive mechanisms coexist**, neither replacing the other:

1. **`sticky_context`** (session_store.py, whole-session-sticky, single-slot per field): `dept_name`, `employee_id`, `employee_name`, `month`, `date_range`, `day_compare_dates`, `month_compare_months`, and (item #92) a SEPARATE `employee_dept_name` slot. Each field is unconditionally overwritten (`push_context()`) whenever a NEW EXPLICIT value is named in the current message — explicit mention always overrides sticky context, never the reverse. This is the mechanism behind "who's in red in Founders Office" → six turns later → "who all are in black?" still correctly resolving to Founders Office.

2. **`query_context`** (added item #84, specifically to fix the gap #83 found — the extraction cascade's own ranking branch previously wrote NO state at all): `{last_operation, last_dimension, last_result_ids, ascending, metric, period_phrase}`. `last_result_ids` is a LIST (not a single value), kept in the SAME order the SQL was sorted, so "the employee with the lowest/highest X" can resolve by index without re-guessing direction. Written by BOTH the extraction cascade's `is_ranking` branch AND (extended item #86) the rule-based `metric_ranking()`/`dept_best`/`dept_worst` handlers.

**Pronoun resolution** ("their"/"them"/"there"/"those employees"):
- A single most-recently-discussed employee: `_extract_employee_ctx()` (item #30), the ONE genuinely general mechanism — any `_INDIVIDUAL_EMP_INTENTS`-class branch inherits it on a bare pronoun.
- "Their weakest areas" right after a multi-employee ranking (failure G, item #83/#84): resolves via `query_context.last_dimension=="employee"` + `last_result_ids`, reporting EACH employee's own weakest area (no aggregate "group weakest area" exists — a per-employee table is returned instead).
- "The employee/department with the lowest/highest X" right after a ranking (failure H): `_SINGULAR_RANKED_REFERENT` regex resolves directly to `query_context.last_result_ids[idx]` instead of re-running a fresh default-10-row ranking.
- 2-entity comparisons ("them"/"both"/"the two"/"either of them", failures I/J): resolve against `comparison_entities` when both tracked entities share the requested dimension's type.
- Department-level weakest-area follow-ups ("...for the department overall?", failure O): `_handle_dept_weakest_area_followup()` resolves "that area"/"the department" from `query_context`/sticky context.

**Explicit new info always overrides old context** — every follow-up mechanism above checks the CURRENT message for an explicit mention first and only falls back to sticky/query context when the current message names nothing of its own kind.

**What's intentionally NOT sticky**: time period is NOT carried forward for fixed current-vs-prior-month comparison intents (`emp_trend`, `score_drop_ranking`, etc. — a `_PERIOD_CONTEXT_BLACKLIST`, since those intents have their own fixed 2-month semantics that a carried-forward period would corrupt). Department scope is NOT inherited by `_INDIVIDUAL_EMP_INTENTS`/`_DUAL_PURPOSE_EMP_INTENTS`/`_DEPT_LEVEL_RANKING_INTENTS` (company-wide-by-definition rankings like `dept_best`/`rm_ranking_best` never inherit a stale department).

**The sticky-context bug (item #91 employee-chain step M) and its fix (item #92)**: item #86's own fix for failure O made the single-employee weakest-area lookup push the resolved employee's OWN department into `sticky_context["dept_name"]` — a convenience so a later "...for the department overall?" follow-up could resolve "the department" without the user re-naming it. But `sticky_context["dept_name"]`'s documented contract (per `session_store.py`'s own comment) is "only what was EXPLICITLY named by the user" — this incidental push violated that contract. The consequence, reproduced live in item #91: K "which employee has the lowest PACE score?" → Divyansh Sharma (IT-Development). L "what is their weakest area?" → correct. M (a FRESH, UNRELATED question, no referential cue at all) "which employee has the highest PACE score?" → wrongly scoped to IT-Development instead of the real company-wide answer, because the general department-scope carry-forward (`app/main.py`'s `dept_name = session_store.get_recent_context(session, "dept_name")`, unconditional, no referential-cue gate) picked up the incidentally-pushed value.

Item #91 itself proposed — and flagged as too risky — gating the GENERAL carry-forward on a referential-cue regex, because that same unconditional carry-forward is relied on by other confirmed-working flows with NO referential word at all (e.g. the red/black Founders-Office example above). Item #92's actual fix was narrower and lower-risk: a SEPARATE sticky-context field, `employee_dept_name`, dedicated ONLY to this one incidental (not-explicitly-named) case. The two call sites that push an employee's own department as a side effect of a weakest/strongest-area answer now call `push_context(session, employee_dept_name=...)` instead of `push_context(session, dept_name=...)`; `_handle_dept_weakest_area_followup()` now reads `employee_dept_name` instead of `dept_name`; every OTHER `push_context(..., dept_name=...)` call site (9 others, individually audited) was left unchanged since each of those pushes a genuine, on-topic, explicitly-relevant department. Live-verified: the employee chain's step M now correctly returns the real company-wide answer (Tanu Mehra, 100), and the department chain (which legitimately needs the old mechanism, via the explicit "there" cue) still works end to end, all 5 steps.

**Two-entity comparisons**: see `comparison_entities` above (section 4).

**Multi-turn K→O chains**: the "K/L/M/N/O" naming refers to a specific 5-turn test pattern used repeatedly across items #86/#88/#91/#92 — e.g. (department chain) K: "which department has the lowest PACE score?" → L: "how many employees are in that department?" → M: "which employee there has the lowest PACE score?" → N: "what is their weakest area?" → O: "is that also the weakest area for the department overall?" — confirmed working end-to-end as of item #92 (both the department-style and employee-style chains, the latter's step M fixed by item #92 as described above).

---

## 9. Important Semantic Decisions

Verbatim/near-verbatim from SESSION_HANDOFF.md, preserved exactly as the business decisions were made — do not silently reinterpret any of these:

- **"Driving performance"** = "the highest-performing employees WITHIN the department, ranked by PACE score" (item #86 Decision 1). Explicitly NOT month-over-month improvement, NOT deviation-from-company-average — both explicitly considered and rejected by the user.
- **Department weakest/strongest area** = the lowest/highest of the 4 AVERAGE sub-metrics (engagement/effectiveness/discipline/working) ACROSS ALL EMPLOYEES in the department for the period (item #86 Decision 2, confirming the pre-existing #76/#78 mechanism).
- **"Highest AND lowest"** in one question = exactly 2 employees/rows: the top-1 and bottom-1 of the SAME filtered population (item #86 Decision 3, the new `rank_both_ends` operation) — NOT two separately-filtered queries, NOT an arbitrary N.
- **"Making progress in PACE" default** (no explicit period named) = latest-20-vs-previous-20 QUALIFYING Standard-shift rows, NOT a calendar-month comparison (item #88's new business rule). An explicit period named in the message still uses the original calendar-month comparison, unchanged.
- **"Top 10 improvers in the last 4 weeks"** preserves the EXISTING gainer/loser ("4 complete weeks vs prior 4 complete weeks") semantics from item #47 — this is a DIFFERENT mechanism from the item #88 progress-ranking default above; the two must not be conflated (confirmed by item #88's own regression test 5, which verified "top 10 PACE improvers in the last 4 weeks" still routes to `gainer_loser_ranking`, not the new `pace_score_progress_ranking`).
- **Capped-vs-percentage business rule** (item #73): bare "X" (engagement/effectiveness/discipline, no "capped", no %/percentage word) → `X_pct`; "capped X" (no %/percentage word) → `capped_X`; "raw capped X" → `capped_X` (same as above); "capped X %"/"capped X percentage" → `X_pct` (DELIBERATE, non-obvious — explicitly NOT `capped_X`). Working Hours only gets the normal-vs-percentage half of this rule (no `capped_working_hours` metric is exposed).
- **`sticky_context["dept_name"]`'s contract**: holds only what was EXPLICITLY named by the user (documented in `session_store.py`'s own comment) — item #86's own O-chain fix violated this incidentally, and item #92 fixed it with a separate `employee_dept_name` slot rather than relaxing the contract.
- **Standard-shift-only grain**: a knowing, accepted tradeoff (undercounts OT-mixed days by ~5-30%, drops the 2.36% of employee-days that are OT-only entirely) made very early in the project (before "Change 2") and never revisited.
- **`pace_chatbot_view` changes are always explicit, case-by-case authorized** — never done silently. Currently on its 4th authorized iteration.

---

## 10. Testing History

The pre-"Change 2" manual-testing failures (items #52-69, briefly, for context — full detail in SESSION_HANDOFF.md):
- Stale greeting/capability text — fixed item #87 (Bug E) with an updated, accurate capability summary.
- Employee-name resolution failures (typo corruption, possessive forms, substring collisions like "product" inside "productive") — fixed across items #3-6 with word-boundary-safe fuzzy matching and a `fallback_text=` raw-text retry pattern used throughout `entities.py`.
- "August performance" silently becoming a trend query instead of a single-month lookup — a recurring failure class, most recently re-confirmed and fixed as item #87 Bug D (single-month lookup wrongly redirected to `full_trend_emp`'s multi-month trend).
- Context loss across turns — the original motivation for `sticky_context` (item #26/#29) and later `query_context` (item #84).
- Engagement-comparison silently becoming company-level — an instance of the same "silent-fallback-to-wrong-scope" failure class documented as the architecture's biggest remaining weakness (item #91 Task 5).
- Bottom-5 returning 10 rows — fixed by broadening `entities.extract_limit()` in item #84 (the "5 lowest"/"lowest 5" phrasing gap).
- Highest-department returning multiple rows — fixed by the `limit=1`-collapse precedent (items #86/#87), applied to `metric_ranking()` and later to `dept_best`/`dept_worst`.
- "Driving performance" clarification needed — item #83 correctly flagged this as a genuine 3-way ambiguity rather than guessing; resolved by explicit user decision in item #86.
- "Highest+lowest" wrong count — resolved by the new `rank_both_ends` operation (item #86 Decision 3).
- Department context loss / weakest-area context loss — the K/L/M/N/O chain work across items #86/#88/#91/#92.
- WFH filter drop — found and root-caused twice (once misdiagnosed as a `build_query()`-adjacent issue in item #91, correctly root-caused as a separate `metric_ranking()` gap in item #92) and fixed for the main ranked-list case in item #92.
- Sticky context leak — found in item #91, fixed in item #92 (see section 8).
- Sparse-attendance weakest-area lookup — the item #89 saga (see section 12 — the FIRST fix attempt diagnosed the wrong root cause).
- "Making progress" default — item #88's new business rule (see section 9).

For each, the CURRENT status is: fixed and live-verified as of the commit noted, UNLESS explicitly listed as a known limitation in section 13 below.

---

## 11. Final 50+20 Validation (item #91)

This is the first (and so far only) full systematic validation run this project performed, against LIVE PRODUCTION only (no DB credentials available that round). Reproduce this from SESSION_HANDOFF.md item #91 directly if exact wording is needed — the original 50-question spec text was not preserved verbatim anywhere in the repo, so item #91's own 50 questions are a best-effort reconstruction from the 8 category *descriptions*, flagged as such rather than presented as verbatim.

**Categories tested**: A — Basic ranking (8/8), B — Multi-metric (6/6), C — Period/change (6/6), D — Filtered period (8/8), E — Company scope (7/7), F/G — the K/L/M/N/O conversational chains (reused from item #86), H — CEO-style composite (5/5). Plus 20 unseen questions (U1-U20).

**PASS/FAIL counts**: Matrix — 40 non-chain questions tested: 24 PASS, 1 pair fixed live during the round (A3/A4, now PASS), 1 unverified without DB access (D4), 14 remaining as documented failures. Unseen questions — 9 PASS, 11 failures (5 ROUTING-INTENT-ERROR, 2 WRONG GRAIN, 2 WRONG FILTER, 1 WRONG PERIOD, 2 OTHER).

**Failure classifications used**: ROUTING-INTENT-ERROR (message misroutes to an unrelated intent/clarification), WRONG GRAIN (returns employee rows when department rows were asked for, or vice versa), WRONG METRIC (silently answers a different metric than named), WRONG FILTER (a named filter is silently dropped or not applied), WRONG PERIOD (period resolves incorrectly), OTHER (unhandled exception, cosmetic leak, or a genuine-but-out-of-scope data condition).

**Root-cause groupings** (per the task's explicit "group by root cause, not by question" instruction):
1. Plural "top/bottom N departments" not recognized as department-dimension → A3, A4, U4, U7(partial) — **FIXED this round**, commit `e7520a3`.
2. Group-pronoun weakest-area lookup missing the qualifying-population-filter relaxation → department-chain step N — **FIXED this round**, commit `2bdae1d`.
3. Company-wide (no entity named) aggregate questions mostly misroute → E1, E2, E3, E5, E6, E7, U19 — **NOT fixed**, flagged as the single biggest gap, needs DB access or a debug-logging pass.
4. WFH/work-mode filters dropped in the plain ranked-list path (but correctly applied in `rank_both_ends`) → D2, U16 — **NOT fixed this round** (fixed in the FOLLOWING round, item #92).
5. No multi-clause/secondary-query composition beyond the one hardcoded driving-performance case → B1, B2, B3, B5, H2, H3 — **NOT fixed**, explicitly out of scope per items #83/#86's own "narrowly scoped, not a general planner" design decision.
6. Sticky department context applied unconditionally to a fresh, unrelated query with no referential cue → employee-chain step M, step O — **NOT fixed this round** (fixed in item #92).
7. Assorted single-instance routing/phrasing misses (C1, C6, U5, U8, U12, U13, U15, U17, U1) — each a narrow, distinct gap, none blind-fixed without DB access.
8. `sql_fallback.py` raw-column-name leakage → U20 — **NOT fixed**, low severity, the path is explicitly labeled unverified.
9. Cosmetic `None` leak in "who improved the least" reply text → U10 — **NOT fixed this round** (fixed in item #92).

**Conversational chain results**: employee-style chain (K/L/M/N/O) — K/L PASS, **M WRONG CONTEXT** (the sticky-leak bug, root-caused live), **N WRONG CONTEXT**, **O ROUTING-INTENT-ERROR/WRONG CONTEXT**. Department-style chain (re-run AFTER this round's group-pronoun fix) — all 5 steps PASS.

**Architectural review findings** (item #91 Task 5, condensed — see section 13 for the full "known limitations" writeup which draws directly from this): the DETERMINISTIC layer (limit/operation extraction, qualifying-population-filter relaxation, capped-metric normalization, `rank_both_ends`) is genuinely generalized. The DIMENSION-ROUTING layer (which regex pattern list an intent falls into) is NOT generalized and was never claimed to be — an honestly-acknowledged "accumulating pile of precedented-but-individually-added patterns," a deliberate and reasonable tradeoff for a system with intermittent DB access, not a design flaw. "Silent-fallback-to-wrong-scope" is named as the architecture's single biggest remaining weakness — three separate confirmed instances existed at the time of this review (department-grain fallback, WFH-filter drop, sticky-context leak); two of the three were fixed in the very next round (item #92).

**Final verdict (item #91's own words)**: **READY WITH KNOWN LIMITATIONS.**

---

## 12. Final Hardening (item #92)

**Issue 1 — WFH filter silently dropped for ranked-list phrasing — FIXED**. See section 7 for the full technical detail (root cause: `_METRIC_INTENTS` dispatch calls the separate `queries.metric_ranking()` function, which had no work-mode parameter at all, unlike `build_query()`). Fix: `filters` param added to `metric_ranking()` and `pace_score_progress_ranking()`, both `app/main.py` call sites now call `_detect_build_query_filters()` and thread the result through their rerun closures. Commit `024cfd9`. Live-verified with 6 tests including a cross-check that the WFH-filtered "top 5"/"bottom 5" and the `rank_both_ends` "highest and lowest" answers agree on the actual lowest WFH employee (Mainak Mukherjee, 33) across both code paths. **Explicitly NOT fixed**: the interaction between the WFH filter and `metric_ranking()`'s missing "latest N qualifying rows" support (test 4 in item #92's transcript: filter applies correctly, but `days_counted` comes back 1-3 instead of up to 20, because `metric_ranking()` only supports a fixed calendar window).

**Issue 2 — sticky-department-context leak (item #91 employee-chain step M) — FIXED**. See section 8 for the full technical detail (the `employee_dept_name` separate-slot fix). Commit `11cf38a`.

**Issue 3 — the sparse-attendance weakest-area saga (items #89/#90) — an important lesson, restated explicitly so a fresh agent does not repeat it**:
- The bug: `"What is Kalpesh Nandkumar Thakur weakest area?"` (and similar, for other sparse-attendance employees) returned `"No data found for [employee] in this period."` even though the employee has real, visible data elsewhere in the app (e.g. `"PACE score of Kalpesh Nandkumar Thakur"` works fine via a different code path, `employee_detail()`).
- **The FIRST diagnosis was WRONG**: commit `23f7873` blamed `build_query()`'s default fixed calendar-60-day window in the single-employee `_area_match` branch of `_extraction_llm_reply()`. The theory: a sparse-attendance employee's worked days could fall entirely outside that fixed window. The fix retried with `latest_n_days=BUILD_QUERY_DEFAULT_PERIOD_DAYS` (a rolling window) instead. This was committed and pushed **with the commit message claiming the bug fixed, but no SESSION_HANDOFF.md entry was written and no live re-test was actually run before that commit.** The very next live test of the exact same question, in the CORRECTING round, reproduced the byte-identical failure — twice, with a wait in between to rule out deploy lag.
- **The SECOND investigation found the REAL root cause**: `queries.build_query()`'s default *qualifying-population filters* (`ps_worked_flag_day=1`, `visit_flag='No'`, `shift_type='Standard'`) are applied IDENTICALLY regardless of which date-window mode is used. Item #89's first-attempt retry only changed the date-window mode — it reused the exact same three filters, which is exactly why redeploying and re-testing that fix produced byte-identical (still-broken) output. For Kalpesh Nandkumar Thakur specifically, a direct live query (`"Is PS working for Kalpesh Nandkumar Thakur?"`) confirmed **`ps_worked_flag_day` is 0/null on 100% of his rows this period** — the default `ps_status="working"` filter alone excludes every single one of his rows, in BOTH the calendar-window query and the latest-N-rows retry.
- **The real fix**: commit `ad01603` — when a single named employee's initial `build_query()` call finds no data, retry ONCE with `ps_status`/`visit_status`/`shift_type` relaxed to `"any"` wherever the user did NOT explicitly request a specific value (an explicit override is preserved via `dict.setdefault`, never clobbered), in ADDITION to (not instead of) the date-window widening from the first attempt. This generalizes to any employee excluded by any of the 3 default filters, not hardcoded to this one employee or to `ps_status` specifically.
- **The lesson, stated explicitly**: this project's own standing rule — "live-verify before reporting a fix done" — exists precisely because of this incident. A fix that is logically plausible from reading the code is not the same as a fix that has been proven to work against real data/production. A fresh agent working on this codebase must NOT report a fix as done without an actual live (or, if DB credentials are available, direct-SQL) re-test of the EXACT originally-failing repro.
- Item #90 covers the same underlying event from a slightly different angle: `3cd2dc0` (an additive, harmless `subscore_compare_emp` fallback that was NOT what fixed the reported case) and `ad01603` (the actual fix, described above).

---

## 13. Current Known Limitations (CRITICAL — do not hide anything)

Every item below is exactly as SESSION_HANDOFF.md items #91/#92 documented it — deliberately left unfixed, with an honest reason given each time, not silently ignored.

| Limitation | Example | Severity | Category | Code location (if known) | Priority note |
|---|---|---|---|---|---|
| Company-wide aggregate questions with no named entity mostly misroute (5 of 7 in item #91's Category E) | "What is the company's average PACE score?" → "I couldn't find that employee..." | High (very common question shape) | Correctness/routing | `classify()`→cascade dispatch order for the "no dimension named at all" case | **Flagged as the single biggest gap** — needs DB access or a debug-logging pass, not a blind fix |
| WFH+qualifying-row-count interaction unsupported in the rule-based ranking path | "top 5 pace score for employees over their latest 20 WFH days" → filter applies, but `days_counted` is 1-3 not up to 20 | Medium | Edge-case | `queries.metric_ranking()` (no `latest_n_days` support at all) | Would require unifying `metric_ranking()`'s query engine with `build_query()`'s qualifying-row mechanism — materially bigger/riskier than a filter-plumbing fix |
| No multi-metric composition | "Which employee has the highest PACE and engagement scores?" → answers ONLY engagement | High (common CEO-style phrasing) | Unsupported | `_extraction_llm_reply()`'s single dispatch, one metric list | Extraction schema has no representation for "AND" between metrics beyond a flat list that silently collapses |
| No general multi-clause/secondary-query composition beyond one hardcoded case | "5 lowest PACE employees and their weakest areas" → ranking only, weakest-areas clause dropped | High | Unsupported (by design) | Only `_handle_driving_performance()` exists as a 2-step composition | Item #83's `secondary` field design was never built generally; explicitly out of scope per the user's "not a general planner" constraint |
| `sql_fallback.py` raw-column-name/internal-mechanism leak risk | "Give me the weakest area for the employee with the lowest PACE score" → correct answer, but leaks `employee_id`, `overall_pace_score`, and a description of its own SQL logic | Low (path is explicitly labeled AI-generated/unverified) | Cosmetic/UX | `app/sql_fallback.py` | Acceptable given the labeling, worth a future cosmetic pass |
| `metric_ranking()` vs `build_query()` tie-order difference | "top 5 WFH employees by pace score" (via `metric_ranking()`, day-level `pace_1` query) vs `rank_both_ends` (via `build_query()`, subquery-restricted `pace_chatbot_view` query) can disagree on WHICH employee tied at the max score sorts first | Low | Cosmetic | both functions, no shared secondary sort key | Would require adding an arbitrary secondary sort key to both engines |
| Company-scope-with-filter "average X for the whole company" pre-dates item #71 and was never fixed | "average engagement percentage for the whole company" (found item #71, re-confirmed as unresolved multiple rounds later) | Medium | Correctness | `average_metric` branch, `app/main.py` | Flagged repeatedly across rounds, never picked up |
| Assorted single-instance routing/phrasing misses | Unhandled exception on "last N weeks" period phrasing for department-decline; compound AND-filters unsupported ("low discipline AND low effectiveness"); "how does X compare to Y" not recognized (only "compare X vs Y" works); count-threshold department filters unsupported ("departments with fewer than 5 employees"); "this week" resolves to a single day not a week span | Mixed | Edge-case | various, see item #91 Task 1/2 tables | Each narrow and distinct, none shares a fixable root cause; none was reproducible with high enough confidence to blind-fix without DB access |
| Cosmetic `None`-leak-adjacent nuance not fixed | "who improved the least in PACE" replies with heading "Who is improving:" instead of "Who is declining:" | Low | Cosmetic | direction-word detection for "improved the least" phrasing | Observed in item #92, not fixed, out of scope |
| Bare-direction follow-up after a WFH-filtered ranking preserves the filter but not the sort-direction flip | "top 5 WFH employees by pace score" → "what about the bottom 5?" returns the SAME sort direction (full-list rerun), not a flipped ascending sort | Low | Edge-case | pre-existing "what about the bottom N" phrasing-recognition nuance | Unrelated to WFH filtering specifically, not fixed |
| D4 (item #91) — possible wrong department filter, unverified | "Top 5 employees by PACE in Ops - Cement in the last 60 days" — headers omit department, and the #1 name appears elsewhere tagged as a different department | Unknown (flagged, not confirmed) | Unverified | — | Could not be independently confirmed without DB access; needs a repro with logging/DB access |

---

## 14. Current Git/Deployment State

- **Latest commit**: `bbaaf45` — "Item #92 (final docs): WFH filter fix, sticky-context leak fix, cosmetic fix, full regression pass, architectural review, and final verdict".
- **Key previous commits** (most load-bearing, in reverse chronological order): `ac3cbc8` (cosmetic None-leak fix), `11cf38a` (sticky-context leak fix), `024cfd9` (WFH filter fix), `f00b6b1` (item #91 final docs), `15cc6bd`/`d5259fc` (item #91 partial docs), `2bdae1d`/`e7520a3` (item #91's 2 code fixes), `0c11fea`/`ad01603`/`3cd2dc0`/`23f7873` (the item #89/90 sparse-attendance saga), `638b28f` (item #87), `544e115`/`a3a43b9`/`23377d4`/`5aa6eaf`/`af0d173` (item #86), `5026e8f`/`a282466`/`46ee958` (item #85), `8119997`→`94a7c01` (item #84, 10 commits), `08f1a97` (item #71, the extraction-LLM cascade's initial build).
- **Branch situation**: working branch is `master`. Repo convention (established since the earliest LLM-migration round) is to push every meaningful round as `git push origin master` followed by `git push origin master:main`, so `origin/master` and `origin/main` are always kept in sync. Confirmed at the time of writing: `master`, `origin/master`, and `origin/main` all point at the identical commit `bbaaf45`.
- **Working-tree state**: clean except one pre-existing, intentionally-untracked file, `PROJECT_BACKUP_2026-09-09.md` — present since before item #91's round started, not created or modified by any round documented here, left untouched by every round since.
- **Deployment state**: live and auto-redeploying on push, at `https://pace-chat-bot.onrender.com` (root — dashboard + chat overlay) and `https://pace-chat-bot.onrender.com/dashboard` (identical page, second URL kept for compatibility per item #37). GitHub repo: `aryangupta-cell/pace_chat_bot`. Render's Start Command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (set explicitly in Render's dashboard, not the default placeholder — see item #34's deployment snag).
- **No secrets included here**: DB credentials (`PACE_DB_HOST`/`PACE_DB_PORT`/`PACE_DB_NAME`/`PACE_DB_USER`/`PACE_DB_PASSWORD`) and API keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `LLM_PROVIDER`) are environment variables set locally/in Render's dashboard, never committed — see local secrets, not committed, per this project's standing convention (SESSION_HANDOFF.md itself follows the same rule). One known, still-open historical exception: the real DB password was committed in plaintext in `SESSION_HANDOFF.md`'s own early git history (3 commits, item #32/#33) before redaction — this has NOT been remediated (rotating the RDS password remains an outstanding, user-actionable recommendation from early in the project, repeatedly noted as still-not-done in later rounds).

---

## 15. Safe Continuation Rules

- **Preserve working behavior.** Do not touch `pace_chatbot_view`, `PACE_SCORE_AGG_SQL`/`PACE_SCORE_FROM_AVGS_SQL`, `build_query()`'s existing filter/dimension/metric semantics, or any of the ~123 existing rule-based intents' own query functions without a specific, live-confirmed reason.
- **Fix forward only.** Do not revert or rewrite items #52-92's architecture. If something looks wrong, add a narrow, additive fix (a new regex pattern, a new optional parameter with a safe default, a new sibling intent that redirects rather than duplicates logic) — the exact pattern used in essentially every fix documented in section 3.
- **Do not create phrase-specific patches when a generalized mechanism already exists.** Extend `_detect_build_query_filters()`, `_detect_pct_capped_metrics()`, `query_context`, `_NEW_VOCAB_OVERRIDE_PATTERN`, etc. rather than hand-coding one more one-off string match — this project explicitly and repeatedly chose "one new sibling function/parameter, reused everywhere" over "another special case," and a fresh agent should keep doing the same.
- **Verify against actual DB/live production, not just plausible-looking output.** This is the single most important rule in this codebase's history — see section 12's item #89 lesson. A code-review-only "this looks right" is not sufficient; a fix must be live-tested against the EXACT originally-failing repro before being reported done.
- **Test grain/period/filters/metric/ranking/limit/context explicitly**, not just the happy path. Every round documented in section 3 includes a regression suite of 5-12+ previously-confirmed-working behaviors, re-run live after every fix — follow the same discipline.
- **Protect the regression floor.** Before considering any round "done," re-run (live) the baseline behaviors listed in item #92's own "Regression floor" table (section 3/section 11) at minimum, plus any behavior your specific change plausibly touches.
- **Document incrementally.** SESSION_HANDOFF.md's own standing rule (set after items #45/#51/#53 each independently discovered this had been skipped): writing the round's own SESSION_HANDOFF.md entry is a MANDATORY LAST STEP of every round, not optional or deferred. Follow the same discipline for this file if you make a change significant enough to affect its content.
- **Commit after meaningful completed work**, not mid-task, and always push both `origin master` and `origin master:main` (this repo's specific branch-sync convention).
- **Investigate root cause before patching.** Item #89's saga (section 12) is the canonical cautionary tale — a plausible-sounding but wrong diagnosis, shipped without a live re-test, wasted an entire round and left the bug live for a full round longer than necessary.
- **When to make a code change**: only after reproducing the reported failure live (or via direct SQL, if credentials are available) AND identifying a root cause you can point to in the actual code (a specific function, line, or regex) — not a guess.
- **When to run a live test**: before AND after every code change that touches routing, filters, conversational state, or any shared query function — this project's live URL is `https://pace-chat-bot.onrender.com/api/chat`.
- **When to add a regression test**: whenever you fix something covered by an existing "confirmed working" claim elsewhere in SESSION_HANDOFF.md — re-verify that claim, don't just trust it was still true.
- **When to update SESSION_HANDOFF.md**: at the end of every round with a substantive finding or fix, per the mandatory-last-step rule above.
- **When to stop and ask**: when you find a genuine business-decision ambiguity (the way item #83 stopped and asked about "driving performance" and "highest AND lowest" rather than guessing) — do not silently pick an interpretation for something with 2+ defensible readings and no existing precedent in this codebase.

---

## 16. Next Recommended Steps

**What's done**: the full "Change 2" generalized semantic-extraction layer (items #70-73) is built, extended with limit/operation/multi-entity conversational state (items #74-86), hardened through 4 rounds of CEO-style bug-hunting (items #87-90), validated with a full 50+20 question matrix for the first time (item #91), and given a final hardening pass fixing the 2 highest-priority bugs that validation found (item #92).

**What's stable**: single-entity lookups, standard top/bottom-N rankings for both employees and departments, period-over-period change rankings, the department-driving-employees composite, WFH highest/lowest via `rank_both_ends`, the full K/L/M/N/O conversational chains for their originally-designed phrasings, and (as of item #92) WFH-filtered ranked lists via the rule-based path too. All confirmed via live production testing, with no known regressions from the baseline items #83-92 established.

**What remains** (directly from item #91/#92's own known-limitations list — see section 13 for full detail, do not invent anything beyond this):
1. Company-wide aggregate questions with no named entity — the single biggest, most consistent gap (5 of 7 failed in item #91's Category E). **Recommended first pick if given a free choice** — needs either DB access or a debug-logging pass to pin down the exact `classify()`→cascade dispatch-order issue, per item #91's own honest assessment that a blind fix here risks repeating item #89's mistake.
2. The WFH-filter / qualifying-row-count interaction in `metric_ranking()`'s rule-based path (section 13, row 2) — a real but lower-priority gap, would need unifying `metric_ranking()`'s query engine with `build_query()`'s qualifying-row mechanism.
3. Multi-metric composition ("PACE and engagement scores") and general multi-clause/secondary-query composition beyond the one hardcoded driving-performance case — both explicitly scoped out as "not a general planner" per the user's own constraint; only revisit if the user explicitly asks for this capability to be generalized.
4. The `average_metric`/"whole company" pre-existing bug (flagged repeatedly, never fixed).
5. The assorted single-instance routing/phrasing misses in section 13's table.

**What should NOT be changed unnecessarily**: `pace_chatbot_view` (any change needs fresh explicit authorization, per the project's standing rule, currently on its 4th iteration), the centralized `PACE_SCORE_AGG_SQL`/`PACE_SCORE_FROM_AVGS_SQL` formula constants (validated correct, item #75), any of the ~123 existing rule-based intents' own query functions (only their pattern LISTS should be extended, never their core logic rewritten, per the "narrow, precedented, additive patterns" discipline every round since item #72 has followed), and `build_query()`'s own filter/dimension/metric semantics (confirmed correct multiple times; bugs found so far have all been in SIBLING functions lacking a capability `build_query()` already had, not in `build_query()` itself).

Do not invent new requirements beyond what SESSION_HANDOFF.md items #83-92 already discussed — if the user's next request doesn't map onto section 13's documented limitations, treat it as new scope requiring its own investigation-first round, following the same "investigate → design → implement → live-verify → document" pattern used throughout this project's history.
