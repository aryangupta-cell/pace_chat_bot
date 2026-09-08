# PACE Chatbot — LLM Reference Doc (condensed schema + rules)

**Purpose / where this is used.** This document is NOT part of the live intent-classification
prompt (that prompt in `app/llm_nlu.py` only needs the short intent-name enum + a few-shot
list, which it already has). This document is prep for a *hypothetical future* SQL-generation
fallback path — triggered only when neither the rule-based matcher (`intents.py`) nor the
Gemini intent classifier (`llm_nlu.py`) can match a question to an existing `queries.py`
function. It is NOT wired up anywhere yet. Building that execution path is a separate,
explicit decision the user has not made — see `SESSION_HANDOFF.md`'s note that "the LLM
never touches the database." If/when that path is built, this doc is the condensed schema +
rules context an LLM would need to either (a) pick the right existing intent/entities, or
(b) draft read-only SQL for a genuinely novel question.

---

## 1. Schema — `public.pace_chatbot_view` (primary query surface)

Grain: 1 row per (employee_id, worked_day), `shift_type = 'Standard'` only. ~26,600+ rows,
grows daily. Built from `public.pace_1` (see §1b for columns available there but not here).

| Column | Type | Meaning |
|---|---|---|
| employee_id | integer | Employee primary key |
| emp_name | varchar | Employee full name |
| emp_code | text | Employee code (alt identifier) |
| dept_id | smallint | Department id |
| dept_name | varchar | Department name |
| reporting_user_id | integer | Manager's employee_id (raw org-chart FK — NOT used for "my team" logic, see §3) |
| reporting_manager_name | varchar | Manager's name |
| designation | varchar | Job title |
| grade | varchar | Employee grade/band |
| doj | date | Date of joining |
| worked_day | date | The calendar day this row is for |
| lc_flag_per_day | integer | 1 if late-coming flagged that day, else 0 |
| el_flag_per_day | integer | 1 if early-leaving flagged that day, else 0 |
| dh_flag_per_day | integer | 1 if deficient-hours flagged that day, else 0 |
| defaulter_count_per_day | integer | 1 if any of LC/EL/DH fired that day |
| overall_std_pace_status | text | Current PACE status bucket (Black/Red/Amber/Green) for the trailing 60-worked-day window — see §2 thresholds |
| worked_hours | numeric | Hours worked that day |
| shift_type | varchar | Always 'Standard' in this view (Overtime rows excluded) |
| productive_and_meeting_min | bigint | Productive + meeting minutes that day |
| non_productive_and_whatsapp_min | bigint | Non-productive + WhatsApp minutes |
| whatsapp_min | bigint | WhatsApp-specific minutes |
| tools_and_mails_min | bigint | Tools/email minutes |
| ai_min | bigint | AI-tool usage minutes |
| engagement_minutes | numeric | Engagement minutes |
| overall_pace_score | numeric | Current PACE score, 60-worked-day rolling window (aliased from `last_60_days_new_pace_score_7_3`) |
| dept_pace_score | numeric | Department-level PACE score |
| engagement_pct | numeric | Daily engagement percentage (ingredient of PACE score formula) |
| effectiveness_pct | numeric | Daily effectiveness percentage (ingredient) |
| discipline_pct | numeric | Daily discipline percentage (ingredient) |
| working_pct | numeric | Daily working-hours percentage (ingredient) |
| pace_score_prev_month | numeric | Prior calendar month's correctly-aggregated PACE score (see §2 formula) |
| pace_score_delta | numeric | Current month avg − prior month avg (correctly aggregated) |

## 1b. Extra columns — `public.pace_1` only (NOT in the view; query `pace_1` directly for these)

`pace_1` is session-grain (not clean 1-row-per-day; ~7.5% of employee-days have both a
Standard and an Overtime row). Filter `shift_type='Standard'` for day-grain unless you
specifically need Overtime rows.

| Column | Type | Meaning |
|---|---|---|
| capped_engagement / capped_effectiveness / capped_discipline / capped_working_hours | numeric | Row-level (session-level) capped sub-metric ingredients used in the PACE score formula (§2). Confirmed genuinely row-level, not duplicated. |
| new_pace_score_7_3_event_level | numeric | Day-level (non-rolling) PACE score for one session. Use this (not `overall_pace_score`) as the raw input when computing your own period aggregates — never average this directly across days for a period total; average the capped ingredients first (§2). |
| new_pace_status_overall_last_60_days_7_3 | varchar | Source column for `overall_std_pace_status` |
| last_60_days_new_pace_score_7_3 | numeric | Source column for `overall_pace_score` (rolling 60-worked-day window, frozen-ish per employee — do not use for month-over-month deltas) |
| email_access | text | Comma-delimited list of email addresses; used for team resolution, see §3 |
| ps_worked_flag_day | integer | 0/1 — whether the employee's PS (productivity-suite) was actually working that day |
| ps_installed_new | integer | 0/1 — whether PS was installed that day (device/offline status) |
| meeting_in_min | bigint | Meeting minutes (not exposed in the view) |
| wfh_status | varchar | 'Work From Home' or other; rare (~180/30,000 rows are WFH) |
| visit_flag | varchar | 'Yes'/other — client visit flag |
| applied_leave_type | varchar | Non-null when the employee is on leave that day |
| d_score | numeric | Quality score; 84% NULL |
| ontime_completion_rate / responsiveness_score / extension_adherence_score | numeric | Task-quality scores; each ~69% NULL (only populated for a task-management-eligible subset) |
| punch_in_ts / punch_out_ts | timestamp | Raw punch times; `punch_in_ts is null` = absent |
| offline_attendance_flag | varchar | 'Offline Attendance' = device/PS offline that day |
| total_calls, meeting_count, tasks_created, tasks_assigned, todos_created, todos_assigned | numeric/int | Activity counts |

**Dead / do-not-use columns** (confirmed via ETL source, always 0 or "No Data"):
`pace_score_overall_last_60_days_04_06_combined`, `pace_status_overall_last_60_days`, and the
OLD score family (`overall_pace_score`/`overall_std_pace_status` as raw `pace_1` columns —
attendance-only formula, superseded by the "7:3" model; do not confuse with the VIEW's
same-named aliased columns, which correctly point at the new formula).

---

## 2. Key formulas and rules (plain English)

- **PACE score / sub-score aggregation over a period (month/week) — CRITICAL, caused multiple
  real bugs**: to get a period's PACE score, you must average the 4 capped ingredients
  (`capped_engagement`, `capped_effectiveness`, `capped_discipline`, `capped_working_hours`)
  across the period's rows FIRST, THEN apply the score formula ONCE to those averages. Do
  **not** average each day's already-computed score (`new_pace_score_7_3_event_level`) across
  the period — that gives a materially different (and wrong) number, because the formula
  multiplies terms together (Jensen's-inequality gap, confirmed non-trivial in practice, e.g.
  99 vs 96 for a real employee-month). The formula itself:
  `score = LEAST(100, ROUND(((avg_capped_engagement * avg_capped_effectiveness * avg_capped_working_hours * 7) + (avg_capped_discipline * 3)) * 10))`.
  This same capped-average-first rule applies to sub-score month-wise breakdowns
  (engagement/effectiveness/discipline %) too — average the underlying `capped_*` column, not
  the daily percentage column. `working_pct` is the one exception: it's a linear rescale of
  `capped_working_hours`, so a simple average of the daily percentage is fine for it.
- **Status bucket thresholds** (applies to `overall_std_pace_status` and any bucketing of a
  PACE score you compute yourself): Black < 50, Red 50–64, Amber 65–79, Green 80–100.
- **WFH**: `wfh_status = 'Work From Home'` (rare, ~0.6% of rows). **Leave**:
  `applied_leave_type is not null`. **Visit**: `visit_flag = 'Yes'`. **PS-working**:
  `ps_worked_flag_day = 1` — a day the employee's productivity-suite tracking was actually
  active; the inverse (`= 0`) should generally be EXCLUDED from engagement/effectiveness/
  discipline/working-hours/productive-time averages when the exclusion would materially change
  the result (established threshold: flag it when PS-off days are ≥25% of a period with ≥3
  days in scope).
- **`shift_type`**: 'Standard' vs 'Overtime (OT)'. Use `shift_type='Standard'` for nearly all
  day-grain queries — it's the only slice confirmed to be exactly 1 row per employee-day
  (zero exceptions). Trade-off knowingly accepted: this undercounts productivity minutes by
  ~5–30% on days that also have OT activity, and drops 2.36% of employee-days that are
  OT-only. OT-specific questions should filter `shift_type = 'Overtime (OT)'` on `pace_1`
  directly (this is also the row-level source for OT-specific sub-score isolation, using the
  same row-level `capped_*` columns).
- **Team resolution via `email_access`**: an employee's "team" is found by taking the FIRST
  email in their own `email_access` list, then reverse-looking-up all employees whose list
  *also* contains that exact email. This is EXACT STRING MATCH ONLY, never fuzzy/substring —
  a documented safety requirement (real collisions were found, e.g. one department name is a
  literal substring of an unrelated word, and one employee's email is a literal substring of
  another's). 9 admin/HR emails resolve to nearly the entire company ("universal access,"
  threshold: team size ≥ 90% of all active employees) and must be treated as ambiguous/
  confirmed-before-answering rather than silently returned as "someone's team."
- **`MIN_DAYS_FOR_DELTA = 10`**: never report a month-over-month delta/trend unless both the
  current and prior month have at least 10 Standard-shift days of data for that employee.
- Other reused thresholds: `CHRONIC_LATE_THRESHOLD = 3` late-coming days; `NEW_JOINER_DAYS =
  20` / `POTENTIAL_NEW_JOINER_DAYS = 40` days since `doj`.

---

## 3. Existing capability categories (already handled by deterministic `queries.py` functions —
do NOT re-derive these with novel SQL; route to the existing function/intent instead)

- **Individual employee lookups**: PACE score, status, sub-scores (engagement/effectiveness/
  discipline/working %), overview/"how is X performing", weakest/strongest area, tenure/
  grade/designation.
- **Rankings / best-worst / top-N / bottom-N**: generic metric ranking (pace score, sub-
  scores, attendance flags, productivity minutes) via one shared `metric_ranking()` pattern,
  optionally scoped by department, team, or a specific manager.
- **Trends / month-wise breakdowns**: current-vs-prior-month delta (individual, department,
  team), full multi-month history, weekly breakdown, month-wise breakdown generalized across
  11 metrics (WFH/visit/leave/late/early-leave/deficient-hours/OT days-or-hours/engagement/
  effectiveness/discipline/working %), score-drop / score-improvement rankings.
- **Status filters**: list/count employees currently in a status bucket (Black/Red/Amber/
  Green), status distribution by department, status transitions month-over-month.
- **Day-specific / period-specific counts and lists**: one shared `DAY_FLAGS` pattern covering
  attendance, absence, leave, WFH, visit, late-coming, early-leaving, overtime, deficient-
  hours, defaulter, offline/PS-not-installed, PS-working, zero-productive-minutes, calls,
  meetings, task activity — filterable by yesterday/today/this-week/last-week/an explicit
  date/date-range/rolling window ("last 3 months")/"week of month", with optional department/
  team scope and an "excluding named person" modifier.
- **PS-exclusion queries**: metric averages/sums with PS-off days optionally excluded, plus a
  PS-working ratio and a "who has PS off the most" ranking.
- **Comparisons**: employee-vs-employee, department-vs-department.
- **Team / department aggregates**: department summary, department ranking, team summary,
  "my team" / named-manager team scoping (via `team.py`, exact-match email resolution).
- **Attendance-specific**: chronic-late list, perfect-attendance list, half-day ranking.
- **Leave / calls / visits / WFH / meetings / tasks-and-todos**: per-employee activity lookup
  and org/department rankings for each category.
- **d_score / task-quality scores**: ranking and trend (small eligible subset, NULLs excluded
  rather than shown).
- **Roster / shift / OT**: OT hours ranking, shift-type lookup.
- **Offline / device status**: PS-installed rate by department, offline-attendance ranking.
- **Org info**: grade breakdown, designation breakdown, average tenure, new-joiner detection.
- **New-joiner detection**: via `doj` + the two day thresholds above.

**What is genuinely NOT covered** (would need a novel SQL-fallback path if built): ad hoc
multi-condition combinations not already composed as a function (e.g. "red AND absent
yesterday" cross-filters), true anomaly-detection-style questions, and any casual/vague
phrasing that doesn't map to one of the categories above even after fuzzy-matching.

---

## 4. Explicitly excluded from this document

Deprecated/dead score columns (§1b), the full raw `CREATE VIEW` SQL (only the column list +
plain-English meaning is included above), the ETL source's inline bug-fix history and
commented-out old logic (`pace.py`/`ps_pace.py` — thousands of lines, not reproduced here),
and the exhaustive list of ~300+ individual example questions already tested (only the
category list in §3 is included — a future model should recognize the *category*, not
pattern-match a memorized example list).
