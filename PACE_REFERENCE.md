# PACE_REFERENCE.md — authoritative semantic knowledge for the PACE dataset

**Status**: this is the single authoritative description of what the PACE data
*means*. It is loaded at runtime by `app/llm_nlu.py` and prepended to BOTH
LLM prompts (`classify()` and `extract_build_query()`), so it is not
documentation about the system — it is part of the system.

**How to read this file (important)**

> The question→meaning mappings in this document, and the 124 named intents in
> section 9, are **REFERENCE EXAMPLES, NOT AN ALLOWLIST**. They exist to show
> *how* natural language maps onto this dataset's columns, formulas, and
> business rules. They are **not** the set of questions that may be answered.
>
> **Novel compositions that never appear anywhere in this file are expected to
> work** whenever the data and the rules support them — any
> {entity} × {metric} × {filter + operator} × {grouping} × {period} ×
> {ranking} × {result cardinality} combination is legitimate, whether or not anyone has written it
> down. Conversely, when the data or the rules genuinely do **not** support
> what was asked, the correct behaviour is the existing controlled
> clarification / "I don't have a metric called X" / "which one did you mean?"
> response — **never** a plausible-looking fabricated number, and never
> silently substituting a nearby metric, period, scope, or filter for the one
> that was asked for.

Sources this file is built from (nothing here is invented): the terminology
sheet `Aryan_Task_sheet - pace chatbot.csv`, the 124 intents in
`app/intents.py`, the formulas/thresholds/default filters in
`app/queries.py`, and sections 5–9 of `PROJECT_HANDOFF.md`.

---

## 1. The data

All data lives in PostgreSQL:

| Object | What it is | Grain |
|---|---|---|
| `public.pace_1` | a VIEW over the ETL table `pace_phase_1_table`. The authoritative source. | session-level: one row per employee per worked day per shift type (an employee with both a Standard and an Overtime punch on one day has two rows) |
| `public.pace_chatbot_view` | a purpose-built convenience VIEW | restricted to `shift_type='Standard'`, so ~1 row per employee-day |

Most queries restrict to `shift_type='Standard'` to get a clean
one-row-per-employee-day grain. This is a **knowing, accepted trade-off**: it
undercounts days that mix Standard and Overtime punches, and drops the ~2.36%
of employee-days that are Overtime-only. It has been the rule since before the
generalized query layer existed and must not be silently changed.

The generalized engine (`queries.build_query()`) queries `pace_1` **directly**,
because it needs columns the view does not expose (`ps_worked_flag_day`,
`visit_flag`, `capped_*`).

---

## 2. The PACE score

PACE score is a composite performance score built from four internal "capped"
ingredients — `capped_engagement`, `capped_effectiveness`, `capped_discipline`,
`capped_working_hours`. These are **non-linear transforms** of the user-facing
percentages, not a rescale (e.g. `effectiveness_pct = 50.00` pairs with
`capped_effectiveness = 0.53`, not 0.50).

```
PACE score = LEAST(100, ROUND(
    ((avg_capped_engagement * avg_capped_effectiveness * avg_capped_working_hours * 7)
     + (avg_capped_discipline * 3)) * 10
))
```

A "7:3" model — 7 parts engagement×effectiveness×working-hours, 3 parts
discipline.

### Recomputing it over an arbitrary period — the one rule that must never break

1. Filter to `shift_type='Standard'` rows in the target period/scope, from
   `public.pace_1`.
2. `AVG()` **each of the four capped ingredients separately** across that row set.
3. Apply the formula **once**, to those four averages.

Averaging a per-row precomputed score across a period is **wrong** (Jensen's
inequality) and has been fixed repeatedly in this codebase's history. The
correct form is centralised as `PACE_SCORE_AGG_SQL` /
`PACE_SCORE_FROM_AVGS_SQL` / `pace_status_sql()` in `app/queries.py` and is
reused from 11+ call sites. **Do not reimplement it.**

### PACE status bands

Computed on the (banded) score: **Black** `<50`, **Red** `50–64`,
**Amber** `65–79`, **Green** `>=80`.

"PACE status" / "banding" / "category" / a colour name always means the status,
never the numeric score.

### Distinct score concepts that must never be conflated

| Concept | Meaning |
|---|---|
| `pace_score` | the period aggregate computed capped-average-first, as above |
| `pace_score_day_level` (`new_pace_score_7_3_event_level`) | the per-row/event-level score, averaged over whatever period is asked. A genuinely different metric — not a "single day only" concept. Triggered by the words "day level" / "day-level" / "event level". |
| `last_60_days_new_pace_score_7_3` (stored) | a rolling **60-WORKED-day** window computed by the external ETL, not by this app |
| `build_query()`'s default period | a fixed **60-CALENDAR-day** window ending today |
| `dept_score_60_days_precomputed` | the ETL-precomputed **department** score column. Triggered by the word "precomputed". |

The stored 60-worked-day column and the live 60-calendar-day recomputation do
**not** always agree, especially for employees with meaningful PS-off or
visit-flagged days. That is a known, documented difference, not a bug to
"fix" by quietly swapping one for the other.

---

## 3. Metric vocabulary (NL term → column)

From the terminology sheet, cross-checked against `app/queries.py`.

| Natural-language term(s) | Column / derivation |
|---|---|
| PACE score, score | `last_60_days_new_pace_score_7_3` (stored) / live recomputation (see §2) |
| PACE status, black/red/amber/green, category, banding | `new_pace_status_overall_last_60_days_7_3` / banded score |
| trend, improving, declining, gainer, loser, month-wise score | `pace_score_prev_month`, `pace_score_delta` |
| who dropped / improved the most | `pace_score_delta` |
| day-level / event-level score | `new_pace_score_7_3_event_level` |
| department PACE score | `dept_score_60_days_7_3` |
| department PACE status | `dept_status_60_days_7_3` |
| engagement, engagement % | `engagement_pct` |
| effectiveness, effectiveness % | `effectiveness_pct` |
| discipline, most/least disciplined | `discipline_pct` |
| working %, working hours % | `working_pct` |
| weakest / strongest area | a DERIVED comparison across those four `*_pct` columns |
| capped engagement / effectiveness / discipline | `capped_engagement` / `capped_effectiveness` / `capped_discipline` (internal) |
| date | `worked_day` |
| late, late coming(s), LC | `lc_flag_per_day` |
| early leaving, left early, EL | `el_flag_per_day` |
| deficient hours, DH | `dh_flag_per_day` |
| defaulter | `defaulter_count_per_day` (1 if LC/EL/DH fired) |
| attendance, punched in / absent | `punch_in_ts` |
| chronic late | repeated `lc_flag_per_day` |
| perfect attendance | no LC/EL/DH and `punch_in_ts` present |
| half day | half-day-specific logic |
| WFH, work from home, remote | `wfh_status = 'Work From Home'` |
| leave, took leave | `applied_leave_type is not null` |
| visit, client visit | `visit_flag` ('Yes'/'No') |
| on OT, overtime | `shift_type = 'Overtime (OT)'` |
| PS not installed | `ps_installed_new = 0` |
| PS install rate | dept-level aggregate of `ps_installed_new` |
| offline, marked offline | `offline_attendance_flag` |
| PS working, PS-working ratio | `ps_worked_flag_day = 1` |
| productive time, productive minutes | `productive_and_meeting_min` |
| non-productive | `non_productive_and_whatsapp_min` |
| WhatsApp minutes | `whatsapp_min` |
| tools and mails, email/tools time | `tools_and_mails_min` |
| AI usage, AI minutes | `ai_min` |
| engagement minutes | `engagement_minutes` (**a different metric from `engagement_pct`** — the word "minutes" is the deciding signal) |
| meeting minutes | `meeting_in_min` |
| meeting count, how many meetings | `meeting_count` |
| todos created / assigned | `todos_created` / `todos_assigned` |
| tasks created / assigned | `tasks_created` / `tasks_assigned` (kept as TWO separate metrics — never summed together) |
| task activity | `tasks_created > 0 or tasks_assigned > 0` (a day flag, distinct from the counts) |
| calls made | `total_calls` |
| on-time completion / responsiveness / extension adherence | `ontime_completion_rate` / `responsiveness_score` / `extension_adherence_score` (each ~69% NULL) |
| d score | `d_score` (~84% NULL) |
| department | `dept_name` / `dept_id` |
| manager, reporting manager | `reporting_manager_name` / `reporting_user_id` |
| grade | `grade` |
| designation, job title | `designation` |
| tenure, how long has X worked here, new joiner | derived from `doj` (new joiner < 20 days; potential new joiner < 40 days) |
| shift type | `shift_type` |
| my team, "[Name]'s team" | `email_access` (comma-delimited, exact-match reverse lookup — NOT `reporting_manager_name`) |

### The capped-vs-percentage business rule (deliberate and non-obvious)

| Wording | Metric |
|---|---|
| "engagement" (bare) | `engagement_pct` |
| "engagement %" / "engagement percentage" | `engagement_pct` |
| "capped engagement" | `capped_engagement` |
| "raw capped engagement" | `capped_engagement` |
| **"capped engagement %" / "capped engagement percentage"** | **`engagement_pct`** — intentional, not a mistake to correct |

Working hours only gets the normal-vs-percentage half of this rule (no
`capped_working_hours` metric is exposed to users).

---

## 4. Default filters (the qualifying population)

Unless the question explicitly says otherwise, the population is:

- `shift_type = 'Standard'`
- `visit_flag = 'No'`
- `ps_worked_flag_day = 1`

**WFH is unfiltered by default** — a work-mode filter applies only when the
message actually asks for it.

Explicit overrides are never clobbered: a filter the user stated (e.g. "during
OT") always survives, and the relaxation retry described below only fills in
filters the user did **not** name. Passing `"any"` for
`ps_status`/`visit_status`/`shift_type` drops that clause entirely.

**The sparse-attendance retry**: when a lookup for ONE named employee returns
nothing, the three default filters are relaxed to `"any"` wherever the user did
not explicitly request a value, and the date window is widened. This exists
because an employee can legitimately have `ps_worked_flag_day = 0` on 100% of
their rows, in which case the default population excludes every row they have
while other parts of the product still show real numbers for them.

---

## 5. Period semantics — four DISTINCT concepts

These are genuinely different and must never be conflated:

1. **Calendar days** — "last 40 days" = a fixed calendar window ending today.
2. **Worked days** — the ETL's own rolling 60-worked-day window (stored columns only).
3. **Qualifying Standard rows** — rows where `shift_type='Standard'` AND all four `capped_*` columns are non-null.
4. **Latest-N-qualifying-rows** — the most recent N rows *per employee*, with all filters applied **before** the N rows are selected, regardless of how many calendar days that spans. This is what "last 10 WFH days" means, and taking a calendar window and filtering afterwards would silently undercount.

Period phrases are always re-parsed by the application's own date parsers.
An LLM must pass a period through **as written** and never compute dates.

**"Making progress in PACE"** with no explicit period named defaults to
**latest-20 vs previous-20 qualifying Standard-shift rows**, not a calendar-month
comparison. An explicitly named period still uses the calendar-month comparison.
**"Top N improvers in the last 4 weeks"** is a *different*, pre-existing
mechanism: 4 complete weeks vs the prior 4 complete weeks. The two must not be
conflated.

---

## 6. The semantic CONCEPTS (this is the part that generalizes)

A question is understood as a **query plan**, not as a fixed question shape.
The plan's fields are independent and freely composable.

### 6.1 Entity — the SUBJECT

`employee` | `department` | `rm` (reporting manager) | `company`.

The entity is *what the question is about*. It is **never** the same thing as a
filter that happens to name a department.

### 6.2 Metrics

A flat list drawn from the vocabulary in §3. Several metrics may be asked for
at once. Never substitute a metric that was not asked for; if a named
metric/KPI concept does not exist in this dataset, say so explicitly rather
than answering with the nearest real one.

### 6.3 Filtering — with OPERATORS

A filter is `{field, operator, value}`. Filterable fields today:
`department`, `employee`, `rm`, `grade`, `designation` (plus the four
population filters in §4). Operators: `eq`, `ne`, `in`, `not_in`, `is_null`,
`is_not_null`.

The operator carries any negation, and it is **generic across every field** —
there is no department-specific exclusion rule.

| Wording | Operator |
|---|---|
| in X, from X, within X, for X, only X, just X, restricted to X, limited to X | `eq` |
| excluding X, except X, other than X, outside X, without X, not in X, apart from X, leaving out X, omit X, besides X, barring X | `ne` |
| several named values | `in` / `not_in` |

An **excluded** name is never the subject of the question. "bottom 10 employees
by engagement excluding Sales - Digital Fleet" is a ranking of **employees**
with a department filter `ne 'Sales - Digital Fleet'` — it is *not* a question
about the Sales - Digital Fleet department, and answering it as one silently
inverts what was asked.

### 6.4 Grouping — distinct from both entity and filter

`group_by` breaks the answer down by a dimension:
`employee` | `department` | `rm` | `day` | `month` | `grade` | `designation`.

The single most important distinction in this system:

| Question | Meaning |
|---|---|
| "employees in Sales" | a **FILTER**: `department eq Sales`; entity stays `employee` |
| "show it department wise" | a **GROUP BY**: `group_by = department`; the subject is unchanged |
| "which department has the lowest score?" | a question **about departments**: `entity = department`, no grouping |

The presence of the word "department" decides nothing on its own.

Grouping phrasings: "X wise", "X-wise", "per X", "for each X", "broken down by
X", "grouped by X", "split by X", "segmented by X".

### 6.5 Ranking

`rank_top` (best/highest first), `rank_bottom` (worst/lowest first),
`rank_both_ends`, plus a row count.

**"Highest AND lowest" in one question** = exactly **two rows**: the top-1 and
the bottom-1 of the **same filtered population**. Not two separately filtered
queries, and not an arbitrary N.

Ranking direction is read from the user's own wording. Direction words come in
opposite pairs (best/worst, most/least, improving/declining, top/bottom) and
getting the direction wrong is a silent correctness failure — when the wording
is genuinely ambiguous, ask rather than guess.

### 6.6 Comparison

An explicit A-vs-B comparison between two periods — two dates, or two months.

Comparison does **not** mean "two named employees". It applies equally to:
an employee population, one department, one employee, the whole company, a
date pair, or a month pair.

A comparison can **also carry a grouping**: "compare 11 Sept with 10 Sept,
department wise" = one row per department, with **both dates and the compare
operation preserved**. Adding a breakdown to a comparison must never be read as
"which department did you mean?".

Month-level comparison of the PACE score uses the capped-average-first
methodology of §2. Single-day comparison does not need it (a single day has no
multi-day averaging to get wrong).

### 6.7 Derived operations

- **strongest / weakest area** — which of the four `*_pct` sub-metrics
  (engagement / effectiveness / discipline / working hours) is highest/lowest
  for ONE scope. For a **department**, this is the lowest/highest of the four
  **averages across all employees** in that department — not the widest spread,
  and not "the sub-metric most employees are individually weakest in".
- **trend** — a period-over-period CHANGE, as opposed to a snapshot value.
- **"driving performance"** for a department = the **highest individual PACE
  scorers within that department**. Explicitly *not* month-over-month
  improvement and *not* deviation from the company average.

### 6.7b Result cardinality — THREE states, never two

How many rows the user wants is part of the plan, and it has **three**
distinct states. Collapsing the third into the first silently answers a
different question than the one asked:

| State | The user said | What must happen |
|---|---|---|
| `unspecified` | nothing about how many ("which employees have low engagement?") | a sensible default row count applies — this is the **only** state in which a default may decide anything |
| `exact` | a specific number ("top 10", "bottom 5", "give me 3") | that exact number is used, unchanged, all the way to SQL |
| `unlimited` | the **whole qualifying population** | no row cap at all beyond a pure safety backstop |

`unlimited` is an **explicit request**, not a missing one. Any wording that
means *everybody who qualifies* belongs here — "all employees", "every
employee", "each department", "the whole company", "the entire team",
"everyone", "company-wide", "the full list", "no limit" — and this list is,
as everywhere in this file, **examples of the meaning, not an allowlist of
phrasings**. Never substitute a guessed number (50, 100, 200, 500…) for
"all": a guessed cap is a wrong answer, not a safe one.

Note the distinction between cardinality and **population scope**: "who has
the lowest PACE score *among all employees*" names a population to search,
and still asks for **one** answer. "Show me *all employees* by PACE score"
asks for the whole list.

This is orthogonal to every other plan field: an explicit count or an
explicit "all" composes freely with any entity, metric, filter, grouping and
period, and survives conversational modification until the user changes it.

### 6.8 Conversational modification

A follow-up **patches** the query under discussion. It changes only what the
user actually changed and preserves everything else — metric, period, filters,
grouping, ranking direction, row count, population.

Supported modifications: `add_filter`, `remove_filter`, `replace_filter`,
`change_metric`, `change_group_by`, `change_period`, `change_ranking`,
`change_population`. Several may occur in one message.

Examples of the SHAPES (not an allowlist): "exclude SCM" (add a negative
filter, keep the ranking, metric, direction and row count); "only SCM"
(replace with a positive filter); "remove that filter" / "show everyone again";
"tell me dept wise" (add a grouping); "show discipline instead" (change the
metric); "same thing for August" (change the period); "make it the top 5"
(change ranking direction and row count).

A genuinely **unrelated** new question resets the context — stale filters,
groupings and scopes from an earlier conversation must not leak into it.

---

## 7. Ambiguity and refusal

- Two or more real entities match a name → list them and ask which one.
- A named metric concept does not exist in this dataset → say so, list what is
  available, and do **not** answer with a different metric.
- A direction is genuinely ambiguous → ask, don't pick a side.
- Nothing matches at all → the existing capability/clarification message.

Never fabricate a number, a column, an employee, or a department.

---

## 8. Confirmed business decisions (do not silently reinterpret)

- "Driving performance" = top individual PACE scorers within the department.
- Department weakest/strongest area = lowest/highest of the four averages across all employees in that department.
- "Highest AND lowest" in one question = exactly 2 rows from the same population.
- "Making progress in PACE" (no period named) = latest-20 vs previous-20 qualifying rows.
- "Top N improvers in the last 4 weeks" = 4 complete weeks vs prior 4 complete weeks (a different mechanism).
- The capped-vs-percentage rule of §3.
- Standard-shift-only grain is a knowing trade-off.
- Sticky department context holds only departments the user **explicitly named**.

---

## 9. The 124 named intents — REFERENCE EXAMPLES, NOT AN ALLOWLIST

`app/intents.py` contains 124 named, regex-matched intents. They encode a great
deal of hard-won, validated business logic in their match order and in their
handlers, and they remain the first thing every message is checked against —
when one of them matches, it answers, exactly as before.

**Their role in this document is different**: they are a catalogue of *how this
company talks about its data*, useful as grounding. They are **not** the
boundary of what can be answered. Any valid composition of §6's concepts over
§3's vocabulary is answerable whether or not an intent exists for it.

The intent names, grouped, as a vocabulary reference:

- **PACE score / status**: `emp_pace_score`, `pace_score_best`, `pace_score_worst`, `status_list`, `status_count`, `status_emp`, `status_distribution`, `status_transitions`, `status_improving`
- **Trend / change**: `improving`, `declining`, `emp_trend`, `emp_trend_2month`, `full_trend_emp`, `full_trend_dept`, `full_trend_team`, `dept_trend`, `team_improving`, `score_drop_ranking`, `score_improvement_alltime`, `gainer_loser_ranking`, `subscore_trend_emp`
- **Sub-scores / areas**: `emp_engagement`, `emp_effectiveness`, `emp_discipline`, `emp_working_pct`, `engagement_high`, `engagement_low`, `effectiveness_high`, `effectiveness_low`, `most_disciplined`, `least_disciplined`, `highest_working_pct`, `lowest_working_pct`, `subscore_compare_emp`, `ot_subscore`, `wfh_subscore`
- **Attendance / discipline**: `attendance_best`, `attendance_worst`, `chronic_late`, `perfect_attendance`, `most_late_comings`, `fewest_late_comings`, `emp_late_comings`, `most_early_leavings`, `emp_early_leavings`, `most_deficient_hours`, `emp_deficient_hours`, `deficit_hours_ranking`, `defaulter_ranking`, `emp_attendance_summary`, `half_day_ranking`
- **Leave / WFH / visits / OT**: `leave_who`, `leave_emp_check`, `leave_by_dept`, `zero_leave`, `wfh_emp`, `wfh_ranking`, `wfh_by_dept`, `fewest_wfh`, `visit_ranking`, `visit_emp`, `zero_visit`, `ot_ranking`, `shift_type_emp`, `breakshift_emp`
- **Productivity / activity**: `productive_high`, `productive_low`, `emp_productive_time`, `most_whatsapp`, `emp_whatsapp`, `emp_ai_usage`, `meeting_count_ranking`, `meeting_min_ranking`, `meeting_ratio_emp`, `meeting_had_emp`, `call_most`, `call_fewest`, `call_emp`, `call_duration`, `todos_created_ranking`, `todos_assigned_ranking`, `tasks_created_ranking`, `tasks_assigned_ranking`, `ontime_completion_ranking`, `responsiveness_ranking`, `extension_adherence_ranking`, `d_score_emp`, `d_score_ranking`, `d_score_trend`
- **Department / manager / team**: `dept_avg`, `dept_best`, `dept_worst`, `dept_count`, `dept_summary`, `dept_compare`, `rm_ranking_best`, `rm_ranking_worst`, `team_how_doing`, `team_lowest_scorers`, `team_compare`
- **Comparison**: `day_compare`, `month_compare`, `employee_compare`, `dept_compare`, `team_compare`
- **PS / device**: `ps_worked_emp`, `ps_worked_ranking`, `ps_install_rate`, `ps_exclude_metric`, `ps_ratio_info`, `ps_explain`, `offline_emp`, `offline_ranking`
- **Org info**: `grade_lookup`, `designation_breakdown`, `avg_tenure`, `new_joiners`, `emp_department`, `emp_manager`, `emp_overview`, `employee_day_summary`, `roster_list`
- **Generic**: `average_metric`, `day_count`, `day_list`

---

## 10. Known limitations (do not pretend these work)

- No general multi-metric composition: asking for 2+ metrics at once can collapse to one.
- No general multi-clause composition ("5 lowest PACE employees **and their weakest areas**") beyond the one hand-built "driving performance" two-step.
- Company-wide aggregate questions naming no entity at all still misroute in several shapes.
- `grade`/`designation` are available as filter fields but are not exposed as group-by output grains.
- The rule-based ranking engine does not support the latest-N-qualifying-rows window mode, only a fixed calendar window.
- Free-form generated SQL is the explicit last resort and its replies are always labelled AI-generated/unverified.
