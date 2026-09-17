"""Offline integration suite for the item #94 query-plan interceptor.

Exercises app/main.py's `_handle_query_plan_message()` / `_execute_plan()`
end to end through the real `handle_message()` entry point, with the DATABASE
and the LLM stubbed out — so it asserts on the SQL arguments the pipeline
actually builds (dimension, metrics, limit, ascending, dimension_filters)
rather than on prose. No credentials required.

    python scripts/test_plan_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import entities, llm_nlu, main, queries, query_plan, session_store  # noqa: E402

DEPTS = ["Sales - Digital Fleet", "Sales - Enterprise", "SCM", "Annotation",
         "IT-Development", "Founders Office", "Control Tower", "Ops - Cement",
         "Walle8", "Admin", "Channel Sales"]
MANAGERS = ["Nikhil Kumar", "Megha Sharma"]

CALLS = []


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------

def fake_build_query(dimension, metrics, **kw):
    CALLS.append(dict(fn="build_query", dimension=dimension, metrics=list(metrics), **kw))
    name_col = {"employee": "emp_name", "department": "dept_name",
                "rm": "reporting_manager_name"}.get(dimension, "dept_name")
    id_col = {"employee": "employee_id", "department": "dept_name",
              "rm": "reporting_manager_name"}.get(dimension, "dept_name")
    rows = []
    n = kw.get("limit") or 10
    for i in range(min(n, 3)):
        r = {name_col: "Row%d" % i, id_col: (100 + i) if dimension == "employee" else "Row%d" % i,
             "dept_name": "Annotation", "n_employees": 5}
        for m in metrics:
            r[m] = 50 + i
        rows.append(r)
    return rows


def fake_compare_grouped(period_a, period_b, group_by, **kw):
    CALLS.append(dict(fn="compare_grouped", period_a=period_a, period_b=period_b,
                      group_by=group_by, **kw))
    return ([{"group_key": "Annotation", "val_a": 70, "val_b": 75, "delta": 5, "n_a": 4, "n_b": 4},
             {"group_key": "SCM", "val_a": 60, "val_b": 55, "delta": -5, "n_a": 3, "n_b": 3}],
            "PACE score", "Department")


def fake_extract_department(text, fallback_text=None):
    t = (text or "").lower().strip()
    hits = [d for d in DEPTS if d.lower() == t] or [d for d in DEPTS if d.lower() in t]
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        hits.sort(key=len, reverse=True)
        if all(h.lower() in hits[0].lower() for h in hits):
            return hits[0], None
        return None, hits
    if fallback_text and fallback_text != text:
        return fake_extract_department(fallback_text)
    return None, None


def fake_extract_manager(text, fallback_text=None):
    t = (text or "").lower()
    hits = [m for m in MANAGERS if m.lower() in t]
    if len(hits) == 1:
        return 1, hits[0]
    return None, None


def fake_extract_employee(text, fallback_text=None):
    return None, None


main.queries.build_query = fake_build_query
main.queries.compare_grouped = fake_compare_grouped
queries.build_query = fake_build_query
queries.compare_grouped = fake_compare_grouped
entities.extract_department = fake_extract_department
entities.extract_manager = fake_extract_manager
entities.extract_employee = fake_extract_employee
llm_nlu.classify = lambda *a, **k: None
llm_nlu.extract_build_query = lambda *a, **k: None
main.sql_fallback.answer = lambda *a, **k: None


def fake_metric_ranking(metric_key, *a, **kw):
    CALLS.append(dict(fn="metric_ranking", metric_key=metric_key, **kw))
    # `metric_value` is the column main.format_metric_rows() reads; the
    # metric-keyed column is what some callers read. Both are supplied so
    # this stub works for every caller of metric_ranking().
    return [{"employee_id": 100, "emp_name": "Row0", "dept_name": "Annotation",
             metric_key: 50, "metric_value": 50, "days_counted": 20}]


queries.metric_ranking = fake_metric_ranking
main.queries.metric_ranking = fake_metric_ranking
main._resolve_population_filter = lambda msg: ("shift_type = 'Standard'", None)
main.spellcheck.correct_typos = lambda m: m
entities.get_dept_names = lambda: tuple(DEPTS)
entities.get_manager_names = lambda: tuple(MANAGERS)
entities.get_employee_names = lambda: ()


FAILURES = []
PASSED = [0]


def check(name, got, want):
    if got == want:
        PASSED[0] += 1
    else:
        FAILURES.append("%s\n     got:  %r\n     want: %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASSED[0] += 1
    else:
        FAILURES.append("%s  %s" % (name, detail))


def seed_ranking_plan(session_id, metric="engagement_pct", limit=10, ascending=True, **kw):
    """Stand in for turn 1 (a ranking answer) without needing a real DB."""
    session = session_store.get_session(session_id)
    session_store.set_current_plan(session, query_plan.new_plan(
        entity="employee", metrics=[metric],
        operation="rank_bottom" if ascending else "rank_top",
        limit=limit, ascending=ascending, **kw))
    return session


def ask(session_id, msg):
    CALLS.clear()
    resp = main.handle_message(msg, session_id)
    return resp, list(CALLS)


def last_bq():
    for c in reversed(CALLS):
        if c["fn"] == "build_query":
            return c
    return None


def last_limit():
    """The row cap the pipeline actually asked the DB layer for, whichever
    query function it routed to (item #95 asserts on the cap itself, not on
    which engine produced it)."""
    for c in reversed(CALLS):
        if c.get("limit") is not None:
            return c["limit"]
    return None


def dim_filters(call):
    return [(f["field"], f["operator"], f["value"]) for f in (call.get("dimension_filters") or [])]


# ==========================================================================
# A. NEGATIVE FILTER, MULTI-TURN  (the headline item #93/#94 repro)
# ==========================================================================

seed_ranking_plan("A1")
resp, calls = ask("A1", "exclude Sales - Digital Fleet")
c = last_bq()
check_true("A1 fired the plan path", c is not None, "no build_query call: %r" % (calls,))
if c:
    check("A1 dimension stays employee", c["dimension"], "employee")
    check("A1 metric preserved", c["metrics"], ["engagement_pct"])
    check("A1 limit preserved", c["limit"], 10)
    check("A1 direction preserved", c["ascending"], True)
    check("A1 negative filter applied", dim_filters(c),
          [("department", "ne", "Sales - Digital Fleet")])
check_true("A1 reply mentions the exclusion", "excluding Sales - Digital Fleet" in resp.reply,
           repr(resp.reply[:200]))

# ==========================================================================
# B. NEGATIVE FILTER, SINGLE TURN  (previously INVERTED: scoped INTO the dept)
# ==========================================================================

resp, calls = ask("B1", "exclude Sales - Digital Fleet dept and then tell bottom 10 emps based on Engagement")
c = last_bq()
check_true("B1 fired the plan path", c is not None, repr(calls))
if c:
    check("B1 dimension", c["dimension"], "employee")
    check("B1 metric", c["metrics"], ["engagement_pct"])
    check("B1 limit", c["limit"], 10)
    check("B1 ascending", c["ascending"], True)
    check("B1 negative filter (NOT a scope)", dim_filters(c),
          [("department", "ne", "Sales - Digital Fleet")])
    check("B1 no positive scope leaked", c.get("scope"), None)

# single-turn and multi-turn must agree on everything that matters
seed_ranking_plan("B2")
ask("B2", "exclude Sales - Digital Fleet")
c_multi = last_bq()
ask("B3", "exclude Sales - Digital Fleet dept and then tell bottom 10 emps based on Engagement")
c_single = last_bq()
for k in ("dimension", "metrics", "limit", "ascending", "dimension_filters"):
    check("single==multi: %s" % k, c_single.get(k), c_multi.get(k))

# ==========================================================================
# C. NEGATIVE FILTER GENERICITY — several real departments + an RM
# ==========================================================================

for dept in ["SCM", "Annotation", "IT-Development", "Founders Office", "Control Tower"]:
    seed_ranking_plan("C-" + dept)
    ask("C-" + dept, "exclude " + dept)
    c = last_bq()
    check_true("C generic exclude %s" % dept,
               c is not None and dim_filters(c) == [("department", "ne", dept)],
               repr(c and dim_filters(c)))

seed_ranking_plan("C-rm")
ask("C-rm", "excluding Nikhil Kumar")
c = last_bq()
check_true("C generic exclude RM (not a department special case)",
           c is not None and dim_filters(c) == [("rm", "ne", "Nikhil Kumar")],
           repr(c and dim_filters(c)))

# ==========================================================================
# D. POSITIVE FILTER FOLLOW-UP
# ==========================================================================

seed_ranking_plan("D1")
ask("D1", "only SCM")
c = last_bq()
check_true("D1 positive filter", c is not None and dim_filters(c) == [("department", "eq", "SCM")],
           repr(c and dim_filters(c)))
if c:
    check("D1 metric preserved", c["metrics"], ["engagement_pct"])
    check("D1 direction preserved", c["ascending"], True)

seed_ranking_plan("D2")
ask("D2", "in SCM")
c = last_bq()
check_true("D2 weak positive marker in follow-up mode",
           c is not None and dim_filters(c) == [("department", "eq", "SCM")],
           repr(c and dim_filters(c)))

# ==========================================================================
# E. FILTER MODIFICATION — replace, add, remove
# ==========================================================================

session = seed_ranking_plan("E1")
ask("E1", "exclude SCM")
ask("E1", "only Annotation")
c = last_bq()
check_true("E1 replace filter", c is not None and dim_filters(c) == [("department", "eq", "Annotation")],
           repr(c and dim_filters(c)))

session = seed_ranking_plan("E2")
ask("E2", "exclude SCM")
ask("E2", "excluding Nikhil Kumar")
c = last_bq()
check_true("E2 add a second filter on a different field",
           c is not None and sorted(dim_filters(c)) ==
           [("department", "ne", "SCM"), ("rm", "ne", "Nikhil Kumar")],
           repr(c and dim_filters(c)))

session = seed_ranking_plan("E3")
ask("E3", "exclude SCM")
ask("E3", "remove that filter")
c = last_bq()
check_true("E3 remove filter", c is not None and dim_filters(c) == [], repr(c and dim_filters(c)))
if c:
    check("E3 metric survives removal", c["metrics"], ["engagement_pct"])
    check("E3 limit survives removal", c["limit"], 10)

session = seed_ranking_plan("E4")
ask("E4", "exclude SCM")
ask("E4", "also exclude Annotation")
c = last_bq()
check_true("E4 'also exclude' accumulates into not_in",
           c is not None and dim_filters(c) == [("department", "not_in", ["SCM", "Annotation"])],
           repr(c and dim_filters(c)))

# ==========================================================================
# F. GROUPING
# ==========================================================================

seed_ranking_plan("F1", metric="pace_score", limit=5, ascending=False)
ask("F1", "show it department wise")
c = last_bq()
check_true("F1 group_by changes the output grain", c is not None and c["dimension"] == "department",
           repr(c and c["dimension"]))
if c:
    check("F1 metric preserved by grouping", c["metrics"], ["pace_score"])
    check("F1 direction preserved by grouping", c["ascending"], False)

seed_ranking_plan("F2")
ask("F2", "manager wise")
c = last_bq()
check_true("F2 manager-wise grouping", c is not None and c["dimension"] == "rm",
           repr(c and c["dimension"]))

seed_ranking_plan("F3")
ask("F3", "employee wise")
c = last_bq()
check_true("F3 employee-wise grouping", c is not None and c["dimension"] == "employee",
           repr(c and c["dimension"]))

# grouping composes with an existing filter
seed_ranking_plan("F4")
ask("F4", "exclude SCM")
ask("F4", "now tell me dept wise")
c = last_bq()
check_true("F4 grouping preserves the filter",
           c is not None and c["dimension"] == "department" and dim_filters(c) == [("department", "ne", "SCM")],
           repr(c and (c["dimension"], dim_filters(c))))

# ==========================================================================
# G. RANKING + METRIC + PERIOD MODIFICATION
# ==========================================================================

seed_ranking_plan("G1")
ask("G1", "exclude SCM")
ask("G1", "make it the top 5")
c = last_bq()
check_true("G1 ranking modification", c is not None and c["limit"] == 5 and c["ascending"] is False,
           repr(c and (c["limit"], c["ascending"])))
check_true("G1 filter survives a ranking change",
           c is not None and dim_filters(c) == [("department", "ne", "SCM")],
           repr(c and dim_filters(c)))

seed_ranking_plan("G2")
ask("G2", "exclude SCM")
ask("G2", "show me effectiveness instead")
c = last_bq()
check_true("G2 metric modification", c is not None and c["metrics"] == ["effectiveness_pct"],
           repr(c and c["metrics"]))
check_true("G2 filter survives a metric change",
           c is not None and dim_filters(c) == [("department", "ne", "SCM")],
           repr(c and dim_filters(c)))

seed_ranking_plan("G3")
ask("G3", "exclude SCM and show me August")
c = last_bq()
check_true("G3 combined filter+period in one message",
           c is not None and c["period"] is not None and dim_filters(c) == [("department", "ne", "SCM")],
           repr(c and (c["period"], dim_filters(c))))

# ==========================================================================
# H. COMPARISON + GROUP BY
# ==========================================================================

session = session_store.get_session("H1")
session_store.set_current_plan(session, query_plan.new_plan(
    entity="employee", metrics=["pace_score"], operation="compare",
    comparison={"kind": "day", "period_a": "2026-09-10", "period_b": "2026-09-11"},
))
resp, calls = ask("H1", "tell me dept wise")
cg = next((c for c in calls if c["fn"] == "compare_grouped"), None)
check_true("H1 comparison + grouping reaches compare_grouped", cg is not None, repr(calls))
if cg:
    check("H1 both dates preserved", (cg["period_a"], cg["period_b"]), ("2026-09-10", "2026-09-11"))
    check("H1 grouped by department", cg["group_by"], "department")
check_true("H1 reply is a comparison, not 'which department?'",
           "Comparing" in resp.reply, repr(resp.reply[:200]))

# comparison + grouping + a filter
session = session_store.get_session("H2")
session_store.set_current_plan(session, query_plan.new_plan(
    entity="employee", metrics=["pace_score"], operation="compare",
    comparison={"kind": "day", "period_a": "2026-09-10", "period_b": "2026-09-11"},
))
ask("H2", "tell me dept wise")
resp, calls = ask("H2", "exclude SCM")
cg = next((c for c in calls if c["fn"] == "compare_grouped"), None)
check_true("H2 comparison keeps grouping AND gains a filter",
           cg is not None and cg["group_by"] == "department"
           and [(f["field"], f["operator"], f["value"]) for f in cg["dimension_filters"]]
           == [("department", "ne", "SCM")],
           repr(cg))

# ==========================================================================
# I. NO REGRESSION — fresh questions must NOT be intercepted
# ==========================================================================

NOT_INTERCEPTED = [
    "top 5 employees by PACE score",
    "which department has the lowest PACE score?",
    "who is making progress in PACE?",
    "highest and lowest PACE score among WFH employees",
    "who's in red in Founders Office?",
    "how many employees are in that department?",
    "what is their weakest area?",
    "top 10 PACE improvers in the last 4 weeks",
    "compare 11 sept with 10 sept for all employees",
    "average engagement percentage for the whole company",
]
for msg in NOT_INTERCEPTED:
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("I-fresh"))
    check_true("I not intercepted (no prior plan): %r" % msg, r is None, repr(r and r.reply[:120]))

# ...and still not intercepted even WITH a prior plan, because the rule
# matcher claims them (the 123 intents keep precedence).
seed_ranking_plan("I2")
for msg in NOT_INTERCEPTED:
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("I2"))
    check_true("I not intercepted (with prior plan): %r" % msg, r is None,
               repr(r and r.reply[:120]))

# ==========================================================================
# J. CONTEXT RESET — an unrelated question must not leak stale filters
# ==========================================================================

seed_ranking_plan("J1")
ask("J1", "exclude SCM")
plan_after_filter = session_store.get_current_plan(session_store.get_session("J1"))
check_true("J1 plan holds the filter", bool(plan_after_filter["filters"]))
# an unrelated question the plan layer declines clears the stored plan
main.handle_message("who is making progress in PACE?", "J1")
sess = session_store.get_session("J1")
check_true("J1 stored plan cleared by an unrelated question",
           sess.get("current_plan") is None, repr(sess.get("current_plan")))

# ==========================================================================
# K. UNSEEN / NOVEL phrasings (none of these appear in intents.py or the spec)
# ==========================================================================

seed_ranking_plan("K1")
ask("K1", "leaving out Annotation please")
c = last_bq()
check_true("K1 unseen negation phrasing",
           c is not None and dim_filters(c) == [("department", "ne", "Annotation")],
           repr(c and dim_filters(c)))

seed_ranking_plan("K2", metric="pace_score", limit=5, ascending=False)
ask("K2", "segmented by department")
c = last_bq()
check_true("K2 unseen grouping phrasing", c is not None and c["dimension"] == "department",
           repr(c and c["dimension"]))

seed_ranking_plan("K3")
ask("K3", "barring Ops - Cement, give me the bottom 3")
c = last_bq()
check_true("K3 unseen negation + inline limit change",
           c is not None and c["limit"] == 3
           and dim_filters(c) == [("department", "ne", "Ops - Cement")],
           repr(c and (c["limit"], dim_filters(c))))

# ==========================================================================
# L. Refinements found by LIVE testing (item #94, round 2)
# ==========================================================================

# explicit multi-value exclusion is a not_in filter, not an ambiguity
resp, calls = ask("L1", "top 10 employees by effectiveness, leaving out Ops - Cement and Annotation")
c = last_bq()
check_true("L1 explicit multi-value exclusion",
           c is not None and dim_filters(c) == [("department", "not_in", ["Ops - Cement", "Annotation"])],
           repr(c and dim_filters(c)))

# an explicit interrogative subject sets the entity; "weakest" means ascending
resp, calls = ask("L2", "which managers have the weakest discipline outside Annotation?")
c = last_bq()
check_true("L2 subject sets entity=rm", c is not None and c["dimension"] == "rm",
           repr(c and c["dimension"]))
check_true("L2 'weakest' ranks ascending", c is not None and c["ascending"] is True,
           repr(c and c["ascending"]))
check_true("L2 exclusion still applied",
           c is not None and dim_filters(c) == [("department", "ne", "Annotation")],
           repr(c and dim_filters(c)))

# an incidental dimension word is NOT mistaken for the subject
resp, calls = ask("L3", "exclude Sales - Digital Fleet dept and then tell bottom 10 emps based on Engagement")
c = last_bq()
check_true("L3 'dept' in a filter phrase does not become the entity",
           c is not None and c["dimension"] == "employee", repr(c and c["dimension"]))

# ==========================================================================
# M. Ordering fix (round 3): "what about excluding X" must reach the plan
#    layer, not the older vague-rescope handler, which cannot express a
#    filter operator and silently re-ran the query unfiltered.
# ==========================================================================

session = seed_ranking_plan("M1")
session_store.set_last_list(session, kind="ranking", answer_kind="list",
                            rerun_same=lambda **kw: ("stale rerun", []),
                            rerun_list=lambda **kw: ("stale rerun", []))
resp, calls = ask("M1", "what about excluding everyone in Control Tower")
c = last_bq()
check_true("M1 'what about excluding X' reaches the plan layer",
           c is not None and dim_filters(c) == [("department", "ne", "Control Tower")],
           repr((c and dim_filters(c), resp.reply[:120])))

# ...while a genuinely vague re-scope still belongs to the older handler
session = seed_ranking_plan("M2")
session_store.set_last_list(session, kind="ranking", answer_kind="list",
                            rerun_same=lambda **kw: ("OLD HANDLER", []),
                            rerun_list=lambda **kw: ("OLD HANDLER", []))
resp, calls = ask("M2", "what about last month")
check_true("M2 vague re-scope still uses the older handler",
           "OLD HANDLER" in resp.reply, repr(resp.reply[:160]))

# ==========================================================================
# N. The plan as a STRUCTURED FALLBACK, ahead of free-form generated SQL
#    (round 3): fresh ranking phrasings no rule intent claims used to land
#    on raw SQL, which leaves no state, stranding every follow-up.
# ==========================================================================

for msg, want_dim in [("bottom 8 by discipline", "employee"),
                      ("worst 6 on effectiveness", "employee"),
                      ("which departments have the best engagement?", "department"),
                      ("rank departments by discipline, skipping Admin", "department")]:
    resp, calls = ask("N-" + msg[:12], msg)
    c = last_bq()
    check_true("N fallback answers %r at %s grain" % (msg[:34], want_dim),
               c is not None and c["dimension"] == want_dim,
               repr((c and c["dimension"], resp.reply[:120])))

resp, calls = ask("N-skip", "rank departments by discipline, skipping Admin")
c = last_bq()
check_true("N 'skipping Admin' applies as a filter",
           c is not None and dim_filters(c) == [("department", "ne", "Admin")],
           repr(c and dim_filters(c)))

# ...and the fallback leaves a plan behind, so a follow-up composes
ask("N-chain", "worst 6 on effectiveness")
ask("N-chain", "break it down by department")
c = last_bq()
check_true("N fallback leaves a plan for follow-ups",
           c is not None and c["dimension"] == "department", repr(c and c["dimension"]))

# spelled-out counts
resp, calls = ask("N-word", "give me the five least effective people, but not anyone in Walle8")
c = last_bq()
check_true("N spelled-out limit + separated negation",
           c is not None and c["limit"] == 5
           and dim_filters(c) == [("department", "ne", "Walle8")],
           repr(c and (c["limit"], dim_filters(c))))

# vague input still falls through rather than being force-answered
r = main._plan_fallback_reply("hello there", "hello there", session_store.get_session("N-vague"))
check_true("N vague input still falls through", r is None, repr(r))

# ==========================================================================
# O. Grain guards (round 4): a plural-ranking shape can never be answered as
#    an individual-employee field lookup, nor at the wrong grain, no matter
#    which matcher proposed it.
# ==========================================================================

resp, calls = ask("O1", "top 5 employees by working hours percentage")
check_true("O1 plural ranking is not an individual lookup",
           "couldn't find that employee" not in resp.reply, repr(resp.reply[:160]))
c = last_bq()
check_true("O1 answered at employee grain with a limit",
           c is not None and c["dimension"] == "employee" and c["limit"] == 5,
           repr(c and (c["dimension"], c["limit"])))

resp, calls = ask("O2", "which departments have the best engagement?")
c = last_bq()
check_true("O2 plural department question answered at department grain",
           c is not None and c["dimension"] == "department", repr(c and c["dimension"]))

# a genuine individual lookup must still route to the individual intent
r = main._handle_query_plan_message("what is the PACE score of Tanu Mehra?",
                                    "what is the PACE score of Tanu Mehra?",
                                    session_store.get_session("O3"))
check_true("O3 single-employee lookup untouched by the plan layer", r is None, repr(r))

# a group-by follow-up reaches the plan layer, not the vague-list handler
session = seed_ranking_plan("O4", metric="engagement_pct", limit=10, ascending=False)
session_store.set_last_list(session, kind="ranking", answer_kind="list",
                            rerun_same=lambda **kw: ("OLD HANDLER", []),
                            rerun_list=lambda **kw: ("OLD HANDLER", []))
resp, calls = ask("O4", "employee wise instead")
check_true("O4 group-by follow-up bypasses the vague-list handler",
           "OLD HANDLER" not in resp.reply, repr(resp.reply[:160]))

# adjective metric forms
seed_ranking_plan("O5")
ask("O5", "give me the five least effective people, but not anyone in Walle8")
c = last_bq()
check_true("O5 adjective metric form 'least effective'",
           c is not None and c["metrics"] == ["effectiveness_pct"] and c["limit"] == 5,
           repr(c and (c["metrics"], c["limit"])))

# ==========================================================================
# P. ITEM #95 — an EXPLICIT cardinality is never overridden by a default
# ==========================================================================
#
# The three states asserted end to end, on the `limit` argument the pipeline
# actually hands to build_query(): "all/every/..." -> the safety ceiling,
# an explicit N -> exactly N, and nothing said -> queries.LIMIT. The
# phrasings below are deliberately a MIX of known and never-before-seen
# wordings — the fix is in the representation, so none of them is special.

CEIL = query_plan.UNLIMITED_CEILING

_UNLIMITED_PHRASINGS = [
    "show me all employees by engagement",
    "list every employee's pace score",
    "rank the whole company by discipline",
    "give me the full list of employees by effectiveness",
    "show me each employee's working hours percentage",
    "list the entire team by pace score",
    "show every single employee ranked by discipline",
    "display all staff by engagement with no limit",
]
for i, q in enumerate(_UNLIMITED_PHRASINGS):
    ask("P_unl_%d" % i, q)
    lim = last_limit()
    check_true("P unlimited: %r -> no semantic cap" % q,
               lim is not None and lim >= CEIL, repr(lim))

_EXACT_PHRASINGS = [
    ("top 10 employees by effectiveness", 10),
    ("bottom 5 employees in SCM by engagement", 5),
    ("bottom 10 employees by engagement", 10),
    ("show me the 3 lowest employees on discipline", 3),
    ("top 25 people by working hours", 25),
    ("bottom 7 employees by engagement", 7),
    ("top 300 employees by pace score", 300),
]
for i, (q, want) in enumerate(_EXACT_PHRASINGS):
    ask("P_exact_%d" % i, q)
    check_true("P exact: %r -> limit %d" % (q, want), last_limit() == want, repr(last_limit()))

# No cardinality named at all -> the established default still applies.
# `None` counts as passing here: the query layer's own `lim = limit or LIMIT`
# turns it into queries.LIMIT. What must NEVER appear is the unlimited
# ceiling, which would mean a default had been mistaken for a request.
for i, q in enumerate(["which employees have the lowest engagement",
                       "which employees have the lowest discipline",
                       "bottom employees by engagement"]):
    ask("P_def_%d" % i, q)
    check_true("P unspecified: %r -> default, not unlimited" % q,
               last_limit() in (None, queries.LIMIT), repr(last_limit()))

# A population SCOPE phrase is not a cardinality request.
check("P 'among all employees' is scope, not cardinality",
      entities.wants_unlimited("who has the lowest pace score among all employees"), False)
check("P 'all 30 days' is a period, not a population",
      entities.extract_limit("how did the team do over all 30 days"), None)

# Multi-turn: each state survives, and each replaces the other on request.
seed_ranking_plan("P_ft1", metric="engagement_pct", limit=10, ascending=True)
ask("P_ft1", "now show me all of them")
check_true("P follow-up 'all of them' lifts the previous limit of 10",
           (last_limit() or 0) >= CEIL, repr(last_limit()))

session = seed_ranking_plan("P_ft2", metric="engagement_pct", limit=10, ascending=True)
session_store.set_current_plan(session, query_plan.new_plan(
    entity="employee", metrics=["engagement_pct"], operation="rank_bottom",
    limit_mode="unlimited", ascending=True))
ask("P_ft2", "make it the top 5")
check_true("P follow-up 'top 5' overrides a previous unlimited request",
           last_limit() == 5, repr(last_limit()))

session = seed_ranking_plan("P_ft3", metric="engagement_pct", limit=10, ascending=True)
session_store.set_current_plan(session, query_plan.new_plan(
    entity="employee", metrics=["engagement_pct"], operation="rank_bottom",
    limit_mode="unlimited", ascending=True))
ask("P_ft3", "exclude Annotation")
check_true("P unlimited survives a filter-only follow-up",
           (last_limit() or 0) >= CEIL, repr(last_limit()))

# The LLM's own tri-state field is honoured even when the regex sees nothing.
_saved_extract = llm_nlu.extract_build_query
llm_nlu.extract_build_query = lambda *a, **k: {
    "dimension": "employee", "dimension_name": None, "metrics": ["engagement_pct"],
    "filters": {}, "period_phrase": None, "unrecognized_metric_phrase": None,
    "limit": None, "limit_mode": "unlimited", "operation": "rank_top",
    "group_by": None, "dimension_filters": [], "context_modification": "none",
}
main.llm_nlu.extract_build_query = llm_nlu.extract_build_query
ask("P_llm", "engagement leaderboard for the entire org please")
check_true("P LLM-reported limit_mode=unlimited is honoured",
           (last_limit() or 0) >= CEIL, repr(last_limit()))
llm_nlu.extract_build_query = _saved_extract
main.llm_nlu.extract_build_query = _saved_extract

# ==========================================================================
# Q. FRESH GROUP BY, no prior plan (item #96) — "each department's X"/"X by
# department"/"X across departments"/"all departments' X" must reach the plan
# path on a FIRST message, not only as a follow-up. Before this fix, these
# fell through the plan interceptor (which only fired on group_by in
# follow_up_mode) to classify()'s closed intent vocabulary, which has no
# "grouped view of every department" concept and picked a single-department-
# required intent instead — landing on "Which department did you mean?" even
# though no specific department was named.
# ==========================================================================

Q_GROUPED = [
    ("show each department's engagement, all of them", "engagement_pct"),
    ("engagement by department", "engagement_pct"),
    ("show engagement for every department", "engagement_pct"),
    ("give me each department's engagement", "engagement_pct"),
    ("department-wise engagement for all departments", "engagement_pct"),
    ("list all departments with their engagement", "engagement_pct"),
    ("compare engagement across departments", "engagement_pct"),
]
for msg, metric in Q_GROUPED:
    resp, calls = ask("Q-" + msg[:20], msg)
    c = last_bq()
    check_true("Q fresh group_by: %r" % msg,
               c is not None and c["dimension"] == "department" and c["metrics"] == [metric],
               repr(c))
    if c:
        # A GROUP BY is not a FILTER: the plan must carry no department
        # name filter at all — every department, not one specific one.
        check_true("Q %r carries no department name filter" % msg,
                   not any(f["field"] == "department" for f in (c.get("dimension_filters") or [])),
                   repr(c.get("dimension_filters")))

# Genuine department FILTERS (one specific, named department) must be
# entirely unaffected by this — group_by must NOT fire for these, so they
# still flow through the normal (pre-existing, already-correct) pipeline.
Q_FILTERS_NOT_GROUPED = [
    "employees in Sales - Digital Fleet",
    "show SCM engagement",
    "what is the engagement of the Annotation department?",
]
for msg in Q_FILTERS_NOT_GROUPED:
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("Q-filt-" + msg[:15]))
    check_true("Q genuine dept filter not treated as group_by: %r" % msg, r is None,
               repr(r and r.reply[:120]))

# Existing rule intents (dept_best/dept_worst/dept_avg etc.) still take
# precedence over the new fresh-fire branch — it only fires when rule_free.
seed_ranking_plan("Q-rulewin")  # prior plan present, but irrelevant here
r = main._handle_query_plan_message(
    "which department has the best PACE score?", "which department has the best PACE score?",
    session_store.get_session("Q-rulewin"))
check_true("Q existing rule intent (dept_best) still wins, plan path declines",
           r is None, repr(r and r.reply[:120]))

# Single-turn fresh grouped question == the same request phrased as a
# follow-up to an existing employee-ranking plan (single-turn/multi-turn
# equivalence, same principle as section B).
resp_fresh, _ = ask("Q-equiv-fresh", "engagement by department")
c_fresh = last_bq()
seed_ranking_plan("Q-equiv-multi", metric="engagement_pct")
ask("Q-equiv-multi", "show it department wise")
c_multi = last_bq()
check("Q fresh vs follow-up: dimension", c_fresh and c_fresh["dimension"], c_multi and c_multi["dimension"])
check("Q fresh vs follow-up: metrics", c_fresh and c_fresh["metrics"], c_multi and c_multi["metrics"])

# ==========================================================================
# R. ITEM #97 — THE GROUP-BY-BLIND INTENT SWEEP
#
# Item #96 redirected exactly two old intents (average_metric, dept_compare)
# and only for group_by == "department". The item #97 sweep found the same
# collision class across a whole FAMILY of intents and on the `rm` grouping
# dimension too. Each assertion below is a LIVE-CONFIRMED before/after.
# ==========================================================================

# R1 — single-employee FIELD intents hijacking a grouped question. Before:
# "I couldn't find that employee" (or, worse, an employee-grain ranking).
R1 = [
    ("department wise whatsapp usage", "department", "whatsapp_min"),
    ("whatsapp usage by manager", "rm", "whatsapp_min"),
    ("calls made by department", "department", "total_calls"),
    ("total calls made by department", "department", "total_calls"),
    ("show working hours percentage department wise", "department", "working_pct"),
    ("working hours percentage by manager", "rm", "working_pct"),
    ("ai usage by department", "department", "ai_min"),
    ("show me all departments by pace score", "department", "pace_score"),
    ("average pace score for each manager", "rm", "pace_score"),
]
for msg, dim, metric in R1:
    ask("R1-" + msg[:18], msg)
    c = last_bq()
    check_true("R1 %r -> %s grain, %s" % (msg, dim, metric),
               c is not None and c["dimension"] == dim and c["metrics"] == [metric],
               repr(c and (c["dimension"], c["metrics"])))

# R2 — aggregate/compare intents that drop the grouping. Before: one
# company-wide row, or "I need two employee names to compare".
R2 = [
    ("average pace score by manager", "rm", "pace_score"),
    ("average deficient hours across managers", "rm", "DH"),
    ("compare engagement across managers", "rm", "engagement_pct"),
    ("meeting minutes by department", "department", "meeting_minutes"),
    ("average late comings per department", "department", "LC"),
]
for msg, dim, metric in R2:
    ask("R2-" + msg[:18], msg)
    c = last_bq()
    check_true("R2 %r -> %s grain, %s" % (msg, dim, metric),
               c is not None and c["dimension"] == dim and c["metrics"] == [metric],
               repr(c and (c["dimension"], c["metrics"])))

# R3 — a two-period COMPARISON that also names a grouping now reaches
# compare_grouped() on a FRESH turn (item #94 could only get there via a
# follow-up patch). Before: one company-wide pair of numbers.
R3 = [
    ("compare July and August across departments", "department", "month"),
    ("compare August and September by manager", "rm", "month"),
    ("how did July compare to August department wise", "department", "month"),
    ("compare 2026-08-01 and 2026-08-02 by department", "department", "day"),
]
for msg, gb, kind in R3:
    CALLS.clear()
    main.handle_message(msg, "R3-" + msg[:18])
    cg = next((c for c in CALLS if c["fn"] == "compare_grouped"), None)
    check_true("R3 %r -> grouped comparison (%s, %s)" % (msg, gb, kind),
               cg is not None and cg["group_by"] == gb and cg.get("kind") == kind,
               repr(cg))

# R3b — an UNGROUPED comparison still belongs to the old engines, untouched.
for msg in ("compare July and August", "compare Accounts vs Billing",
            "was august better or july"):
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("R3b-" + msg[:14]))
    check_true("R3b ungrouped comparison untouched: %r" % msg, r is None,
               repr(r and r.reply[:120]))

# R4 — metric REPRESENTABILITY. A metric the plan has no column for must
# make it DECLINE, not answer a PACE-score question nobody asked.
for msg in ("leaves by manager", "visits by department", "wfh days by manager",
            "half days by department"):
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("R4-" + msg[:14]))
    check_true("R4 unrepresentable metric declines: %r" % msg, r is None,
               repr(r and r.reply[:120]))

# ...and a period-over-period CHANGE question is likewise not the plan's to
# answer; the exclusion is threaded into the progress engine instead.
for msg in ("who is improving the most excluding SCM",
            "who is declining the most excluding Annotation",
            "who improved the most excluding IT-Development"):
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("R4b-" + msg[:16]))
    check_true("R4 change-ranking declines in the plan: %r" % msg, r is None,
               repr(r and r.reply[:120]))

# R5 — every guard that keeps the old intents' real purpose intact.
# (a) a named EMPLOYEE keeps the single-employee intent, grouping words or not
entities.extract_employee = lambda t, fallback_text=None: (
    (7, "Aryan Gupta") if "aryan" in (t or "").lower() else (None, None))
for msg in ("whatsapp usage of Aryan Gupta by month",
            "working hours percentage for Aryan Gupta department wise"):
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("R5-" + msg[:16]))
    check_true("R5 named employee keeps its intent: %r" % msg, r is None,
               repr(r and r.reply[:120]))
entities.extract_employee = fake_extract_employee
# (b) grouping BY EMPLOYEE is a no-op and must never strip an intent
for msg in ("what is the engagement percentage for every employee in SCM",):
    ask("R5b", msg)
    c = last_bq()
    check_true("R5 employee-grouping never redirects: %r" % msg, c is None, repr(c))
# (c) a population filter with no grouping still reads as a PACE-score
#     ranking (the item #92 WFH behaviour)
d = main._plan_seed_delta_from_message("top 5 wfh employees", "top 5 wfh employees")
check_true("R5 population filter without grouping stays representable",
           main._plan_metric_is_representable("top 5 wfh employees", d), repr(d))

# R6 — the four newly-expressible activity metrics, each on >1 dimension.
for msg, dim, metric in [
        ("calls made by department", "department", "total_calls"),
        ("sum of calls made per manager", "rm", "total_calls"),
        ("whatsapp minutes by department", "department", "whatsapp_min"),
        ("whatsapp usage by manager", "rm", "whatsapp_min"),
        ("ai usage by department", "department", "ai_min"),
        ("most calls made excluding SCM", "employee", "total_calls")]:
    ask("R6-" + msg[:18], msg)
    c = last_bq()
    check_true("R6 %r -> %s/%s" % (msg, dim, metric),
               c is not None and c["dimension"] == dim and c["metrics"] == [metric],
               repr(c and (c["dimension"], c["metrics"])))
check_true("R6 all four metrics exist in BUILD_QUERY_METRICS",
           all(k in queries.BUILD_QUERY_METRICS
               for k in ("total_calls", "whatsapp_min", "ai_min", "tools_and_mails_min")))

# R7 — the metric-translation gap (item #94's own known limitation): a
# follow-up after a rule-based metric_ranking() answer used to find an
# untranslatable metric key, decline, and let an unrelated engine answer a
# different question. Each pair below is turn 1 (old rule intent) then a
# plan-shaped follow-up that must KEEP the metric.
for first, follow, metric in [
        ("who has the most late comings", "now exclude Annotation", "LC"),
        ("who has the most deficient hours", "exclude IT-Development", "DH"),
        ("who has the highest discipline", "show it department wise", "discipline_pct")]:
    sid = "R7-" + first[:16] + follow[:8]
    ask(sid, first)
    ask(sid, follow)
    c = last_bq()
    check_true("R7 %r -> %r keeps metric %s" % (first, follow, metric),
               c is not None and c["metrics"] == [metric], repr(c and c["metrics"]))

# A metric with NO build_query() equivalent must still make the plan decline.
check("R7 untranslatable metric still declines",
      main._plan_metrics_for_build_query(["d_score"]), ([], False))
# ...and every rule-path metric key that DOES have an equivalent translates.
check("R7 metric_ranking keys translate",
      [main._plan_metrics_for_build_query([k])[0][0]
       for k in ("late_comings", "early_leavings", "deficient_hours_days",
                 "whatsapp_min", "ai_min", "tools_and_mails_min", "productive_min")],
      ["LC", "EL", "DH", "whatsapp_min", "ai_min", "tools_and_mails_min",
       "productive_minutes"])

# R8 — EMPLOYEE-GRAIN ranking intents that drop a grouping. Before: an
# employee ranking with the "by department"/"by manager" silently ignored.
for msg, dim, metric in [
        ("all departments by late comings", "department", "LC"),
        ("most defaulters by department", "department", "defaulter_days"),
        ("meeting count by department", "department", "meeting_count"),
        ("meeting minutes by department", "department", "meeting_minutes"),
        ("which manager has the most late comings", "rm", "LC"),
        ("most whatsapp by department", "department", "whatsapp_min"),
        ("highest engagement by department", "department", "engagement_pct")]:
    ask("R8-" + msg[:18], msg)
    c = last_bq()
    check_true("R8 %r -> %s/%s" % (msg, dim, metric),
               c is not None and c["dimension"] == dim and c["metrics"] == [metric],
               repr(c and (c["dimension"], c["metrics"])))

# ...and the very same intents keep their own (ungrouped) question, including
# the item #92 WFH-filtered ranking, which must still reach metric_ranking().
for msg in ("who has the highest discipline among WFH employees",
            "top 5 wfh employees by pace score",
            "who has the most late comings"):
    CALLS.clear()
    main.handle_message(msg, "R8b-" + msg[:16])
    check_true("R8 ungrouped ranking still on its own engine: %r" % msg,
               any(c["fn"] == "metric_ranking" for c in CALLS), repr([c["fn"] for c in CALLS]))

# R9 — PACE STATUS questions are not the plan's to answer either.
for msg in ("who are the black status employees excluding SCM",
            "how many employees are in red status excluding Annotation"):
    r = main._handle_query_plan_message(msg, msg, session_store.get_session("R9-" + msg[:16]))
    check_true("R9 status question declines in the plan: %r" % msg, r is None,
               repr(r and r.reply[:120]))

# R10 — an EXCLUDED department must never also become the positive scope.
# Found by live-testing R4/R9 above: once the plan correctly declined those
# messages, the rule intents that took them over scoped INTO the excluded
# department ("Who is improving in SCM ... (excluding SCM)", 0 rows;
# "Employees currently Black in SCM"). One guard, in answer_intent(), where
# every rule intent's department scope is resolved.
for msg, dept in [("who is improving the most excluding SCM", "SCM"),
                  ("who are the black status employees excluding Annotation", "Annotation"),
                  ("who is declining the most except IT-Development", "IT-Development"),
                  ("most late comings other than Walle8", "Walle8")]:
    check_true("R10 %r excludes, never scopes into, %s" % (msg, dept),
               dept in main._excluded_department_names(msg, msg), repr(msg))
# ...and a department named POSITIVELY is still a scope.
for msg in ("who all are in black in SCM", "top 5 in Annotation by engagement"):
    check("R10 positive scope unaffected: %r" % msg,
          main._excluded_department_names(msg, msg), set())

# ==========================================================================
# S. ITEM #98 — THE CONVERSATIONAL MODIFICATION MATRIX
#
# One rule, asserted per dimension and per prior-plan SHAPE:
#
#     FINAL PLAN = previous plan + ONLY the fields the follow-up changed
#
# Every case below asserts BOTH halves: the intended field really changed,
# and every other field survived byte for byte (`survived()` diffs the whole
# stored plan, so a field nobody thought to name in an assertion still fails
# the test if a follow-up silently erased it).
# ==========================================================================

IGNORED_PLAN_FIELDS = ("source",)


def plan_of(sid):
    return session_store.get_current_plan(session_store.get_session(sid))


def changed_fields(before, after):
    """Every plan field whose value differs — the whole schema, not a list
    someone maintained by hand."""
    out = set()
    for k in set(before or {}) | set(after or {}):
        if k in IGNORED_PLAN_FIELDS:
            continue
        if (before or {}).get(k) != (after or {}).get(k):
            out.add(k)
    return out


def survived(name, before, after, expect_changed):
    """The core invariant: NOTHING outside `expect_changed` moved.

    A subset rather than an equality because a step can legitimately be a
    no-op for one of the fields it is allowed to touch ("make it the top 7"
    after another exact limit moves `limit` but not `limit_mode`). What must
    never happen is a field the follow-up never mentioned changing, which is
    exactly what the subset test catches; the assertions at each call site
    pin down what DID change.
    """
    got = changed_fields(before, after)
    check_true("S %s -> nothing outside %s changed" % (name, sorted(expect_changed)),
               got <= set(expect_changed) and (bool(got) or not expect_changed),
               "actually changed: %r" % sorted(got))


def seed_compare_plan(sid, group_by=None, metrics=("pace_score",),
                      period_a="2026-09-10", period_b="2026-09-11", kind="day", **kw):
    session = session_store.get_session(sid)
    session_store.set_current_plan(session, query_plan.new_plan(
        entity="company", metrics=list(metrics), operation="compare",
        group_by=group_by,
        comparison={"kind": kind, "period_a": period_a, "period_b": period_b,
                    "group_by": group_by},
        **kw))
    return session


def step(sid, msg):
    """One conversational turn; returns (plan_before, plan_after)."""
    before = copy_plan(plan_of(sid))
    ask(sid, msg)
    return before, copy_plan(plan_of(sid))


def copy_plan(p):
    import copy as _c
    return _c.deepcopy(p) if p else p


# --------------------------------------------------------------------------
# S1. THE REPORTED REPRO + its single-turn twin, at the plan level
# --------------------------------------------------------------------------

seed_compare_plan("S1")
before, after = step("S1", "for all employees")
check_true("S1 comparison survives a population-grain follow-up",
           after is not None and after["operation"] == "compare"
           and after["comparison"]["period_a"] == "2026-09-10"
           and after["comparison"]["period_b"] == "2026-09-11",
           repr(after))
check_true("S1 grain became employee (top level AND the nested mirror)",
           after and after["group_by"] == "employee"
           and after["comparison"]["group_by"] == "employee", repr(after))
check_true("S1 explicit 'all' reached the comparison engine as a cardinality",
           any(c["fn"] == "compare_grouped" and (c.get("limit") or 0) >= CEIL
               for c in CALLS), repr(CALLS))
survived("S1", before, after, {"group_by", "comparison", "limit_mode"})

# --------------------------------------------------------------------------
# S2. EVERY DIMENSION, as a BARE follow-up to a RANKING-shaped plan
# --------------------------------------------------------------------------

S2_RANKING = [
    # (message, fields that may change)
    ("show me effectiveness instead",        {"metrics"}),
    ("engagement and discipline please",     {"metrics"}),
    ("make it the top 5",                    {"limit", "limit_mode", "operation", "ascending"}),
    ("show me all of them",                  {"limit", "limit_mode"}),
    ("department wise",                      {"group_by"}),
    ("per manager",                          {"group_by"}),
    ("only SCM",                             {"filters"}),
    ("excluding Annotation",                 {"filters"}),
    ("for August",                           {"period", "period_phrase"}),
]
for msg, allowed in S2_RANKING:
    sid = "S2-" + msg[:16]
    seed_ranking_plan(sid, metric="engagement_pct", limit=10, ascending=True)
    before, after = step(sid, msg)
    check_true("S2 %r modified the plan at all" % msg,
               after is not None and changed_fields(before, after), repr(after))
    got = changed_fields(before, after)
    check_true("S2 %r touched nothing outside %s" % (msg, sorted(allowed)),
               got <= allowed, "changed: %r" % sorted(got))
    check_true("S2 %r kept the metric unless it named one" % msg,
               "metrics" in allowed or (after and after["metrics"] == ["engagement_pct"]),
               repr(after and after["metrics"]))

# --------------------------------------------------------------------------
# S3. EVERY APPLICABLE DIMENSION, as a BARE follow-up to a COMPARISON plan
# --------------------------------------------------------------------------

S3_COMPARE = [
    ("for all employees",        {"group_by", "comparison", "limit_mode", "limit"}),
    ("department wise",          {"group_by", "comparison"}),
    ("manager wise",             {"group_by", "comparison"}),
    ("across departments",       {"group_by", "comparison"}),
    ("show me engagement instead", {"metrics"}),
    ("just the top 5",           {"limit", "limit_mode", "operation", "ascending", "comparison", "group_by"}),
]
for msg, allowed in S3_COMPARE:
    sid = "S3-" + msg[:16]
    seed_compare_plan(sid, group_by="department")
    before, after = step(sid, msg)
    check_true("S3 %r keeps BOTH periods and the compare operation" % msg,
               after is not None and after["operation"] == "compare"
               and after["comparison"]["period_a"] == "2026-09-10"
               and after["comparison"]["period_b"] == "2026-09-11", repr(after))
    got = changed_fields(before, after)
    check_true("S3 %r touched nothing outside %s" % (msg, sorted(allowed)),
               got <= allowed, "changed: %r" % sorted(got))

# a comparison + a FILTER (positive and negative), each preserving the rest
for msg, want in [("only SCM", ("department", "eq", "SCM")),
                  ("excluding Annotation", ("department", "ne", "Annotation"))]:
    sid = "S3f-" + msg[:12]
    seed_compare_plan(sid, group_by="department")
    before, after = step(sid, msg)
    check_true("S3 comparison + %r keeps the comparison and gains the filter" % msg,
               after and after["operation"] == "compare"
               and after["comparison"]["period_b"] == "2026-09-11"
               and [(f["field"], f["operator"], f["value"]) for f in after["filters"]] == [want],
               repr(after))
    survived("S3 filter %r" % msg, before, after, {"filters"})

# a comparison whose PERIODS change and nothing else
sid = "S3p"
seed_compare_plan(sid, group_by="department")
before, after = step(sid, "now compare 2026-08-01 and 2026-08-02")
check_true("S3 period-pair change keeps the grouping",
           after and after["group_by"] == "department"
           and str(after["comparison"]["period_a"]) == "2026-08-01"
           and str(after["comparison"]["period_b"]) == "2026-08-02", repr(after))
survived("S3 period-pair change", before, after, {"comparison"})

# a MONTH comparison keeps its kind through an unrelated follow-up
sid = "S3m"
seed_compare_plan(sid, group_by="department", kind="month",
                  period_a="2026-07", period_b="2026-08")
before, after = step(sid, "show me engagement instead")
check_true("S3 month comparison keeps kind='month' and both months",
           after and after["comparison"] == {"kind": "month", "period_a": "2026-07",
                                             "period_b": "2026-08", "group_by": "department"},
           repr(after and after["comparison"]))
survived("S3 month metric change", before, after, {"metrics"})

# --------------------------------------------------------------------------
# S4. CHAINS — A->B, A->C, A->B->C, and a 4-step chain
# --------------------------------------------------------------------------

# A->B and A->C from the same A: the two branches must not interfere
seed_ranking_plan("S4-ab", metric="engagement_pct", limit=10, ascending=True)
step("S4-ab", "exclude SCM")
b_before, b_after = step("S4-ab", "make it the top 5")
survived("S4 A->B (limit after filter)", b_before, b_after,
         {"limit", "limit_mode", "operation", "ascending"})
check_true("S4 A->B kept the filter",
           [(f["field"], f["operator"], f["value"]) for f in b_after["filters"]]
           == [("department", "ne", "SCM")], repr(b_after["filters"]))

seed_ranking_plan("S4-ac", metric="engagement_pct", limit=10, ascending=True)
step("S4-ac", "exclude SCM")
c_before, c_after = step("S4-ac", "show me discipline instead")
survived("S4 A->C (metric after filter)", c_before, c_after, {"metrics"})

# A->B->C
seed_ranking_plan("S4-abc", metric="engagement_pct", limit=10, ascending=True)
step("S4-abc", "exclude SCM")
step("S4-abc", "make it the top 5")
p_before, p_after = step("S4-abc", "department wise")
survived("S4 A->B->C (grouping last)", p_before, p_after, {"group_by"})
check_true("S4 A->B->C final plan holds all three modifications",
           p_after["group_by"] == "department" and p_after["limit"] == 5
           and [(f["field"], f["operator"], f["value"]) for f in p_after["filters"]]
           == [("department", "ne", "SCM")], repr(p_after))

# 4 independent sequential modifications, asserted after EVERY step
seed_ranking_plan("S4-4", metric="engagement_pct", limit=10, ascending=True)
for msg, expect in [("exclude Annotation", {"filters"}),
                    ("show me discipline instead", {"metrics"}),
                    ("make it the top 7", {"limit", "limit_mode", "operation", "ascending"}),
                    ("manager wise", {"group_by"})]:
    b, a = step("S4-4", msg)
    survived("S4 4-chain step %r" % msg, b, a, expect)
final = plan_of("S4-4")
check_true("S4 4-chain final plan is the union of all four steps",
           final["group_by"] == "rm" and final["limit"] == 7
           and final["metrics"] == ["discipline_pct"]
           and [(f["field"], f["operator"], f["value"]) for f in final["filters"]]
           == [("department", "ne", "Annotation")], repr(final))

# a chain on a COMPARISON plan
seed_compare_plan("S4-cmp")
step("S4-cmp", "for all employees")
b, a = step("S4-cmp", "department wise")
survived("S4 comparison chain: grain -> grouping", b, a, {"group_by", "comparison"})
b, a = step("S4-cmp", "exclude SCM")
survived("S4 comparison chain: + filter", b, a, {"filters"})
check_true("S4 comparison chain still compares the original two days",
           a["comparison"]["period_a"] == "2026-09-10"
           and a["comparison"]["period_b"] == "2026-09-11", repr(a["comparison"]))

# --------------------------------------------------------------------------
# S5. EXPLICIT REMOVAL vs. OMISSION
# --------------------------------------------------------------------------

seed_ranking_plan("S5-g", metric="engagement_pct", limit=10, ascending=True)
step("S5-g", "department wise")
b, a = step("S5-g", "remove the department grouping")
check_true("S5 explicit removal clears the grouping", a["group_by"] is None, repr(a["group_by"]))
survived("S5 grouping removal", b, a, {"group_by"})

# ...on a COMPARISON plan the nested mirror is cleared too (this is the exact
# resurrection bug the one-slot rule exists to prevent).
seed_compare_plan("S5-cg", group_by="department")
b = copy_plan(plan_of("S5-cg"))
a = query_plan.patch(b, {"group_by": query_plan.CLEAR})
check_true("S5 removal clears the nested comparison mirror too",
           a["group_by"] is None and a["comparison"]["group_by"] is None, repr(a["comparison"]))

seed_ranking_plan("S5-f", metric="engagement_pct", limit=10, ascending=True)
step("S5-f", "exclude SCM")
b, a = step("S5-f", "remove that filter")
check_true("S5 explicit filter removal clears filters", a["filters"] == [], repr(a["filters"]))

# an UNRELATED follow-up clears nothing by omission
seed_ranking_plan("S5-o", metric="engagement_pct", limit=10, ascending=True)
step("S5-o", "exclude SCM")
step("S5-o", "department wise")
b, a = step("S5-o", "show me discipline instead")
check_true("S5 omission never clears: filter+grouping survive a metric change",
           a["group_by"] == "department" and a["filters"], repr(a))
survived("S5 omission", b, a, {"metrics"})

# --------------------------------------------------------------------------
# S6. SINGLE-TURN vs MULTI-TURN EQUIVALENCE (structural plan equality)
# --------------------------------------------------------------------------

def plan_shape(p):
    return {k: v for k, v in (p or {}).items() if k not in IGNORED_PLAN_FIELDS}


EQUIV = [
    # (single-turn message, [turn-1, follow-ups...])
    ("top 5 employees by effectiveness excluding SCM",
     ["top 5 employees by effectiveness", "exclude SCM"]),
    ("engagement by department excluding Annotation",
     ["engagement by department", "exclude Annotation"]),
    ("top 5 employees by effectiveness, department wise",
     ["top 5 employees by effectiveness", "department wise"]),
]
for i, (single, turns) in enumerate(EQUIV):
    ask("S6-single-%d" % i, single)
    ps = plan_of("S6-single-%d" % i)
    for t in turns:
        ask("S6-multi-%d" % i, t)
    pm = plan_of("S6-multi-%d" % i)
    check_true("S6 single-turn produced a plan: %r" % single, ps is not None, repr(ps))
    check("S6 single==multi plan: %r" % single, plan_shape(ps), plan_shape(pm))

# ...and the item #98 headline pair, at the level the plan layer controls:
# the multi-turn form must reach the SAME operation/grain/periods as asking
# for a per-employee comparison in one breath.
seed_compare_plan("S6-cmp")
ask("S6-cmp", "for all employees")
p = plan_of("S6-cmp")
check_true("S6 comparison + grain: per-employee comparison of the same two days",
           p["operation"] == "compare" and p["group_by"] == "employee"
           and (p["comparison"]["period_a"], p["comparison"]["period_b"])
           == ("2026-09-10", "2026-09-11"), repr(p))

# --------------------------------------------------------------------------
# S7. CONTEXT RESET after a chain
# --------------------------------------------------------------------------

seed_ranking_plan("S7", metric="engagement_pct", limit=10, ascending=True)
ask("S7", "exclude SCM")
ask("S7", "department wise")
main.handle_message("who is making progress in PACE?", "S7")
check_true("S7 chain state does not survive an unrelated question",
           session_store.get_session("S7").get("current_plan") is None,
           repr(session_store.get_session("S7").get("current_plan")))

seed_compare_plan("S7c")
ask("S7c", "for all employees")
main.handle_message("who is making progress in PACE?", "S7c")
check_true("S7 comparison chain state does not survive either",
           session_store.get_session("S7c").get("current_plan") is None,
           repr(session_store.get_session("S7c").get("current_plan")))

# --------------------------------------------------------------------------
# S8. NOVEL PHRASINGS — none of these appears in intents.py, in the item #98
#     bug report, or anywhere else in this suite.
# --------------------------------------------------------------------------

# (message, prior plan shape, assertion on the resulting plan)
S8 = [
    ("for every manager", "compare", lambda p: p["group_by"] == "rm"),
    ("across each department please", "compare", lambda p: p["group_by"] == "department"),
    ("for all the departments", "ranking", lambda p: p["group_by"] == "department"),
    ("per individual", "ranking", lambda p: p["group_by"] == "employee"),
    ("narrow it to Walle8", "ranking",
     lambda p: [(f["field"], f["operator"], f["value"]) for f in p["filters"]]
     == [("department", "eq", "Walle8")]),
    ("make it the best 12", "ranking", lambda p: p["limit"] == 12),
    ("expand it to everyone", "ranking", lambda p: p["limit_mode"] == "unlimited"),
    ("swap the metric to effectiveness", "ranking", lambda p: p["metrics"] == ["effectiveness_pct"]),
    ("get rid of the row limit", "ranking", lambda p: p["limit_mode"] == "unspecified"),
    ("clear the exclusions", "ranking_filtered", lambda p: p["filters"] == []),
    ("omit Channel Sales as well", "ranking_filtered",
     lambda p: any(f["operator"] in ("ne", "not_in") for f in p["filters"])),
]
for i, (msg, shape, assertion) in enumerate(S8):
    sid = "S8-%d" % i
    if shape == "compare":
        seed_compare_plan(sid, group_by="department")
    else:
        seed_ranking_plan(sid, metric="engagement_pct", limit=10, ascending=True)
        if shape == "ranking_filtered":
            ask(sid, "exclude SCM")
    before = copy_plan(plan_of(sid))
    ask(sid, msg)
    after = plan_of(sid)
    check_true("S8 novel phrasing %r" % msg, after is not None and assertion(after),
               repr(after and {k: after[k] for k in ("group_by", "limit", "limit_mode",
                                                     "metrics", "filters", "operation")}))
    if after is not None and shape == "compare":
        check_true("S8 %r preserved the comparison" % msg,
                   after["operation"] == "compare"
                   and after["comparison"]["period_a"] == "2026-09-10", repr(after))

# --------------------------------------------------------------------------
# S9. The MECHANISM itself, unit level (no pipeline, no stubs)
# --------------------------------------------------------------------------

# a nested dict field is patched KEY BY KEY, not replaced wholesale
p = query_plan.new_plan(entity="employee", metrics=["pace_score"],
                        population_filters={"shift_type": "Standard", "work_mode": "WFH"})
p2 = query_plan.patch(p, {"population_filters": {"work_mode": "WFO"}})
check("S9 nested dict merges key-wise", p2["population_filters"],
      {"shift_type": "Standard", "work_mode": "WFO"})
p3 = query_plan.patch(p, {"population_filters": {"work_mode": query_plan.CLEAR}})
check("S9 per-key CLEAR removes exactly one key", p3["population_filters"],
      {"shift_type": "Standard"})
p4 = query_plan.patch(p, {"population_filters": query_plan.CLEAR})
check("S9 whole-field CLEAR empties it", p4["population_filters"], {})

# a partial comparison delta keeps the keys it did not name
pc = query_plan.new_plan(operation="compare", metrics=["pace_score"], group_by="department",
                         comparison={"kind": "day", "period_a": "2026-09-10",
                                     "period_b": "2026-09-11", "group_by": "department"})
pc2 = query_plan.patch(pc, {"comparison": {"period_b": "2026-09-12"}})
check("S9 partial comparison delta keeps period_a/kind/group_by", pc2["comparison"],
      {"kind": "day", "period_a": "2026-09-10", "period_b": "2026-09-12",
       "group_by": "department"})

# the grouping slot is single-valued: nested and top-level can never diverge
pc3 = query_plan.patch(pc, {"group_by": "rm"})
check("S9 grouping patch reaches the nested mirror",
      (pc3["group_by"], pc3["comparison"]["group_by"]), ("rm", "rm"))
pc4 = query_plan.normalize({"operation": "compare", "metrics": ["pace_score"],
                            "comparison": {"kind": "day", "period_a": "a", "period_b": "b",
                                           "group_by": "rm"}})
check("S9 a nested-only grouping is lifted to the top level", pc4["group_by"], "rm")

# the grain detector is dimension-generic, not phrase-specific
for phrase, want in [("for all employees", "employee"), ("per manager", "rm"),
                     ("across departments", "department"), ("for each grade", "grade"),
                     ("by designation", "designation"), ("for every person", "employee"),
                     ("among all the teams", "department")]:
    check("S9 grain %r" % phrase, query_plan.detect_population_grain(phrase), want)
check("S9 grain detector says nothing about a metric phrase",
      query_plan.detect_population_grain("by engagement"), None)

# explicit-removal detection is slot-generic
for phrase, want in [("remove the department grouping", {"group_by"}),
                     ("drop the breakdown", {"group_by"}),
                     ("clear the exclusions", {"filters"}),
                     ("get rid of the row limit", {"limit"}),
                     ("forget the date range", {"period"})]:
    check_true("S9 removal %r -> %s" % (phrase, want),
               query_plan.detect_explicit_removals(phrase) >= want,
               repr(query_plan.detect_explicit_removals(phrase)))
check("S9 an ordinary follow-up removes nothing",
      query_plan.detect_explicit_removals("show me effectiveness instead"), set())

# ==========================================================================


# ==========================================================================
# T. ITEM #99 — GENERALIZED SHAPES THAT NO LONGER DEPEND ON classify()
#
# Root cause A: a message no rule regex claims was routed ONLY by
# llm_nlu.classify()'s closed 124-name vocabulary, which cannot decline.
# Root cause B: _plan_fallback_reply() gated "is this a ranking?" on a
# hardcoded verb list.
#
# `llm_nlu.classify` is stubbed to None for this whole file, which is exactly
# the condition these tests need: NOTHING may depend on a classifier guess.
# ==========================================================================

def fake_status_list(statuses, dept_name=None, employee_ids=None, limit=None,
                     dimension_filters=None):
    CALLS.append(dict(fn="status_list", statuses=statuses, dept_name=dept_name,
                      employee_ids=employee_ids, limit=limit,
                      dimension_filters=dimension_filters))
    return [{"employee_id": 100, "emp_name": "Row0", "dept_name": "Annotation",
             "overall_std_pace_status": (statuses or ["Red"])[0]}]


def fake_status_count(statuses, dept_name=None, employee_ids=None, dimension_filters=None):
    CALLS.append(dict(fn="status_count", statuses=statuses, dept_name=dept_name,
                      employee_ids=employee_ids, dimension_filters=dimension_filters))
    return 7


queries.status_list = fake_status_list
queries.status_count = fake_status_count
main.queries.status_list = fake_status_list
main.queries.status_count = fake_status_count


def last_call(fn):
    for c in reversed(CALLS):
        if c["fn"] == fn:
            return c
    return None


# --- T1: the STATUS shape detector, 12 novel phrasings + 8 non-status ------
for phrase, want_statuses, want_kind in [
    ("who's in red in Founders Office?", ["Red"], "list"),
    ("which people are sitting in the amber band right now", ["Amber"], "list"),
    ("anyone flagged black in Annotation?", ["Black"], "list"),
    ("give me the names of everybody currently green", ["Green"], "list"),
    ("list the folks in the red zone", ["Red"], "list"),
    ("show me the amber and red guys in SCM", ["Amber", "Red"], "list"),
    ("how many staff are black company wide", ["Black"], "count"),
    ("count of people in green", ["Green"], "count"),
    ("who all sit in amber in Control Tower", ["Amber"], "list"),
    ("name the employees rated red", ["Red"], "list"),
    ("which team members are black", ["Black"], "list"),
    ("tell me who is green in Walle8", ["Green"], "list"),
]:
    got = query_plan.detect_status_shape(phrase)
    check("T1 status shape %r" % phrase, got,
          {"statuses": want_statuses, "kind": want_kind})

for phrase in [
    "which department has the best PACE score?",
    "compare Accounts vs Billing",
    "top 10 employees by engagement",
    "who is improving the most",
    "what is the red engagement percentage",
    "which department has the most red employees",
    "who moved from red to amber last month",
    "bottom 5 on discipline",
]:
    check("T1 NOT a status shape %r" % phrase, query_plan.detect_status_shape(phrase), None)

# --- T2: the AREA shape detector, 12 novel phrasings + 8 non-area ---------
for phrase, want in [
    ("what is their weakest area?", "weakest"),
    ("what are their weakest areas?", "weakest"),
    ("worst-performing category for that person", "weakest"),
    ("which of the four sub-scores is dragging them down", "weakest"),
    ("where is she strongest across the four components", "strongest"),
    ("what's his best area", "strongest"),
    ("which dimension is the department weakest in", "weakest"),
    ("which aspect is lagging for them", "weakest"),
    ("their standout area", "strongest"),
    ("which sub-metric is poorest for that team", "weakest"),
    ("what is the strongest aspect of their performance", "strongest"),
    ("which category are they struggling in most", "weakest"),
]:
    check("T2 area shape %r" % phrase, query_plan.detect_area_shape(phrase), want)

for phrase in [
    "bottom 10 employees by engagement",
    "who are the weakest employees",
    "top 5 employees by effectiveness",
    "which department has the worst discipline",
    "who is in red in SCM",
    "compare July and August across departments",
    "rank the whole company by discipline",
    "who improved the most",
]:
    check("T2 NOT an area shape %r" % phrase, query_plan.detect_area_shape(phrase), None)

# --- T3: status questions route to the verified status engines -------------
resp, calls = ask("T3a", "who's in red in Founders Office?")
c = last_call("status_list")
check_true("T3a reached status_list", c is not None, repr(calls))
if c:
    check("T3a statuses", c["statuses"], ["Red"])
    check("T3a department scope", c["dept_name"], "Founders Office")

resp, calls = ask("T3b", "which people are sitting in the amber band right now")
c = last_call("status_list")
check_true("T3b reached status_list", c is not None, repr(calls))
if c:
    check("T3b statuses", c["statuses"], ["Amber"])
    check("T3b no spurious department scope", c["dept_name"], None)

resp, calls = ask("T3c", "how many staff are black company wide")
check_true("T3c reached status_count", last_call("status_count") is not None, repr(calls))

resp, calls = ask("T3d", "anyone flagged black in Annotation?")
c = last_call("status_list")
check_true("T3d reached status_list", c is not None, repr(calls))
if c:
    check("T3d department scope", c["dept_name"], "Annotation")

resp, calls = ask("T3e", "show me the amber and red guys in SCM")
c = last_call("status_list")
check_true("T3e reached status_list", c is not None, repr(calls))
if c:
    check("T3e both statuses kept", c["statuses"], ["Amber", "Red"])

resp, calls = ask("T3f", "who is in red, excluding SCM")
c = last_call("status_list")
check_true("T3f reached status_list", c is not None, repr(calls))
if c:
    check("T3f exclusion applied", dim_filters(c), [("department", "ne", "SCM")])
    check("T3f the excluded dept is not also the scope", c["dept_name"], None)

# --- T4: the legacy status phrasing still goes through the rule intent -----
resp, calls = ask("T4a", "who is red status in Founders Office")
c = last_call("status_list")
check_true("T4a legacy phrasing still answered by status_list", c is not None, repr(calls))
if c:
    check("T4a legacy statuses", c["statuses"], ["Red"])
    check("T4a legacy dept scope", c["dept_name"], "Founders Office")

# --- T5: rankings introduced by verbs OUTSIDE the old hardcoded list -------
for sid, msg, want_metric in [
    ("T5a", "I want every single employee sorted by effectiveness, no limit", "effectiveness_pct"),
    ("T5b", "every employee ordered by discipline, no limit", "discipline_pct"),
    ("T5c", "arrange all employees by engagement, no limit", "engagement_pct"),
    ("T5d", "organize the entire company by effectiveness, no limit", "effectiveness_pct"),
    ("T5e", "put every employee in order of discipline, no limit", "discipline_pct"),
    ("T5f", "I need all employees sequenced by engagement with no limit", "engagement_pct"),
]:
    resp, calls = ask(sid, msg)
    c = last_bq()
    check_true("%s fired the plan path (%r)" % (sid, msg), c is not None, repr(resp.reply[:120]))
    if c:
        check("%s metric" % sid, c["metrics"], [want_metric])
        check("%s cardinality is unlimited" % sid, c["limit"], query_plan.UNLIMITED_CEILING)

resp, calls = ask("T5g", "the 12 employees with the lowest discipline")
# whichever engine owns it, the explicit row count must reach the DB layer
# and the answer must be a real ranking, not the help fallback.
check("T5g explicit row count reached the DB layer", last_limit(), 12)
check_true("T5g answered with a ranking", "discipline" in resp.reply.lower(),
           repr(resp.reply[:160]))

# --- T6: an AGGREGATE over the whole population is NOT a list -------------
for sid, msg in [
    ("T6a", "what's the average effectiveness across all employees"),
    ("T6b", "what is the overall engagement for the whole company"),
    ("T6c", "mean discipline across every employee"),
]:
    resp, calls = ask(sid, msg)
    c = last_bq()
    check_true("%s did not become an unlimited list (%r)" % (sid, msg),
               c is None or c.get("limit") != query_plan.UNLIMITED_CEILING,
               repr(c))

# --- T7: an AREA question is never turned into a ranking ------------------
for sid, msg in [
    ("T7a", "what is their weakest area?"),
    ("T7b", "which of the four sub-scores is dragging them down"),
    ("T7c", "worst-performing category for that person"),
    ("T7d", "their standout area"),
    ("T7e", "which aspect is lagging for them"),
]:
    resp, calls = ask(sid, msg)
    c = last_bq()
    check_true("%s the plan layer did not answer with a ranking (%r)" % (sid, msg),
               c is None or c.get("limit") == 1, repr(c))
    d = main._plan_seed_delta_from_message(msg, msg)
    check("%s delta operation is strongest_weakest" % sid, d.get("operation"), "strongest_weakest")

d = main._plan_seed_delta_from_message(
    "which managers have the weakest discipline outside Annotation?", None)
check("T7f 'weakest discipline' is still a ranking direction", d.get("operation"), "rank_bottom")

# --- T8: _execute_plan declines the area operation rather than guessing ---
check("T8 executor declines strongest_weakest",
      main._execute_plan(query_plan.new_plan(entity="employee", operation="strongest_weakest",
                                             metrics=["engagement_pct"]), None), None)

# --- T9: the populations sweep — every dimension still routes -------------
for sid, msg, want_dim in [
    ("T9a", "rank every department by engagement, no limit", "department"),
    ("T9b", "list all managers by discipline with no limit", "rm"),
    ("T9c", "sort every employee by engagement, no limit", "employee"),
]:
    resp, calls = ask(sid, msg)
    c = last_bq()
    check_true("%s fired (%r)" % (sid, msg), c is not None, repr(resp.reply[:120]))
    if c:
        check("%s dimension" % sid, c["dimension"], want_dim)

# --- T10: newly-generalized shapes compose with exclusion ------------------
resp, calls = ask("T10a", "sort every employee by engagement excluding Annotation, no limit")
c = last_bq()
check_true("T10a fired", c is not None, repr(resp.reply[:120]))
if c:
    check("T10a exclusion applied", dim_filters(c), [("department", "ne", "Annotation")])
    check("T10a still unlimited", c["limit"], query_plan.UNLIMITED_CEILING)



# --- T11: the AREA operation, answered deterministically -------------------
# No classifier and no extraction LLM in the path (both are stubbed to None
# for this whole file), so every check here proves the deterministic route.
EMPLOYEES = {"aarna jain": (100, "Aarna Jain"), "rahul kanwaria": (101, "Rahul Kanwaria")}


def fake_extract_employee_named(text, fallback_text=None):
    t = (text or "").lower()
    for k, v in EMPLOYEES.items():
        if k in t:
            return v
    if fallback_text and fallback_text != text:
        return fake_extract_employee_named(fallback_text)
    return None, None


entities.extract_employee = fake_extract_employee_named
main.entities.extract_employee = fake_extract_employee_named

for sid, msg, want_dim, want_name in [
    ("T11a", "worst-performing category for Rahul Kanwaria", "employee", 101),
    ("T11b", "which aspect is lagging for Aarna Jain", "employee", 100),
    ("T11c", "what is Aarna Jain's standout area", "employee", 100),
    ("T11d", "which area is Rahul Kanwaria weakest in", "employee", 101),
    ("T11e", "which dimension is Annotation weakest in", "department", "Annotation"),
    ("T11f", "what is the strongest aspect of SCM", "department", "SCM"),
]:
    resp, calls = ask(sid, msg)
    c = last_bq()
    check_true("%s reached the area engine (%r)" % (sid, msg),
               c is not None and c["metrics"] == main._AREA_METRICS, repr(resp.reply[:140]))
    if c:
        check("%s dimension" % sid, c["dimension"], want_dim)
        check("%s scope" % sid, c["name_filter"], want_name)
    check_true("%s reply names an area" % sid,
               ("weakest area is" in resp.reply or "strongest area is" in resp.reply),
               repr(resp.reply[:140]))

# a PRONOUN resolves against the conversational state, not the wording
sess = session_store.get_session("T11g")
session_store.set_query_context(sess, last_operation="value", last_dimension="employee",
                                last_result_ids=[100], metric=["pace_score"])
session_store.push_context(sess, employee_id=100, employee_name="Aarna Jain")
resp, calls = ask("T11g", "what is their weakest area?")
c = last_bq()
check_true("T11g pronoun resolved to the employee under discussion",
           c is not None and c["dimension"] == "employee" and c["name_filter"] == 100,
           repr(resp.reply[:140]))
check_true("T11g answered with an area", "weakest area is" in resp.reply, repr(resp.reply[:140]))

# ... and the answer leaves the state the dept follow-up reads
qc = session_store.get_query_context(session_store.get_session("T11g")) or {}
check("T11g records the operation", qc.get("last_operation"), "strongest_weakest")
check_true("T11g records which area", (qc.get("metric") or [None])[0] in main._AREA_METRICS,
           repr(qc.get("metric")))

# with NO resolvable scope at all it declines instead of guessing
check("T11h declines with nothing to scope to",
      main._handle_area_shape("what is the weakest area?", "what is the weakest area?",
                              session_store.get_session("T11h")), None)

entities.extract_employee = fake_extract_employee
main.entities.extract_employee = fake_extract_employee



# --- T12: the SINGLE-ENTITY status form ------------------------------------
entities.extract_employee = fake_extract_employee_named
main.entities.extract_employee = fake_extract_employee_named


def fake_status_emp_list(statuses, dept_name=None, employee_ids=None, limit=None,
                         dimension_filters=None):
    CALLS.append(dict(fn="status_list", statuses=statuses, dept_name=dept_name,
                      employee_ids=employee_ids, limit=limit,
                      dimension_filters=dimension_filters))
    return [{"employee_id": (employee_ids or [100])[0], "emp_name": "Aarna Jain",
             "dept_name": "Annotation", "overall_std_pace_status": "Amber"}]


queries.status_list = fake_status_emp_list
main.queries.status_list = fake_status_emp_list

for sid, msg, want_id in [
    ("T12a", "is Rahul Kanwaria in the red band?", 101),
    ("T12b", "is Aarna Jain black right now", 100),
]:
    resp, calls = ask(sid, msg)
    c = last_call("status_list")
    check_true("%s routed to the single-employee status lookup (%r)" % (sid, msg),
               c is not None and c["employee_ids"] == [want_id], repr(resp.reply[:140]))

check("T12c a band word with no resolvable person is not a single-entity lookup",
      query_plan.detect_status_shape("is he red"), None)
check("T12d mentions_status_band ignores a distribution question",
      query_plan.mentions_status_band("which department has the most red employees"), [])
check("T12e mentions_status_band finds the band",
      query_plan.mentions_status_band("is Rahul Kanwaria in the red band?"), ["Red"])

queries.status_list = fake_status_list
main.queries.status_list = fake_status_list
entities.extract_employee = fake_extract_employee
main.entities.extract_employee = fake_extract_employee


print("plan-pipeline offline suite: %d passed, %d failed" % (PASSED[0], len(FAILURES)))
for f in FAILURES:
    print("  FAIL " + f)
sys.exit(1 if FAILURES else 0)
