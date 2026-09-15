"""Offline regression suite for app/query_plan.py (item #94).

Runs with NO database and NO LLM — entity resolution is injected, so every
assertion here is deterministic and reproducible in any sandbox. Covers the
13 categories the item #94 mandate requires of the plan layer itself:
positive filters, negative filters, filter modification (add/remove/replace),
grouping, ranking, metric modification, period modification, comparison,
multi-turn composition with per-step invariant assertions, single-turn vs
multi-turn equivalence, unseen phrasings, negation across several real
department names, and context reset.

    python scripts/test_query_plan.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import query_plan as qp  # noqa: E402

# Real department names from the production DB (confirmed present in live
# baseline answers captured before this round's changes).
DEPTS = [
    "Sales - Digital Fleet", "Sales - Enterprise", "SCM", "Annotation",
    "Channel Sales", "Customer Success - CPL", "Customer Success - Enterprise",
    "IT-Development", "IT-Projects", "Ops - Inbound", "Ops - Cement",
    "Founders Office", "Control Tower", "Walle8", "Admin", "Accounts", "CRM",
    "Data Science & Analytics", "Solutions - Enterprise", "Product Support - AT",
]
MANAGERS = ["Nikhil Kumar", "Megha Sharma"]
EMPLOYEES = {"Tanu Mehra": 1001, "Divyansh Sharma": 1002}


def _fake_department(text):
    t = (text or "").lower().strip()
    hits = [d for d in DEPTS if d.lower() == t]
    if hits:
        return hits[0], None
    hits = [d for d in DEPTS if d.lower() in t]
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        # mimic entities.extract_department(): longest wins if one contains
        # the others, otherwise genuinely ambiguous
        hits.sort(key=len, reverse=True)
        if all(h.lower() in hits[0].lower() for h in hits):
            return hits[0], None
        return None, hits
    # loose token-overlap tier, same idea as entities._extract_department_single
    stop = {"the", "and", "of", "for", "-", "&"}
    cands = []
    words = set(re.split(r"[^a-z0-9&]+", t))
    for d in DEPTS:
        toks = [x for x in re.split(r"[\s\-/]+", d.lower()) if len(x) > 2 and x not in stop]
        if toks and any(x in words for x in toks):
            cands.append(d)
    if len(cands) == 1:
        return cands[0], None
    if len(cands) > 1:
        return None, cands
    return None, None


def _fake_rm(text):
    t = (text or "").lower()
    hits = [m for m in MANAGERS if m.lower() in t]
    if len(hits) == 1:
        return hits[0], None
    return None, None


def _fake_employee(text):
    t = (text or "").lower()
    hits = [(n, i) for n, i in EMPLOYEES.items() if n.lower() in t]
    if len(hits) == 1:
        return hits[0][1], None
    return None, None


RESOLVERS = {"department": _fake_department, "rm": _fake_rm, "employee": _fake_employee}

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


def filters_of(text, weak=False):
    f, amb = qp.detect_dimension_filters(text, RESOLVERS, allow_weak_positive=weak)
    return [(x["field"], x["operator"], x["value"]) for x in f], amb


# ---------------------------------------------------------------------------
# 1. NEGATIVE filters — generic across markers AND across departments
# ---------------------------------------------------------------------------

NEG_MARKERS = [
    "exclude {d}", "excluding {d}", "except {d}", "other than {d}",
    "outside {d}", "outside of {d}", "without {d}", "not in {d}",
    "besides {d}", "apart from {d}", "omit {d}", "leave out {d}",
    "barring {d}",
]
for dept in ["Sales - Digital Fleet", "SCM", "Annotation", "IT-Development", "Founders Office"]:
    for tpl in NEG_MARKERS:
        msg = tpl.format(d=dept)
        check("neg %-22r" % msg, filters_of(msg)[0], [("department", "ne", dept)])

# negation inside a fuller single-turn sentence
check(
    "neg single-turn sentence",
    filters_of("exclude Sales - Digital Fleet dept and then tell bottom 10 emps based on Engagement")[0],
    [("department", "ne", "Sales - Digital Fleet")],
)
check(
    "neg mid-sentence",
    filters_of("bottom 10 employees by engagement, excluding SCM")[0],
    [("department", "ne", "SCM")],
)
check(
    "neg rm dimension (generic field, not dept-specific)",
    filters_of("top 5 employees by pace score excluding Nikhil Kumar")[0],
    [("rm", "ne", "Nikhil Kumar")],
)

# ---------------------------------------------------------------------------
# 2. POSITIVE filters
# ---------------------------------------------------------------------------

for tpl in ["only {d}", "just {d}", "solely {d}", "restricted to {d}", "limited to {d}"]:
    for dept in ["SCM", "Annotation", "Sales - Enterprise"]:
        msg = tpl.format(d=dept)
        check("pos %-22r" % msg, filters_of(msg)[0], [("department", "eq", dept)])

# weak positive markers only act in follow-up mode
check("weak positive OFF by default", filters_of("in SCM")[0], [])
check("weak positive ON in follow-up mode", filters_of("in SCM", weak=True)[0],
      [("department", "eq", "SCM")])
check("weak positive: 'from SCM'", filters_of("from SCM", weak=True)[0],
      [("department", "eq", "SCM")])

# a fresh question with normal English "in"/"for" must NOT gain a filter in
# default (non-follow-up) mode — this is the regression guard for the
# whole 123-intent population.
for msg in [
    "top 5 employees by PACE score",
    "who is making progress in PACE?",
    "which department has the lowest PACE score?",
    "what is the PACE score of Tanu Mehra?",
    "average engagement percentage for the whole company",
]:
    check("no false filter %-45r" % msg, filters_of(msg)[0], [])

# ---------------------------------------------------------------------------
# 3. GROUP BY vs FILTER — the critical distinction
# ---------------------------------------------------------------------------

check("group_by 'tell me dept wise'", qp.detect_group_by("tell me dept wise"), "department")
check("group_by 'show it department wise'", qp.detect_group_by("show it department wise"), "department")
check("group_by 'employee wise'", qp.detect_group_by("employee wise"), "employee")
check("group_by 'manager wise'", qp.detect_group_by("manager wise"), "rm")
check("group_by 'day wise'", qp.detect_group_by("day wise"), "day")
check("group_by 'month wise'", qp.detect_group_by("month wise"), "month")
check("group_by 'break it down by department'",
      qp.detect_group_by("break it down by department"), "department")
check("group_by 'per employee'", qp.detect_group_by("give me it per employee"), "employee")
check("group_by 'grouped by manager'", qp.detect_group_by("grouped by manager"), "rm")
check("group_by 'for each department'", qp.detect_group_by("for each department"), "department")
check("group_by 'split by month'", qp.detect_group_by("split by month"), "month")

# a METRIC after "by" is never a grouping
check("group_by not from 'by engagement'", qp.detect_group_by("bottom 10 employees by engagement"), None)
check("group_by not from 'by PACE score'", qp.detect_group_by("top 5 employees by PACE score"), None)
# "employees in Sales" is a filter, not a grouping
check("group_by not from 'employees in Sales - Enterprise'",
      qp.detect_group_by("employees in Sales - Enterprise"), None)
# "which department?" is an entity question, not a grouping
check("group_by not from 'which department has the lowest PACE score?'",
      qp.detect_group_by("which department has the lowest PACE score?"), None)

# ---------------------------------------------------------------------------
# 4. Plan normalization
# ---------------------------------------------------------------------------

p = qp.new_plan(entity="employee", metrics=["engagement_pct"], operation="rank_bottom", limit=10)
check("normalize sets ascending from rank_bottom", p["ascending"], True)
check("normalize keeps limit", p["limit"], 10)
check("normalize bad entity -> employee", qp.new_plan(entity="nonsense")["entity"], "employee")
check("normalize bad operation -> value", qp.new_plan(operation="nonsense")["operation"], "value")
check("normalize bad group_by -> None", qp.new_plan(group_by="colour")["group_by"], None)
check("normalize drops bogus filter field",
      qp.new_plan(filters=[{"field": "colour", "operator": "eq", "value": "red"}])["filters"], [])
check("normalize drops bogus operator",
      qp.new_plan(filters=[{"field": "department", "operator": "sorta", "value": "SCM"}])["filters"], [])
check("normalize clamps limit", qp.new_plan(limit=99999)["limit"], 500)

# ---------------------------------------------------------------------------
# 5. PATCH invariants — unchanged fields literally unchanged
# ---------------------------------------------------------------------------

base = qp.new_plan(
    entity="employee", metrics=["engagement_pct"], operation="rank_bottom",
    limit=10, period_phrase="last 60 days",
)
after = qp.patch(base, {"filters": [{"field": "department", "operator": "ne",
                                     "value": "Sales - Digital Fleet"}]})
for field in ("entity", "metrics", "operation", "limit", "ascending", "period_phrase", "group_by"):
    check("patch(add_filter) preserves %s" % field, after[field], base[field])
check("patch(add_filter) sets filter", [(f["field"], f["operator"], f["value"]) for f in after["filters"]],
      [("department", "ne", "Sales - Digital Fleet")])

# add a SECOND filter on a different field -> both survive
after2 = qp.patch(after, {"filters": [{"field": "rm", "operator": "ne", "value": "Nikhil Kumar"}]})
check("patch(add second filter) keeps both",
      sorted((f["field"], f["operator"], f["value"]) for f in after2["filters"]),
      [("department", "ne", "Sales - Digital Fleet"), ("rm", "ne", "Nikhil Kumar")])

# REPLACE a filter on the same field
after3 = qp.patch(after, {"filters": [{"field": "department", "operator": "eq", "value": "SCM"}]})
check("patch replaces same-field filter",
      [(f["field"], f["operator"], f["value"]) for f in after3["filters"]],
      [("department", "eq", "SCM")])

# REMOVE all filters
after4 = qp.patch(after2, {"filters": qp.CLEAR})
check("patch(CLEAR filters) empties them", after4["filters"], [])
check("patch(CLEAR filters) preserves metric", after4["metrics"], ["engagement_pct"])
check("patch(CLEAR filters) preserves limit", after4["limit"], 10)

# REMOVE one named filter field
after5 = qp.patch(after2, {"filters": [{"field": "rm", "operator": "ne", "value": "Nikhil Kumar"}]},
                  filter_mode="remove")
check("patch(remove one field) leaves the other",
      [(f["field"], f["operator"], f["value"]) for f in after5["filters"]],
      [("department", "ne", "Sales - Digital Fleet")])

# CHANGE GROUPING only
after6 = qp.patch(after, {"group_by": "department"})
check("patch(group_by) sets grouping", after6["group_by"], "department")
for field in ("entity", "metrics", "operation", "limit", "ascending"):
    check("patch(group_by) preserves %s" % field, after6[field], after[field])
check("patch(group_by) preserves filters", after6["filters"], after["filters"])

# CHANGE METRIC only
after7 = qp.patch(after6, {"metrics": ["effectiveness_pct"]})
check("patch(metric) changes metric", after7["metrics"], ["effectiveness_pct"])
check("patch(metric) preserves group_by", after7["group_by"], "department")
check("patch(metric) preserves filters", after7["filters"], after6["filters"])
check("patch(metric) preserves limit", after7["limit"], 10)

# CHANGE RANKING direction + limit
after8 = qp.patch(after7, {"operation": "rank_top", "limit": 5})
check("patch(ranking) flips direction", after8["ascending"], False)
check("patch(ranking) sets limit", after8["limit"], 5)
check("patch(ranking) preserves metric", after8["metrics"], ["effectiveness_pct"])
check("patch(ranking) preserves filters", after8["filters"], after7["filters"])

# ascending alone keeps operation consistent
after9 = qp.patch(after8, {"ascending": True})
check("patch(ascending) syncs operation", after9["operation"], "rank_bottom")

# CHANGE PERIOD only
after10 = qp.patch(after8, {"period_phrase": "August"})
check("patch(period) sets phrase", after10["period_phrase"], "August")
check("patch(period) preserves metric", after10["metrics"], ["effectiveness_pct"])
check("patch(period) preserves filters", after10["filters"], after8["filters"])

# COMBINED modification in one message
after11 = qp.patch(after, {"metrics": ["discipline_pct"], "limit": 3, "operation": "rank_top",
                           "group_by": "department"})
check("combined patch metric", after11["metrics"], ["discipline_pct"])
check("combined patch limit", after11["limit"], 3)
check("combined patch direction", after11["ascending"], False)
check("combined patch group_by", after11["group_by"], "department")
check("combined patch preserves filters", after11["filters"], after["filters"])

# accumulate mode: "also exclude Y" unions into a not_in list
acc = qp.patch(
    qp.new_plan(entity="employee", metrics=["engagement_pct"], operation="rank_bottom", limit=10,
                filters=[{"field": "department", "operator": "ne", "value": "SCM"}]),
    {"filters": [{"field": "department", "operator": "ne", "value": "Annotation"}]},
    filter_mode="accumulate")
check("accumulate unions same-field negatives",
      [(f["field"], f["operator"], f["value"]) for f in acc["filters"]],
      [("department", "not_in", ["SCM", "Annotation"])])
check("accumulate describe()", qp.describe(acc), "excluding SCM, Annotation")

# ---------------------------------------------------------------------------
# 6. Single-turn vs multi-turn EQUIVALENCE
# ---------------------------------------------------------------------------

multi = qp.patch(
    qp.new_plan(entity="employee", metrics=["engagement_pct"], operation="rank_bottom", limit=10),
    {"filters": [{"field": "department", "operator": "ne", "value": "Sales - Digital Fleet"}]},
)
single = qp.new_plan(
    entity="employee", metrics=["engagement_pct"], operation="rank_bottom", limit=10,
    filters=[{"field": "department", "operator": "ne", "value": "Sales - Digital Fleet"}],
)
check("single-turn == multi-turn plan", multi, single)

# ---------------------------------------------------------------------------
# 7. COMPARISON with group_by
# ---------------------------------------------------------------------------

cmp_plan = qp.new_plan(
    entity="employee", operation="compare",
    comparison={"kind": "day", "period_a": "2026-09-10", "period_b": "2026-09-11"},
)
check("comparison normalizes", cmp_plan["comparison"]["kind"], "day")
check("comparison forces operation", cmp_plan["operation"], "compare")
cmp2 = qp.patch(cmp_plan, {"group_by": "department"})
check("comparison + group_by keeps both dates",
      (cmp2["comparison"]["period_a"], cmp2["comparison"]["period_b"]),
      ("2026-09-10", "2026-09-11"))
check("comparison + group_by keeps compare op", cmp2["operation"], "compare")
check("comparison + group_by sets grouping", cmp2["group_by"], "department")

# comparison over a department population, not two named employees
cmp3 = qp.new_plan(entity="department", operation="compare",
                   comparison={"kind": "month", "period_a": "2026-07", "period_b": "2026-08",
                               "group_by": "department"})
check("dept-population comparison", (cmp3["entity"], cmp3["comparison"]["group_by"]),
      ("department", "department"))

# ---------------------------------------------------------------------------
# 8. context_modification classification
# ---------------------------------------------------------------------------

check("ctxmod remove", qp.detect_context_modification("remove that filter"), "remove_filter")
check("ctxmod remove (all depts)",
      qp.detect_context_modification("show all departments again"), "remove_filter")
check("ctxmod remove (everyone)",
      qp.detect_context_modification("include everyone"), "remove_filter")
check("ctxmod add", qp.detect_context_modification("exclude SCM", has_filters=True), "add_filter")
check("ctxmod replace",
      qp.detect_context_modification("only SCM instead", has_filters=True), "replace_filter")
check("ctxmod group_by",
      qp.detect_context_modification("tell me dept wise", has_group_by=True), "change_group_by")
check("ctxmod metric",
      qp.detect_context_modification("show effectiveness instead", has_metric=True), "change_metric")
check("ctxmod ranking",
      qp.detect_context_modification("make it top 5", has_limit=True), "change_ranking")
check("ctxmod none", qp.detect_context_modification("top 5 employees by PACE"), "none")

# ---------------------------------------------------------------------------
# 9. from_query_context bridge (rule-based answers seed a plan too)
# ---------------------------------------------------------------------------

qc = {"last_operation": "rank_bottom", "last_dimension": "employee",
      "last_result_ids": [1, 2, 3], "ascending": True, "metric": ["engagement"],
      "period_phrase": None, "limit": 10, "dept_filter": None}
bridged = qp.from_query_context(qc)
check("bridge entity", bridged["entity"], "employee")
check("bridge operation", bridged["operation"], "rank_bottom")
check("bridge limit", bridged["limit"], 10)
check("bridge ascending", bridged["ascending"], True)
check("bridge metric", bridged["metrics"], ["engagement"])

qc2 = dict(qc, dept_filter={"operator": "ne", "value": "SCM"})
check("bridge dept_filter",
      [(f["field"], f["operator"], f["value"]) for f in qp.from_query_context(qc2)["filters"]],
      [("department", "ne", "SCM")])

# ---------------------------------------------------------------------------
# 10. describe()
# ---------------------------------------------------------------------------

check("describe negative", qp.describe(after), "excluding Sales - Digital Fleet")
check("describe grouping+filter", qp.describe(after6), "grouped by department, excluding Sales - Digital Fleet")

# ---------------------------------------------------------------------------
# 11. Unseen / novel phrasings (not copied from the 123 intents)
# ---------------------------------------------------------------------------

check("unseen: 'leaving out Annotation'", filters_of("leaving out Annotation")[0],
      [("department", "ne", "Annotation")])
check("unseen: 'apart from Control Tower'", filters_of("apart from Control Tower")[0],
      [("department", "ne", "Control Tower")])
check("unseen: 'designation wise'", qp.detect_group_by("give me designation wise numbers"), "designation")
check("unseen: 'segmented by department'",
      qp.detect_group_by("segmented by department please"), "department")

# marker vocabulary found by live testing (round 3)
for msg, want in [
    ("give me the five least effective people, but not anyone in Walle8", "Walle8"),
    ("rank departments by discipline, skipping Admin", "Admin"),
    ("dropping Annotation from that", "Annotation"),
    ("not in SCM", "SCM"),
    ("not from Annotation", "Annotation"),
]:
    check("live-found negation %r" % msg[:40], filters_of(msg)[0],
          [("department", "ne", want)])

# bare "dropped" is a TREND word, never an exclusion
check("bare 'dropped' is not an exclusion",
      filters_of("whose score dropped the most in Annotation")[0], [])

for msg, want in [
    ("restrict that to Annotation", "Annotation"),
    ("limit it to SCM", "SCM"),
    ("narrow it down to Annotation", "Annotation"),
    ("keep it to SCM", "SCM"),
]:
    check("live-found positive %r" % msg[:40], filters_of(msg)[0],
          [("department", "eq", want)])

# ---------------------------------------------------------------------------
# 12. Ambiguity is surfaced, never guessed
# ---------------------------------------------------------------------------

amb_filters, amb = qp.detect_dimension_filters("exclude Sales", RESOLVERS)
check_true("ambiguous dept surfaces candidates", bool(amb) and amb_filters == [],
           "got filters=%r amb=%r" % (amb_filters, amb))

# multi-value negative filter: an explicit LIST is not an ambiguity
check("multi-value not_in from an explicit list",
      filters_of("leaving out Ops - Cement and Annotation")[0],
      [("department", "not_in", ["Ops - Cement", "Annotation"])])
check("multi-value in from an explicit list",
      filters_of("only SCM and Annotation")[0],
      [("department", "in", ["SCM", "Annotation"])])
# a genuinely ambiguous fragment is still surfaced, not guessed
_f, _a = filters_of("exclude Sales")
check_true("genuinely ambiguous name still asks", _f == [] and len(_a) > 1, repr((_f, _a)))

# ---------------------------------------------------------------------------
# 13. Context reset
# ---------------------------------------------------------------------------

fresh = qp.new_plan(entity="department", metrics=["pace_score"], operation="rank_bottom", limit=1)
check_true("fresh plan carries no stale filters", fresh["filters"] == [])
check_true("fresh plan carries no stale grouping", fresh["group_by"] is None)
check_true("patching does not mutate the base plan", base["filters"] == [],
           "base was mutated: %r" % (base["filters"],))


# ---------------------------------------------------------------------------

print("query_plan offline suite: %d passed, %d failed" % (PASSED[0], len(FAILURES)))
for f in FAILURES:
    print("  FAIL " + f)
sys.exit(1 if FAILURES else 0)
