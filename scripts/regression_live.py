"""Live regression suite for the PACE chatbot (item #94).

Runs whole CONVERSATIONS against a running instance (production by default)
and asserts on the replies. Complements the two offline suites:

    scripts/test_query_plan.py     — plan vocabulary/patching, pure python
    scripts/test_plan_pipeline.py  — interceptor + executor, stubbed DB
    scripts/regression_live.py     — this file: end to end against real data

Usage:
    python scripts/regression_live.py                    # production
    python scripts/regression_live.py http://127.0.0.1:8010
    python scripts/regression_live.py <host> --only baseline
    python scripts/regression_live.py <host> --verbose

Categories (the 13 the item #94 mandate requires, plus a baseline group of
previously-confirmed-working behaviours that must not regress):
    baseline, pos_filter, neg_filter, filter_mod, grouping, ranking,
    metric_mod, period_mod, comparison, multiturn, equivalence, unseen,
    neg_generic, context_reset
"""

import json
import re
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

DEFAULT_HOST = "https://pace-chat-bot.onrender.com"


def ask(host, msg, session):
    url = host.rstrip("/") + "/api/chat"
    body = json.dumps({"message": msg, "session_id": session}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return (json.loads(r.read().decode()).get("reply") or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(4)
    return "<<ERROR %s>>" % last


def plain(text):
    """Strip HTML so assertions are written against readable content."""
    t = re.sub(r"<[^>]+>", " | ", text or "")
    t = t.replace("&amp;", "&")
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------

def contains(*needles):
    def _f(reply):
        p = plain(reply).lower()
        missing = [n for n in needles if n.lower() not in p]
        return (not missing), ("missing %r" % missing if missing else "")
    return _f


def excludes(*needles):
    def _f(reply):
        p = plain(reply).lower()
        present = [n for n in needles if n.lower() in p]
        return (not present), ("unexpectedly contains %r" % present if present else "")
    return _f


def all_of(*checks):
    def _f(reply):
        for c in checks:
            ok, why = c(reply)
            if not ok:
                return False, why
        return True, ""
    return _f


def rows_at_least(n):
    def _f(reply):
        count = plain(reply).count("|") // 2
        return count >= n, "too few table cells (%d)" % count
    return _f


NO_FAIL_TEXT = excludes(
    "i couldn't confidently answer",
    "i couldn't find that employee",
    "which department did you mean",
    "i need two employee names",
    "<<error",
)

# --------------------------------------------------------------------------
# The suite: (category, [ (question, assertion_or_None), ... ])
# Each list is ONE conversation, run in its own session.
# --------------------------------------------------------------------------

SUITE = [
    # ---- baseline: previously-confirmed-working behaviour -----------------
    ("baseline", [("top 5 employees by PACE score",
                   all_of(NO_FAIL_TEXT, contains("pace score"), rows_at_least(5)))]),
    ("baseline", [("which department has the lowest PACE score?",
                   all_of(NO_FAIL_TEXT, contains("pace score")))]),
    ("baseline", [("bottom 10 emps based on Engagement",
                   all_of(NO_FAIL_TEXT, contains("engagement")))]),
    ("baseline", [("who is making progress in PACE?",
                   all_of(NO_FAIL_TEXT, contains("improving")))]),
    ("baseline", [("highest and lowest PACE score among WFH employees",
                   all_of(NO_FAIL_TEXT, contains("highest", "lowest")))]),
    ("baseline", [("top 5 WFH employees by pace score",
                   all_of(NO_FAIL_TEXT, contains("wfh")))]),
    ("baseline", [("who's in red in Founders Office?",
                   all_of(NO_FAIL_TEXT, contains("founders office", "red")))]),
    ("baseline", [("top 10 PACE improvers in the last 4 weeks",
                   all_of(NO_FAIL_TEXT, contains("gainer")))]),
    ("baseline", [("average engagement percentage for the whole company",
                   all_of(NO_FAIL_TEXT, contains("company", "engagement")))]),
    ("baseline", [("which employee has the lowest PACE score?", NO_FAIL_TEXT),
                  ("what is their weakest area?",
                   all_of(NO_FAIL_TEXT, contains("weakest")))]),
    ("baseline", [("which department has the lowest PACE score?", NO_FAIL_TEXT),
                  ("how many employees are in that department?",
                   all_of(NO_FAIL_TEXT, contains("employees"))),
                  ("which employee there has the lowest PACE score?", NO_FAIL_TEXT),
                  ("what is their weakest area?", all_of(NO_FAIL_TEXT, contains("weakest"))),
                  ("is that also the weakest area for the department overall?",
                   all_of(NO_FAIL_TEXT, contains("weakest area")))]),

    # ---- 1. positive filters --------------------------------------------
    ("pos_filter", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("only SCM", all_of(NO_FAIL_TEXT, contains("scm")))]),
    ("pos_filter", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("just Annotation", all_of(NO_FAIL_TEXT, contains("annotation")))]),

    # ---- 2. negative filters --------------------------------------------
    ("neg_filter", [("bottom 10 emps based on Engagement", NO_FAIL_TEXT),
                    ("exclude Sales - Digital Fleet",
                     all_of(NO_FAIL_TEXT, contains("excluding sales - digital fleet"),
                            excludes("employee_id", "select ")))]),
    ("neg_filter", [("exclude Sales - Digital Fleet dept and then tell bottom 10 emps based on Engagement",
                     all_of(NO_FAIL_TEXT, contains("excluding sales - digital fleet", "engagement")))]),
    ("neg_filter", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("exclude SCM", all_of(NO_FAIL_TEXT, contains("excluding scm")))]),

    # ---- 3. filter modification -----------------------------------------
    ("filter_mod", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("exclude SCM", contains("excluding scm")),
                    ("only SCM", all_of(NO_FAIL_TEXT, contains("scm only")))]),
    ("filter_mod", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("exclude SCM", contains("excluding scm")),
                    ("remove that filter and show everyone again",
                     all_of(NO_FAIL_TEXT, excludes("excluding scm")))]),
    ("filter_mod", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("exclude SCM", contains("excluding scm")),
                    # both exclusions accumulate into one not_in filter, so
                    # the footer reads "excluding SCM, Annotation"
                    ("also exclude Annotation",
                     all_of(NO_FAIL_TEXT, contains("excluding", "scm", "annotation")))]),

    # ---- 4. grouping -----------------------------------------------------
    ("grouping", [("top 5 employees by PACE score", NO_FAIL_TEXT),
                  ("show it department wise",
                   all_of(NO_FAIL_TEXT, contains("department")))]),
    ("grouping", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                  ("tell me dept wise", all_of(NO_FAIL_TEXT, contains("department")))]),
    ("grouping", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                  ("manager wise", NO_FAIL_TEXT)]),

    # ---- 5. ranking ------------------------------------------------------
    ("ranking", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                 ("exclude SCM", contains("excluding scm")),
                 ("make it the top 5", all_of(NO_FAIL_TEXT, contains("excluding scm")))]),

    # ---- 6. metric modification -----------------------------------------
    ("metric_mod", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("exclude SCM", contains("excluding scm")),
                    ("show me effectiveness instead",
                     all_of(NO_FAIL_TEXT, contains("effectiveness", "excluding scm")))]),

    # ---- 7. period modification -----------------------------------------
    ("period_mod", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                    ("exclude SCM", contains("excluding scm")),
                    ("same thing for August", all_of(NO_FAIL_TEXT, contains("excluding scm")))]),

    # ---- 8. comparison ---------------------------------------------------
    ("comparison", [("compare 11 sept with 10 sept for all employees",
                     all_of(NO_FAIL_TEXT, excludes("i need two employee names")))]),
    ("comparison", [("compare 11 sept with 10 sept for all employees", NO_FAIL_TEXT),
                    ("tell me dept wise",
                     all_of(NO_FAIL_TEXT, contains("comparing"),
                            excludes("which department did you mean")))]),
    ("comparison", [("compare 2 sept vs 7 sept", NO_FAIL_TEXT)]),

    # ---- 9. multi-turn composition --------------------------------------
    ("multiturn", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                   ("exclude Sales - Digital Fleet", contains("excluding sales - digital fleet")),
                   ("make it the top 5", all_of(contains("excluding sales - digital fleet"), NO_FAIL_TEXT)),
                   ("show me discipline instead",
                    all_of(contains("discipline", "excluding sales - digital fleet"), NO_FAIL_TEXT)),
                   ("now tell me dept wise",
                    all_of(contains("department", "excluding sales - digital fleet"), NO_FAIL_TEXT))]),

    # ---- 10. single-turn vs multi-turn equivalence -----------------------
    ("equivalence", [("bottom 10 employees by engagement excluding SCM",
                      all_of(NO_FAIL_TEXT, contains("engagement", "excluding scm")))]),

    # ---- 11. unseen / novel questions ------------------------------------
    ("unseen", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                ("barring Annotation", all_of(NO_FAIL_TEXT, contains("excluding annotation")))]),
    ("unseen", [("top 5 employees by PACE score", NO_FAIL_TEXT),
                ("segmented by department", all_of(NO_FAIL_TEXT, contains("department")))]),
    ("unseen", [("which employees have the weakest discipline outside Ops - Cement?",
                 all_of(NO_FAIL_TEXT, contains("discipline")))]),

    # ---- 12. negation across several different departments ---------------
    ("neg_generic", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                     ("exclude IT-Development", contains("excluding it-development"))]),
    ("neg_generic", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                     ("exclude Annotation", contains("excluding annotation"))]),
    ("neg_generic", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                     ("exclude Channel Sales", contains("excluding channel sales"))]),

    # ---- 13. context reset ----------------------------------------------
    ("context_reset", [("bottom 10 employees by engagement", NO_FAIL_TEXT),
                       ("exclude SCM", contains("excluding scm")),
                       ("which department has the highest engagement?",
                        all_of(NO_FAIL_TEXT, excludes("excluding scm")))]),
    ("context_reset", [("which employee has the lowest PACE score?", NO_FAIL_TEXT),
                       ("what is their weakest area?", NO_FAIL_TEXT),
                       ("which employee has the highest PACE score?",
                        all_of(NO_FAIL_TEXT, contains("pace score")))]),
]


def run_conversation(host, category, steps, verbose=False):
    session = "reg-" + uuid.uuid4().hex[:12]
    out = []
    for question, check in steps:
        reply = ask(host, question, session)
        ok, why = (True, "")
        if check is not None:
            ok, why = check(reply)
        out.append({"category": category, "q": question, "reply": reply, "ok": ok, "why": why})
    return out


def main():
    args = [a for a in sys.argv[1:]]
    verbose = "--verbose" in args
    args = [a for a in args if a != "--verbose"]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1]
        args = args[:i] + args[i + 2:]
    host = args[0] if args else DEFAULT_HOST

    suite = [(c, s) for c, s in SUITE if only is None or c == only]
    print("Running %d conversations against %s\n" % (len(suite), host))

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda cs: run_conversation(host, cs[0], cs[1], verbose), suite))

    flat = [r for conv in results for r in conv]
    failed = [r for r in flat if not r["ok"]]
    by_cat = {}
    for r in flat:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += 1
        if not r["ok"]:
            c[1] += 1

    for conv in results:
        if verbose or any(not r["ok"] for r in conv):
            print("=" * 72)
            for r in conv:
                print("[%s] Q: %s" % ("PASS" if r["ok"] else "FAIL", r["q"]))
                print("     A: %s" % plain(r["reply"])[:700])
                if not r["ok"]:
                    print("     >> %s" % r["why"])

    print("\n" + "=" * 72)
    for cat in sorted(by_cat):
        total, bad = by_cat[cat]
        print("  %-14s %2d checks, %d failed" % (cat, total, bad))
    print("TOTAL: %d checks, %d failed" % (len(flat), len(failed)))

    with open("regression_live_results.json", "w", encoding="utf-8") as fh:
        json.dump(flat, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
