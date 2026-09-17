"""Item #94 — the normalized QUERY PLAN layer.

WHY THIS MODULE EXISTS
----------------------
Before this round the chatbot had, in effect, one "question shape" per
intent: 123 hand-written regex intents (`app/intents.py`), a closed-
vocabulary intent-name classifier (`llm_nlu.classify()`), and — only if both
of those found nothing — a structured extraction step whose schema had no
concept of a filter OPERATOR, no concept of GROUP BY (as distinct from the
query's subject), and no concept of a COMPARISON. Conversational state was
spread across four loosely-coupled stores (`sticky_context`, `query_context`,
`comparison_entities`, `last_list`), none of which described "the query
currently under discussion" as a single object a follow-up could PATCH.

The concrete symptom (SESSION_HANDOFF.md items #93/#94): "bottom 10 employees
by engagement" → "exclude Sales - Digital Fleet" silently became a
single-department VALUE lookup, because a department name had nowhere to live
in the schema except as the query's primary subject. Item #93 patched that
*inside* `_extraction_llm_reply()` — the 4th of 5 cascade stages — so a rule
match or a `classify()` guess routed around it before it ever ran.

THE MODEL
---------
    entity      — the SUBJECT of the question (employee | department | rm |
                  company). Never conflated with a filter.
    metrics     — flat list of BUILD_QUERY_METRICS keys.
    group_by    — a DIMENSION to break the answer down by. Distinct from
                  `entity`: "employees in Sales" is a FILTER on department;
                  "show it department wise" is a GROUP BY on department;
                  "which department?" is a question about the entity.
    operation   — value | rank_top | rank_bottom | rank_both_ends |
                  strongest_weakest | trend | compare
    ranking     — limit + ascending
    time        — period (start,end) | latest_n_days | period_phrase
    filters     — a LIST of {field, operator, value}, operators
                  eq | ne | in | not_in | is_null | is_not_null. This is the
                  core fix: the operator is carried GENERICALLY for ANY
                  filterable dimension, never as a department-specific
                  special case.
    population_filters — the pre-existing 4-key qualifying-population dict
                  (ps_status / visit_status / work_mode / shift_type) that
                  `queries.build_query()` already understands. Deliberately
                  kept as its own field rather than folded into `filters`,
                  because those four have established default semantics
                  (section 7 of PROJECT_HANDOFF.md) that must not change.
    comparison  — {kind, period_a, period_b, group_by} for day/month compare.
    context_modification — how a follow-up relates to the previous plan.

EVERYTHING IN THIS MODULE IS PURE PYTHON — no DB, no LLM, no imports from
`main`/`queries`/`entities`. Entity resolution is INJECTED by the caller
(`resolvers=`) so the whole module is unit-testable offline; see
`scripts/test_query_plan.py`.
"""

import copy
import re

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ENTITIES = ("employee", "department", "rm", "company")

OPERATIONS = (
    "value", "rank_top", "rank_bottom", "rank_both_ends",
    "strongest_weakest", "trend", "compare",
)

FILTER_OPERATORS = ("eq", "ne", "in", "not_in", "is_null", "is_not_null")

NEGATIVE_OPERATORS = ("ne", "not_in", "is_null")

# Filterable DIMENSION fields (as opposed to the 4 population filters below).
# `column` is the real public.pace_1 column the filter is applied to by
# queries.build_query()'s `dimension_filters` param.
FILTER_FIELDS = {
    "department":  {"column": "dept_name", "resolver": "department"},
    "employee":    {"column": "employee_id", "resolver": "employee"},
    "rm":          {"column": "reporting_manager_name", "resolver": "rm"},
    "grade":       {"column": "grade", "resolver": None},
    "designation": {"column": "designation", "resolver": None},
}

# The 4 pre-existing qualifying-population filters build_query() already
# owns (PROJECT_HANDOFF.md section 7) — listed here only so callers can tell
# the two families apart; their semantics are NOT redefined by this module.
POPULATION_FILTER_KEYS = ("ps_status", "visit_status", "work_mode", "shift_type")

# group_by dimensions. "day"/"month" are TIME groupings (only meaningful for
# trend/compare operations); the other three are entity groupings.
GROUP_BY_DIMENSIONS = ("employee", "department", "rm", "day", "month", "grade", "designation")

# ---------------------------------------------------------------------------
# Item #95: result cardinality is a THREE-state field, not an integer
# ---------------------------------------------------------------------------
#
# `limit: int | None` could only say "N rows" or "the user said nothing",
# and every executor turned the second into queries.LIMIT (10). There is a
# third, semantically different state — "the user explicitly asked for the
# ENTIRE population" — which was being silently folded into the default,
# so "rank the whole company by discipline" answered with 10 rows.
#
# `limit_mode` makes the three states distinct and explicit:
#   "unspecified" — the user named no cardinality; the EXECUTOR's default
#                   applies (this is the only state a default may fill in).
#   "exact"       — the user named N; N is authoritative and must survive
#                   every downstream layer untouched.
#   "unlimited"   — the user explicitly asked for everything; only the
#                   UNLIMITED_CEILING safety backstop applies.
LIMIT_MODES = ("unspecified", "exact", "unlimited")

#: Pure safety backstop against a pathological/unbounded scan. Deliberately
#: far above any realistic population in pace_1 (company headcount is in the
#: hundreds) so it can never act as a disguised semantic default — contrast
#: 10/50/100/200, every one of which is a number a user might actually have
#: meant. Mirrors entities.UNLIMITED.
UNLIMITED_CEILING = 100000

CONTEXT_MODIFICATIONS = (
    "none", "add_filter", "remove_filter", "replace_filter", "change_metric",
    "change_group_by", "change_period", "change_ranking", "change_population",
)


# ---------------------------------------------------------------------------
# Generic filter-operator detection
# ---------------------------------------------------------------------------
#
# The mandate is explicit: "exclude" must NOT be a department-specific special
# case. So the marker regexes below are operator-generic — they say only
# "what follows is a NEGATIVE (or POSITIVE) filter value" — and the field the
# value belongs to is decided by WHICH resolver recognises it. Adding a new
# filterable dimension is therefore a one-line addition to FILTER_FIELDS plus
# a resolver, never a new regex per phrase.

_NEGATION_MARKER = re.compile(
    r"\b(?:exclude|excluding|excluded|except(?:ing)?|other\s+than|outside(?:\s+of)?|"
    r"without|besides|apart\s+from|leave\s+out|leaving\s+out|omit(?:ting)?|"
    # NOTE: "dropping" only, never bare "drop"/"dropped" — "whose score
    # dropped the most in Annotation" is a trend question, not an exclusion.
    r"minus|barring|ignore|ignoring|skip(?:ping)?|dropping|"
    # "but not anyone in Walle8" / "not in X" / "not from X" — the negation
    # word and the preposition can be separated by a quantifier phrase.
    r"but\s+not|"
    r"not\s+(?:in|from|any(?:one|body|\s+of)?(?:\s+(?:in|from))?)"
    r")\b",
    re.IGNORECASE,
)

_POSITIVE_MARKER = re.compile(
    r"\b(?:only|just|solely|"
    r"restrict(?:ed)?(?:\s+(?:it|that|this|them))?\s+to|"
    r"limit(?:ed)?(?:\s+(?:it|that|this|them))?\s+to|"
    r"narrow(?:ed)?(?:\s+(?:it|that|this|them))?(?:\s+down)?\s+to|"
    r"keep(?:\s+(?:it|that|this))?\s+to|"
    r"confined\s+to|within|inside|in|from|for)\b",
    re.IGNORECASE,
)

# Markers that are unambiguous enough to act on in ANY message (not just a
# conversational follow-up). "only"/"in"/"from" are far too common in normal
# English to treat as filter markers on a fresh question — the existing
# scope-resolution machinery already handles those correctly — so they are
# restricted to follow-up mode by the caller.
_STRONG_POSITIVE_MARKER = re.compile(
    r"\b(?:only|just|solely|"
    r"restrict(?:ed)?(?:\s+(?:it|that|this|them))?\s+to|"
    r"limit(?:ed)?(?:\s+(?:it|that|this|them))?\s+to|"
    r"narrow(?:ed)?(?:\s+(?:it|that|this|them))?(?:\s+down)?\s+to|"
    r"keep(?:\s+(?:it|that|this))?\s+to)\b",
    re.IGNORECASE,
)

# Words that may trail a filter value without being part of it.
_VALUE_TRAILING_FILLER = re.compile(
    r"\b(?:dept|depts|department|departments|team|teams|employees?|emps?|staff|people|"
    r"guys|folks|members?|only|please|pls|too|also|as\s+well)\b",
    re.IGNORECASE,
)

# Clause terminators — a filter value never spans one of these.
_CLAUSE_BREAK = re.compile(
    r"[,;.\?!]|\band\s+then\b|\bthen\b|\band\s+(?:tell|show|give|list|rank|display)\b|"
    r"\bbut\b|\bhowever\b",
    re.IGNORECASE,
)

_MAX_VALUE_TOKENS = 8


def _value_span(text, start):
    """Text from `start` up to the first clause break (or end of string)."""
    rest = text[start:]
    m = _CLAUSE_BREAK.search(rest)
    if m:
        rest = rest[:m.start()]
    return rest.strip()


def _candidate_windows(span):
    """Progressively shorter leading windows of `span`, longest first.

    Resolvers scan free text, so handing them a whole trailing clause risks
    matching an entity mentioned later in the sentence for a different reason.
    Trying the longest window first and shrinking gives multi-word names
    ("Sales - Digital Fleet") a chance while still bounding how far a match
    can reach.
    """
    tokens = span.split()
    if not tokens:
        return []
    out = []
    for n in range(min(len(tokens), _MAX_VALUE_TOKENS), 0, -1):
        window = " ".join(tokens[:n])
        window = _VALUE_TRAILING_FILLER.sub("", window).strip(" -–—:")
        if window and window not in out:
            out.append(window)
    return out


def _resolve_window(window, resolvers):
    """Try every configured resolver against `window`.

    Returns (field, value) or None, or the string "ambiguous" plus the
    candidate list as ("__ambiguous__", candidates).
    """
    for field, spec in FILTER_FIELDS.items():
        rname = spec.get("resolver")
        if not rname:
            continue
        fn = (resolvers or {}).get(rname)
        if fn is None:
            continue
        try:
            value, candidates = fn(window)
        except Exception:
            continue
        if candidates:
            return ("__ambiguous__", (field, candidates))
        if value:
            return (field, value)
    return None


def detect_dimension_filters(text, resolvers, allow_weak_positive=False):
    """Operator-aware, field-generic filter extraction.

    Returns (filters, ambiguous_candidates):
      filters  — list of {"field", "operator", "value", "marker"} dicts
      ambiguous_candidates — list of names when a value matched >1 entity
                             (the caller should ask the user to disambiguate)

    `resolvers` — {"department": fn, "employee": fn, "rm": fn}; each fn takes
    a text fragment and returns (value_or_None, candidates_or_None). This is
    the SAME contract `entities.extract_department()` already has, so the
    real resolvers are passed straight through with no adapter.

    `allow_weak_positive` — include the broad positive markers ("in", "from",
    "for", "within"). Off by default: on a fresh question those words are
    normal English and the existing scope machinery already handles them
    correctly; on a bare conversational follow-up ("only SCM", "in SCM")
    they are the whole message and are safe to act on.
    """
    filters = []
    seen = set()
    text = text or ""

    positive_re = _POSITIVE_MARKER if allow_weak_positive else _STRONG_POSITIVE_MARKER

    for marker_re, operator in ((_NEGATION_MARKER, "ne"), (positive_re, "eq")):
        for m in marker_re.finditer(text):
            span = _value_span(text, m.end())
            if not span:
                continue
            for window in _candidate_windows(span):
                hit = _resolve_window(window, resolvers)
                if hit is None:
                    continue
                field, value = hit
                if field == "__ambiguous__":
                    # A resolver reporting several candidates usually means a
                    # genuinely ambiguous name ("exclude Sales") and the user
                    # must be asked. But when EVERY candidate is spelled out
                    # in the window ("leaving out Ops - Cement and
                    # Annotation"), the user named a LIST — that is an
                    # in/not_in filter, not an ambiguity. Resolved here, in
                    # the one generic detector, so it works for any field.
                    amb_field, candidates = value
                    named = sorted((c for c in candidates if c.lower() in window.lower()),
                                   key=lambda c: window.lower().index(c.lower()))
                    if len(named) > 1 and len(named) == len(candidates):
                        filters.append({
                            "field": amb_field,
                            "operator": "not_in" if operator == "ne" else "in",
                            "value": named, "marker": m.group(0).lower(),
                        })
                        break
                    return [], list(candidates)
                key = (field, operator, value)
                if key in seen:
                    break
                # A value already claimed by an EARLIER (negative) marker is
                # never re-claimed by a later positive one — negation is the
                # more specific signal and is scanned first.
                if any(f["field"] == field and f["value"] == value for f in filters):
                    break
                seen.add(key)
                filters.append({
                    "field": field, "operator": operator, "value": value,
                    "marker": m.group(0).lower(),
                })
                break
    return filters, []


# ---------------------------------------------------------------------------
# GROUP BY detection ("dept wise", "per employee", "broken down by manager")
# ---------------------------------------------------------------------------

_GROUP_BY_WORDS = {
    "employee": "employee", "employees": "employee", "emp": "employee",
    "emps": "employee", "person": "employee", "people": "employee",
    "individual": "employee", "head": "employee", "name": "employee",
    "department": "department", "departments": "department", "dept": "department",
    "depts": "department", "team": "department", "teams": "department",
    "manager": "rm", "managers": "rm", "rm": "rm", "rms": "rm",
    "reporting manager": "rm", "supervisor": "rm",
    "day": "day", "days": "day", "date": "day", "dates": "day", "daily": "day",
    "month": "month", "months": "month", "monthly": "month",
    "grade": "grade", "grades": "grade",
    "designation": "designation", "designations": "designation",
}

_GROUP_BY_TOKEN = r"(?:reporting\s+manager|employees?|emps?|person|people|individual|" \
                  r"departments?|depts?|teams?|managers?|rms?|supervisor|" \
                  r"days?|dates?|daily|months?|monthly|grades?|designations?)"

# "<dim> wise" / "<dim>-wise" — the dominant Indian-English phrasing in this
# product's real usage ("tell me dept wise").
_GROUP_BY_WISE = re.compile(r"\b(" + _GROUP_BY_TOKEN + r")[\s\-]*wise\b", re.IGNORECASE)

# "by <dim>" / "per <dim>" / "for each <dim>" / "grouped by <dim>" /
# "broken down by <dim>" / "split by <dim>". Deliberately restricted to the
# dimension vocabulary above so "by engagement" (a METRIC) can never be
# mistaken for a grouping.
_GROUP_BY_PREP = re.compile(
    r"\b(?:group(?:ed)?\s+by|broken\s+down\s+by|break\s+(?:it\s+)?down\s+by|split\s+by|"
    # "each <dim>" with no preposition ("show each department's engagement")
    # is the same grouping request as "for each <dim>" — item #95.
    r"segmented\s+by|for\s+each|each|every|per|by)\s+(" + _GROUP_BY_TOKEN + r")\b",
    re.IGNORECASE,
)

# "all <dim>s" / "across <dim>s" ("list all departments with their
# engagement", "compare engagement across departments") — item #96. Restricted
# to a token list WITHOUT the employee synonyms deliberately: "all employees"/
# "across employees" already means something else entirely (the item #95
# UNLIMITED-cardinality signal on the default employee entity, not a request
# to GROUP employees by employee, which would be a meaningless no-op grouping
# that risks colliding with that unrelated, already-verified mechanism).
_GROUP_BY_TOKEN_NO_EMP = r"(?:reporting\s+manager|departments?|depts?|teams?|managers?|rms?|" \
                         r"supervisor|days?|dates?|daily|months?|monthly|grades?|designations?)"
_GROUP_BY_ALL_ACROSS = re.compile(
    r"\b(?:all|across)\s+(" + _GROUP_BY_TOKEN_NO_EMP + r")\b", re.IGNORECASE)


#: Item #98 — POPULATION GRAIN.
#
# `detect_group_by()` above answers "did this message ask for a BREAKDOWN?".
# It deliberately refuses the employee vocabulary in its `all/across` branch
# (see the comment on _GROUP_BY_TOKEN_NO_EMP) because "all employees" is also
# the item #95 UNLIMITED-cardinality signal.
#
# But a message can state the GRAIN the answer should be produced at without
# asking for a "breakdown" in those words — "for all employees", "per
# employee", "for each manager", "across departments". On a RANKING plan whose
# entity is already employee that grain is a no-op, which is why nothing
# depended on it before. On a COMPARISON plan it is the whole difference
# between one company-wide pair of numbers and one row per employee, and it
# was unrepresentable: the grain had nowhere to go in the delta at all.
#
# This is the same dimension vocabulary as the grouping detector, over the
# FULL dimension set including employee, and it is consulted only when
# building a follow-up's delta (never to decide whether a FRESH question is
# intercepted) — so no message that works today changes route because of it.
_POPULATION_GRAIN = re.compile(
    r"\b(?:for|across|among|amongst|over|by|per|of)\s+"
    r"(?:all\s+(?:the\s+)?|every\s+|each\s+|the\s+(?:whole|entire|full)\s+set\s+of\s+)?"
    r"(" + _GROUP_BY_TOKEN + r")\b",
    re.IGNORECASE,
)


def detect_population_grain(text):
    """The DIMENSION a message says the answer should be produced at, or None.

    Field-generic over GROUP_BY_DIMENSIONS — "for all employees", "per
    manager", "across departments" and "for each grade" are all the same
    signal about the same slot, and none of them is special-cased.
    """
    text = text or ""
    m = _POPULATION_GRAIN.search(text)
    if not m:
        return None
    word = re.sub(r"\s+", " ", m.group(1).strip().lower())
    return _GROUP_BY_WORDS.get(word)


def detect_group_by(text):
    """Returns a GROUP_BY_DIMENSIONS value, or None.

    NOTE the critical distinction this function exists to protect:
      "employees in Sales"      -> a FILTER (handled by detect_dimension_filters)
      "show it department wise" -> a GROUP BY (handled here)
      "which department?"       -> neither; that is the ENTITY of a question
    The presence of the word "department" alone decides nothing.
    """
    text = text or ""
    for rx in (_GROUP_BY_WISE, _GROUP_BY_PREP, _GROUP_BY_ALL_ACROSS):
        m = rx.search(text)
        if m:
            word = re.sub(r"\s+", " ", m.group(1).strip().lower())
            dim = _GROUP_BY_WORDS.get(word)
            if dim:
                return dim
    return None


# ---------------------------------------------------------------------------
# Context-modification detection
# ---------------------------------------------------------------------------

_REMOVE_FILTER = re.compile(
    r"\b(?:remove|drop|clear|forget|undo|reset|cancel)\b[^.?!]{0,30}\b(?:filter|exclusion|"
    r"restriction|exclude|scope|condition)\b"
    r"|\b(?:no|without)\s+(?:the\s+)?filters?\b"
    r"|\b(?:include|show|across|for)\s+(?:every(?:one|body)|all\s+(?:the\s+)?"
    r"(?:departments?|depts?|employees?|teams?))\b"
    r"|\bcompany[\s-]?wide\s+again\b|\bback\s+to\s+(?:all|everyone|the\s+full\s+list)\b",
    re.IGNORECASE,
)

_RANKING_CHANGE = re.compile(
    r"\b(?:make\s+it|change\s+it\s+to|instead\s+(?:show|give)|show\s+me)?\s*"
    r"\b(?:top|bottom|highest|lowest|best|worst|first|last)\s+(\d{1,3})\b",
    re.IGNORECASE,
)

_INSTEAD = re.compile(r"\binstead\b|\brather\b|\bswitch\s+to\b|\bchange\s+to\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Item #98 — EXPLICIT REMOVAL, as a first-class signal
# ---------------------------------------------------------------------------
#
# The patch rule is "absent means unchanged", which leaves exactly one way to
# say "erase this": say so. `_REMOVE_FILTER` above could only ever express
# that for FILTERS. This detector is the same idea generalized over the plan's
# own field vocabulary — a removal VERB plus the name of a plan slot — so
# "remove the department grouping", "drop the date range", "forget the metric"
# and "clear the row limit" each clear precisely the slot they name and
# nothing else. The caller turns each hit into a `CLEAR` sentinel in the
# delta, which is the only thing in the whole pipeline that erases a field.
_REMOVAL_VERB = (r"(?:remove|drop|clear|forget|undo|cancel|get\s+rid\s+of|"
                 r"stop|no\s+more|without|take\s+(?:off|away)|lose)")

_REMOVAL_SLOTS = {
    "group_by": r"(?:group(?:ing|ed)?(?:\s+by)?|breakdown|break\s+down|"
                r"(?:department|dept|manager|rm|employee|grade|designation|day|month)"
                r"[\s\-]*wise)",
    "filters": r"(?:filters?|exclusions?|restrictions?|conditions?|scope)",
    "limit": r"(?:limits?|caps?|row\s+limits?|cut\s*-?\s*off)",
    "period": r"(?:date\s+ranges?|periods?|time\s+ranges?|months?\s+filter|dates?)",
    "population_filters": r"(?:population\s+filters?|wfh\s+filter|shift\s+filter|"
                          r"visit\s+filter|ps\s+filter)",
    "metrics": r"(?:metrics?)",
}

_REMOVAL_RES = {
    field: re.compile(_REMOVAL_VERB + r"\b[^.?!]{0,24}?\b" + pat + r"\b", re.IGNORECASE)
    for field, pat in _REMOVAL_SLOTS.items()
}


def detect_explicit_removals(text):
    """Plan fields this message explicitly asks to CLEAR. Returns a set."""
    text = text or ""
    out = set()
    for field, rx in _REMOVAL_RES.items():
        if rx.search(text):
            out.add(field)
    return out


# ---------------------------------------------------------------------------
# Item #99 — SEMANTIC SHAPE detection (the closed-vocabulary bypass)
# ---------------------------------------------------------------------------
#
# Everything above this point describes how a message MODIFIES a query. The
# two detectors below describe two whole QUESTION SHAPES that the product
# genuinely supports but that, until this round, only reached their correct
# engine when `intents.match_intent()`'s hand-written regexes happened to
# claim the exact phrasing:
#
#   * a PACE-STATUS (Black/Red/Amber/Green) membership question — "who is in
#     red in Founders Office", "which people are sitting in the amber band",
#     "anyone flagged black in Annotation";
#   * a STRONGEST/WEAKEST-AREA question — which of the four PACE component
#     sub-scores is an employee's / a department's best or worst.
#
# When the regexes did not claim the phrasing, the message fell through to
# `llm_nlu.classify()`, which MUST pick one of ~124 fixed intent names and
# has no way to say "this is a generalized question, let the plan layer have
# it". It therefore guessed, and a wrong guess produced a confident wrong
# answer (a department PACE aggregate instead of a status list; an unrelated
# PACE ranking instead of an area lookup — both live-confirmed).
#
# These detectors are deterministic and VOCABULARY-based, in exactly the same
# style as `_GROUP_BY_WORDS`/`_NEGATION_MARKER` above: they describe the
# CONCEPT (a status band; one of the four component areas) rather than any
# phrasing, so a novel wording of the same question is recognised by
# construction rather than by adding another pattern. The caller uses them
# only when `match_intent()` found NOTHING — i.e. precisely the population of
# messages whose routing was previously a `classify()` guess.

#: The four PACE status bands. Same vocabulary as queries.status_list()'s
#: `statuses` argument (capitalised there).
PACE_STATUS_WORDS = ("black", "red", "amber", "green")

_STATUS_WORD = re.compile(r"\b(" + "|".join(PACE_STATUS_WORDS) + r")\b", re.IGNORECASE)

#: A reference to PEOPLE — the thing a status question asks for. Without one,
#: a colour word is not a membership question (it could be a department name
#: fragment, a metric, or small talk).
_STATUS_POPULATION_REF = re.compile(
    r"\b(?:who|whom|whose|which|list|name|names|show|give|display|tell\s+me|"
    r"employees?|emps?|people|persons?|folks?|staff|members?|headcount|"
    r"anyone|anybody|everyone|everybody|guys|team)\b",
    re.IGNORECASE)

#: Words that make the question a COUNT rather than a list. Reuses the same
#: split the existing status_count/status_list intents already draw.
_STATUS_COUNT_CUE = re.compile(
    r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b|\bhow\s+much\b", re.IGNORECASE)

#: A status question is about the BAND itself; a question that also names one
#: of the four component metrics is a metric question that merely happens to
#: contain a colour word, and belongs to the engines that own those metrics.
_STATUS_DISQUALIFIER = re.compile(
    r"\bengagement\b|\beffectiveness\b|\bdiscipline\b|\bworking\s*hours?\b|"
    r"\blate[\s-]?coming\b|\bearly[\s-]?leav\w*\b|\bwhatsapp\b|\bmeeting\b|"
    r"\btasks?\b|\btodos?\b|\bproductiv\w*\b|\battendance\b|\bwfh\b|"
    r"\bwork\s+from\s+home\b|\bleaves?\b|\bvisits?\b|\bimprov\w*\b|\bdeclin\w*\b|"
    r"\bmoved\s+from\b|\bdistribution\b|\bpercentage\b|\bratio\b|\btrend\b|"
    r"\bpace\s+score\b",
    re.IGNORECASE)

#: "which department has the MOST red employees" is a status DISTRIBUTION
#: question (a ranking of groups), not a membership list. A superlative of
#: quantity is the generic signal for that, and the status_distribution
#: intent family owns it.
_STATUS_RANKING_CUE = re.compile(
    r"\b(?:most|least|fewest|largest|biggest|smallest)\b", re.IGNORECASE)


def detect_status_shape(text):
    """`{"statuses": [...], "kind": "list"|"count"}` when this message asks
    WHICH PEOPLE are in one or more PACE status bands, else None.

    Shape-level, not phrase-level: a status-band word plus a reference to
    people. Every phrasing of that pair is the same question, and the answer
    is always `queries.status_list()`/`status_count()` — the engines the
    existing status_list/status_count intents already call, unchanged.
    """
    text = text or ""
    if _STATUS_DISQUALIFIER.search(text) or _STATUS_RANKING_CUE.search(text):
        return None
    words =[m.group(1).lower() for m in _STATUS_WORD.finditer(text)]
    if not words:
        return None
    if not _STATUS_POPULATION_REF.search(text):
        return None
    statuses = list(dict.fromkeys(w.capitalize() for w in words))
    kind = "count" if _STATUS_COUNT_CUE.search(text) else "list"
    return {"statuses": statuses, "kind": kind}


def mentions_status_band(text):
    """The PACE status bands this message names, ignoring whether it also
    refers to a population. Used for the SINGLE-ENTITY form of the same
    question ("is X in the red band?"), where the subject is a resolved
    employee rather than a population reference.
    """
    text = text or ""
    if _STATUS_DISQUALIFIER.search(text) or _STATUS_RANKING_CUE.search(text):
        return []
    return list(dict.fromkeys(m.group(1).capitalize() for m in _STATUS_WORD.finditer(text)))


#: "status" as the PACE BAND, as opposed to the several other statuses this
#: product talks about (PS status, shift status, visit status, work status).
#: Naming the band without naming a colour is the other half of the same
#: single-entity question ("what is X's pace status").
_STATUS_BAND_NOUN = re.compile(
    r"\bpace\s+status\b|\bstatus\s+(?:band|banding|categor\w+)\b|"
    r"\bcolou?r\s+(?:band|categor\w+|code)\b|"
    r"(?<!ps\s)(?<!shift\s)(?<!visit\s)(?<!work\s)(?<!install\s)\bstatus\b",
    re.IGNORECASE)


def asks_for_status_band(text):
    """True when the message asks about the PACE STATUS BAND itself (no
    colour named). The caller still requires a resolvable subject before
    acting on it, so this never decides anything on its own."""
    text = text or ""
    if _STATUS_DISQUALIFIER.search(text) or _STATUS_RANKING_CUE.search(text):
        return False
    return _STATUS_BAND_NOUN.search(text) is not None


#: The four PACE components are collectively "areas" in this product's
#: language. This is the full noun vocabulary for that concept.
_AREA_NOUN = (r"(?:areas?|sub[-\s]?scores?|subscores?|sub[-\s]?metrics?|"
              r"components?|categor(?:y|ies)|dimensions?|aspects?|pillars?|"
              r"parameters?|factors?|metrics?)")

#: "which area is worst" direction vocabulary — the two poles, each a family
#: of words rather than a literal superlative.
_AREA_WEAK = (r"(?:weak\w*|worst|lowest|poor\w*|bottom|lagging|lags|trailing|"
              r"dragging|drags|drag|struggl\w*|deficien\w*|underperform\w*|"
              r"problem\w*|hurting|holding\s+\w+\s+back|behind)")
_AREA_STRONG = (r"(?:strong\w*|best|highest|top|greatest|strength\w*|leading|"
                r"excel\w*|standout|shin\w*)")

_AREA_WINDOW = r"[^.?!]{0,45}?"
_AREA_RES = (
    ("weakest", re.compile(_AREA_WEAK + _AREA_WINDOW + r"\b" + _AREA_NOUN + r"\b", re.IGNORECASE)),
    ("weakest", re.compile(r"\b" + _AREA_NOUN + r"\b" + _AREA_WINDOW + _AREA_WEAK, re.IGNORECASE)),
    ("strongest", re.compile(_AREA_STRONG + _AREA_WINDOW + r"\b" + _AREA_NOUN + r"\b", re.IGNORECASE)),
    ("strongest", re.compile(r"\b" + _AREA_NOUN + r"\b" + _AREA_WINDOW + _AREA_STRONG, re.IGNORECASE)),
)

#: An area question asks WHICH of the four components is best/worst. A message
#: that already NAMES one of them is asking about that component, not asking
#: which one — so it is a ranking/value question and must keep its engine.
_AREA_DISQUALIFIER = re.compile(
    r"\bengagement\b|\beffectiveness\b|\bdiscipline\b|\bworking\s*hours?\b|"
    r"\bworking\s*(?:%|percent\w*)|\bpace\s+score\b|\battendance\b|"
    r"\blate[\s-]?coming\w*\b|\bearly[\s-]?leav\w*\b|\bwhatsapp\b|"
    r"\bimprov\w*\b|\bdeclin\w*\b|\bprogress\w*\b",
    re.IGNORECASE)


def detect_area_shape(text):
    """`"weakest"` / `"strongest"` when this message asks WHICH of the four
    PACE component areas is the best/worst one, else None.

    The pair (an area NOUN, a direction WORD) is the whole signal — "their
    weakest area", "worst-performing category", "which of the four sub-scores
    is dragging them down" and "where is she strongest across the four
    components" are all the same question and all resolve here without a
    pattern of their own.
    """
    text = text or ""
    if _AREA_DISQUALIFIER.search(text):
        return None
    for direction, rx in _AREA_RES:
        if rx.search(text):
            return direction
    return None


def detect_context_modification(text, has_filters=False, has_group_by=False,
                                has_metric=False, has_period=False, has_limit=False):
    """Classify how a follow-up message relates to the previous plan.

    Returns one of CONTEXT_MODIFICATIONS. Signals are passed IN (rather than
    re-derived here) so this stays a pure, order-independent classifier over
    what the caller's own detectors actually found.
    """
    text = text or ""
    if _REMOVE_FILTER.search(text):
        return "remove_filter"
    if has_filters:
        return "replace_filter" if _INSTEAD.search(text) else "add_filter"
    if has_group_by:
        return "change_group_by"
    if has_metric:
        return "change_metric"
    if has_period:
        return "change_period"
    if has_limit:
        return "change_ranking"
    return "none"


# ---------------------------------------------------------------------------
# The plan object
# ---------------------------------------------------------------------------

_PLAN_DEFAULTS = {
    "entity": "employee",
    "metrics": None,              # list of BUILD_QUERY_METRICS keys
    "group_by": None,
    "operation": "value",
    "limit": None,
    "limit_mode": "unspecified",   # item #95: see LIMIT_MODES above
    "ascending": False,
    "period": None,               # (date, date) | None
    "period_phrase": None,
    "latest_n_days": None,
    "population_filters": None,   # dict; build_query()'s own 4 keys
    "filters": None,              # list of {field, operator, value}
    "comparison": None,           # {kind, period_a, period_b, group_by}
    "name_filter": None,
    "name_label": None,
    "source": None,               # which engine produced/owns this plan
}


def new_plan(**kwargs):
    plan = copy.deepcopy(_PLAN_DEFAULTS)
    plan["metrics"] = []
    plan["population_filters"] = {}
    plan["filters"] = []
    for k, v in kwargs.items():
        if k not in _PLAN_DEFAULTS:
            raise KeyError("unknown QueryPlan field: %r" % k)
        plan[k] = v
    return normalize(plan)


def normalize(plan):
    """Deterministic validation/normalization — application code, never the
    LLM, decides what a legal plan looks like. Invalid values are coerced to
    safe defaults rather than raising, per this codebase's standing
    fail-safe-not-crash contract."""
    plan = dict(plan or {})
    for k, v in _PLAN_DEFAULTS.items():
        plan.setdefault(k, copy.deepcopy(v))

    if plan.get("entity") not in ENTITIES:
        plan["entity"] = "employee"
    if plan.get("operation") not in OPERATIONS:
        plan["operation"] = "value"
    if plan.get("group_by") not in GROUP_BY_DIMENSIONS:
        plan["group_by"] = None

    plan["metrics"] = [m for m in (plan.get("metrics") or []) if isinstance(m, str)]
    plan["population_filters"] = {
        k: v for k, v in (plan.get("population_filters") or {}).items()
        if k in POPULATION_FILTER_KEYS and v
    }

    clean_filters = []
    for f in (plan.get("filters") or []):
        if not isinstance(f, dict):
            continue
        field, op, val = f.get("field"), f.get("operator"), f.get("value")
        if field not in FILTER_FIELDS or op not in FILTER_OPERATORS:
            continue
        if op in ("is_null", "is_not_null"):
            val = None
        elif val in (None, "", []):
            continue
        clean_filters.append({"field": field, "operator": op, "value": val})
    plan["filters"] = clean_filters

    # ---- result cardinality (item #95) ---------------------------------
    # The mode is the authority; `limit` is only meaningful in "exact" mode.
    # Note the old code clamped to 500 — which was itself an instance of the
    # bug being fixed here, since an explicit "top 600" (or an unlimited
    # request routed through `limit`) was silently rewritten to 500.
    mode = plan.get("limit_mode")
    limit = plan.get("limit")
    valid_int = isinstance(limit, int) and not isinstance(limit, bool)
    if mode not in LIMIT_MODES:
        # Infer the mode for a plan built by older code that only set `limit`
        # (from_query_context(), and any caller predating this field).
        mode = "exact" if valid_int else "unspecified"
    if mode == "unlimited":
        plan["limit_mode"] = "unlimited"
        plan["limit"] = None
    elif valid_int:
        plan["limit_mode"] = "exact"
        plan["limit"] = max(1, min(limit, UNLIMITED_CEILING))
    else:
        plan["limit_mode"] = "unspecified"
        plan["limit"] = None

    plan["ascending"] = bool(plan.get("ascending"))

    # Ranking direction and operation must agree — `operation` is the one
    # that carries meaning downstream, `ascending` is its SQL expression.
    if plan["operation"] == "rank_bottom":
        plan["ascending"] = True
    elif plan["operation"] == "rank_top":
        plan["ascending"] = False

    cmp_ = plan.get("comparison")
    if cmp_ is not None:
        if not isinstance(cmp_, dict) or not cmp_.get("period_a") or not cmp_.get("period_b"):
            plan["comparison"] = None
        else:
            # ---- item #98: ONE grouping slot, not two -------------------
            # `comparison["group_by"]` used to be a SECOND, independent home
            # for the grouping dimension, reachable only by replacing the
            # whole `comparison` object. `patch()` merges top-level fields,
            # so a follow-up that changed the grain wrote it to the
            # top-level slot while the executor read `plan["group_by"] or
            # comparison["group_by"]` — two sources of truth that nothing
            # kept in step. The nested copy is now purely a MIRROR of the
            # top-level field: whichever side carries a value, both end up
            # holding it, and every patch of the grouping dimension is an
            # ordinary top-level patch no matter which operation the plan
            # is running. That is what makes "nested state didn't get
            # patched" structurally impossible here rather than fixed once.
            gb = plan.get("group_by")
            if gb not in GROUP_BY_DIMENSIONS:
                gb = cmp_.get("group_by")
            gb = gb if gb in GROUP_BY_DIMENSIONS else None
            plan["group_by"] = gb
            plan["comparison"] = {
                "kind": cmp_.get("kind") or "day",
                "period_a": cmp_["period_a"],
                "period_b": cmp_["period_b"],
                "group_by": gb,
            }
            plan["operation"] = "compare"
    return plan


def is_ranking(plan):
    return (plan or {}).get("operation") in ("rank_top", "rank_bottom", "rank_both_ends")


def effective_limit(plan, default):
    """The row count to hand to queries.build_query() for this plan.

    The ONE place the three cardinality states become a single SQL number,
    so no executor has to re-derive the rule (and no executor can quietly
    reintroduce `plan["limit"] or <magic number>`, which is exactly how the
    item #95 bug worked):

        "exact"       -> the user's own N, never overridden.
        "unlimited"   -> the safety ceiling only.
        "unspecified" -> the caller's `default` — the only state in which a
                         deterministic default is allowed to decide anything.
    """
    plan = plan or {}
    mode = plan.get("limit_mode")
    if mode not in LIMIT_MODES:
        mode = "exact" if isinstance(plan.get("limit"), int) else "unspecified"
    if mode == "unlimited":
        return UNLIMITED_CEILING
    if mode == "exact" and isinstance(plan.get("limit"), int):
        return plan["limit"]
    return default


# ---------------------------------------------------------------------------
# Follow-up PATCHING — the heart of the conversational-state design
# ---------------------------------------------------------------------------

def merge_filters(existing, incoming, mode="add"):
    """Compose a follow-up's filters with the plan's existing ones.

    mode="add"     — keep existing filters, add/override per FIELD
                     (a second filter on the same field replaces the first,
                     so "exclude SCM" then "exclude Annotation" behaves the
                     way a user means it: one department filter at a time per
                     field unless they explicitly say "also").
    mode="replace" — the incoming filters become the complete set.
    mode="remove"  — drop the named fields (or all, if `incoming` is empty).
    mode="accumulate" — "ALSO exclude Y" after "exclude X": same-field filters
                     of the same polarity are UNIONED into an in/not_in list
                     rather than replacing one another.
    """
    existing = [dict(f) for f in (existing or [])]
    incoming = [dict(f) for f in (incoming or [])]

    if mode == "replace":
        return incoming
    if mode == "remove":
        if not incoming:
            return []
        drop = {f["field"] for f in incoming}
        return [f for f in existing if f["field"] not in drop]

    if mode == "accumulate":
        out = []
        by_field = {}
        for f in existing:
            by_field.setdefault(f["field"], []).append(f)
        for f in incoming:
            prev = by_field.pop(f["field"], [])
            negative_new = f["operator"] in NEGATIVE_OPERATORS
            values = []
            for p in prev:
                if (p["operator"] in NEGATIVE_OPERATORS) == negative_new:
                    v = p.get("value")
                    values.extend(v if isinstance(v, list) else [v])
            v = f.get("value")
            values.extend(v if isinstance(v, list) else [v])
            values = [x for i, x in enumerate(values) if x is not None and x not in values[:i]]
            if len(values) > 1:
                out.append({"field": f["field"],
                            "operator": "not_in" if negative_new else "in",
                            "value": values})
            else:
                out.append(dict(f))
        for rest in by_field.values():
            out.extend(rest)
        return out

    out = []
    incoming_fields = {f["field"] for f in incoming}
    for f in existing:
        if f["field"] not in incoming_fields:
            out.append(f)
    out.extend(incoming)
    return out


#: Every field a follow-up may patch. Anything NOT named in the delta is
#: carried forward from the previous plan untouched — that invariant is what
#: the multi-turn regression tests assert on, step by step.
PATCHABLE_FIELDS = (
    "entity", "metrics", "group_by", "operation", "limit", "limit_mode", "ascending",
    "period", "period_phrase", "latest_n_days", "population_filters",
    "name_filter", "name_label", "comparison",
)


def patch(plan, delta, filter_mode="add"):
    """Return a NEW plan: `plan` with only the fields present in `delta`
    changed. `delta` may contain any PATCHABLE_FIELDS key plus "filters".

    A value of None in `delta` means "not specified by this follow-up" and
    therefore preserves the previous value — the one exception being the
    explicit sentinel `CLEAR`, which sets the field back to its default.
    """
    base = normalize(plan)
    out = dict(base)
    delta = dict(delta or {})
    # Item #95: the two cardinality fields move together. A follow-up that
    # names a NUMBER ("make it the top 5") after an unlimited request must
    # switch the mode back to "exact", or normalize() — which treats the
    # mode as authoritative — would drop the number on the floor.
    if "limit" in delta and "limit_mode" not in delta:
        v = delta["limit"]
        if v is CLEAR or v is None:
            pass  # "not mentioned"/"reset": leave the previous mode alone
        else:
            delta["limit_mode"] = "exact"
    for k in PATCHABLE_FIELDS:
        if k not in (delta or {}):
            continue
        v = delta[k]
        if v is CLEAR:
            out[k] = copy.deepcopy(_PLAN_DEFAULTS[k])
            if k == "metrics":
                out[k] = []
            elif k == "population_filters":
                out[k] = {}
        elif v is not None:
            # Item #98: a STRUCTURED field is patched key-by-key, exactly as
            # the plan as a whole is. A follow-up that names one key of a
            # nested object ("just the WFH ones" -> population_filters
            # {work_mode: ...}; "make the second date the 12th" ->
            # comparison {period_b: ...}) must not erase the sibling keys it
            # never mentioned — the same unmentioned-is-not-removed rule that
            # governs the top level, applied one level down. A per-key CLEAR
            # removes exactly that key.
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                merged = dict(out[k])
                for kk, vv in v.items():
                    if vv is CLEAR:
                        merged.pop(kk, None)
                    elif vv is not None:
                        merged[kk] = vv
                out[k] = merged
            else:
                out[k] = v

    # The grouping dimension has exactly ONE home (see normalize()). When a
    # follow-up patches it — including an EXPLICIT removal — the mirror inside
    # a comparison sub-object follows it, so a cleared grouping can never be
    # resurrected from the nested copy.
    if "group_by" in (delta or {}) and isinstance(out.get("comparison"), dict):
        out["comparison"] = dict(out["comparison"])
        out["comparison"]["group_by"] = out.get("group_by")

    if "filters" in (delta or {}):
        v = delta["filters"]
        if v is CLEAR:
            out["filters"] = []
        elif v is not None:
            out["filters"] = merge_filters(base.get("filters"), v, mode=filter_mode)

    # Keep operation/ascending consistent when a follow-up flips only one of
    # them (e.g. "make it the top 5" after a bottom-10 ranking).
    if "ascending" in (delta or {}) and "operation" not in (delta or {}):
        if is_ranking(out):
            out["operation"] = "rank_bottom" if out["ascending"] else "rank_top"
    return normalize(out)


class _Clear(object):
    """Sentinel: explicitly reset a field to its default in patch()/delta."""
    __slots__ = ()

    def __repr__(self):
        return "<query_plan.CLEAR>"


CLEAR = _Clear()


# ---------------------------------------------------------------------------
# Interop with the pre-existing conversational stores
# ---------------------------------------------------------------------------

def from_query_context(qc):
    """Build a plan from the older `session_store.query_context` object.

    The four legacy stores are not deleted (that would mean rewriting every
    one of the ~123 intents' bookkeeping); instead this adapter lets a plan
    be recovered from state that the RULE-BASED ranking handlers already
    write, so a plan-shaped follow-up composes with a rule-based answer just
    as well as with a plan-produced one. New code writes a real plan via
    `session_store.set_current_plan()`; this is the bridge for everything
    that predates it.
    """
    qc = qc or {}
    op = qc.get("last_operation")
    if op not in OPERATIONS:
        op = "value"
    dept_filter = qc.get("dept_filter") or None
    filters = []
    if dept_filter and dept_filter.get("value"):
        filters.append({
            "field": "department",
            "operator": dept_filter.get("operator") or "eq",
            "value": dept_filter["value"],
        })
    return new_plan(
        entity=qc.get("last_dimension") or "employee",
        metrics=list(qc.get("metric") or []),
        operation=op,
        limit=qc.get("limit"),
        ascending=bool(qc.get("ascending")),
        period_phrase=qc.get("period_phrase"),
        filters=filters,
        source="query_context",
    )


def describe(plan):
    """Short human-readable description of the plan — used in reply footers
    so the user can see what filter/grouping is actually in effect, and in
    test assertions."""
    plan = normalize(plan)
    bits = []
    if plan.get("limit_mode") == "unlimited":
        bits.append("all matching rows")
    if plan["group_by"]:
        bits.append("grouped by %s" % plan["group_by"])
    for f in plan["filters"]:
        val = f["value"]
        val = ", ".join(str(v) for v in val) if isinstance(val, list) else val
        if f["operator"] in NEGATIVE_OPERATORS:
            bits.append("excluding %s" % val)
        else:
            bits.append("%s only" % val)
    return ", ".join(bits)
