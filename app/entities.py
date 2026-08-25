import re
import datetime
from functools import lru_cache

from rapidfuzz import fuzz

from .db import run_query

VIEW = "public.pace_chatbot_view"

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_STOPWORDS_IN_DEPT_MATCH = {"the", "team", "dept", "department", "in", "of"}


@lru_cache(maxsize=1)
def get_dept_names():
    rows = run_query(f"SELECT DISTINCT dept_name FROM {VIEW} WHERE dept_name IS NOT NULL ORDER BY dept_name")
    return [r["dept_name"] for r in rows]


@lru_cache(maxsize=1)
def get_manager_names():
    rows = run_query(
        f"SELECT DISTINCT reporting_manager_name, reporting_user_id FROM {VIEW} "
        f"WHERE reporting_manager_name IS NOT NULL ORDER BY reporting_manager_name"
    )
    return [(r["reporting_manager_name"], r["reporting_user_id"]) for r in rows]


def refresh_entity_cache():
    """Call if the underlying view's distinct values might have changed."""
    get_dept_names.cache_clear()
    get_manager_names.cache_clear()
    get_employee_names.cache_clear()


def extract_limit(text, default=None):
    """'top 5' / 'bottom 3' / 'top5' -> 5 / 3. Returns default if not present."""
    m = re.search(r"\b(?:top|bottom)\s*(\d{1,3})\b", text.lower())
    if m:
        n = int(m.group(1))
        return max(1, min(n, 100))
    return default


def split_comparison(text):
    """Splits 'compare X vs Y' / 'X versus Y' / 'X vs. Y' style text into
    (left, right) substrings, or (None, None) if no comparison marker found."""
    m = re.search(r"\bcompare\s+(.+?)\s+(?:vs\.?|versus|and|to|with)\s+(.+)", text, re.I)
    if not m:
        m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", text, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


class Ambiguous(Exception):
    def __init__(self, kind, candidates):
        self.kind = kind
        self.candidates = candidates
        super().__init__(f"Ambiguous {kind}: {candidates}")


def _contains_word(text_l, phrase_l):
    """True if phrase_l appears in text_l as a whole word/phrase, not as a
    substring of a longer word (e.g. 'product' must not match inside 'productive')."""
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_l) + r"(?![a-z0-9])"
    return re.search(pattern, text_l) is not None


# --- Fuzzy last-resort tier (offline, rapidfuzz) -----------------------------
# Only used inside extract_department/extract_employee/extract_manager, and
# only AFTER their existing exact + loose-token tiers both come up empty.
# Deliberately NOT used anywhere in team.py - the email-access resolution
# there must stay exact-match only (confirmed real substring-collision risk
# in that data), which is why this lives here instead of being a shared
# generic utility both modules reach for.
#
# Uses partial-ratio ALIGNMENT (not a plain score) rather than token_set_ratio,
# because token_set_ratio is too conservative for genuine truncation (e.g.
# "founder offi" -> "Founders Office" only scores ~49, well under threshold)
# while a bare partial-ratio score is unsafe on its own: "Product" scores
# 100 against "...most productive employee..." because "product" is a
# literal substring of "productive" - the exact same class of bug fixed
# earlier for exact matching (_contains_word). The alignment lets us check
# WHERE the match landed in the message and reject it unless both ends sit
# on a real word boundary (start-of-string/whitespace/punctuation), which
# correctly accepts the truncation case and rejects the substring case.
FUZZY_NAME_THRESHOLD = 80
FUZZY_NAME_MIN_COVERAGE = 0.7  # matched span must cover at least this fraction of the candidate name's own length
FUZZY_NAME_MARGIN = 5  # candidates within this of the best score are treated as tied -> ask, don't guess


def _is_word_boundary(text_l, start, end):
    before_ok = start == 0 or not text_l[start - 1].isalnum()
    after_ok = end >= len(text_l) or not text_l[end].isalnum()
    return before_ok and after_ok


def _fuzzy_name_candidates(text_l, names):
    """names: real display strings (e.g. actual dept_name/emp_name casing).
    Returns the subset scoring >= threshold as (name, score) pairs, best first,
    after the word-boundary + coverage safety checks above."""
    hits = []
    for name in names:
        normalized = re.sub(r"[\-/]", " ", name.lower())
        alignment = fuzz.partial_ratio_alignment(text_l, normalized)
        if alignment.score < FUZZY_NAME_THRESHOLD:
            continue
        if not _is_word_boundary(text_l, alignment.src_start, alignment.src_end):
            continue
        matched_len = alignment.dest_end - alignment.dest_start
        if matched_len / max(len(normalized), 1) < FUZZY_NAME_MIN_COVERAGE:
            continue
        hits.append((name, alignment.score))
    hits.sort(key=lambda h: -h[1])
    return hits


def _fuzzy_resolve(text_l, names):
    """Returns (single_match, None) / (None, candidates) / (None, None)."""
    hits = _fuzzy_name_candidates(text_l, names)
    if not hits:
        return None, None
    best_score = hits[0][1]
    close = [name for name, score in hits if best_score - score <= FUZZY_NAME_MARGIN]
    if len(close) == 1:
        return close[0], None
    return None, close


def extract_department(text, fallback_text=None):
    """Returns (dept_name or None, candidates or None).
    - dept_name set -> confident single match
    - candidates set (list) -> ambiguous, caller should ask user to pick
    - both None -> no department mentioned at all

    `fallback_text`, if given, is retried (from scratch, all tiers) when
    `text` produces no match at all. Exists because dictionary spellcheck
    upstream can occasionally corrupt a genuinely truncated department
    fragment into an unrelated short word (e.g. "offi" -> "off"), which
    would otherwise silently kill a match that the RAW pre-spellcheck text
    could still resolve.
    """
    result = _extract_department_single(text)
    if result == (None, None) and fallback_text and fallback_text != text:
        return _extract_department_single(fallback_text)
    return result


def _extract_department_single(text):
    text_l = text.lower()
    depts = get_dept_names()

    # exact (case-insensitive) full-name match first
    for d in depts:
        if d.lower() == text_l.strip():
            return d, None

    # whole-phrase match: department name appears in the text as a full word/phrase
    exact_hits = [d for d in depts if _contains_word(text_l, d.lower())]
    if len(exact_hits) == 1:
        return exact_hits[0], None
    if len(exact_hits) > 1:
        return None, exact_hits

    # loose token-overlap match: any significant whole word from a dept name appears in text
    candidates = []
    for d in depts:
        tokens = [t for t in re.split(r"[\s\-/]+", d.lower()) if t and t not in _STOPWORDS_IN_DEPT_MATCH and len(t) > 2]
        if tokens and any(_contains_word(text_l, t) for t in tokens):
            candidates.append(d)

    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, candidates

    # last resort: fuzzy match (typo tolerance), only reached if exact and
    # loose-token matching both found nothing at all
    return _fuzzy_resolve(text_l, depts)


def extract_month(text, default_to_current=True):
    """Returns (month_str 'YYYY-MM' or None, was_mentioned bool).
    If the user mentioned no time reference at all, defaults to the current month
    (when default_to_current=True) and was_mentioned=False.
    If the user mentioned a time reference but it couldn't be resolved, returns
    (None, True) so the caller can ask for clarification instead of guessing.
    """
    text_l = text.lower()

    # YYYY-MM explicit
    m = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", text_l)
    if m:
        return f"{m.group(1)}-{m.group(2)}", True

    # "this month" / "current month"
    if re.search(r"\b(this month|current month)\b", text_l):
        now = datetime.date.today()
        return f"{now.year:04d}-{now.month:02d}", True

    # "last month"
    if re.search(r"\blast month\b", text_l):
        now = datetime.date.today()
        first_of_this_month = now.replace(day=1)
        last_month_end = first_of_this_month - datetime.timedelta(days=1)
        return f"{last_month_end.year:04d}-{last_month_end.month:02d}", True

    # month name (optionally with year)
    for name, num in MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", text_l):
            year_m = re.search(r"\b(20\d{2})\b", text_l)
            year = int(year_m.group(1)) if year_m else datetime.date.today().year
            return f"{year:04d}-{num:02d}", True

    # no time mention found at all
    if default_to_current:
        now = datetime.date.today()
        return f"{now.year:04d}-{now.month:02d}", False
    return None, False


def extract_months(text, default_to_current=True):
    """Returns (list_of_month_strs, was_mentioned bool) — like extract_month()
    but collects EVERY distinct month mentioned in the text instead of
    stopping at the first match (e.g. "wfh in june, july, august" ->
    (["2026-06", "2026-07", "2026-08"], True)).

    Order follows calendar order (Jan..Dec), which also naturally dedupes
    "jun"/"june" style double-mentions of the same month. If the user
    mentioned no time reference at all, defaults to the current month alone
    (when default_to_current=True) with was_mentioned=False, mirroring
    extract_month()'s no-mention behavior exactly so single/no-mention
    callers see identical results either way.
    """
    text_l = text.lower()
    months = []

    def _add(s):
        if s not in months:
            months.append(s)

    # YYYY-MM explicit (collect all distinct occurrences)
    for m in re.finditer(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", text_l):
        _add(f"{m.group(1)}-{m.group(2)}")

    # "this month" / "current month"
    if re.search(r"\b(this month|current month)\b", text_l):
        now = datetime.date.today()
        _add(f"{now.year:04d}-{now.month:02d}")

    # "last month"
    if re.search(r"\blast month\b", text_l):
        now = datetime.date.today()
        first_of_this_month = now.replace(day=1)
        last_month_end = first_of_this_month - datetime.timedelta(days=1)
        _add(f"{last_month_end.year:04d}-{last_month_end.month:02d}")

    # month names — collect ALL distinct months named, not just the first
    year_m = re.search(r"\b(20\d{2})\b", text_l)
    year = int(year_m.group(1)) if year_m else datetime.date.today().year
    for name, num in MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", text_l):
            _add(f"{year:04d}-{num:02d}")

    if months:
        return months, True

    if default_to_current:
        now = datetime.date.today()
        return [f"{now.year:04d}-{now.month:02d}"], False
    return [], False


_ORDINAL_WORD_TO_NUM = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "last": -1,
}

_MONTH_DAY_RE = (
    r"(?:(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s+(?P<m1>[a-z]+)|"
    r"(?P<m2>[a-z]+)\s+(?P<d2>\d{1,2})(?:st|nd|rd|th)?)"
)


def _parse_single_date_token(token, default_year=None):
    """Best-effort parse of a single date expression into a datetime.date,
    or None if it can't be resolved. Supports 'YYYY-MM-DD', '15th August',
    'August 15', 'Aug 15 2026'. Defaults the year to `default_year` (or the
    current year) when not stated in the token itself."""
    token = token.strip().strip(".,")
    today = datetime.date.today()
    if default_year is None:
        default_year = today.year

    m = re.match(r"^(20\d{2})-(\d{1,2})-(\d{1,2})$", token)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](20\d{2})$", token)
    if m:
        try:
            return datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None

    year_m = re.search(r"\b(20\d{2})\b", token)
    year = int(year_m.group(1)) if year_m else default_year

    m = re.search(_MONTH_DAY_RE, token, re.I)
    if m:
        day = m.group("d1") or m.group("d2")
        month_name = (m.group("m1") or m.group("m2") or "").lower()
        month_num = MONTH_NAMES.get(month_name)
        if month_num and day:
            try:
                return datetime.date(year, month_num, int(day))
            except ValueError:
                return None

    return None


def _week_of_month(text_l, today=None):
    """Handles 'first/second/third/fourth/last week of <month>' — maps to
    fixed 7-day calendar blocks starting on day 1 of that month (Aug 1-7,
    8-14, 15-21, 22-28, with 'last week' being the remaining tail days
    through month-end), NOT ISO Mon-Sun weeks. This is a deliberate,
    documented judgment call: "first week of August" in everyday usage
    means "the first ~7 days of August", not "the first ISO week that
    overlaps August" (which could dip into July). Returns (start, end,
    True) or None if no such phrase is present."""
    if today is None:
        today = datetime.date.today()
    m = re.search(
        r"\b(first|1st|second|2nd|third|3rd|fourth|4th|last)\s+week\s+of\s+([a-z]+)\b(?:\s+(20\d{2}))?",
        text_l,
    )
    if not m:
        return None
    ordinal = _ORDINAL_WORD_TO_NUM.get(m.group(1))
    month_num = MONTH_NAMES.get(m.group(2))
    if ordinal is None or month_num is None:
        return None
    year = int(m.group(3)) if m.group(3) else today.year

    if month_num == 12:
        next_month_first = datetime.date(year + 1, 1, 1)
    else:
        next_month_first = datetime.date(year, month_num + 1, 1)
    month_last_day = (next_month_first - datetime.timedelta(days=1)).day

    if ordinal == -1:  # "last week of <month>"
        start_day = ((month_last_day - 1) // 7) * 7 + 1
        start = datetime.date(year, month_num, start_day)
        end = datetime.date(year, month_num, month_last_day)
        return start, end, True

    start_day = (ordinal - 1) * 7 + 1
    if start_day > month_last_day:
        return None
    end_day = min(start_day + 6, month_last_day)
    start = datetime.date(year, month_num, start_day)
    end = datetime.date(year, month_num, end_day)
    return start, end, True


def _rolling_last_n_months(text_l, today=None):
    """Handles '(over/in) the last/past N months' — a bounded ROLLING window
    distinct from the full-history trend feature: start = the 1st of the
    month that is (N-1) months before the current month, end = today (so
    the window includes the current, still-partial month). E.g. today =
    2026-08-24, N=3 -> 2026-06-01 .. 2026-08-24. Returns (start, end, True)
    or None. Deliberately checked BEFORE the plain 'last month' /
    'last week' phrases at call sites can't collide since those require the
    literal word immediately after 'last' with no number in between."""
    if today is None:
        today = datetime.date.today()
    m = re.search(r"\b(?:last|past)\s+(\d{1,2})\s+months?\b", text_l)
    if not m:
        return None
    n = max(1, int(m.group(1)))
    y, mo = today.year, today.month
    mo -= (n - 1)
    while mo <= 0:
        mo += 12
        y -= 1
    start = datetime.date(y, mo, 1)
    return start, today, True


def extract_date_range(text):
    """Returns (start_date, end_date, was_mentioned bool) for day/week-level
    relative time references: "yesterday", "today", "last week", "this week",
    PLUS custom dates/ranges/week-of-month/rolling-N-months (see helpers
    above). Returns (None, None, False) if none of these are mentioned -
    callers should fall back to extract_month() in that case. Deliberately a
    separate function (rather than folding into extract_month) so existing
    month-based callers are untouched; new day/week-granularity callers
    (Category A/B/C/F etc.) check this FIRST and fall back to extract_month
    for month-level or no-mention cases.

    Every branch below ultimately returns a plain (start_date, end_date)
    tuple, which is the SAME shape consumed by queries._period_filter() -
    custom dates/ranges therefore flow through the exact same query
    plumbing as "yesterday"/"last week" with no parallel code path."""
    text_l = text.lower()
    today = datetime.date.today()

    if re.search(r"\byesterday\b", text_l):
        d = today - datetime.timedelta(days=1)
        return d, d, True

    if re.search(r"\btoday\b", text_l):
        return today, today, True

    # Rolling "last/past N months" (N >= 1 with an explicit number) - must be
    # checked before the plain "last week"/"last month" phrases below since
    # it's a more specific pattern; the word-boundary regexes don't actually
    # collide (no digit sits between "last" and "week"/"month" in those), but
    # checking this first keeps the more specific/newer feature authoritative.
    rolling = _rolling_last_n_months(text_l, today)
    if rolling:
        return rolling

    if re.search(r"\blast week\b", text_l):
        # ISO week: Monday-Sunday. "Last week" = the previous full ISO week.
        this_week_start = today - datetime.timedelta(days=today.weekday())
        last_week_start = this_week_start - datetime.timedelta(days=7)
        last_week_end = this_week_start - datetime.timedelta(days=1)
        return last_week_start, last_week_end, True

    if re.search(r"\bthis week\b", text_l):
        this_week_start = today - datetime.timedelta(days=today.weekday())
        return this_week_start, today, True

    # "first/second/third/fourth/last week of <month>"
    week_of_month = _week_of_month(text_l, today)
    if week_of_month:
        return week_of_month

    # Custom date RANGE: "between X and Y" / "from X to Y"
    m = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[.?!]|$)", text_l
    ) or re.search(
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[.?!]|$)", text_l
    )
    if m:
        start = _parse_single_date_token(m.group(1))
        end = _parse_single_date_token(m.group(2), default_year=start.year if start else None)
        if start and end:
            if end < start:
                start, end = end, start
            return start, end, True

    # Custom single date: "on 2026-08-15", "on 15th August", "on August 15"
    m = re.search(r"\bon\s+(" + _MONTH_DAY_RE + r"|20\d{2}-\d{1,2}-\d{1,2})", text_l, re.I)
    if m:
        d = _parse_single_date_token(m.group(1))
        if d:
            return d, d, True

    # Bare ISO date with no leading "on" (e.g. "attendance for 2026-08-15")
    m = re.search(r"\b(20\d{2}-\d{1,2}-\d{1,2})\b", text_l)
    if m:
        d = _parse_single_date_token(m.group(1))
        if d:
            return d, d, True

    return None, None, False


def is_self_referential(text):
    """True if the user is asking about 'my'/'our' team rather than naming
    an explicit department or manager."""
    return re.search(r"\b(my|our)\b", text.lower()) is not None


@lru_cache(maxsize=1)
def get_employee_names():
    rows = run_query(f"SELECT DISTINCT employee_id, emp_name FROM {VIEW} WHERE emp_name IS NOT NULL")
    return [(r["emp_name"].strip(), r["employee_id"]) for r in rows]


def extract_employee(text, fallback_text=None):
    """Returns (employee_id, emp_name) or (None, None) if no employee is
    named in the text, or raises Ambiguous if multiple employees match.
    Whole-name match first (avoids 'Abhay' matching 10 people), falling back
    to a loose token match same as extract_manager.

    `fallback_text`, if given, is retried when `text` matches nothing at
    all (same rationale as extract_department's fallback_text)."""
    try:
        result = _extract_employee_single(text)
    except Ambiguous:
        raise  # an ambiguous match IS a match - don't override with fallback_text
    if result == (None, None) and fallback_text and fallback_text != text:
        return _extract_employee_single(fallback_text)
    return result


def _extract_employee_single(text):
    text_l = text.lower()
    employees = get_employee_names()

    hits = [(name, eid) for name, eid in employees if name.lower() in text_l]
    if len(hits) == 1:
        return hits[0][1], hits[0][0]
    if len(hits) > 1:
        # prefer the longest matching name (e.g. "Abhay Gupta" over a
        # coincidental single-token overlap) before giving up as ambiguous
        hits.sort(key=lambda h: -len(h[0]))
        if len(hits[0][0]) > len(hits[1][0]):
            return hits[0][1], hits[0][0]
        raise Ambiguous("employee", [h[0] for h in hits])

    # last resort: fuzzy match (typo tolerance), only reached if the exact
    # whole-name match above found nothing at all
    name_to_id = dict(employees)
    match, candidates = _fuzzy_resolve(text_l, [name for name, _eid in employees])
    if match:
        return name_to_id[match], match
    if candidates:
        raise Ambiguous("employee", candidates)

    return None, None


def extract_manager(text, fallback_text=None):
    """Returns (reporting_user_id, name) or (None, None) if no mention,
    or raises Ambiguous if multiple managers match.

    `fallback_text`, if given, is retried when `text` matches nothing at
    all (same rationale as extract_department's fallback_text)."""
    try:
        result = _extract_manager_single(text)
    except Ambiguous:
        raise
    if result == (None, None) and fallback_text and fallback_text != text:
        return _extract_manager_single(fallback_text)
    return result


def _extract_manager_single(text):
    text_l = text.lower()
    managers = get_manager_names()

    hits = [(name, uid) for name, uid in managers if name.lower() in text_l]
    if len(hits) == 1:
        return hits[0][1], hits[0][0]
    if len(hits) > 1:
        raise Ambiguous("manager", [h[0] for h in hits])

    # loose match: a candidate only qualifies if a STRICT MAJORITY of their
    # own name tokens (len>2) appear as standalone words in the query text -
    # not just any single shared token. This is what stops a bare shared
    # surname fragment (e.g. querying "Aryan Gupta" - not a manager at all -
    # loosely matching BOTH "Fanish Kumar Gupta" and "Lokesh Gupta" purely
    # because they all contain "gupta") from either guessing a match or
    # raising a false ambiguity prompt: neither candidate's OTHER name
    # tokens ("fanish"/"kumar", "lokesh") appear anywhere in the query, so
    # neither clears the >50% coverage bar and both are correctly excluded -
    # this yields (None, None), not a guess and not a false "which one?".
    # A query that genuinely supplies a person's full name (e.g. "Ajay
    # Gaur's team") still matches normally, since 100% of that candidate's
    # tokens are present.
    loose_hits = []
    for name, uid in managers:
        tokens = [t for t in name.lower().split() if len(t) > 2]
        if not tokens:
            continue
        matched = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", text_l))
        if matched / len(tokens) > 0.5:
            loose_hits.append((name, uid))
    if len(loose_hits) == 1:
        return loose_hits[0][1], loose_hits[0][0]
    if len(loose_hits) > 1:
        # genuine tie between two comparably-strong loose matches (e.g. two
        # managers with the same full name) - prefer the longest/most
        # complete match first, only raise Ambiguous on a real tie.
        loose_hits.sort(key=lambda h: -len(h[0]))
        if len(loose_hits[0][0]) > len(loose_hits[1][0]):
            return loose_hits[0][1], loose_hits[0][0]
        raise Ambiguous("manager", [h[0] for h in loose_hits])

    # last resort: fuzzy match (typo tolerance), only reached if exact and
    # loose-token matching both found nothing at all
    name_to_id = dict(managers)
    match, candidates = _fuzzy_resolve(text_l, [name for name, _uid in managers])
    if match:
        return name_to_id[match], match
    if candidates:
        raise Ambiguous("manager", candidates)

    return None, None
