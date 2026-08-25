"""Offline typo correction (pyspellchecker) run once on raw chat input,
before intent/entity matching. Fully local - no network calls, no LLM.

Protects domain vocabulary (department names, employee names, product
jargon) from being "corrected" into nonsense by loading it into the
spellchecker's known-word set first.
"""
import re

from spellchecker import SpellChecker

_spell = SpellChecker()

# Product/domain jargon that isn't standard English but must never be
# treated as a typo.
_DOMAIN_WORDS = {
    "pace", "whatsapp", "chatgpt", "dept", "emp", "ai",
    # New-category (A-J) domain terms that pyspellchecker's general English
    # dictionary doesn't know and would otherwise "correct" into unrelated
    # words (discovered the same way the "offi"->"off" truncation bug was:
    # live testing, e.g. "todos" -> "dodos").
    "todos", "todo", "wfh", "ot", "d_score", "dscore",
}

_domain_vocab_loaded = False


def _load_domain_vocab():
    # Imported lazily to avoid a circular import (entities.py doesn't
    # import this module, but importing at module load time would still
    # force a DB round-trip before the app is ready).
    from . import entities

    words = set(_DOMAIN_WORDS)
    for d in entities.get_dept_names():
        words.update(re.findall(r"[a-zA-Z]+", d.lower()))
    for name, _eid in entities.get_employee_names():
        words.update(re.findall(r"[a-zA-Z]+", name.lower()))
    return words


def _ensure_domain_vocab():
    global _domain_vocab_loaded
    if not _domain_vocab_loaded:
        _spell.word_frequency.load_words(_load_domain_vocab())
        _domain_vocab_loaded = True


_WORD_OR_GAP = re.compile(r"[A-Za-z']+|[^A-Za-z']+")


_POSSESSIVE_SUFFIX = re.compile(r"('s|s')$", re.IGNORECASE)


def correct_typos(text):
    """Token-level typo correction. Only touches alphabetic words that:
    - are longer than 3 characters (short words are ambiguous to correct
      and rarely the source of a failed match anyway)
    - pyspellchecker considers unknown
    - aren't in the domain whitelist (dept/employee names, product jargon)
    - have a confident single correction available

    Numbers, punctuation, and department/employee names are always left
    exactly as typed.

    Possessive names ("Ashok Yadav's team") are handled by stripping the
    trailing 's/s' BEFORE the whitelist/correction check and reattaching it
    after - otherwise "yadav's" (the whole token, apostrophe included) is
    never found in the whitelist even though "yadav" itself is, and gets
    silently "corrected" into an unrelated word (confirmed real case:
    "yadav's" -> "adam's"). This is a general fix, not specific to any one
    name - it protects every whitelisted word's possessive form the same way
    the bare word is already protected.
    """
    _ensure_domain_vocab()
    out = []
    for tok in _WORD_OR_GAP.findall(text):
        if len(tok) > 3 and re.fullmatch(r"[A-Za-z']+", tok):
            suffix_match = _POSSESSIVE_SUFFIX.search(tok)
            suffix = suffix_match.group(1) if suffix_match else ""
            base = tok[: len(tok) - len(suffix)] if suffix else tok
            lower = base.lower()
            if len(base) <= 3 or _spell.known([lower]):
                out.append(tok)
                continue
            suggestion = _spell.correction(lower)
            if suggestion and suggestion != lower:
                corrected_base = suggestion.capitalize() if tok[0].isupper() else suggestion
                out.append(corrected_base + suffix)
            else:
                out.append(tok)
        else:
            out.append(tok)
    return "".join(out)
