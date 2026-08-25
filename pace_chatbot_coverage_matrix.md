# PACE Chatbot — Full Coverage Matrix (Metrics × Time × Phrasing)

Goal: cover every combination of (metric) × (time period) × (sentence
pattern/voice), not just a fixed list of sentences — genuinely "any
English the user might type" is infinite, so the practical way to get
there is: enumerate the axes, let Claude Code combine them
systematically for trigger-pattern generation, AND strengthen the
fuzzy-match/semantic fallback so novel phrasings not explicitly listed
still route correctly.

## Axis 1 — Metrics/topics to cover
- PACE score (overall)
- PACE status (Black/Red/Amber/Green category)
- Discipline %
- Engagement %
- Effectiveness %
- Working hours %
- Attendance (present/absent)
- Late-coming
- Early-leaving
- Deficient hours
- Leave (on leave / leave count)
- WFH (work from home)
- Client visits
- Overtime (OT) hours
- Productive minutes
- WhatsApp/non-productive minutes
- AI tool usage minutes
- Tools & mail minutes
- Meetings
- Calls
- Tasks/todos completed
- d_score / quality score
- Roster/shift/OT status
- Device/PS online-offline status
- New joiner status
- Team/department/manager scope of any of the above

## Axis 2 — Time periods to cover
- Yesterday
- Today
- Current week (this week, ISO Mon-Sun)
- Last week
- Current month
- Last month
- A named month (e.g. "in July", "for August")
- A custom date (e.g. "on 2026-08-15", "on 15th August")
- A custom date range (e.g. "between Aug 1 and Aug 10")
- A custom week range (e.g. "first week of August")
- Multiple named months summed (e.g. "June, July, August")
- All-time / full available history
- "so far this month" (partial-month, in-progress)
- "over the last 3 months" (rolling trend)
- No time period stated at all (defaults to current month)

## Axis 3 — Sentence patterns & voice
For each (metric × time) combination, cover these query SHAPES:
- WHO: "who is/was/has [metric] [time]" (active)
- WHO (passive): "who has been [verb-ed] [time]" e.g. "who was marked absent yesterday"
- HOW MANY (count): "how many employees [verb] [metric] [time]"
- HOW MANY (passive): "how many employees were [verb-ed] [time]"
- IS/DOES (yes-no about a specific employee): "is [employee] on WFH today", "did [employee] visit a client yesterday"
- WHAT (value lookup): "what is [employee]'s [metric] [time]", "what was [employee]'s pace score last month"
- SHOW/LIST (explicit list request): "show me who was on leave yesterday", "list employees in red status"
- TELL ME (open request): "tell me about [employee]'s attendance this week"
- RANKING (superlative): "who has the highest/lowest/best/worst/most/least [metric] [time]"
- COMPARISON: "compare [employee/dept] and [employee/dept] on [metric] [time]"
- TREND: "how has [employee]'s [metric] changed [time]", "is [employee] improving in [metric]"
- CASUAL/VAGUE: "how's [employee] doing", "anyone slacking off [time]", "who's been solid [time]"
- SCOPED variants of all the above: + "in [department]", + "in my team", + "in [manager]'s team", + company-wide (no scope named)

## How Claude Code should use this
1. Treat this as a GENERATION SPEC, not a literal list — programmatically
   or manually combine Axis 1 × Axis 2 × Axis 3 to produce trigger
   regex patterns, prioritizing:
   - Metrics/times NOT yet covered by any existing intent (day-specific
     WFH/visit/leave/attendance counts and lists, status-category
     filters, custom date ranges, full trend history — per the
     200-question gap list already sent and implemented)
   - The most natural/common phrasing shapes first (WHO, HOW MANY,
     WHAT, RANKING) before rarer ones (CASUAL/VAGUE)
2. For combinations that are genuinely rare or awkward in natural
   English (e.g. "passive voice + custom date range + AI tool usage"),
   don't force literal trigger patterns for every one — instead ensure
   the FUZZY-MATCH FALLBACK (_fuzzy_match_intent in intents.py) has
   strong enough canonical-phrase coverage per metric+time combination
   that novel/unlisted phrasings still route correctly to the right
   intent and entity extraction (metric, time period, scope) still
   works even when the exact wording wasn't explicitly triggered.
3. Report back: which (metric × time) combinations got explicit trigger
   coverage vs which are relying on fuzzy fallback, so gaps are visible
   rather than silently assumed covered.

## Verification
- Spot-test at least 2 combinations per metric across at least 4
  different time periods (yesterday, last week, last month, custom
  date range) — confirm correct real-data responses
- Test at least 3 phrasing SHAPES (e.g. WHO, HOW MANY, RANKING) for the
  same underlying metric+time to confirm shape-independence
- Test at least 5 deliberately NOT-explicitly-listed phrasings to
  confirm the fuzzy fallback catches them correctly (this tests whether
  coverage generalizes, not just memorizes the list)
- Confirm opposite-direction safety margins still hold (best/worst,
  improving/declining) across all new combinations
