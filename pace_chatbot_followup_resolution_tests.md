# PACE Chatbot — Follow-Up Resolution Bug Class: Test Cases

## The bug (confirmed live)

Sequence: "how many employees were on visit on 25 august?" (count answer) → "can you share their names list" → WRONG: answered with the Red/Black company-wide list (stale context from several messages earlier) instead of the names behind the 37-person visit count just given.

Root cause hypothesis: a vague follow-up ("share their names", "list them", "who are they") needs to resolve to whatever the MOST RECENT answer was about — but the app appears to be resolving it against some older/stickier tracked context (e.g. last-mentioned status category) instead of the true most-recent answer. This is a different bug from the department/employee/time-period stickiness fixed earlier — that was about missing scope in a NEW question; this is about "expand/list the thing I was JUST shown" resolving to the wrong prior turn.

## Test case categories (~70 total)

### A. Count → "show me their names" (should expand THAT count's list) — 15

1. "how many on WFH yesterday" → "show me their names"
2. "how many on leave today" → "list them"
3. "how many were absent yesterday" → "who are they"
4. "how many came late yesterday" → "can you share the names"
5. "how many did OT yesterday" → "who were they"
6. "how many are red in SCM" → "give me the list"
7. "how many are black company-wide" → "share names please"
8. "how many are on visit this week" → "list their names"
9. "how many took leave this month" → "who took leave"
10. "how many had zero productive minutes yesterday" → "names?"
11. "how many were offline yesterday" → "who was offline"
12. "how many new joiners this month" → "who are the new joiners"
13. "how many are amber in IT-Development" → "show me who"
14. "how many attended yesterday in Founders Office" → "list them"
15. "how many left early today" → "names please"

### B. Ranking → "give me the full list" (should expand THAT ranking) — 10

16. "top 5 by pace score in Accounts" → "show me the full list, not just 5"
17. "who has the worst attendance in SCM" → "show more"
18. "biggest score drop in Founders Office" → "who else"
19. "most improved this month" → "give me the whole list"
20. "top 10 productive employees" → "show 20 instead"
21. "worst 5 in Control Tower" → "expand this"
22. "who is declining this month" → "how many total"
23. "highest WFH days in Sales" → "who's next after them"
24. "lowest discipline % in IT-Development" → "list everyone"
25. "who has the most OT hours" → "full list please"

### C. Individual employee answer → vague follow-up referring to a DIFFERENT prior list-type answer (the exact bug pattern found) — 15

26. "aryan gupta score" → [earlier: "how many red in SCM"] → "list them" (should resolve to the SCM red list, NOT Aryan)
27. "how many on visit yesterday" → "megha sharma score" → "list them" (ambiguous — should probably ask for clarification, not guess)
28. "who is red in Accounts" → "aryan gupta's pace score" → "show me the names" (should go back to Accounts red list, not something stale from further back)
29. "how many on leave today" → "is aryan improving" → "who are they" (should resolve to the leave-today list, not aryan-related)
30. "top 5 in IT-Development" → "megha sharma score" → "give me the rest of the list" (should resolve to IT-Development ranking)
31-40. [same pattern, 10 more combinations across different metric pairs: WFH-count + individual-lookup + "list them"; red-category + individual-lookup + "who are they"; visit-count + individual-lookup + "share names"; etc.]

### D. Ambiguous follow-ups after MULTIPLE list-type answers in a row (should resolve to the MOST RECENT one) — 10

41. "how many on WFH yesterday" → "how many on leave yesterday" → "list them" (should be the leave list, the most recent count, not WFH)
42. "who is red in SCM" → "who is black in SCM" → "show me the names" (should be black, not red)
43. "top 5 in Accounts" → "top 5 in Billing" → "give me the list" (should be Billing, not Accounts)
44. "how many on visit today" → "how many absent today" → "who are they" (should be absent, not visit)
45-50. [5 more triple-chain variants: three list-type answers in a row, vague follow-up should always target the 3rd/most recent]

### E. Vague follow-up with NO prior list-type answer at all (should ask for clarification, never guess/hallucinate a list) — 10

51. Fresh session → "list them" (nothing to expand — should ask "list who?")
52. "aryan gupta score" (single-value answer, no list) → "show me their names" (there's no "them" — should clarify)
53. "is he improving" (single-value trend) → "who are they" (should clarify, not guess)
54-60. [6 more: various single-value answers followed by a vague plural follow-up that has nothing valid to expand]

### F. Explicit re-scoping mid-follow-up (should override, not blend) — 10

61. "how many on WFH yesterday" → "list them for Founders Office only" (should filter the WFH list to Founders Office, not ignore the department)
62. "top 5 in Accounts" → "show me the black ones instead" (should switch to a status filter, not try to blend with the ranking)
63. "how many red in SCM" → "what about last month" (should re-run the red-category count for last month, not the current sticky department alone)
64-70. [7 more re-scoping variants]

## What Claude Code should verify

For each category, confirm:

- The follow-up resolves to the CORRECT prior answer (most recent list-producing turn), not a stale/earlier one
- Category E: no hallucinated list ever appears — always a clarification question when there's genuinely nothing to expand
- Category F: explicit re-scoping in the follow-up correctly overrides/filters, rather than being ignored or blended incorrectly
- This must work consistently regardless of whether individual-employee lookups (which don't produce a "list") are interspersed in between
