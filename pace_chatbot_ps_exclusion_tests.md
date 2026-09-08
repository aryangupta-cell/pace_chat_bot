# PACE Chatbot — PS (ps_worked_flag_day) Exclusion Questions

Covers: "exclude PS non-working days, then show me [metric]" pattern, using `pace_1`'s `ps_worked_flag_day` column. When PS wasn't working that day, engagement/effectiveness/productivity numbers for that day are unreliable (no real usage data captured) — these questions ask the chatbot to filter those days OUT before computing the answer.

## A. Direct "exclude PS non-working days" + metric — 25

1. Excluding PS non-working days, what is Aryan Gupta's engagement this month?
2. Remove PS non-working days and tell me effectiveness for Megha Sharma
3. If PS wasn't working, ignore those days — what's the engagement % for [employee]?
4. Exclude days where PS was not working, show discipline % for [employee]
5. Filter out PS-not-working days, what's [employee]'s working hours %?
6. Ignoring days PS was off, what's [employee]'s productive minutes?
7. Excluding non-PS-working days, top 5 by engagement in [department]
8. Remove PS not working days, worst effectiveness in [department]
9. Excluding PS-off days, what's [employee]'s PACE score this month?
10. If PS was not working, exclude that day — [employee]'s engagement trend
11. Only count days PS was working, show [employee]'s effectiveness
12. Excluding PS-not-working days, compare [employee] vs [employee] on engagement
13. Remove days without PS data, [employee]'s discipline %
14. Excluding PS off days, [department]'s average engagement
15. Filter PS non-working days out, [employee]'s working hours trend
16. Excluding PS-not-working, who has the highest effectiveness in [department]
17. Ignore PS-off days, [employee]'s engagement month wise
18. Excluding days PS wasn't working, [employee]'s productive vs non-productive minutes
19. Remove PS not working days, my team's average effectiveness
20. Excluding PS-off days, [employee]'s AI tool usage minutes
21. Filter out non-PS-working days, [employee]'s WhatsApp minutes
22. Excluding PS non-working, [employee]'s engagement last week
23. Ignore days without PS, [employee]'s effectiveness this week
24. Remove PS-off days, [department]'s discipline % ranking
25. Excluding PS-not-working days, [employee]'s tools and mail minutes

## B. "How many PS non-working days" (informational, no metric) — 15

26. How many days was PS not working for [employee] this month?
27. How many PS non-working days does [employee] have?
28. Count of days PS was off for [employee]
29. How many days did [employee] have no PS data this month?
30. PS not working count for [employee]
31. Who has the most PS non-working days in [department]?
32. Which employees have zero PS-working days this month?
33. How many PS-working days does [employee] have this month?
34. PS working ratio for [employee]
35. What percentage of days was PS working for [employee]?
36. List employees with more than 5 PS non-working days
37. Who has the fewest PS-working days in [department]?
38. Show me [employee]'s PS working ratio trend by month
39. How many total PS non-working days company-wide this month?
40. Which department has the most PS non-working days?

## C. "PS not working" as an implicit data-quality caveat (chatbot should proactively note it, without being asked to exclude) — 15

41. What is [employee]'s engagement this month? (chatbot should flag if significant PS-off days exist and may be skewing the number)
42. [employee]'s effectiveness for [month] (same caveat check)
43. Top 5 by engagement in [department] (flag employees in the list with notable PS-off days)
44. [employee]'s productive minutes this month (flag PS-off day count)
45. Compare [employee] vs [employee] on effectiveness (flag if one has materially more PS-off days than the other, affecting fairness of comparison)
46-55. [10 more variants: same pattern across engagement/effectiveness/discipline/working-hours/productive-minutes, different employees/departments/months]

## D. Combining PS-exclusion with other existing filters — 15

56. Excluding PS non-working days, [employee]'s engagement in [department] for [month]
57. Excluding PS-off days, top 5 by effectiveness in my team
58. Remove PS non-working days, [employee]'s effectiveness last week
59. Excluding PS-not-working, who is red or black in [department] (does PS status factor into status determination — clarify if not)
60. Excluding PS-off days, [employee]'s month-wise engagement trend
61. Filter PS non-working, compare [department] vs [department] on effectiveness
62. Excluding days PS wasn't working, [employee]'s weekly effectiveness breakdown
63-70. [8 more combination variants]

## E. Edge cases — 10

71. Exclude PS non-working days for [employee] who has ZERO PS-working days this month (should return "no data" gracefully, not an error)
72. Excluding PS off days, [employee]'s engagement — [employee] has 100% PS-working days (should be identical to unfiltered result)
73. What does "PS not working" even mean? (chatbot should be able to explain the concept if asked directly)
74. Excluding PS non-working days AND excluding OT days, [employee]'s effectiveness (double filter combination)
75-80. [6 more edge cases: no PS data at all for an employee, PS-off for an entire month, mixing PS-exclusion with day-specific "yesterday" queries, etc.]

## What Claude Code should verify

- Confirm `ps_worked_flag_day=0` (or however "not working" is represented) rows are correctly excluded from the underlying aggregation for every metric category above, cross-check at least 2 with direct SQL
- Confirm the caveat-flagging behavior in category C is helpful, not noisy — only flag when PS-off days are a meaningful fraction (your judgment on threshold, but document it), not every single query
- Confirm category E edge cases degrade gracefully (no crash, sensible "no data" messaging) rather than erroring
