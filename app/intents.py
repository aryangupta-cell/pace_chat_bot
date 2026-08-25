import re

from rapidfuzz import fuzz

# (intent_name, compiled regex list) — first match wins, checked in order.
# Order matters: more specific patterns are placed before generic ones that
# could otherwise shadow them (e.g. "compare" and "employee detail" style
# questions must be checked before generic best/worst ranking patterns).

_ATTENDANCE_BEST_PATTERNS = [
    r"\bbest attendance\b",
    r"\bmost punctual\b",
    r"\btop attendance\b",
    r"\bleast (late[- ]?comers?|late)\b",
    r"\bwho.*best attendance\b",
    # casual phrasing round 3
    r"\bwho'?s (most|the most) punctual\b",
    r"\bwho comes? on time\b", r"\bwho'?s never late\b",
    r"\bgreat attendance\b", r"\bstrongest attendance\b",
    r"\bwho has the best attendance\b",
]

_ATTENDANCE_WORST_PATTERNS = [
    r"\bworst attendance\b",
    r"\bmost absent\b",
    r"\bwho is defaulting\b",
    r"\bpoor attendance\b",
    r"\bmost late[- ]?comers?\b",
    r"\bbad attendance\b",
    r"\battendance (problems?|issues?)\b",
    # casual phrasing round 3
    r"\bwho keeps? missing (work|days)\b",
    r"\bwho has the worst attendance\b", r"\battendance is (bad|poor|terrible)\b",
    r"\bwho'?s absent (the )?most\b", r"\bmost absences?\b",
]

_PRODUCTIVE_HIGH_PATTERNS = [
    r"\bmost productiv(e|ity)( time)?\b",
    r"\bwho works? the most\b",
    r"\bhighest productiv(e|ity)(\s+minutes?|\s+time)?\b",
    r"\btop performer by time\b",
    r"\bworking hard\b", r"\bworks? hard\b",
    r"\bpulling (their|his|her) weight\b",
    # casual phrasing round 3
    r"\bwho'?s the most productive\b", r"\bmost hard[- ]?working\b",
    r"\bbusiest employees?\b", r"\bwho puts? in the most (time|hours)\b",
    r"\btop performers? by productivity\b",
]

_PRODUCTIVE_LOW_PATTERNS = [
    r"\bleast productiv(e|ity)( time)?\b",
    r"\bwho works? the least\b",
    r"\blowest productiv(e|ity)(\s+minutes?|\s+time)?\b",
    r"\bslack(ing)? off\b",
    r"\bwasting time\b",
    r"\bnot pulling (their|his|her) weight\b",
    # casual phrasing round 3
    r"\bwho'?s the least productive\b", r"\bleast hard[- ]?working\b",
    r"\bidlest employees?\b", r"\bwho puts? in the least (time|hours)\b",
    r"\bbarely working\b",
]

_IMPROVING_PATTERNS = [
    r"\bwho is improving\b",
    r"\bwho'?s getting better\b",
    r"\btop improvers?\b",
    r"\bwho improved( the most)?\b",
    r"\bimprovement this month\b",
    r"\bwho is trending up\b",
    # casual phrasing round 3
    r"\bwho'?s on the (up|upward) (swing|trend)\b", r"\bwho'?s doing better\b",
    r"\bwho'?s picking up\b", r"\bwho'?s bouncing back\b",
]

_DECLINING_PATTERNS = [
    r"\bwho is declining\b",
    r"\bwho'?s getting worse\b",
    r"\bbiggest drop\b",
    r"\bwho declined( the most)?\b",
    r"\bwho is trending down\b",
    r"\bwho fell behind\b",
    # casual phrasing round 3
    r"\bwho'?s on the (down|downward) (swing|trend)\b", r"\bwho'?s doing worse\b",
    r"\bwho'?s slipping\b", r"\bwho'?s sliding\b",
]

# --- Casual phrasing (new round): performance/attendance/productivity/score/
# comparison/team/general natural-language triggers for EXISTING intents.
# No new query logic here - every pattern below resolves to an intent whose
# handler already exists in main.py's answer_intent(). Grouped to mirror the
# 8 phrase groups from the task; each block augments the pattern lists above
# or below, defined here up top only when the target list is also above.

# --- Category A: single-employee lookups -----------------------------------
_EMP_PACE_SCORE_PATTERNS = [
    r"\bpace score (of|for)\b", r"\bwhat is .* pace score\b",
    r"\bwhat'?s .* score\b", r"\bhow'?s .* score\b",
    r"\b(green|red|black) zone\b", r"\bwhat zone\b", r"\bwhat status\b.*\bin\b",
    # casual phrasing round 3
    r"\bshow me .* pace score\b", r"\bcan you tell me .* pace score\b",
    r"\bpace score (please|pls)\b", r"\bcurrent (pace )?score of\b",
    r"\bwhat'?s the (pace )?score for\b",
    r"\bhow'?s .* doing on (their|his|her) score\b",
    r"\bwhat'?s the (deal|story) with .* score\b",
    # Bug 1 fix: bare "score" (no "pace" prefix) should default to PACE score
    # for individual lookups, same as "pace score of X" already works.
    r"\bwhat is .*'?s? score\b", r"\bwhat is the score (of|for)\b",
    r"\bscore (of|for)\b", r"\bwhats .* score\b",
]
_EMP_ATTENDANCE_SUMMARY_PATTERNS = [
    r"\battendance (summary|of|for)\b",
    r"\btell me about\b(?!.*\bdepartment\b)(?!.*\bteam\b)",
    # casual phrasing round 3
    r"\bhow (was|is) .* attendance\b", r"\bgive me .* attendance\b",
    r"\battendance record (of|for)\b", r"\battendance details? (of|for)\b",
]
_EMP_LATE_COMINGS_PATTERNS = [
    r"\blate[- ]?comings?\b.*\b(for|of)\b", r"\bhow many times.*late\b",
    r"\bhow often (is|was|has)\b.*\blate\b", r"\bhow many late[- ]?comings?\b",
]
_EMP_EARLY_LEAVINGS_PATTERNS = [
    r"\bearly leav(ing|ings)\b",
    r"\bhow many times.*left early\b", r"\bhow often.*(left|leaves) early\b",
]
_EMP_PRODUCTIVE_TIME_PATTERNS = [
    r"\bproductiv(e|ity) time (of|for)\b", r"\bhow productive (is|was)\b",
    r"\bproductive minutes (of|for)\b", r"\bhow much time.*productive\b",
]
_EMP_WHATSAPP_PATTERNS = [
    r"\bwhatsapp usage\b", r"\bwhatsapp (minutes|time) (of|for)\b",
    r"\bhow much whatsapp\b", r"\btime (spent )?on whatsapp\b",
]
_EMP_AI_USAGE_PATTERNS = [
    r"\bai (tool )?usage\b", r"\bai (minutes|time) (of|for)\b",
    r"\bhow much ai (tool )?usage\b", r"\btime (spent )?on ai tools?\b",
]
_EMP_TREND_PATTERNS = [
    r"\bis .* (improving|declining)\b", r"\bimproving or declining\b",
    # casual phrasing: "is aryan performance increasing or decreasing?" etc.
    r"\bincreasing or decreasing\b",
    r"\bgetting better or worse\b",
    r"\bgoing up or down\b",
    r"\bhow'?s .* trending\b",
    r"\bis .* performance (going up|going down)\b",
]
_EMP_DEPARTMENT_PATTERNS = [
    r"\bdepartment of\b", r"\bwhich department (is|does)\b", r"\bwhat department\b",
    r"\bwhich team is\b.*\bin\b", r"\bwhat dept is\b",
]
_EMP_MANAGER_PATTERNS = [
    r"\breporting manager of\b", r"\bwho does .* report to\b", r"\bmanager of\b",
    r"\bwho'?s .* manager\b", r"\bwho does .* report into\b",
]
_EMP_DISCIPLINE_PATTERNS = [
    r"\bdiscipline (%|percent|percentage) (of|for)\b", r"\bdiscipline of\b",
    r"\bhow disciplined (is|was)\b",
]
_EMP_ENGAGEMENT_PATTERNS = [
    r"\bengagement (%|percent|percentage) (of|for)\b",
    r"\bhow engaged (is|was)\b",
]
_EMP_EFFECTIVENESS_PATTERNS = [
    r"\beffectiveness (%|percent|percentage) (of|for)\b",
    r"\bhow effective (is|was)\b",
]
_EMP_DEFICIENT_HOURS_PATTERNS = [
    r"\bdeficient hours? (of|for)\b", r"\bhow many deficient hours?\b",
]
_EMP_WORKING_PCT_PATTERNS = [
    r"\bworking hours? (%|percent|percentage) (of|for)\b",
    r"\bworking hours? percentage\b",
]

# --- Category B/C: generic best/worst metric rankings -----------------------
# JUDGMENT CALL: a bare "top/bottom N employees" mention with no metric named
# (e.g. "bottom 5 emp in IT-Development") defaults to ranking by PACE score -
# same default already established for dept_best/dept_worst when no metric is
# named. Flagged here rather than silently added.
_PACE_SCORE_BEST_PATTERNS = [
    r"\bbest pace score\b", r"\bhighest pace score\b", r"\btop\b.*\bpace score\b",
    r"\btop\s*\d+\b.*\b(emp|employee|employees|people)\b",
    r"\bbest performers?\b", r"\btop performers?\b(?!\s+by\s+time)",
    # casual phrasing round 3
    r"\bwho'?s (doing|performing) (the )?best\b", r"\bstar performers?\b",
    r"\bwho'?s crushing it\b", r"\btop scorers?\b",
    # superlative-synonym coverage (gap category E): most/highest/best/strongest
    r"\bmost pace score\b", r"\bstrongest performer\b", r"\bwho is doing the best\b",
    r"\btop performer in my team\b",
    # Bug 1 fix: bare "score" ranking synonyms (no "pace" prefix)
    r"\bhighest score\b", r"\bbest score\b", r"\btop\b.*\bscore\b",
    r"\bwho has the highest score\b",
]
_PACE_SCORE_WORST_PATTERNS = [
    r"\bworst pace score\b", r"\blowest pace score\b", r"\bbottom\b.*\bpace score\b",
    r"\bbottom\s*\d+\b.*\b(emp|employee|employees|people)\b",
    r"\bworst performers?\b", r"\bwho needs? help\b", r"\bwho'?s struggling\b",
    r"\banything (i )?should (i )?flag\b", r"\bany red flags?\b", r"\banything concerning\b",
    # casual phrasing round 3
    r"\bwho'?s (doing|performing) (the )?worst\b", r"\blowest scorers?\b",
    r"\bwho'?s falling behind\b", r"\bwho needs? (the most )?attention\b",
    # superlative-synonym coverage (gap category E): least/lowest/worst/weakest/fewest
    r"\bleast pace score\b", r"\bwho has the least pace score\b",
    r"\bweakest performer\b", r"\bwho is struggling the most\b",
    r"\bbottom performer in my team\b",
    # Bug 1 fix: bare "score" ranking synonyms (no "pace" prefix)
    r"\blowest score\b", r"\bworst score\b",
    r"\bwho has the lowest score\b",
]
_ENGAGEMENT_HIGH_PATTERNS = [
    r"\bhighest engagement\b", r"\bmost engaged\b", r"\btop\b.*\bengagement\b",
    r"\bwho'?s the most engaged\b", r"\bmost engaged employee\b",
]
_ENGAGEMENT_LOW_PATTERNS = [
    r"\blowest engagement\b", r"\bleast engaged\b", r"\bbottom\b.*\bengagement\b",
    r"\bwho'?s the least engaged\b", r"\bleast engaged employee\b",
]
_EFFECTIVENESS_HIGH_PATTERNS = [
    r"\bhighest effectiveness\b", r"\bmost effective\b", r"\btop\b.*\beffectiveness\b",
    r"\bwho'?s the most effective\b",
]
_EFFECTIVENESS_LOW_PATTERNS = [
    r"\blowest effectiveness\b", r"\bleast effective\b", r"\bbottom\b.*\beffectiveness\b",
    r"\bwho'?s the least effective\b",
]
_MOST_LATE_COMINGS_PATTERNS = [r"\bmost late[- ]?comings?\b", r"\bwho'?s late the most\b", r"\bmost late[- ]?coming days\b"]
_FEWEST_LATE_COMINGS_PATTERNS = [r"\bfewest late[- ]?comings?\b", r"\bfewest late[- ]?coming days\b", r"\bleast late[- ]?comings?\b"]
_MOST_EARLY_LEAVINGS_PATTERNS = [r"\bmost early leav(ing|ings)\b", r"\bwho leaves early the most\b"]
_MOST_DEFICIENT_HOURS_PATTERNS = [r"\bmost deficient hours?\b", r"\bwho has the most deficient hours?\b"]
_MOST_DISCIPLINED_PATTERNS = [r"\bmost disciplined\b", r"\bhighest discipline\b", r"\bwho'?s the most disciplined\b", r"\bhighest discipline %\b"]
_LEAST_DISCIPLINED_PATTERNS = [r"\bleast disciplined\b", r"\blowest discipline\b", r"\blowest discipline %\b"]
_MOST_WHATSAPP_PATTERNS = [r"\bhighest whatsapp\b", r"\bmost whatsapp\b", r"\bwho uses whatsapp the most\b"]
_LOWEST_WORKING_PCT_PATTERNS = [r"\blowest working hours? %\b", r"\blowest working hours? percentage\b"]
_HIGHEST_WORKING_PCT_PATTERNS = [r"\bhighest working hours? %\b", r"\bhighest working hours? percentage\b"]
_FEWEST_WFH_PATTERNS = [r"\bfewest wfh days?\b", r"\bleast wfh days?\b"]
_MOST_WFH_DAYS_PATTERNS = [r"\bmost wfh days?\b"]
_FEWEST_VISITS_PATTERNS = [r"\bfewest visits?\b", r"\bleast visits?\b"]
_SMALLEST_SCORE_CHANGE_PATTERNS = [r"\bsmallest score change\b"]
_BIGGEST_SCORE_CHANGE_PATTERNS = [r"\bbiggest score change\b"]
_LEAST_IMPROVED_PATTERNS = [r"\bleast improved employee\b"]
_MOST_IMPROVED_PATTERNS = [r"\bmost improved employee\b"]
_WORST_SUBSCORE_COMPANY_PATTERNS = [r"\bworst sub[- ]?score across the whole company\b", r"\bworst sub[- ]?score company[- ]?wide\b"]

# --- Category D: improving/declining extensions -----------------------------
_TREND_2MONTH_PATTERNS = [
    r"\bover the last (two|2) months?\b", r"\bchanged over.*months?\b", r"\blast 2 months\b",
    r"\bpast (two|2) months?\b", r"\bhow'?s .* changed over the last.*months?\b",
]
_TEAM_IMPROVING_PATTERNS = [
    r"\bis my team improving\b", r"\bhow.*my team.*(improving|doing)\b.*trend",
    r"\bis my team getting better\b", r"\bhow'?s my team trending\b",
    r"\bis my team doing better or worse\b",
    r"\bis my team getting worse\b", r"\bis my team declining\b",
]
_DEPT_TREND_PATTERNS = [
    r"\bwhich department is (improving|declining)\b",
    r"\bwhich department is trending (up|down)\b",
    r"\bwhich department is (getting better|getting worse)\b",
]

# --- Category E: department aggregates --------------------------------------
_DEPT_AVG_PATTERNS = [
    r"\baverage pace score\b.*\bdepartment\b", r"\bdepartment.*average\b",
    r"\bavg pace score\b.*\bdepartment\b", r"\bmean pace score\b.*\bdepartment\b",
]
_DEPT_BEST_PATTERNS = [
    r"\bbest department\b", r"\btop department\b", r"\bwhich department is (the )?best\b",
    r"\bstrongest department\b", r"\bwhich department is doing (the )?best\b",
]
_DEPT_WORST_PATTERNS = [
    r"\bworst department\b", r"\bbottom department\b", r"\bwhich department is (the )?worst\b",
    r"\bweakest department\b", r"\bwhich department is doing (the )?worst\b",
]
_DEPT_COUNT_PATTERNS = [
    r"\bhow many (employees|people).*(department|in)\b", r"\bemployee count\b", r"\bheadcount\b",
    r"\bhow many people work in\b", r"\bnumber of employees in\b",
]
_DEPT_SUMMARY_PATTERNS = [
    r"\bdepartment summary\b", r"\bhow is .* department doing\b",
    r"\bhow'?s .* department (looking|performing)\b",
]
_DEPT_COMPARE_PATTERNS = [
    r"\bcompare\b.*\bdepartments?\b", r"\bcompare .* (vs\.?|versus) .*department\b",
    r"\bwhich department is better\b.*\bvs\.?\b",
]

# --- Category F: attendance specifics ----------------------------------------
_CHRONIC_LATE_PATTERNS = [
    r"\bchronically late\b", r"\bhabitually late\b",
    r"\balways late\b", r"\bconstantly late\b",
    r"\b(coming|comes) late a lot\b", r"\bkeeps? coming late\b",
    # casual phrasing round 3
    r"\bchronic latecomers?\b", r"\brepeat(edly)? late\b",
]
_PERFECT_ATTENDANCE_PATTERNS = [
    r"\bperfect attendance\b",
    r"\bnever misses? (a )?days?\b", r"\bnever absent\b",
    r"\b100% attendance\b", r"\bfull attendance\b",
]
_DEFAULTER_RANKING_PATTERNS = [
    r"\bmost defaulter\b", r"\bdefaulter ranking\b", r"\btop defaulters?\b",
    r"\bworst defaulters?\b",
]
_DEFICIT_HOURS_RANKING_PATTERNS = [
    r"\bdeficit hours? ranking\b", r"\bmost deficit hours?\b",
    r"\bwho has the most deficit hours?\b",
]

# --- Category G: productivity/usage specifics --------------------------------
_MEETING_MIN_PATTERNS = [
    r"\bmeeting (minutes|time)\b", r"\btime (spent )?in meetings\b",
    r"\bhow much time in meetings\b",
]
_COMPARE_EMPLOYEES_PATTERNS = [
    r"\bcompare\b(?!.*\bdepartments?\b)",
    r"\bwho'?s better\b(?!.*\bteam\b)", r"\bwho is better\b(?!.*\bteam\b)",
    r"\bhow do .* stack up\b(?!.*\bteam\b)(?!.*\bdepartment\b)",
]

# --- Category H: team/manager view -------------------------------------------
_TEAM_HOW_DOING_PATTERNS = [
    r"\bhow is my team doing\b", r"\bhow'?s my team\b", r"\bmy team.*average\b",
    r"\bhow'?s my team looking\b", r"\bmy team looking\b",
    r"\bhow'?s my team performing\b", r"\bmy team'?s performance\b",
    r"\bgive me a summary of my team\b",
    r"\bcatch me up on my team\b", r"\bwhat'?s up with my team\b",
    r"\bgive me the rundown on my team\b",
]
_TEAM_LOWEST_SCORERS_PATTERNS = [
    r"\blowest scorers? in my team\b", r"\bworst.*my team\b.*pace score",
    r"\bwho needs? help (in|on) my team\b",
    r"\bwho'?s struggling (on|in) my team\b", r"\bmy team'?s weakest\b",
]
_TEAM_COMPARE_PATTERNS = [
    r"\bcompare my team\b",
    r"\bis my team better than\b", r"\bmy team.*(better|worse) than\b",
    r"\bhow does my team stack up (against|to)\b",
]
_NEW_JOINERS_PATTERNS = [
    r"\bnew joiners?\b", r"\brecently joined\b",
    r"\bwho joined recently\b", r"\bnew hires?\b", r"\bnew employees?\b",
]

# --- Category A (new): Leave & absence ---------------------------------------
_LEAVE_WHO_PATTERNS = [
    r"\bwho\S* (is|are|was|were)\s+on leave\b", r"\bwho'?s on leave\b",
    r"\bwho took leave\b", r"\bwho'?s off (today|this week)\b",
]
_LEAVE_EMP_CHECK_PATTERNS = [
    r"\bwas\b.*\bon leave\b", r"\bis\b.*\bon leave\b",
    r"\bdid\b.*\btake (any )?leave\b",
    r"\bhas\b.*\bbeen on leave\b",
]
_HALF_DAY_RANKING_PATTERNS = [
    r"\bhalf[- ]?day\b.*\b(count|ranking|most)\b", r"\bmost half[- ]?days?\b",
    r"\bwho has the most half[- ]?days?\b",
]
_LEAVE_BY_DEPT_PATTERNS = [
    r"\bleave (count|counts|totals?)\b.*\bdepartment\b", r"\bleaves? by department\b",
    r"\bhow many leaves? (does|did)\b.*\bdepartment\b",
]
_ZERO_LEAVE_PATTERNS = [
    r"\bzero leaves?\b", r"\bno leaves? taken\b", r"\bwithout (any )?leave\b",
    r"\bwho hasn'?t taken (any )?leave\b", r"\bnever took leave\b",
]

# --- Category B (new): Calls ---------------------------------------------------
_CALL_MOST_PATTERNS = [
    r"\bmost calls\b", r"\bhighest (number of )?calls\b", r"\btop calls\b",
    r"\bwho made the most calls\b", r"\bmost phone calls\b",
]
_CALL_FEWEST_PATTERNS = [
    r"\bfewest calls\b", r"\bleast calls\b", r"\blowest (number of )?calls\b",
    r"\bwho made the fewest calls\b",
]
_CALL_DURATION_PATTERNS = [
    r"\bcall duration\b", r"\baverage call\b", r"\blongest calls?\b",
    r"\bhow long (are|were) the calls\b",
]
_CALL_EMP_PATTERNS = [
    r"\bcalls? (for|of|made by)\b", r"\bhow many calls\b",
    r"\bnumber of calls (for|of|by)\b",
]

# --- Category C (new): Visits ---------------------------------------------------
_VISIT_RANKING_PATTERNS = [
    r"\bmost visits?\b", r"\bvisit ranking\b", r"\btop visits?\b",
    r"\bwho made the most visits?\b",
]
_ZERO_VISIT_PATTERNS = [
    r"\bzero visits?\b", r"\bno visits?\b", r"\bnever visited\b",
    r"\bwho hasn'?t made (any )?visits?\b",
]
_VISIT_EMP_PATTERNS = [
    r"\bvisits? (for|of|by)\b", r"\bdid\b.*\bvisit\b",
    r"\bhow many visits? (did|has)\b",
]

# --- Category D (new): WFH ------------------------------------------------------
_WFH_EMP_PATTERNS = [
    r"\bwfh\b.*\b(for|of)\b", r"\bwork(ed|ing)? from home\b.*\b(for|of)\b", r"\bwas\b.*\bwfh\b",
    r"\b(was|is)\b.*\bworking from home\b",
    r"\bhow many wfh days (did|has)\b", r"\bwfh days (of|for)\b",
]
_WFH_RANKING_PATTERNS = [
    r"\bmost wfh\b", r"\bmost work from home\b", r"\bwfh ranking\b",
    r"\bwho works? from home the most\b",
]
_WFH_BY_DEPT_PATTERNS = [
    r"\bwfh\b.*\bdepartment\b", r"\bwork from home\b.*\bdepartment\b",
    r"\bhow much wfh\b.*\bdepartment\b",
]

# --- Category E (new): Tasks/todos ----------------------------------------------
_TODOS_CREATED_PATTERNS = [
    r"\bmost todos? created\b", r"\btodos? created\b.*\branking\b",
    r"\bwho created the most todos?\b",
]
_TODOS_ASSIGNED_PATTERNS = [
    r"\bmost todos? assigned\b", r"\btodos? assigned\b.*\branking\b",
    r"\bwho was assigned the most todos?\b", r"\bwho got the most todos?\b",
]
_TASKS_CREATED_PATTERNS = [
    r"\bmost tasks? created\b", r"\btasks? created\b.*\branking\b",
    r"\bwho created the most tasks?\b",
]
_TASKS_ASSIGNED_PATTERNS = [
    r"\bmost tasks? assigned\b", r"\btasks? assigned\b.*\branking\b",
    r"\bwho was assigned the most tasks?\b", r"\bwho got the most tasks?\b",
]
_ONTIME_COMPLETION_PATTERNS = [
    r"\bon[- ]?time completion\b", r"\bwho completes? (tasks|todos) on time\b",
]
_RESPONSIVENESS_PATTERNS = [
    r"\bresponsiveness score\b", r"\bwho'?s (most|the most) responsive\b",
]
_EXTENSION_ADHERENCE_PATTERNS = [
    r"\bextension adherence\b", r"\bhow often.*extensions? (are|were) followed\b",
]

# --- Category F (new/extended): Meetings ----------------------------------------
_MEETING_COUNT_PATTERNS = [
    r"\bmost meetings\b", r"\bmeeting count\b", r"\bnumber of meetings\b",
    r"\bwho has the most meetings\b", r"\bwho attends? the most meetings\b",
]
_MEETING_RATIO_PATTERNS = [
    r"\bmeeting.{0,10}ratio\b", r"\bmeeting.{0,15}productive.{0,10}(ratio|share)\b",
    r"\bwhat (share|portion|fraction) of time.*meetings\b",
]

# --- Category G (new): Quality / d_score -----------------------------------------
_D_SCORE_RANKING_PATTERNS = [
    r"\bd[- ]?score ranking\b", r"\bbest d[- ]?score\b", r"\bhighest d[- ]?score\b", r"\bworst d[- ]?score\b", r"\blowest d[- ]?score\b",
    r"\bwho has the (best|highest) d[- ]?score\b", r"\bwho has the (worst|lowest) d[- ]?score\b",
]
_D_SCORE_EMP_PATTERNS = [
    r"\bd[- ]?score (of|for)\b", r"\bwhat'?s .* d[- ]?score\b",
]
_D_SCORE_TREND_PATTERNS = [
    r"\bd[- ]?score.*(improv|declin)\b", r"\bis .* d[- ]?score (improving|declining)\b",
]

# --- Category H (new): Roster/shift/OT -------------------------------------------
_SHIFT_TYPE_EMP_PATTERNS = [
    r"\bshift type (of|for)\b", r"\bwhat shift\b",
    r"\bwhich shift (is|does)\b",
]
_OT_RANKING_PATTERNS = [
    r"\bmost (ot|overtime) hours?\b", r"\bovertime ranking\b",
    r"\bwho works? the most overtime\b", r"\bwho has the most ot\b",
]
_BREAKSHIFT_EMP_PATTERNS = [
    r"\bbreak[- ]?shift\b", r"\bis .* on break[- ]?shift\b",
]

# --- Category I (new): Offline/device status --------------------------------------
_OFFLINE_RANKING_PATTERNS = [
    r"\bmost offline attendance\b", r"\boffline attendance ranking\b",
    r"\bwho has the most offline (attendance|days)\b",
]
_OFFLINE_EMP_PATTERNS = [
    r"\boffline attendance (of|for)\b", r"\bps installed\b.*\bfor\b",
    r"\bhow many offline days (does|did)\b",
]
_PS_INSTALL_RATE_PATTERNS = [
    r"\bps install(ation|ed)? rate\b", r"\bpace software install\b",
    r"\bwhat percent(age)? have ps installed\b",
]

# --- Category J (new): Org info ---------------------------------------------------
_GRADE_LOOKUP_PATTERNS = [
    r"\bemployees? (with|in) grade\b", r"\bgrade\s+\w+\b.*\bemployees?\b",
    r"\bwho'?s in grade\b", r"\blist (employees|people) (with|in) grade\b",
]
_DESIGNATION_BREAKDOWN_PATTERNS = [
    r"\bdesignation breakdown\b", r"\bdesignations? by department\b",
    r"\bwhat designations? (are|exist)\b",
]
_AVG_TENURE_PATTERNS = [
    r"\baverage tenure\b", r"\bhow long have employees been here\b",
    r"\bhow long (has|have).*(worked|been working) here\b",
]

# --- Category K (new round 2) -------------------------------------------------
_SCORE_DROP_PATTERNS = [
    r"\bbiggest (score )?drop\b", r"\bbiggest decline\b", r"\bworst score drop\b",
    r"\bwhose score dropped( the most)?\b", r"\bwho(se)? score dropped\b",
    r"\bwho dropped( the most)?\b", r"\bwhose score fell( the most)?\b",
    r"\bwho'?s score dropped( the most)?\b",
    r"\bwhose pace score dropped( the most)?\b",
    r"\bbiggest score drop\b", r"\blargest score drop\b",
    r"\bwho had the biggest (score )?drop\b",
    r"\bwhose score (went down|decreased)( the most)?\b",
    r"\bscore dropped the most\b", r"\bscore fell the most\b",
]
_SCORE_IMPROVEMENT_ALLTIME_PATTERNS = [
    r"\bimproved the most\b.*\b(overall|ever|total|all time|since (they|he|she) joined)\b",
    r"\b(overall|all time) (improvement|improver)\b",
    r"\bwho improved the most overall\b",
    r"\bwhose score improved( the most)?\b", r"\bwho(se)? score improved\b",
    r"\bwhose score (went up|increased)( the most)?\b",
    r"\bscore improved the most\b", r"\bwhose pace score improved( the most)?\b",
    r"\bwho had the biggest (score )?improvement\b",
    r"\bbiggest score improvement\b", r"\blargest score improvement\b",
]
_EMP_OVERVIEW_PATTERNS = [
    r"\bhow is\b(?!.*\bmy team\b)(?!.*\bdepartment\b).*\bperforming\b",
    r"\bhow'?s\b(?!.*\bmy team\b).*\bperforming\b",
    r"\bgive me an overview of\b", r"\boverview of\b(?!.*\bdepartment\b)",
    r"\bhow is\b(?!.*\bmy team\b)(?!.*\bdepartment\b).*\bdoing\b",
    # casual phrasing round 3
    r"\bsum up\b.*\bperformance\b", r"\bfull picture (of|for)\b",
    r"\bhow'?s\b(?!.*\bmy team\b)(?!.*\bdepartment\b).*\bdoing\b",
]
_SUBSCORE_COMPARE_PATTERNS = [
    r"\b(weakest|strongest)\b.*\b(engagement|effectiveness|discipline)\b",
    r"\bwhich (is|of).*(weakest|strongest)\b",
    r"\bengagement.{0,20}effectiveness.{0,20}discipline\b",
    r"\bwhat'?s .* weakest (area|score|sub[- ]?score)\b",
    r"\bwhat'?s .* strongest (area|score|sub[- ]?score)\b",
    # "which area is X lacking the most" / "where is X weakest" / "is X
    # lacking in discipline or engagement or effectiveness" and similar
    # natural variants - all resolve to the SAME raw-current-value
    # weakest/strongest comparison as the patterns above (see
    # queries.subscore_compare_for_employee / the subscore_compare_emp
    # branch in main.answer_intent), not a month-over-month trend.
    r"\bwhich (area|part)\b.*\b(lacking|weak|weakest|declining)\b",
    r"\bin which (area|part)\b.*\blacking\b",
    r"\bwhere\b.*\b(is|are)\b.*\bweakest\b",
    r"\b(what|which) is\b.*\bweakest (area|score|sub[- ]?score)\b",
    r"\bis\b.*\blacking in\b.*\b(discipline|engagement|effectiveness)\b",
    r"\b(discipline|engagement|effectiveness)\b.*\bor\b.*\b(discipline|engagement|effectiveness)\b.*\bor\b.*\b(discipline|engagement|effectiveness)\b",
    r"\bweakest area\b", r"\bstrongest area\b",
    # Named employee + "declining/lacking" + abbreviated or full sub-score
    # area words joined by "or" - e.g. "aryan is declining in which
    # discipline eng or eff" / "aryan declining in eng or eff". Without
    # this, these phrasings matched NO regex pattern at all (the triple-
    # "or" pattern above needs 3 full-word areas; abbreviations like
    # "eng"/"eff" and a 2-area "X or Y" weren't covered), so they fell
    # through to the fuzzy fallback, which confidently mis-guessed the
    # unrelated org-wide "declining" ranking intent instead of this
    # per-employee weakest/strongest sub-score comparison.
    r"\b(declining|lacking|weak(est)?)\b.*\b(discipline|disc|engagement|eng|effectiveness|eff)\b.*\bor\b.*\b(discipline|disc|engagement|eng|effectiveness|eff)\b",
    r"\b(discipline|disc|engagement|eng|effectiveness|eff)\b.*\bor\b.*\b(discipline|disc|engagement|eng|effectiveness|eff)\b.*\b(declining|lacking|weak(est)?)\b",
]
_SUBSCORE_TREND_PATTERNS = [
    r"\b(engagement|effectiveness|discipline)\b.*\b(improv\w*|declin\w*)\b",
    r"\bhas\b.*\b(engagement|effectiveness|discipline)\b.*\bchanged\b",
    r"\bis\b.*\b(engagement|effectiveness|discipline)\b.*\b(going up|going down)\b",
]
_STATUS_IMPROVING_PATTERNS = [
    r"\b(black|red)\b.*\bimproving\b", r"\bimproving\b.*\b(black|red)\b",
    r"\b(black|red) status\b.*\bimprov",
    r"\bwhich (black|red) status employees? are (getting better|improving)\b",
]
_OT_SUBSCORE_PATTERNS = [
    r"\b(engagement|effectiveness|discipline)\b.*\b(during|in|on)\b.*\b(ot|overtime)\b",
    r"\b(ot|overtime)\b.*\b(engagement|effectiveness|discipline)\b",
]
_WFH_SUBSCORE_PATTERNS = [
    r"\b(engagement|effectiveness|discipline)\b.*\b(during|in|on|while)\b.*\bwfh\b",
    r"\b(engagement|effectiveness|discipline)\b.*\bwork(ing)? from home\b",
    r"\bwfh\b.*\b(engagement|effectiveness|discipline)\b",
]
_PS_WORKED_EMP_PATTERNS = [
    r"\b(is|was)\b.*\bps\b.*\bworking\b", r"\bps working status\b",
    r"\bhow often\b.*\bps\b.*\bwork(ing|ed)?\b",
    r"\bis .* pace software working\b",
]
_PS_WORKED_RANKING_PATTERNS = [
    r"\bmost days? (ps|pace software) (not )?working\b",
    r"\bps (not )?working ranking\b", r"\bfewest days? ps working\b",
]


# --- NEW capability 1: day-specific COUNT / LIST queries (Category A/B of
# pace_chatbot_gap_questions.md). Both require a day/week reference
# ("yesterday"/"today"/"this week"/"last week"/an explicit date) — checked
# via entities.extract_date_range in main.py; the actual "which flag"
# (attendance/leave/WFH/visit/late/...) is resolved from the message text by
# main._detect_day_flag(), not baked into the regex here, so ONE pair of
# intents covers every flag in queries.DAY_FLAGS instead of one regex+intent
# per flag.
_CUSTOM_DATE_PHRASE = (
    r"(?:yesterday|today|this week|last week|this month|last month|"
    r"on\s+\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+|on\s+[a-z]+\s+\d{1,2}|on\s+20\d{2}-\d{1,2}-\d{1,2}|"
    r"between\s+.+\s+and\s+.+|from\s+.+\s+to\s+.+|"
    r"(?:first|second|third|fourth|last)\s+week\s+of\s+[a-z]+|"
    r"(?:last|past)\s+\d{1,2}\s+months?|"
    r"in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec))"
)
_DAY_COUNT_PATTERNS = [
    rf"\bhow many\b.*\b(employees?|people|staff)\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    rf"\bhow many\b.*\b{_CUSTOM_DATE_PHRASE}\b.*\b(employees?|people|staff)\b",
    r"\battendance count for (yesterday|today)\b",
    r"\bhow many\b.*\bpunch(ed)?\b.*\b(yesterday|today)\b",
    r"\bhow many\b.*\b(employees?|people)\b.*\bon (20\d{2}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2})\b",
]
_DAY_LIST_PATTERNS = [
    rf"\bwho (was|were|is|are)\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    r"\bwho\S* (was|were)n'?t\b.*\b(yesterday|today)\b",
    rf"\blist employees\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    rf"\bnames of employees\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    rf"\bshow me who\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    r"\bwho (did|didn'?t)\b.*\b(yesterday|today)\b",
    rf"\bwho (took|had)\b.*\b(leave|wfh)\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    r"\bwho took leave on\b", r"\bwho was on wfh on\b",
    rf"\bwho (came late|left early|did overtime|worked ot)\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    rf"\bwho visited clients\b.*\b{_CUSTOM_DATE_PHRASE}\b",
    r"\bwho attended\b.*\b(yesterday|today)\b.*\bnot (today|yesterday)\b",
]

# --- NEW capability 2: status-category filters (Black/Red/Amber/Green). ----
_STATUS_LIST_PATTERNS = [
    r"\bwho (all )?(is|are)\b.*\b(black|red|amber|green)\b(\s+(status|category))?\b",
    r"\bwhich employees are\b.*\b(black|red|amber|green)\b",
    r"\blist (all )?(black|red|amber|green) employees\b",
    r"\bshow me (black|red|amber|green) employees\b",
    r"\bwho is (black|red|amber|green) in my team\b",
]
_STATUS_COUNT_PATTERNS = [
    r"\bhow many\b.*\b(employees?|people|in my team)\b.*\b(are|is)\b.*\b(black|red|amber|green)\b",
    r"\bcount of (black|red|amber|green) employees\b",
    r"\bhow many in my team are (black|red|amber|green)\b",
]
_STATUS_DISTRIBUTION_PATTERNS = [
    r"\bwhich department(s)? (has|have) the most (red|black|amber|green) employees\b",
    r"\bwhich department has the (worst|best) pace status distribution\b",
    r"\bpercentage (of employees )?in each pace status\b",
    r"\bpercentage (red|black|amber|green)\b",
]
_STATUS_TRANSITION_PATTERNS = [
    r"\bwhich employees moved from (black|red|amber|green) to (black|red|amber|green)\b",
    r"\bwho (is|was) currently (black|red|amber|green)\b.*\b(last month|was)\b",
    r"\bwho (moved|improved|worsened)\b.*\bfrom (black|red|amber|green)\b",
    r"\bwho has been (black|red|amber|green) for multiple months\b",
]
_STATUS_EMP_PATTERNS = [
    r"\bis\b.*\b(red|black|amber|green)\b.*\bor\b.*\b(red|black|amber|green)\b",
    r"\bwhat status is\b.*\bin\b", r"\bwhat status\b.*\bcurrently\b",
]

# --- NEW capability 3: full multi-month trend history -----------------------
_FULL_TREND_EMP_PATTERNS = [
    r"\bmonth[- ]on[- ]month\b.*\b(pace )?score\b",
    r"\bpace score (trend|history)\b",
    r"\bscore trend over the last\b", r"\bfull score history\b",
    r"\bscore trend since\b", r"\bwhat has\b.*\bscore been each month\b",
    r"\bhow has\b.*\bscore changed over time\b",
    r"\b(discipline|engagement|effectiveness)\b.*\btrend (over|since)\b",
]
_FULL_TREND_DEPT_PATTERNS = [
    r"\bdepartment'?s (average )?pace score trend\b",
    r"\bdepartment'?s month[- ]on[- ]month\b",
    r"\bhas\b.*\bdepartment'?s score been (improving|declining) over time\b",
]
_FULL_TREND_TEAM_PATTERNS = [
    r"\bmy team'?s pace score trend\b",
]


_INTENTS = [
    ("day_count", _DAY_COUNT_PATTERNS),
    ("day_list", _DAY_LIST_PATTERNS),
    ("status_transitions", _STATUS_TRANSITION_PATTERNS),
    ("status_distribution", _STATUS_DISTRIBUTION_PATTERNS),
    ("status_count", _STATUS_COUNT_PATTERNS),
    ("status_list", _STATUS_LIST_PATTERNS),
    ("status_emp", _STATUS_EMP_PATTERNS),
    ("full_trend_team", _FULL_TREND_TEAM_PATTERNS),
    ("full_trend_dept", _FULL_TREND_DEPT_PATTERNS),
    ("full_trend_emp", _FULL_TREND_EMP_PATTERNS),
    # comparisons and very specific phrasings must come before generic ones
    # Category K (new round 2) — checked first: these combine keywords
    # ("engagement"+"OT", "black"+"improving", etc.) that would otherwise be
    # shadowed by the broader single-keyword patterns further down.
    ("score_drop_ranking", _SCORE_DROP_PATTERNS),
    ("score_improvement_alltime", _SCORE_IMPROVEMENT_ALLTIME_PATTERNS),
    ("subscore_compare_emp", _SUBSCORE_COMPARE_PATTERNS),
    ("ot_subscore", _OT_SUBSCORE_PATTERNS),
    ("wfh_subscore", _WFH_SUBSCORE_PATTERNS),
    ("subscore_trend_emp", _SUBSCORE_TREND_PATTERNS),
    ("status_improving", _STATUS_IMPROVING_PATTERNS),
    ("ps_worked_emp", _PS_WORKED_EMP_PATTERNS),
    ("ps_worked_ranking", _PS_WORKED_RANKING_PATTERNS),
    ("emp_overview", _EMP_OVERVIEW_PATTERNS),
    ("dept_compare", _DEPT_COMPARE_PATTERNS),
    ("team_compare", _TEAM_COMPARE_PATTERNS),
    ("employee_compare", _COMPARE_EMPLOYEES_PATTERNS),
    ("new_joiners", _NEW_JOINERS_PATTERNS),
    ("leave_who", _LEAVE_WHO_PATTERNS),
    ("leave_emp_check", _LEAVE_EMP_CHECK_PATTERNS),
    ("half_day_ranking", _HALF_DAY_RANKING_PATTERNS),
    ("leave_by_dept", _LEAVE_BY_DEPT_PATTERNS),
    ("zero_leave", _ZERO_LEAVE_PATTERNS),
    ("call_most", _CALL_MOST_PATTERNS),
    ("call_fewest", _CALL_FEWEST_PATTERNS),
    ("call_duration", _CALL_DURATION_PATTERNS),
    ("call_emp", _CALL_EMP_PATTERNS),
    ("visit_ranking", _VISIT_RANKING_PATTERNS),
    ("zero_visit", _ZERO_VISIT_PATTERNS),
    ("visit_emp", _VISIT_EMP_PATTERNS),
    ("wfh_emp", _WFH_EMP_PATTERNS),
    ("wfh_ranking", _WFH_RANKING_PATTERNS),
    ("wfh_by_dept", _WFH_BY_DEPT_PATTERNS),
    ("todos_created_ranking", _TODOS_CREATED_PATTERNS),
    ("todos_assigned_ranking", _TODOS_ASSIGNED_PATTERNS),
    ("tasks_created_ranking", _TASKS_CREATED_PATTERNS),
    ("tasks_assigned_ranking", _TASKS_ASSIGNED_PATTERNS),
    ("ontime_completion_ranking", _ONTIME_COMPLETION_PATTERNS),
    ("responsiveness_ranking", _RESPONSIVENESS_PATTERNS),
    ("extension_adherence_ranking", _EXTENSION_ADHERENCE_PATTERNS),
    ("meeting_count_ranking", _MEETING_COUNT_PATTERNS),
    ("meeting_ratio_emp", _MEETING_RATIO_PATTERNS),
    ("d_score_trend", _D_SCORE_TREND_PATTERNS),
    ("d_score_ranking", _D_SCORE_RANKING_PATTERNS),
    ("d_score_emp", _D_SCORE_EMP_PATTERNS),
    ("shift_type_emp", _SHIFT_TYPE_EMP_PATTERNS),
    ("ot_ranking", _OT_RANKING_PATTERNS),
    ("breakshift_emp", _BREAKSHIFT_EMP_PATTERNS),
    ("offline_ranking", _OFFLINE_RANKING_PATTERNS),
    ("offline_emp", _OFFLINE_EMP_PATTERNS),
    ("ps_install_rate", _PS_INSTALL_RATE_PATTERNS),
    ("grade_lookup", _GRADE_LOOKUP_PATTERNS),
    ("designation_breakdown", _DESIGNATION_BREAKDOWN_PATTERNS),
    ("avg_tenure", _AVG_TENURE_PATTERNS),
    ("team_how_doing", _TEAM_HOW_DOING_PATTERNS),
    ("team_lowest_scorers", _TEAM_LOWEST_SCORERS_PATTERNS),
    ("meeting_min_ranking", _MEETING_MIN_PATTERNS),
    ("chronic_late", _CHRONIC_LATE_PATTERNS),
    ("perfect_attendance", _PERFECT_ATTENDANCE_PATTERNS),
    ("defaulter_ranking", _DEFAULTER_RANKING_PATTERNS),
    ("deficit_hours_ranking", _DEFICIT_HOURS_RANKING_PATTERNS),
    ("dept_trend", _DEPT_TREND_PATTERNS),
    ("team_improving", _TEAM_IMPROVING_PATTERNS),
    ("emp_trend_2month", _TREND_2MONTH_PATTERNS),
    ("dept_avg", _DEPT_AVG_PATTERNS),
    ("dept_best", _DEPT_BEST_PATTERNS),
    ("dept_worst", _DEPT_WORST_PATTERNS),
    ("dept_count", _DEPT_COUNT_PATTERNS),
    ("dept_summary", _DEPT_SUMMARY_PATTERNS),
    ("emp_pace_score", _EMP_PACE_SCORE_PATTERNS),
    ("emp_attendance_summary", _EMP_ATTENDANCE_SUMMARY_PATTERNS),
    ("emp_late_comings", _EMP_LATE_COMINGS_PATTERNS),
    ("emp_early_leavings", _EMP_EARLY_LEAVINGS_PATTERNS),
    ("emp_productive_time", _EMP_PRODUCTIVE_TIME_PATTERNS),
    ("emp_whatsapp", _EMP_WHATSAPP_PATTERNS),
    ("emp_ai_usage", _EMP_AI_USAGE_PATTERNS),
    ("emp_trend", _EMP_TREND_PATTERNS),
    ("emp_department", _EMP_DEPARTMENT_PATTERNS),
    ("emp_manager", _EMP_MANAGER_PATTERNS),
    ("emp_discipline", _EMP_DISCIPLINE_PATTERNS),
    ("emp_engagement", _EMP_ENGAGEMENT_PATTERNS),
    ("emp_effectiveness", _EMP_EFFECTIVENESS_PATTERNS),
    ("emp_deficient_hours", _EMP_DEFICIENT_HOURS_PATTERNS),
    ("emp_working_pct", _EMP_WORKING_PCT_PATTERNS),
    ("pace_score_best", _PACE_SCORE_BEST_PATTERNS),
    ("pace_score_worst", _PACE_SCORE_WORST_PATTERNS),
    ("engagement_high", _ENGAGEMENT_HIGH_PATTERNS),
    ("engagement_low", _ENGAGEMENT_LOW_PATTERNS),
    ("effectiveness_high", _EFFECTIVENESS_HIGH_PATTERNS),
    ("effectiveness_low", _EFFECTIVENESS_LOW_PATTERNS),
    ("most_late_comings", _MOST_LATE_COMINGS_PATTERNS),
    ("fewest_late_comings", _FEWEST_LATE_COMINGS_PATTERNS),
    ("most_early_leavings", _MOST_EARLY_LEAVINGS_PATTERNS),
    ("most_deficient_hours", _MOST_DEFICIENT_HOURS_PATTERNS),
    ("most_disciplined", _MOST_DISCIPLINED_PATTERNS),
    ("least_disciplined", _LEAST_DISCIPLINED_PATTERNS),
    ("most_whatsapp", _MOST_WHATSAPP_PATTERNS),
    ("lowest_working_pct", _LOWEST_WORKING_PCT_PATTERNS),
    ("highest_working_pct", _HIGHEST_WORKING_PCT_PATTERNS),
    ("fewest_wfh", _FEWEST_WFH_PATTERNS),
    ("wfh_ranking", _MOST_WFH_DAYS_PATTERNS),
    ("zero_visit", _FEWEST_VISITS_PATTERNS),
    ("score_drop_ranking", _SMALLEST_SCORE_CHANGE_PATTERNS + _BIGGEST_SCORE_CHANGE_PATTERNS),
    ("score_drop_ranking", _LEAST_IMPROVED_PATTERNS),
    ("score_improvement_alltime", _MOST_IMPROVED_PATTERNS),
    ("pace_score_worst", _WORST_SUBSCORE_COMPANY_PATTERNS),
    ("declining", _DECLINING_PATTERNS),
    ("improving", _IMPROVING_PATTERNS),
    ("attendance_worst", _ATTENDANCE_WORST_PATTERNS),
    ("attendance_best", _ATTENDANCE_BEST_PATTERNS),
    ("productive_low", _PRODUCTIVE_LOW_PATTERNS),
    ("productive_high", _PRODUCTIVE_HIGH_PATTERNS),
]


# --- Fuzzy fallback (offline, rapidfuzz) -------------------------------------
# Only used when NO exact regex matched at all (the caller already runs
# dictionary-based typo correction - see spellcheck.py - before calling
# match_intent, so this fuzzy pass exists purely to catch trigger phrases the
# spellchecker couldn't fix, e.g. a garbled/transposed keyword). Hand-curated
# short canonical phrases per intent, deliberately NOT exhaustive - only the
# short, high-signal trigger intents are covered, since long/argument-heavy
# intents (single-employee lookups, comparisons) are already well covered by
# spellcheck-then-regex and are riskier to fuzzy-match reliably.
_CANONICAL_PHRASES = {
    "attendance_best": ["best attendance", "most punctual", "top attendance"],
    "attendance_worst": ["worst attendance", "most absent", "poor attendance", "attendance problems"],
    "productive_high": ["most productive time", "highest productive minutes", "most productivity", "working hard"],
    "productive_low": ["least productive time", "lowest productive minutes", "least productivity", "slacking off", "wasting time"],
    "improving": ["who is improving", "who is getting better", "top improvers"],
    "declining": ["who is declining", "who is getting worse", "biggest drop"],
    "pace_score_best": ["best pace score", "highest pace score", "best performer"],
    "pace_score_worst": ["worst pace score", "lowest pace score", "worst performer", "who needs help"],
    "engagement_high": ["highest engagement", "most engaged"],
    "engagement_low": ["lowest engagement", "least engaged"],
    "effectiveness_high": ["highest effectiveness", "most effective"],
    "effectiveness_low": ["lowest effectiveness", "least effective"],
    "most_late_comings": ["most late comings"],
    "most_early_leavings": ["most early leavings"],
    "most_deficient_hours": ["most deficient hours"],
    "most_disciplined": ["most disciplined", "highest discipline"],
    "most_whatsapp": ["highest whatsapp usage", "most whatsapp"],
    "chronic_late": ["chronically late", "habitually late", "always late"],
    "perfect_attendance": ["perfect attendance"],
    "defaulter_ranking": ["most defaulters", "top defaulters"],
    "new_joiners": ["new joiners", "recently joined"],
    "team_how_doing": ["how is my team doing"],
    "team_lowest_scorers": ["lowest scorers in my team"],
    "meeting_min_ranking": ["meeting minutes"],
    "dept_best": ["best department", "top department"],
    "dept_worst": ["worst department", "bottom department"],
    "dept_avg": ["average pace score by department"],
    "dept_count": ["employee count", "headcount"],
    "dept_summary": ["department summary"],
    "score_drop_ranking": ["biggest score drop", "whose score dropped the most", "who dropped the most"],
    "score_improvement_alltime": ["overall improvement", "whose score improved the most", "who improved the most overall"],
    # Round 3 additions: metrics called out as priority (PACE score/status,
    # attendance, WFH, leave, visits, productivity) that previously had NO
    # fuzzy-fallback entry at all, so a genuinely novel phrasing for these
    # (not matching any literal regex) fell through to the generic
    # FALLBACK_MESSAGE instead of routing correctly.
    "wfh_ranking": ["most wfh days", "who works from home the most", "most work from home"],
    "fewest_wfh": ["fewest wfh days", "least wfh days", "who works from home the least"],
    "wfh_emp": ["was on wfh", "working from home", "wfh status of employee"],
    "leave_who": ["who is on leave", "who took leave", "who is off today"],
    "leave_emp_check": ["was on leave", "did take leave", "has been on leave"],
    "zero_leave": ["zero leaves taken", "no leaves taken", "never took leave"],
    "visit_ranking": ["most client visits", "who made the most visits", "top visits"],
    "zero_visit": ["zero visits", "no client visits", "never visited a client"],
    "visit_emp": ["visits for employee", "did visit a client", "how many visits"],
    "call_most": ["most calls", "highest number of calls", "who made the most calls"],
    "call_fewest": ["fewest calls", "least calls", "who made the fewest calls"],
    "status_list": ["who is red status", "list black employees", "show me green employees"],
    "status_count": ["how many employees are red status", "count of black employees"],
    "status_emp": ["what status is the employee", "is employee red or green"],
    "day_count": ["how many employees yesterday", "how many people today", "attendance count for yesterday"],
    "day_list": ["who was absent yesterday", "who was present today", "list employees on leave this week"],
    "d_score_ranking": ["best d score", "highest d score", "worst d score"],
    "d_score_emp": ["d score of employee", "what is the d score"],
    "emp_pace_score": ["pace score of employee", "what is the pace score", "current score of employee"],
    "emp_attendance_summary": ["attendance summary of employee", "how was attendance"],
    "ot_ranking": ["most overtime hours", "who works the most overtime", "overtime ranking"],
    "full_trend_emp": ["pace score trend", "score trend over time", "month on month score"],
}

# Intent pairs whose canonical phrases are close enough (share a metric word,
# differ only on the direction word) that a fuzzy match must be REFUSED
# rather than guessed if both score too closely - this is what stops "top 5"
# from ever being fuzzy-confused with "bottom 5".
_OPPOSITE_INTENTS = {
    "attendance_best": "attendance_worst", "attendance_worst": "attendance_best",
    "productive_high": "productive_low", "productive_low": "productive_high",
    "improving": "declining", "declining": "improving",
    "pace_score_best": "pace_score_worst", "pace_score_worst": "pace_score_best",
    "engagement_high": "engagement_low", "engagement_low": "engagement_high",
    "effectiveness_high": "effectiveness_low", "effectiveness_low": "effectiveness_high",
    "dept_best": "dept_worst", "dept_worst": "dept_best",
    "score_drop_ranking": "score_improvement_alltime", "score_improvement_alltime": "score_drop_ranking",
    # Round 3 additions: same safety-margin treatment for the newly-added
    # metric opposite pairs, so a fuzzy match can't confuse "most WFH" with
    # "fewest WFH" (and similarly for visits/calls) when both score closely.
    "wfh_ranking": "fewest_wfh", "fewest_wfh": "wfh_ranking",
    "visit_ranking": "zero_visit", "zero_visit": "visit_ranking",
    "call_most": "call_fewest", "call_fewest": "call_most",
}

FUZZY_INTENT_THRESHOLD = 82  # conservative: must be a strong, confident match
FUZZY_OPPOSITE_MARGIN = 8    # if the opposite intent scores within this of the winner, refuse rather than guess


def _fuzzy_match_intent(text_l):
    scores = {}
    for intent_name, phrases in _CANONICAL_PHRASES.items():
        scores[intent_name] = max(fuzz.token_set_ratio(text_l, p) for p in phrases)

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    if best_score < FUZZY_INTENT_THRESHOLD:
        return None

    opposite = _OPPOSITE_INTENTS.get(best_intent)
    if opposite and scores.get(opposite, 0) >= best_score - FUZZY_OPPOSITE_MARGIN:
        return None  # too close to call - don't risk a wrong-direction answer

    return best_intent


def match_intent(text):
    text_l = text.lower()
    for intent_name, patterns in _INTENTS:
        for pat in patterns:
            if re.search(pat, text_l):
                return intent_name
    return _fuzzy_match_intent(text_l)


FALLBACK_MESSAGE = (
    "I can currently answer questions about attendance, PACE score, productivity, "
    "engagement, effectiveness, discipline, WhatsApp/AI tool usage, department "
    "comparisons, team summaries, and more — for an individual employee, a "
    "department, 'my team', or a specific manager's team. For example:\n"
    "- \"pace score of Aarna Jain\"\n"
    "- \"worst attendance in Accounts this month\"\n"
    "- \"top 5 by pace score in IT-Development\"\n"
    "- \"compare Accounts vs Billing\"\n"
    "- \"how is my team doing\"\n"
    "- \"new joiners in my team\"\n\n"
    "You can optionally mention a department, employee name, and/or a month; "
    "if you leave them out I'll default to all departments / the current month."
)
