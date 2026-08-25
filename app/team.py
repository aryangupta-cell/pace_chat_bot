"""Employee identity + "my team" resolution via pace_1.email_access.

Confirmed by prior investigation:
- pace_1.email_access is a pure passthrough of emp_level_email_access.email_access_1
  (0 mismatches across all active employees) - simpler to query pace_1 directly.
- email_access is a comma-delimited list of emails; the employee's "own" email is
  the first entry in their own list.
- Reverse lookup (email -> emp_codes it can see) MUST use exact trimmed matching,
  not ILIKE/substring - 'atul.mangal@axestrack.com' is a literal substring of
  'pratul.mangal@axestrack.com', a confirmed real collision risk in this data.
- A handful of emails (confirmed ~9 at investigation time, but not hardcoded here
  since the list could change) map to essentially the entire active employee
  population - a company-wide admin/HR access grant, not a real "team". Detected
  dynamically by comparing team size against total active employee count.
"""
from .db import run_query

PACE_1 = "public.pace_1"

# If an email's reverse-lookup team covers at least this fraction of all active
# employees, treat it as universal/admin access rather than a real team.
UNIVERSAL_ACCESS_THRESHOLD = 0.9


def _first_email(email_access_value):
    if not email_access_value:
        return None
    first = email_access_value.split(",")[0].strip()
    return first or None


def resolve_employee_email(name_or_code):
    """Look up an employee in pace_1 by exact emp_code or emp_name match.
    Returns one of:
      (email, emp_name, emp_code) - confident single match
      (None, candidates, None)    - ambiguous, candidates is a list of "Name (CODE)" strings
      (None, None, None)          - no match found at all
    """
    q = name_or_code.strip()

    rows = run_query(
        f"""
        select distinct emp_code, emp_name, email_access
        from {PACE_1}
        where email_access is not null
          and (upper(emp_code) = upper(%(q)s) or lower(emp_name) = lower(%(q)s))
        """,
        {"q": q},
    )
    if not rows:
        rows = run_query(
            f"""
            select distinct emp_code, emp_name, email_access
            from {PACE_1}
            where email_access is not null and emp_name ilike %(q)s
            """,
            {"q": f"%{q}%"},
        )

    if len(rows) == 1:
        r = rows[0]
        return _first_email(r["email_access"]), r["emp_name"], r["emp_code"]
    if len(rows) > 1:
        candidates = sorted({f"{r['emp_name']} ({r['emp_code']})" for r in rows})
        return None, candidates, None
    return None, None, None


def total_active_employees():
    rows = run_query(f"select count(distinct employee_id) as n from {PACE_1}")
    return rows[0]["n"] or 0


def find_team(email):
    """Reverse-lookup: employees (within pace_1's active population) whose
    email_access list contains `email`, via exact trimmed matching only."""
    return run_query(
        f"""
        select distinct e.employee_id, e.emp_code, e.emp_name, e.dept_name
        from {PACE_1} e,
             unnest(string_to_array(e.email_access, ',')) as x
        where trim(x) = %(email)s
        order by e.emp_name
        """,
        {"email": email},
    )


def resolve_team(email):
    """Returns (employee_ids, is_universal_access).
    employee_ids is the list of employee_id ints visible to this email.
    is_universal_access is True when this looks like a company-wide admin
    grant rather than a real team (team size >= UNIVERSAL_ACCESS_THRESHOLD
    of all active employees)."""
    team_rows = find_team(email)
    employee_ids = [r["employee_id"] for r in team_rows]
    total = total_active_employees()
    is_universal = total > 0 and (len(employee_ids) / total) >= UNIVERSAL_ACCESS_THRESHOLD
    return employee_ids, is_universal


def resolve_named_person_team(name_or_code):
    """Resolve ANY named person's team using the exact same identity->team
    pipeline as 'my team' (resolve_employee_email then resolve_team) - so a
    named-manager query ("Nikhil Kumar's team") and that same person asking
    about "my team" always agree, instead of the former silently falling
    back to a shallower reporting_user_id-only definition.

    Returns one of:
      (employee_ids, is_universal, resolved_name, None) - success
      (None, False, None, candidates)                   - ambiguous name match
      (None, False, None, None)                          - no employee found for this name at all
    """
    email, name_or_candidates, _code = resolve_employee_email(name_or_code)
    if email:
        employee_ids, is_universal = resolve_team(email)
        return employee_ids, is_universal, name_or_candidates, None
    if name_or_candidates:  # ambiguous match -> this is the candidates list
        return None, False, None, name_or_candidates
    return None, False, None, None
