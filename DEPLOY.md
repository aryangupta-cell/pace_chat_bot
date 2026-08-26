# Deploying PACE Chatbot to Render

This is a handoff document — nothing in it has been executed. You are not
connected to GitHub or Render from this environment, so every step below is
something you do yourself.

---

## 0. IMPORTANT — rotate the DB password before going further

While preparing this deployment, a grep of the full repo found the real RDS
password (`PACE_DB_PASSWORD`) hardcoded in `SESSION_HANDOFF.md` in **3
places**. It has been redacted from the file in the working tree, but it was
already committed in earlier commits, so **it exists in this repo's git
history** (`git log --all -p | grep ...` still finds it in past commits even
after the redaction commit). Removing it from history (e.g. via
`git filter-repo`/BFG) is possible but was not attempted here since it
rewrites commit hashes — that's a decision for you to make.

Practical options, in order of simplicity:
1. **Rotate the DB password** (change it in RDS / with your DBA) before or
   right after making the GitHub repo, especially if the repo will ever be
   public or shared. This makes the old value in history harmless.
2. If the repo stays private forever and you're comfortable with that risk,
   you could skip rotation — but this is a judgment call, not a
   recommendation.

No hardcoded credentials remain anywhere else in the tracked codebase — see
the verification section at the bottom of this handoff for the exact grep
commands used to confirm this (app code, requirements, static files, etc. are
all clean; only the one doc file had the issue, now fixed going forward).

---

## 1. Push this code to GitHub

```bash
cd "/c/Users/user/Claude Dashboard/pace chatbot"
git init   # already a repo — skip if `git status` already works
git add .
git commit -m "Prepare for Render deployment"

# Create a new empty repo on github.com first (via the GitHub website —
# do NOT initialize it with a README/gitignore/license), then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

If `origin` already exists, use `git remote set-url origin <url>` instead of
`git remote add`.

---

## 2. Create a Render account and Web Service

1. Sign up / log in at https://render.com (GitHub login is easiest since
   you'll be connecting a GitHub repo anyway).
2. In the Render dashboard: **New +** → **Web Service**.
3. Connect your GitHub account if prompted, then select the repo you just
   pushed.
4. Configure the service:
   - **Name**: anything, e.g. `pace-chatbot`
   - **Region**: pick whatever's closest to your users (this affects which
     outbound IP range Render uses — see the database access section below)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**:
     ```
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
     (A `Procfile` with `web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8010}`
     is also committed in the repo root — Render can pick this up
     automatically, but it's safest to also set the Start Command explicitly
     in the dashboard as shown above, since the two aren't guaranteed to be
     evaluated identically by every Render deploy path.)
   - **Instance type**: Free is fine to start.

5. **Environment variables** — in the same "New Web Service" form (or later
   under the service's *Environment* tab), add these, one per row. Set the
   VALUE for each from your own secrets store — **never paste them into any
   file in this repo**:

   | Key | Value |
   |---|---|
   | `PACE_DB_HOST` | *(your RDS host)* |
   | `PACE_DB_PORT` | `5432` |
   | `PACE_DB_NAME` | *(your DB name)* |
   | `PACE_DB_USER` | *(your DB user)* |
   | `PACE_DB_PASSWORD` | *(your DB password — rotate first, see section 0)* |
   | `GEMINI_API_KEY` | *(your Gemini API key)* |

   Do **not** set `PORT` yourself — Render injects it automatically at
   runtime, and the app now reads it via the `$PORT`/`${PORT:-8010}`
   mechanism in the Start Command / Procfile (falls back to 8010 only when
   `PORT` isn't set, i.e. for local dev).

6. Click **Create Web Service**. Render will build and deploy; watch the
   build/deploy logs in the dashboard for errors.

---

## 3. Database network access — a decision you need to make

The app connects out to your RDS Postgres instance from wherever Render runs
the container. **Render's free tier does not have a fixed outbound IP
address** — outbound traffic comes from a shared, region-wide IP range that
can change, and Render does not publish these ranges in static docs; you can
only see a service's *current* outbound IP ranges from its own dashboard page
(**Connect** dropdown → **Outbound** tab), and only after the service exists.
Render also now offers a paid **dedicated outbound IP** add-on if you want a
truly fixed IP to allowlist.

Your RDS security group currently likely only allows inbound port 5432 from
specific known IPs (e.g. your own machine). Once deployed on Render, the app
will get connection-refused/timeout errors against the DB until you address
this. Two options — **pick one, this is your call, not something pre-decided
for you**:

**Option A — open port 5432 to the whole internet (`0.0.0.0/0`)**
Simplest, works immediately, and doesn't need you to keep the allowlist in
sync if Render's IP ranges change later. Only reasonable because:
- The DB user (`aryangupta_ds`) this app uses is a low-privilege,
  effectively read-only app user, not an admin credential.
- The password should be strong (see rotation note in section 0).
Still, this exposes the Postgres port to internet-wide scanning/brute-force
attempts. If you go this route, keep the password strong and consider
Postgres-level connection logging/alerting if your RDS setup supports it.

**Option B — scope the security group to Render's published outbound IP
ranges for your chosen region**
More restrictive, but the ranges are shared across all Render customers in
that region (not exclusive to you) and can change over time — you'd need to
periodically re-check Render's dashboard (**Connect → Outbound** on your
deployed service) and update the RDS security group if they change. Get the
exact current ranges only after the service is created and deployed (they
are not published anywhere fixed you can copy in advance). This is the
better option if avoiding `0.0.0.0/0` matters for your compliance posture.

Neither option was implemented for you — this doc only lays out the tradeoff.

---

## 4. This is a public, no-authentication deployment

As scoped, this app has **no login and no access control** — anyone with the
Render URL can use the chatbot and see whatever employee/attendance/PACE data
it returns. This mirrors the app's existing local design (in-memory session
store, no auth, "single shared access prototype," per `SESSION_HANDOFF.md`
section 4) — deploying to Render just makes that same design reachable over
the public internet instead of only `127.0.0.1`. If that's not acceptable for
this data, add authentication (e.g. Render's built-in basic auth options, a
reverse proxy, or app-level login) before sharing the URL — that work is out
of scope for this handoff and was not implemented.

---

## 5. Post-deploy checks

- Visit `https://<your-service>.onrender.com/api/health` → should return
  `{"status":"ok"}`.
- Visit `https://<your-service>.onrender.com/` for the standalone chat UI,
  and `/dashboard` for the Looker Studio embed + chat bubble.
- Check the Render **Logs** tab if anything 500s — the app already has a
  global exception handler that returns valid JSON instead of crashing the
  frontend, but the real traceback is only visible in the Render logs.

---

## Verification performed locally before this handoff was written

Grep commands run against the full repo (not just `app/`) to confirm no
credentials are hardcoded in tracked source/config files:

```bash
grep -rn "<the actual DB password>" .                              # DB password literal
grep -rn "<the actual Gemini API key>" .                           # Gemini key literal
grep -rnE 'PACE_DB_PASSWORD\s*=\s*["'"'"']' --include="*.py" .     # assignment pattern
grep -rnE 'GEMINI_API_KEY\s*=\s*["'"'"']' --include="*.py" .       # assignment pattern
```
(Run these with the real secret values substituted in locally — never commit the actual values into this file.)

Result: all four came back empty against the current working tree, **except**
the DB-password literal, which was found in `SESSION_HANDOFF.md` (3 lines)
and has been redacted — see section 0 above for why this still matters
(git history).

The port-binding change (Procfile's `${PORT:-8010}`) was verified live:
started with `PORT=9999` set and confirmed the app bound to 9999 (health
check succeeded on 9999, connection failed on 8010); then restarted with no
`PORT` set and confirmed it fell back to 8010 as before, so local dev per
`SESSION_HANDOFF.md` section 4 is unaffected.
