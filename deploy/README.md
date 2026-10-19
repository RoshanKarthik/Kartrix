# Deploying the free Kartrix demo (₹0, no card)

The demo is the Kartrix server in **hosted mode** on free plans:

- **Render** — free web service that runs the Docker image. No payment method needed: Render only bills an account
  that has one; without it, a free service that hits a limit is stopped, never charged.
- **Neon** — free Postgres with pgvector ("no credit card required", 1 GB storage, 100 compute-hours a month).
- **cron-job.org** — free pings every 10 minutes so the free service doesn't go to sleep.

What the demo is (and isn't): invite-only accounts, the model picker, live run streaming. The free machine has
0.1 CPU (slow — minutes per request), its disk resets on every restart, and it has no command sandbox, so agents
there **write and review code but never run it** (code-running commands are refused). For real work, use
`kartrix --local` on your own machine.

**Never add a card anywhere in these steps.** If a page asks for one, stop — it isn't the free path.
Button and menu names on these sites change now and then; if one differs, look for the closest match.

## 1. Neon: the database (about 5 minutes)

1. Go to <https://neon.com> → **Sign up** (with GitHub or Google).
2. Create a project: name `kartrix`, Postgres **17**, region **AWS Asia Pacific (Singapore)**.
3. On the project dashboard click **Connect**. In the dialog:
   - Branch `main`, database `neondb`, role `neondb_owner` (the defaults).
   - **Switch "Connection pooling" OFF** (Kartrix needs the direct connection).
   - Copy the connection string. It looks like
     `postgresql://neondb_owner:…@ep-xxxx.ap-southeast-1.aws.neon.tech/neondb?sslmode=require`.
   Keep it private — it contains the database password.

Kartrix creates its tables itself on first start (including the pgvector extension).

## 2. GitHub: put the code where Render can build it

Render builds the image from a Git repository (a **private** GitHub repo is fine and free).

1. Create a private repo on GitHub, e.g. `kartrix`.
2. Push this project to it (after removing anything you don't want published). `render.yaml` and `Dockerfile` must
   be in the root of the repo.

## 3. Render: the server (about 10 minutes, the first build takes a few)

1. Go to <https://render.com> → **Get Started** → sign up **with GitHub**. Don't add a payment method.
2. In the dashboard: **New +** → **Blueprint**.
3. **Connect** your GitHub account if asked, and give Render access to the `kartrix` repo only.
4. Pick the repo. Render reads `render.yaml` and shows one service, `kartrix`, plan **Free**.
5. Fill in the variables it asks for:
   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | the Neon connection string from step 1 |
   | `KARTRIX_ADMIN_EMAIL` | your email (your login) |
   | `KARTRIX_ADMIN_PASSWORD` | a new password, 8+ characters, used only for Kartrix |
   | `NVIDIA_API_KEY`, `HF_TOKEN`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | copy from your local `.env` |
6. Click **Apply** (or **Deploy Blueprint**). Watch **Logs**: it builds the image, then prints
   `alembic upgrade` lines, `kartrix serve: hosted mode …` and `Application startup complete`.
7. Your address is at the top of the service page: `https://kartrix-xxxx.onrender.com`.
   Open `https://kartrix-xxxx.onrender.com/v1/health` — it should show `{"name":"kartrix",…}`.

Every push to the repo redeploys automatically (and resets the server's disk).

## 4. cron-job.org: keep it awake

Render puts a free service to sleep after 15 minutes without requests; waking takes about a minute.

1. Go to <https://cron-job.org> → **Sign up** (free) and confirm your email.
2. **Create cronjob**: title `kartrix`, URL `https://kartrix-xxxx.onrender.com/v1/health`, schedule **every 10
   minutes**. Save.

One service awake all month uses about 744 of the 750 free hours Render gives each month. If a ping fails right
after a restart (cron-job.org gives up after 30 s while Render wakes up), the next one succeeds.

## 5. Use it from your terminal

```bash
kartrix login --server https://kartrix-xxxx.onrender.com
```

Log in with the admin email and password from step 3. Then:

```bash
kartrix models
```

```bash
kartrix
```

`kartrix` opens a workspace **on the server** named after your current folder (your local files are not sent).
`kartrix --local` keeps working on your machine as before; `kartrix logout` switches back to local for good.

## Accounts

Sign-up is closed on the demo. To add someone, run this on your PC with the Neon connection string (the account is
created straight in the database):

```bash
DATABASE_URL="postgresql://…neon.tech/neondb?sslmode=require" uv run kartrix admin create-user --email friend@example.com
```

`kartrix admin set-password --email …` resets a password (and signs that user out); `kartrix admin list-users`
lists the accounts. Every account uses **your** free model keys and their limits.

## When something goes wrong

- **Build fails on Render** — read the build log; most often a missing file in the repo (`uv.lock`, `migrations/`).
- **`DatabaseConfigError` or connection errors at start** — `DATABASE_URL` is wrong or uses the pooled host
  (`-pooler` in the name): copy it again with pooling off.
- **Service "suspended"** — a free limit was reached (Render hours/bandwidth or Neon compute-hours); it comes back
  at the start of next month, nothing is charged or deleted.
- **Very slow answers** — expected on 0.1 CPU; Groq models (`/model groq-gpt-oss-120b`) are the fastest.
