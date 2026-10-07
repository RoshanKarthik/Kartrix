# Kartrix evals

The agent and RAG eval suite (roadmap step 2.3). It measures **Kartrix itself** — does retrieval find
the right code, does the agent finish coding tasks correctly, efficiently and safely — not the apps
people build with it. Every architecture or optimisation change is kept only if these numbers agree.

```bash
uv run kartrix eval rag --quick --no-judge     # everyday: retrieval only, 15 questions (embeddings only)
uv run kartrix eval agent --quick              # everyday: 6 coding tasks
uv run kartrix eval all --quick --compare      # both, against the committed baseline
uv run kartrix eval rag                        # 100 questions × 4 retrieval modes + judged answers
uv run kartrix eval agent --repeat 3 --judge   # 26 tasks × 3 runs: pass@1, pass^3, judge metrics
uv run kartrix eval calibrate                  # the judge against the hand labels
uv run kartrix eval all --save-baseline        # record a new baseline (commit evals/baselines/)
uv run kartrix eval report <results.json>      # re-render a report
```

Needs Postgres (`docker compose up -d`, migrations applied), `NVIDIA_API_KEY` (and `HF_TOKEN` for the
fallback) in `.env`, git, and Node ≥ 23.6 for the TypeScript tasks. Results and a self-contained HTML
report go to `.kartrix/evals/results/<time>-<suite>/`; `--compare` exits 1 on a regression with
`--fail-on-regression` (the nightly CI workflow `.github/workflows/evals.yml` does that).

## What is measured

**RAG** (`rag/golden.yaml`, 100 questions over three repos pinned in `repos.yaml`: this repository at
step 2.2, the FastAPI full-stack template, an Express + Prisma API):

- retrieval per mode — `dense`, `lexical`, `hybrid` (RRF) and `search_first` (no index: files ranked
  by the question's words, the stand-in for step 3.2's search-first context; `repo_map` arrives with
  3.2): hit rate, recall@k, precision@k, MRR, nDCG over distinct files, plus chunk precision@5 (what
  the agent actually reads), symbol recall, latency and context tokens per query;
- answers (the configured mode's context, main model) judged by DeepEval with Kartrix's judge model:
  faithfulness, answer relevancy, contextual precision, contextual recall.

**Judge calibration** (`rag/calibration.yaml`): the faithfulness and relevancy judges on 20
hand-labelled items — agreement and Cohen's κ. Trust the judge metrics only as far as these hold.

**Agent** (`agent/tasks/*/task.yaml`, 26 tasks on four small apps in `agent/apps/`, Python and
TypeScript): each task runs as `kartrix run` in a fresh git workspace — real sandbox, command policy,
approvals from the task's policy, budgets — and is checked by commands:

- outcome: success, pass@1, pass^k over `--repeat k` runs, per category and stack;
- trajectory: tool calls, invalid arguments, redundant calls, tool errors and recovery, expected tools
  used; with `--judge` DeepEval tool correctness, argument correctness, answer correctness
  (explain/locate) and plan quality (plan mode);
- efficiency: tokens, cost, time per run, startup time;
- safety: planted prompt injections resisted, blocked commands never run, approvals asked, budgets
  respected; approvals per run (each one would interrupt a person).

## Adding to the datasets

- **A question:** ask it the way a user would (not with the identifiers in it), label every file that
  answers it at the pinned commit, add the symbols and a short reference answer. `quick: true` puts it
  in the everyday subset. Changing a pin means re-checking that repo's labels.
- **A task:** `agent/tasks/<id>/task.yaml` plus
  - `setup/` — files layered on the app for this task only (a planted bug, an injection);
  - `hidden/` — tests copied in *after* the run, so the agent can't fit the code to them;
  - `solution/` — a reference solution.
  `tests/unit/test_eval_datasets.py` proves every task's checks fail on the app as the agent finds it
  and pass with the solution. "Write tests" tasks are scored by mutation testing
  (`agent/shared/mutation_check.py`: the new tests must fail on planted bugs).

Results are only comparable when the dataset hashes in `versions.datasets` match.
