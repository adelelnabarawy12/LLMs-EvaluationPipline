# Prompt Reliability Eval Harness

A lean, dependency-light test harness for measuring LLM response consistency — focusing on **near-duplicate query flakiness**.

---

## Quick Start

```bash
git clone <repo>
cd eval_harness
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running Commands

### Run the test suite (no API key needed)

```bash
# All 63 tests — scorers, pipeline, CLI, output persistence
pytest test_pipeline.py -v

# Scorer unit tests only
pytest test_pipeline.py -v -k "TestExactCI or TestRegex or TestSetMatch or TestJsonSet or TestSemantic"

# Pipeline integration tests only
pytest test_pipeline.py::TestRunCasesPipeline -v

# A single test class
pytest test_pipeline.py::TestEdgeCases -v

# Short traceback on failure
pytest test_pipeline.py --tb=short

# Stop on first failure
pytest test_pipeline.py -x
```

---

### Run the harness (requires API key)

```bash
# Single pass — all 16 cases
python prompt_eval_harness.py

# Dry-run — no API calls, validates routing and output format
python prompt_eval_harness.py --dry-run

# 5-run consistency sweep — detects flaky cases across repeated runs
python prompt_eval_harness.py --runs 5

# Run only invariance variants (INV-001a/b/c)
python prompt_eval_harness.py --group invariance

# Run only perturbation variants (PERT-001a/b/c)
python prompt_eval_harness.py --group perturbation

# Run only baseline cases (SYN-* and RW-*)
python prompt_eval_harness.py --group baseline

# Filter by tag
python prompt_eval_harness.py --tag mcq
python prompt_eval_harness.py --tag binary
python prompt_eval_harness.py --tag edge-case

# Use OpenAI instead of Anthropic
python prompt_eval_harness.py --provider openai --model gpt-4o

# Use a specific Anthropic model
python prompt_eval_harness.py --model claude-haiku-4-5-20251001

# Save results to a custom path
python prompt_eval_harness.py --output results/my_run.json

# 3-run consistency sweep on invariance cases only
python prompt_eval_harness.py --group invariance --runs 3

# Adjust rate-limit delay between calls (default 0.5s)
python prompt_eval_harness.py --delay 1.0
```

---

### Inspect results

```bash
# Pretty-print the latest result file
cat results/$(ls -t results/ | head -1) | python -m json.tool

# See only failures
cat results/run_*.json | python -c "
import json, sys
d = json.load(sys.stdin)
failures = [r for r in d['results'] if not r['pass']]
print(f'{len(failures)} failure(s):')
for r in failures:
    print(f\"  {r['id']}: {r['score_detail']}\")
"

# Check flaky cases from a multi-run result
cat results/run_*.json | python -c "
import json, sys
d = json.load(sys.stdin)
if d.get('consistency'):
    for cid, s in d['consistency'].items():
        if s['flaky']:
            print(f\"FLAKY {cid}: {s['pass_rate']*100:.0f}% pass rate over {s['runs']} runs\")
"
```

---

### GitHub Actions (CI gate)

```yaml
# .github/workflows/eval.yml
name: Prompt Reliability
on: [push, pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      # Smoke-test the harness itself (no API key needed)
      - run: pytest test_pipeline.py -v --tb=short
      # Full eval run against the API (set secret in repo settings)
      - run: python prompt_eval_harness.py --runs 3 --delay 1.0
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## Test Set Design (10 cases + 5 variant cases = 15 total prompts)

### Synthetic cases (5) — generation rules

| ID | Category | Rule applied |
|---|---|---|
| SYN-001 | Factual Q&A | Single unambiguous geographic fact; regex/exact scoring |
| SYN-002 | Arithmetic | Unit conversion with fixed decimal precision; numeric regex |
| SYN-003 | Classification | Grammar binary — sentence has clear error; Yes/No |
| SYN-004 | NER extraction | Named entities from fixed sentence; set-equality scoring |
| SYN-005 | Math binary | Primality check with known answer; Yes/No |

All synthetic cases were generated under these explicit rules:
1. Ground truth is objectively verifiable without model judgment.
2. Scoring uses regex or set equality — no human review needed.
3. Prompts contain no ambiguous quantifiers ("usually", "often").
4. Expected output is ≤ 5 tokens to minimise extraction noise.

### Real-world / edge cases (5) — redacted from production usage

| ID | Why it's tricky |
|---|---|
| RW-001 | Models hedge on "multiple inheritance" with caveats instead of Yes |
| RW-002 | IEEE 754 float rounding — `round(2.675, 2)` returns `2.67`, not `2.68` |
| RW-003 | Format compliance: JSON array required; models often add preamble |
| RW-004 | Open-ended explanation; scored by keyword coverage, not exact match |
| RW-005 | Simple date arithmetic that occasionally trips day-of-week offsets |

---

## Invariance Test (INV-001 family)

**Hypothesis**: The answer to "capital of France?" should be Paris regardless of surface form.

Three variants tested:
- `INV-001a` — canonical phrasing
- `INV-001b` — paraphrase ("Which city serves as…")
- `INV-001c` — deliberate typos + word scramble ("Waht is the captial ciyt of Frnace?")

**Pass criterion**: All three variants return "Paris". A model is **not invariant** if it fails the typo variant but passes the canonical.

---

## Perturbation Test (PERT-001 family)

**Hypothesis**: A model that knows HTTP status codes should pick 404 regardless of which letter (A/B/C/D) it's assigned, and regardless of distractor codes.

Three variants tested:
- `PERT-001a` — 404 at position C (canonical)
- `PERT-001b` — 404 shuffled to position A (tests position bias)
- `PERT-001c` — Distractors 399 and 612 added (tests plausible-fake resistance)

**Common failure mode**: Models anchored to position C from training data on MCQ format answer "C" even when 404 is now at position A.

---

## Scoring Methods

| Method | When used | How it works |
|---|---|---|
| `exact_ci` | Yes/No, city names, day names | First word/line, case-insensitive comparison |
| `regex` | Numeric outputs | Pattern match anywhere in response |
| `set_match` | List extraction | All expected tokens present (order-free) |
| `json_set` | Format-constrained JSON | Parse → set comparison |
| `semantic_keywords` | Open-ended explanations | ≥ N domain keywords present |

**Not yet implemented** (see roadmap): embedding cosine similarity for full semantic scoring.

---

## Consistency Metric

When `--runs N` is set, each case is evaluated N times. The harness computes:

- **pass_rate**: fraction of runs that passed
- **consistent**: True if every run gave the same result (all-pass or all-fail)
- **flaky**: True if results differ across runs (the primary signal of reliability issues)

A case is considered **flaky** if pass_rate ∈ (0, 1) — i.e., it sometimes passes and sometimes fails at temperature 0. This shouldn't happen but does, due to nondeterminism in hosted inference.

---

## What Ships First vs Later

### Ship first (MVP, this sprint)

- ✅ This harness as-is — covers exact, regex, set, and keyword scoring
- ✅ Invariance and perturbation test groups
- ✅ JSON result persistence + per-run consistency tracking
- ✅ CI integration: exit code 1 on any failure, making it gate-able in GitHub Actions

### Ship next (v1.1)

- **Embedding-based semantic scorer** using `sentence-transformers` — currently falls back to keyword matching; proper cosine similarity would catch paraphrase-identical answers scored incorrectly
- **Flakiness threshold config** — flag a case as flaky only if pass_rate < 0.8 (configurable, not hardcoded)
- **HTML report** — currently outputs JSON; a simple Jinja2 template would make results reviewable without parsing

### Ship later (v2.0)

- **LLM-as-judge scorer** — for open-ended answers (like RW-004) an Anthropic grading call with a rubric is more reliable than keywords
- **Test case generation pipeline** — given a production prompt log, auto-synthesize near-duplicates via paraphrase model
- **Regression tracking** — compare runs across model versions; alert on cases that newly flake
- **Multi-provider fan-out** — run the same suite against Claude, GPT-4o, Gemini simultaneously and diff results

---

## File Structure

```
eval_harness/
├── prompt_eval_harness.py   # Main runner
├── test_pipeline.py         # pytest suite (63 tests, no API key needed)
├── test_cases.json          # Test set (10 base + 6 variant cases)
├── requirements.txt
├── results/                 # Auto-created; one JSON per run
└── README.md                # This file
```

---

## Design Decisions

**Why not pytest?** The cases are data-driven and the output format needs to be machine-readable JSON for downstream dashboards. A plain script is easier to integrate into CI as a subprocess and avoids pytest's XML output format.

**Why temperature=0 by default?** Flakiness at temperature 0 is a stronger reliability signal than flakiness at temperature 1. If a model is nondeterministic at temp 0, it's a hosting/sampling issue worth flagging independently.

**Why hand-rolled scoring instead of an eval framework?** Frameworks like `evals` or `promptfoo` add significant config overhead. For an 8–15 case set this is counterproductive. The scorers here cover ~90% of realistic cases and are readable/debuggable in < 50 lines each.
