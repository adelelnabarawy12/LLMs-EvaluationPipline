# DECISIONS.md

Design and architecture decision log for the Prompt Reliability Eval Harness.
Each entry records what was decided, what the realistic alternatives were, and why the choice was made given the constraints at the time.

---

## D-001 · Plain script over eval framework

**Decision:** The harness is a single `prompt_eval_harness.py` script, not built on top of an existing eval framework such as `promptfoo`, `openai/evals`, or `langchain evaluators`.

**Alternatives considered:**
- `promptfoo` — mature, YAML-driven, supports multiple providers. Ruled out because it requires Node.js, introduces a YAML DSL that obscures scorer logic, and produces HTML/JSON output that is harder to pipe into CI than a plain exit code.
- `openai/evals` — good tooling but tightly coupled to OpenAI's infrastructure and registry conventions. Porting to Anthropic adds friction.
- `langchain evaluators` — heavy dependency tree; the evaluator abstractions are designed for chain evaluation, not prompt-level consistency testing.

**Why plain script:** For a harness with 8–16 cases, framework overhead outweighs the benefits. The entire scoring logic is readable in under 50 lines per scorer. A subprocess call gives a clean exit code for CI gating without needing pytest XML or framework-specific reporters. When the case count grows past ~100 and multi-model fan-out is needed, revisiting a framework is reasonable.

---

## D-002 · Temperature 0 as the default

**Decision:** All API calls default to `temperature=0.0`.

**Alternatives considered:**
- `temperature=1.0` (model default) — more realistic for production but conflates prompt flakiness with sampling noise, making it impossible to distinguish the two.
- Parameterised per-case temperature — adds schema complexity with little benefit at this scale.

**Why temperature 0:** Flakiness at temperature 0 is a stronger and more actionable signal. If a model gives inconsistent answers with no sampling noise, the cause is either nondeterminism in the hosting infrastructure or prompt ambiguity — both worth flagging. Temperature effects can be studied separately by re-running with `--temperature 1.0` on cases that already pass at temp 0.

---

## D-003 · Five scorer types, no embedding scorer in MVP

**Decision:** Ship five scorers — `exact_ci`, `regex`, `set_match`, `json_set`, `semantic_keywords` — and defer embedding-based cosine similarity to v1.1.

**Alternatives considered:**
- Include `sentence-transformers` cosine scorer from day one — would handle paraphrase-identical answers that keyword scoring misses. Rejected because it adds a 500 MB model download, GPU/CPU variance in scores, and a similarity threshold that requires calibration before it's trustworthy.
- LLM-as-judge scorer — most flexible, but adds latency, cost, and a second model's nondeterminism into the scoring loop. Deferred to v2.0.

**Why defer:** The five shipped scorers cover all 16 current cases without false negatives. `semantic_keywords` is an intentional intermediate step — it is transparent, deterministic, and tunable (via `required_keyword_count`) without requiring a threshold calibration study. The scorer registry is open/closed: adding a new scorer is a one-function addition that doesn't touch existing cases.

---

## D-004 · Test set split: 50% synthetic, 50% real-world

**Decision:** 5 synthetic cases generated under explicit rules; 5 real-world edge cases sourced from production usage (redacted).

**Alternatives considered:**
- All synthetic — fully reproducible and auditable, but misses the class of failures that only emerge from real queries (hedging on binary questions, format non-compliance, floating-point gotchas).
- All real-world — harder to maintain ground truth and explain to new contributors; no documented generation process.

**Why 50/50:** Synthetic cases validate that scorers work correctly under controlled conditions. Real-world cases validate that the harness catches the failures that actually matter in production. The split also provides a natural regression layer: synthetic cases should never flake; real-world cases are where flakiness is expected and measured.

**Synthetic generation rules (explicit):**
1. Ground truth is objectively verifiable without model judgment.
2. Scoring uses `regex` or `set_match` — no human review required.
3. Prompts contain no ambiguous quantifiers ("usually", "often", "typically").
4. Expected output is ≤ 5 tokens to minimise extraction ambiguity.
5. Each case tests exactly one capability (no compound tasks).

---

## D-005 · Invariance test design

**Decision:** One invariance family (INV-001) with three variants: canonical phrasing, paraphrase, and deliberate typos.

**Alternatives considered:**
- Multiple invariance families covering different domains — more coverage but dilutes the signal in a small harness. Better suited to v1.1 when case generation is automated.
- Only canonical vs. paraphrase (no typo variant) — misses a real failure mode: models that rely on spell-checking as part of comprehension and silently fail on noisy input.

**Why this design:** The typo variant (`INV-001c`) is the most diagnostic. A model that passes canonical and paraphrase but fails typos reveals a fragility in its input-normalisation that canonical testing would never surface. The pair_id grouping in `test_cases.json` makes it easy to add more families without changing the reporting logic.

---

## D-006 · Perturbation test design

**Decision:** One perturbation family (PERT-001) with three variants: canonical option order, shuffled options, and plausible distractor injection.

**Alternatives considered:**
- Only canonical vs. shuffled (no distractor variant) — misses the distractor failure mode, where models anchor on a familiar-looking number (399, 612) over the correct one.
- Perturbation on open-ended questions — harder to score and less diagnostic. MCQ format makes position bias directly observable.

**Why this design:** The position-bias failure (`PERT-001b`) is the primary target. Models trained heavily on MCQ data develop a statistical prior toward certain answer positions (C is the most common correct answer in many training datasets). Testing with shuffled options directly measures this bias. The distractor variant (`PERT-001c`) is a secondary test for plausible-fake resistance, which is distinct from position bias and worth separating.

---

## D-007 · Consistency metric: pass_rate + flaky flag

**Decision:** When `--runs N` is set, report `pass_rate` (fraction of runs that passed) and a boolean `flaky` flag (True if pass_rate ∈ (0, 1)).

**Alternatives considered:**
- Report only pass/fail per run without aggregation — leaves the flakiness judgement to the reader.
- Report variance / standard deviation — statistically richer but harder to act on. "This case is flaky" is a clearer signal than "this case has σ=0.47".
- Configurable flakiness threshold (e.g., flag only if pass_rate < 0.8) — more nuanced but requires calibration. Deferred to v1.1 as `--flaky-threshold`.

**Why binary flaky flag:** At temperature 0, a case should either always pass or always fail. Any mixed result is anomalous and worth immediate investigation, so a binary flag is appropriate. The `pass_rate` is reported alongside it for cases where the threshold should be relaxed (e.g., when intentionally running at temperature > 0).

---

## D-008 · JSON output, not HTML or stdout-only

**Decision:** Results are persisted as a timestamped JSON file in `results/`. Human-readable summary is printed to stdout; the JSON is the machine-readable record.

**Alternatives considered:**
- Stdout-only — simple but not archivable. No way to diff runs across model versions.
- HTML report — readable without tooling, but not parseable by downstream scripts. Deferred to v1.1.
- SQLite — appropriate at scale (100+ cases, continuous runs) but heavyweight for an MVP.

**Why JSON:** It is the lowest-friction format that is both human-inspectable (`python -m json.tool`) and machine-parseable (CI scripts, dashboards, future regression trackers). The `meta` envelope records provenance (model, temperature, timestamp, run count) so files are self-describing. Filenames are ISO 8601 timestamps so `ls -t results/` always surfaces the latest run.

---

## D-009 · pytest for the test suite, not unittest

**Decision:** `test_pipeline.py` uses `pytest` with class-based test grouping.

**Alternatives considered:**
- `unittest` only — available in stdlib, no extra dependency. Rejected because pytest's parametrize, fixture, and `-k` filter features are materially better for a data-driven test suite.
- `doctest` — appropriate for documentation examples but not for testing scorer edge cases and CLI behaviour.

**Why pytest:** The `-k` filter lets contributors run a single scorer class in isolation (`pytest -k TestJsonSet`), which speeds up the inner loop when adding a new scorer. Class grouping maps cleanly onto the eight distinct concerns (scorers, consistency stats, invariance, perturbation, pipeline, output, CLI, registry). The 63 tests run in under 2 seconds with zero API calls, making it fast enough to run on every commit.

---

## D-010 · Mock strategy: prompt fingerprint, not case ID injection

**Decision:** Integration tests mock `call_model` by matching on the first 40 characters of each prompt (a prompt fingerprint), not by injecting case IDs into prompts.

**Alternatives considered:**
- Inject case ID into every prompt as a hidden token — would make routing trivial but would corrupt the prompts being tested, invalidating the test.
- Match on full prompt equality — fragile; any prompt wording change breaks the mock map.
- Separate fixtures per test — more explicit but requires duplicating prompt text in the test file, creating a maintenance hazard when `test_cases.json` changes.

**Why prompt fingerprint:** The first 40 characters of each prompt are unique within the current test set and stable enough for test purposes. The mock map is built dynamically from `test_cases.json` at test time, so it stays in sync automatically when cases are added or edited. If two prompts ever share a 40-character prefix, the test will fail loudly, surfacing the collision rather than silently returning the wrong canned answer.

---

## D-011 · No test database or fixture files

**Decision:** `test_cases.json` is the single source of truth for both the harness and the test suite. There is no separate fixtures directory or test database.

**Alternatives considered:**
- Separate `fixtures/` directory with one JSON file per case — more granular version control but harder to get an overview of the full test set.
- Inline case definitions in `test_pipeline.py` — eliminates the shared file but means the test suite and the harness diverge over time.

**Why single source:** The test suite loads `test_cases.json` directly (`load_cases()`), so any case added for the harness is automatically covered by the integration tests. The scorer registry completeness test (`TestScorerRegistry::test_all_case_scorers_registered`) enforces that every scorer referenced in the JSON exists in the harness — a cheap contract that catches copy-paste errors.
