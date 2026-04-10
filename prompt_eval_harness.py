#!/usr/bin/env python3
"""
prompt_eval_harness.py
======================
Lean evaluation harness for measuring LLM response consistency.

Usage
-----
  # Single run (all cases):
  python prompt_eval_harness.py

  # Multiple runs for consistency measurement:
  python prompt_eval_harness.py --runs 5

  # Filter by group or tag:
  python prompt_eval_harness.py --group invariance
  python prompt_eval_harness.py --tag mcq

  # Use a different model:
  python prompt_eval_harness.py --model gpt-4o

  # Dry-run (no API calls, shows prompts):
  python prompt_eval_harness.py --dry-run

Environment
-----------
  Set ANTHROPIC_API_KEY or OPENAI_API_KEY depending on --provider.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Optional deps — graceful degradation ───────────────────────────────────────
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

CASES_PATH = Path(__file__).parent / "test_cases.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Scoring functions ──────────────────────────────────────────────────────────

def score_exact_ci(response: str, case: dict) -> dict:
    """Case-insensitive exact match (first token / line heuristic)."""
    expected = case["expected"].strip().lower()
    # Extract first non-empty line to handle preamble-y answers
    first_line = next((l.strip() for l in response.splitlines() if l.strip()), "")
    # Also try first word
    first_word = first_line.split()[0].rstrip(".,;:") if first_line else ""
    hit = expected in {first_line.lower(), first_word.lower(), response.strip().lower()}
    return {"pass": hit, "extracted": first_word, "expected": case["expected"]}


def score_regex(response: str, case: dict) -> dict:
    """Regex match anywhere in the response."""
    pattern = case.get("pattern", re.escape(case.get("expected", "")))
    m = re.search(pattern, response)
    return {"pass": bool(m), "match": m.group(0) if m else None, "pattern": pattern}


def score_set_match(response: str, case: dict) -> dict:
    """All expected set items appear (comma/newline separated response)."""
    expected = {e.strip().lower() for e in case["expected_set"]}
    # Normalise response to tokens
    tokens = {t.strip().lower().strip('"\'') for t in re.split(r"[,\n]+", response) if t.strip()}
    found = expected & tokens
    missing = expected - tokens
    return {"pass": len(missing) == 0, "found": list(found), "missing": list(missing)}


def score_json_set(response: str, case: dict) -> dict:
    """Parse response as JSON array; compare as set."""
    expected = {e.lower() for e in case["expected_set"]}
    # Strip markdown fences
    clean = re.sub(r"```[a-z]*\n?", "", response).strip().strip("`")
    try:
        parsed = json.loads(clean)
        got = {str(v).lower() for v in parsed}
        missing = expected - got
        extra = got - expected
        return {"pass": len(missing) == 0, "missing": list(missing), "extra": list(extra)}
    except json.JSONDecodeError as e:
        return {"pass": False, "error": str(e), "raw": clean[:120]}


def score_semantic_keywords(response: str, case: dict) -> dict:
    """At least N required keywords appear in the response."""
    keywords = case["expected_keywords"]
    required = case.get("required_keyword_count", 2)
    resp_lower = response.lower()
    found = [k for k in keywords if k in resp_lower]
    return {
        "pass": len(found) >= required,
        "found": found,
        "required": required,
        "total_keywords": keywords,
    }


SCORERS = {
    "exact_ci": score_exact_ci,
    "regex": score_regex,
    "set_match": score_set_match,
    "json_set": score_json_set,
    "semantic_keywords": score_semantic_keywords,
}

# ── LLM clients ───────────────────────────────────────────────────────────────

def call_anthropic(prompt: str, model: str, temperature: float = 0.0) -> str:
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def call_openai(prompt: str, model: str, temperature: float = 0.0) -> str:
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        max_tokens=256,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def call_model(prompt: str, provider: str, model: str, temperature: float) -> str:
    if provider == "anthropic":
        return call_anthropic(prompt, model, temperature)
    elif provider == "openai":
        return call_openai(prompt, model, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ── Consistency analysis ───────────────────────────────────────────────────────

def consistency_stats(results_per_run: list[list[dict]]) -> dict:
    """
    Given N runs × M cases, compute per-case pass-rate and overall consistency.
    Returns a dict keyed by case_id.
    """
    by_case: dict[str, list[bool]] = defaultdict(list)
    for run in results_per_run:
        for r in run:
            by_case[r["id"]].append(r["pass"])

    stats = {}
    for cid, passes in by_case.items():
        n = len(passes)
        n_pass = sum(passes)
        stats[cid] = {
            "runs": n,
            "pass_count": n_pass,
            "pass_rate": round(n_pass / n, 3),
            "consistent": len(set(passes)) == 1,  # same answer every run
            "flaky": len(set(passes)) > 1,
        }
    return stats


def invariance_report(all_results: list[dict]) -> dict:
    """For invariance pairs, check that all variants give the same pass/fail."""
    pairs: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        if r.get("group") == "invariance" and "pair_id" in r:
            pairs[r["pair_id"]].append(r)
    report = {}
    for pair_id, members in pairs.items():
        passes = [m["pass"] for m in members]
        report[pair_id] = {
            "variants": [m["id"] for m in members],
            "pass_per_variant": dict(zip([m["id"] for m in members], passes)),
            "invariant": len(set(passes)) == 1,
        }
    return report


def perturbation_report(all_results: list[dict]) -> dict:
    """For perturbation pairs, surface which variants caused failures."""
    pairs: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        if r.get("group") == "perturbation" and "pair_id" in r:
            pairs[r["pair_id"]].append(r)
    report = {}
    for pair_id, members in pairs.items():
        passes = [m["pass"] for m in members]
        report[pair_id] = {
            "variants": {m["id"]: {"pass": m["pass"], "variant": m.get("variant")} for m in members},
            "all_pass": all(passes),
            "failure_modes": [m["id"] for m in members if not m["pass"]],
        }
    return report

# ── Core runner ───────────────────────────────────────────────────────────────

def run_cases(
    cases: list[dict],
    provider: str,
    model: str,
    temperature: float,
    dry_run: bool,
    delay: float,
) -> list[dict]:
    results = []
    for i, case in enumerate(cases):
        cid = case["id"]
        prompt = case["prompt"]
        scorer_name = case["scoring"]
        scorer = SCORERS.get(scorer_name)

        if scorer is None:
            print(f"  [SKIP] {cid}: unknown scorer '{scorer_name}'")
            continue

        print(f"  [{i+1:02d}/{len(cases)}] {cid} ({case['type']}, {scorer_name})", end=" ... ", flush=True)

        if dry_run:
            response = f"[DRY-RUN: {cid}]"
            score = {"pass": None, "dry_run": True}
        else:
            try:
                response = call_model(prompt, provider, model, temperature)
                score = scorer(response, case)
            except Exception as e:
                response = ""
                score = {"pass": False, "error": str(e)}

        status = "✓" if score.get("pass") else ("?" if score.get("pass") is None else "✗")
        print(status)

        results.append({
            "id": cid,
            "type": case["type"],
            "group": case.get("group", "baseline"),
            "pair_id": case.get("pair_id"),
            "variant": case.get("variant"),
            "category": case["category"],
            "tags": case.get("tags", []),
            "prompt": prompt[:120] + ("…" if len(prompt) > 120 else ""),
            "response": response[:300],
            "scorer": scorer_name,
            "score_detail": score,
            "pass": score.get("pass", False),
        })

        if not dry_run and delay > 0:
            time.sleep(delay)

    return results

# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], consistency: dict | None = None):
    total = len(all_results)
    passed = sum(1 for r in all_results if r["pass"])
    print("\n" + "═" * 60)
    print(f"  RESULTS  {passed}/{total} passed  ({100*passed//total if total else 0}%)")
    print("═" * 60)

    by_group: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_group[r["group"]].append(r)

    for group, items in sorted(by_group.items()):
        gpass = sum(1 for x in items if x["pass"])
        print(f"\n  [{group.upper()}]  {gpass}/{len(items)}")
        for r in items:
            icon = "✓" if r["pass"] else "✗"
            flaky = ""
            if consistency and r["id"] in consistency:
                cs = consistency[r["id"]]
                if cs["flaky"]:
                    flaky = f"  ⚠ flaky ({cs['pass_rate']*100:.0f}% pass rate)"
            print(f"    {icon} {r['id']:14s}  {r['category']}{flaky}")
            if not r["pass"]:
                detail = r["score_detail"]
                print(f"             └─ {detail}")

    # Invariance block
    inv = invariance_report(all_results)
    if inv:
        print("\n  [INVARIANCE PAIRS]")
        for pair_id, info in inv.items():
            icon = "✓" if info["invariant"] else "✗"
            print(f"    {icon} {pair_id}  invariant={info['invariant']}")
            for vid, p in info["pass_per_variant"].items():
                print(f"       {'✓' if p else '✗'} {vid}")

    # Perturbation block
    pert = perturbation_report(all_results)
    if pert:
        print("\n  [PERTURBATION PAIRS]")
        for pair_id, info in pert.items():
            icon = "✓" if info["all_pass"] else "✗"
            print(f"    {icon} {pair_id}  all_pass={info['all_pass']}")
            if info["failure_modes"]:
                print(f"       Failures: {', '.join(info['failure_modes'])}")

    print()

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prompt Reliability Eval Harness")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model", default="claude-opus-4-20250514")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=1, help="Repeat all cases N times for consistency tracking")
    p.add_argument("--group", help="Filter cases by group (baseline|invariance|perturbation)")
    p.add_argument("--tag", help="Filter cases by tag substring")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    p.add_argument("--dry-run", action="store_true", help="Skip API calls, print prompts only")
    p.add_argument("--cases-path", default=str(CASES_PATH))
    p.add_argument("--output", help="Save JSON results to this path")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.cases_path) as f:
        data = json.load(f)
    cases: list[dict] = data["cases"]

    # Filtering
    if args.group:
        cases = [c for c in cases if c.get("group") == args.group]
    if args.tag:
        cases = [c for c in cases if any(args.tag in t for t in c.get("tags", []))]

    if not cases:
        print("No cases match the filter.")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Prompt Reliability Eval Harness")
    print(f"  Provider : {args.provider}  |  Model: {args.model}")
    print(f"  Cases    : {len(cases)}  |  Runs: {args.runs}")
    print(f"  Temp     : {args.temperature}  |  Dry-run: {args.dry_run}")
    print(f"{'─'*60}\n")

    all_run_results: list[list[dict]] = []
    flat_results: list[dict] = []

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n── Run {run_idx + 1}/{args.runs} ──")
        run_results = run_cases(
            cases,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            dry_run=args.dry_run,
            delay=args.delay,
        )
        all_run_results.append(run_results)
        flat_results.extend(run_results)

    consistency = consistency_stats(all_run_results) if args.runs > 1 else None
    print_summary(flat_results, consistency)

    # Persist results
    now = datetime.now(timezone.utc)
    output_path = args.output or str(
        RESULTS_DIR / f"run_{now.strftime('%Y%m%dT%H%M%S')}.json"
    )
    payload = {
        "meta": {
            "timestamp": now.isoformat(),
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "runs": args.runs,
            "cases_total": len(cases),
        },
        "results": flat_results,
        "consistency": consistency,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved → {output_path}\n")


if __name__ == "__main__":
    main()
