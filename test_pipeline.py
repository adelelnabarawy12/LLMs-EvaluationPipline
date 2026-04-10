#!/usr/bin/env python3
"""
test_pipeline.py
================
Example test suite for the prompt_eval_harness pipeline.

Covers:
  1. Every scorer function (unit tests — no API calls)
  2. The full run_cases() pipeline with a mock LLM (integration)
  3. Invariance and perturbation reporting logic
  4. Consistency stats across multiple runs
  5. JSON output persistence
  6. CLI flag behaviour (--group, --tag, --dry-run)

Run with:
  pytest test_pipeline.py -v
  pytest test_pipeline.py -v -k "scorer"      # scorer unit tests only
  pytest test_pipeline.py -v -k "pipeline"    # pipeline integration only
  pytest test_pipeline.py --tb=short          # brief traceback on failure
"""

import json
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── make the harness importable from the same directory ────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from prompt_eval_harness import (
    score_exact_ci,
    score_regex,
    score_set_match,
    score_json_set,
    score_semantic_keywords,
    consistency_stats,
    invariance_report,
    perturbation_report,
    run_cases,
    SCORERS,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. SCORER UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestExactCI:
    """score_exact_ci — case-insensitive first-token matching."""

    def _case(self, expected):
        return {"expected": expected}

    def test_exact_match(self):
        r = score_exact_ci("Paris", self._case("Paris"))
        assert r["pass"] is True

    def test_case_insensitive(self):
        r = score_exact_ci("paris", self._case("Paris"))
        assert r["pass"] is True

    def test_first_word_extraction(self):
        # Model adds trailing punctuation or a sentence
        r = score_exact_ci("Yes, that is correct.", self._case("Yes"))
        assert r["pass"] is True

    def test_preamble_first_line(self):
        # Model gives answer on first line then explains
        r = score_exact_ci("Monday\n\nBecause Wednesday + 5 days = Monday.", self._case("Monday"))
        assert r["pass"] is True

    def test_wrong_answer(self):
        r = score_exact_ci("London", self._case("Paris"))
        assert r["pass"] is False

    def test_empty_response(self):
        r = score_exact_ci("", self._case("Paris"))
        assert r["pass"] is False

    def test_yes_no_variants(self):
        assert score_exact_ci("No", {"expected": "No"})["pass"] is True
        assert score_exact_ci("YES", {"expected": "Yes"})["pass"] is True
        assert score_exact_ci("no.", {"expected": "No"})["pass"] is True


class TestRegex:
    """score_regex — pattern match anywhere in response."""

    def _case(self, pattern):
        return {"pattern": pattern}

    def test_exact_numeric(self):
        r = score_regex("The answer is 22.2 degrees.", self._case(r"22\.2"))
        assert r["pass"] is True
        assert r["match"] == "22.2"

    def test_no_match(self):
        r = score_regex("The answer is 22.8 degrees.", self._case(r"22\.2"))
        assert r["pass"] is False
        assert r["match"] is None

    def test_pattern_at_end(self):
        r = score_regex("Result: 2.67", self._case(r"2\.67"))
        assert r["pass"] is True

    def test_pattern_embedded(self):
        r = score_regex("round(2.675, 2) evaluates to 2.67 in Python.", self._case(r"2\.67"))
        assert r["pass"] is True

    def test_fallback_to_expected_key(self):
        # When no 'pattern' key, falls back to re.escape(expected)
        case = {"expected": "2.67"}
        r = score_regex("The result is 2.67.", case)
        assert r["pass"] is True


class TestSetMatch:
    """score_set_match — all items from expected_set appear in response."""

    def _case(self, items):
        return {"expected_set": items}

    def test_all_present_comma_separated(self):
        r = score_set_match("marie curie, warsaw, paris, sorbonne", self._case(
            ["marie curie", "warsaw", "paris", "sorbonne"]
        ))
        assert r["pass"] is True
        assert r["missing"] == []

    def test_partial_match_fails(self):
        r = score_set_match("marie curie, warsaw", self._case(
            ["marie curie", "warsaw", "paris", "sorbonne"]
        ))
        assert r["pass"] is False
        assert "paris" in r["missing"]

    def test_order_independent(self):
        r = score_set_match("sorbonne, paris, warsaw, marie curie", self._case(
            ["marie curie", "warsaw", "paris", "sorbonne"]
        ))
        assert r["pass"] is True

    def test_newline_separated(self):
        r = score_set_match("red\ngreen\nblue", self._case(["red", "green", "blue"]))
        assert r["pass"] is True

    def test_extra_items_ok(self):
        # Extra items in response are fine — we only check coverage
        r = score_set_match("red, green, blue, purple", self._case(["red", "green", "blue"]))
        assert r["pass"] is True

    def test_empty_response_fails(self):
        r = score_set_match("", self._case(["red", "green"]))
        assert r["pass"] is False


class TestJsonSet:
    """score_json_set — parse JSON array, compare as set."""

    def _case(self, items):
        return {"expected_set": items}

    def test_valid_json_array(self):
        r = score_json_set('["red", "green", "blue"]', self._case(["red", "green", "blue"]))
        assert r["pass"] is True
        assert r["missing"] == []

    def test_case_insensitive(self):
        r = score_json_set('["Red", "Green", "Blue"]', self._case(["red", "green", "blue"]))
        assert r["pass"] is True

    def test_markdown_fence_stripped(self):
        response = "```json\n[\"red\", \"green\", \"blue\"]\n```"
        r = score_json_set(response, self._case(["red", "green", "blue"]))
        assert r["pass"] is True

    def test_missing_item_fails(self):
        r = score_json_set('["red", "green"]', self._case(["red", "green", "blue"]))
        assert r["pass"] is False
        assert "blue" in r["missing"]

    def test_invalid_json_fails(self):
        r = score_json_set("red, green, blue", self._case(["red", "green", "blue"]))
        assert r["pass"] is False
        assert "error" in r

    def test_extra_items_noted(self):
        r = score_json_set('["red", "green", "blue", "yellow"]', self._case(["red", "green", "blue"]))
        assert r["pass"] is True
        assert "yellow" in r["extra"]


class TestSemanticKeywords:
    """score_semantic_keywords — keyword coverage scoring."""

    def _case(self, keywords, required=2):
        return {"expected_keywords": keywords, "required_keyword_count": required}

    def test_enough_keywords(self):
        response = "A REST API is a stateless HTTP interface where a client requests resources from a server."
        r = score_semantic_keywords(response, self._case(["http", "stateless", "resource", "client", "server"]))
        assert r["pass"] is True
        assert len(r["found"]) >= 2

    def test_below_threshold_fails(self):
        response = "An API is a way for programs to talk to each other."
        r = score_semantic_keywords(response, self._case(["http", "stateless", "resource", "client", "server"], required=2))
        assert r["pass"] is False

    def test_exact_threshold(self):
        response = "A REST API uses http and is stateless."
        r = score_semantic_keywords(response, self._case(["http", "stateless", "resource"], required=2))
        assert r["pass"] is True
        assert len(r["found"]) == 2

    def test_case_insensitive_keywords(self):
        response = "The CLIENT sends HTTP requests."
        r = score_semantic_keywords(response, self._case(["http", "client"], required=2))
        assert r["pass"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSISTENCY STATS
# ══════════════════════════════════════════════════════════════════════════════

class TestConsistencyStats:
    """consistency_stats — multi-run flakiness detection."""

    def _result(self, cid, passed):
        return {"id": cid, "pass": passed}

    def test_all_pass(self):
        runs = [
            [self._result("SYN-001", True)],
            [self._result("SYN-001", True)],
            [self._result("SYN-001", True)],
        ]
        stats = consistency_stats(runs)
        assert stats["SYN-001"]["pass_rate"] == 1.0
        assert stats["SYN-001"]["consistent"] is True
        assert stats["SYN-001"]["flaky"] is False

    def test_all_fail(self):
        runs = [
            [self._result("SYN-001", False)],
            [self._result("SYN-001", False)],
        ]
        stats = consistency_stats(runs)
        assert stats["SYN-001"]["pass_rate"] == 0.0
        assert stats["SYN-001"]["consistent"] is True  # consistently failing
        assert stats["SYN-001"]["flaky"] is False

    def test_flaky_case(self):
        runs = [
            [self._result("RW-002", True)],
            [self._result("RW-002", False)],
            [self._result("RW-002", True)],
        ]
        stats = consistency_stats(runs)
        s = stats["RW-002"]
        assert s["flaky"] is True
        assert s["consistent"] is False
        assert round(s["pass_rate"], 2) == round(2/3, 2)

    def test_multiple_cases(self):
        runs = [
            [self._result("A", True), self._result("B", False)],
            [self._result("A", True), self._result("B", True)],
        ]
        stats = consistency_stats(runs)
        assert stats["A"]["flaky"] is False
        assert stats["B"]["flaky"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 3. INVARIANCE & PERTURBATION REPORT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

class TestInvarianceReport:
    def _result(self, cid, group, pair_id, passed):
        return {"id": cid, "group": group, "pair_id": pair_id, "pass": passed}

    def test_fully_invariant(self):
        results = [
            self._result("INV-001a", "invariance", "INV-001", True),
            self._result("INV-001b", "invariance", "INV-001", True),
            self._result("INV-001c", "invariance", "INV-001", True),
        ]
        report = invariance_report(results)
        assert report["INV-001"]["invariant"] is True

    def test_variant_breaks_invariance(self):
        results = [
            self._result("INV-001a", "invariance", "INV-001", True),
            self._result("INV-001b", "invariance", "INV-001", True),
            self._result("INV-001c", "invariance", "INV-001", False),  # typo variant fails
        ]
        report = invariance_report(results)
        assert report["INV-001"]["invariant"] is False

    def test_ignores_non_invariance_group(self):
        results = [
            self._result("SYN-001", "baseline", None, True),
            self._result("INV-001a", "invariance", "INV-001", True),
        ]
        report = invariance_report(results)
        assert "INV-001" in report
        # SYN-001 has no pair_id so shouldn't appear
        assert len(report) == 1


class TestPerturbationReport:
    def _result(self, cid, pair_id, variant, passed):
        return {"id": cid, "group": "perturbation", "pair_id": pair_id, "variant": variant, "pass": passed}

    def test_all_pass(self):
        results = [
            self._result("PERT-001a", "PERT-001", "order_A", True),
            self._result("PERT-001b", "PERT-001", "order_B", True),
            self._result("PERT-001c", "PERT-001", "distractor", True),
        ]
        report = perturbation_report(results)
        assert report["PERT-001"]["all_pass"] is True
        assert report["PERT-001"]["failure_modes"] == []

    def test_position_bias_failure(self):
        results = [
            self._result("PERT-001a", "PERT-001", "order_A", True),
            self._result("PERT-001b", "PERT-001", "order_B", False),  # shuffled options fail
            self._result("PERT-001c", "PERT-001", "distractor", True),
        ]
        report = perturbation_report(results)
        assert report["PERT-001"]["all_pass"] is False
        assert "PERT-001b" in report["PERT-001"]["failure_modes"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. INTEGRATION — run_cases() with a mock LLM
# ══════════════════════════════════════════════════════════════════════════════

# Canned responses keyed by case ID — what a "perfect" model would say
MOCK_RESPONSES = {
    "SYN-001": "Paris",
    "SYN-002": "22.2",
    "SYN-003": "No",
    "SYN-004": "marie curie, warsaw, paris, sorbonne",
    "SYN-005": "Yes",
    "RW-001": "Yes",
    "RW-002": "2.67",
    "RW-003": '["red", "green", "blue"]',
    "RW-004": "A REST API is a stateless http interface where a client requests resources from a server.",
    "RW-005": "Monday",
    "INV-001a": "Paris",
    "INV-001b": "Paris",
    "INV-001c": "Paris",
    "PERT-001a": "C",
    "PERT-001b": "A",
    "PERT-001c": "C",
}


def mock_call_model(prompt, provider, model, temperature):
    """Intercept call_model; return canned answer based on prompt content."""
    for case_id, response in MOCK_RESPONSES.items():
        # Match by a unique string each prompt contains
        if case_id.replace("-", " ") in prompt or case_id in prompt:
            return response
    # Fallback: echo first 10 chars so scorer sees something
    return prompt[:10]


def load_cases(path=None):
    cases_path = path or Path(__file__).parent / "test_cases.json"
    with open(cases_path) as f:
        return json.load(f)["cases"]


class TestRunCasesPipeline:
    """Integration tests: run_cases() with mocked LLM responses."""

    def _run(self, cases, responses=None):
        resp_map = responses or MOCK_RESPONSES

        def _mock(prompt, provider, model, temperature):
            for cid, text in resp_map.items():
                if cid in prompt or cid.replace("-", " ") in prompt:
                    return text
            return "Paris"  # safe fallback for geography cases

        with patch("prompt_eval_harness.call_model", side_effect=_mock):
            return run_cases(cases, provider="anthropic", model="test",
                             temperature=0.0, dry_run=False, delay=0)

    def test_all_baseline_cases_pass_with_perfect_responses(self):
        """
        Simulate a 'perfect' model: for each case, derive the ideal response
        directly from the case definition and key it on the prompt's first 40 chars
        (unique within our small test set).
        """
        cases = load_cases()
        baseline = [c for c in cases if c.get("group") == "baseline"]

        prompt_to_response = {}
        for c in baseline:
            key = c["prompt"][:40]
            scoring = c["scoring"]
            if scoring == "exact_ci":
                answer = c["expected"]
            elif scoring == "regex":
                answer = c.get("expected", "22.2")
            elif scoring == "set_match":
                answer = ", ".join(c["expected_set"])
            elif scoring == "json_set":
                answer = json.dumps(list(c["expected_set"]))
            elif scoring == "semantic_keywords":
                kws = c["expected_keywords"]
                answer = f"A REST API uses {kws[0]} and is {kws[1]}, involving {kws[2]} and {kws[3]}."
            else:
                answer = "Paris"
            prompt_to_response[key] = answer

        def _perfect(prompt, provider, model, temperature):
            return prompt_to_response.get(prompt[:40], "Paris")

        with patch("prompt_eval_harness.call_model", side_effect=_perfect):
            results = run_cases(baseline, "anthropic", "test", 0.0, False, 0)

        passed = [r for r in results if r["pass"]]
        assert len(passed) == len(baseline), (
            f"Expected all {len(baseline)} baseline cases to pass, got {len(passed)}.\n"
            f"Failures: {[(r['id'], r['score_detail']) for r in results if not r['pass']]}"
        )

    def test_dry_run_skips_api(self):
        cases = load_cases()[:3]
        with patch("prompt_eval_harness.call_model") as mock_llm:
            results = run_cases(cases, "anthropic", "test", 0.0, dry_run=True, delay=0)
        mock_llm.assert_not_called()
        for r in results:
            assert r["score_detail"].get("dry_run") is True

    def test_wrong_answers_fail(self):
        cases = [c for c in load_cases() if c["id"] == "SYN-001"]

        def _wrong(prompt, provider, model, temperature):
            return "London"

        with patch("prompt_eval_harness.call_model", side_effect=_wrong):
            results = run_cases(cases, "anthropic", "test", 0.0, False, 0)

        assert results[0]["pass"] is False

    def test_api_error_is_caught(self):
        cases = [c for c in load_cases() if c["id"] == "SYN-001"]

        def _boom(prompt, provider, model, temperature):
            raise ConnectionError("API timeout")

        with patch("prompt_eval_harness.call_model", side_effect=_boom):
            results = run_cases(cases, "anthropic", "test", 0.0, False, 0)

        assert results[0]["pass"] is False
        assert "error" in results[0]["score_detail"]

    def test_result_schema(self):
        """Each result dict has the required keys."""
        cases = load_cases()[:2]
        with patch("prompt_eval_harness.call_model", return_value="Paris"):
            results = run_cases(cases, "anthropic", "test", 0.0, False, 0)
        required_keys = {"id", "type", "group", "category", "tags", "prompt",
                         "response", "scorer", "score_detail", "pass"}
        for r in results:
            assert required_keys.issubset(r.keys()), f"Missing keys in {r['id']}: {required_keys - r.keys()}"

    def test_group_filtering_works(self):
        all_cases = load_cases()
        inv_cases = [c for c in all_cases if c.get("group") == "invariance"]
        assert len(inv_cases) == 3
        assert all(c["id"].startswith("INV") for c in inv_cases)

    def test_tag_filtering_works(self):
        all_cases = load_cases()
        mcq_cases = [c for c in all_cases if "mcq" in c.get("tags", [])]
        assert len(mcq_cases) >= 3  # PERT-001a/b/c
        assert all("PERT" in c["id"] for c in mcq_cases)


# ══════════════════════════════════════════════════════════════════════════════
# 5. JSON OUTPUT PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputPersistence:
    def test_results_json_is_valid(self, tmp_path):
        """run the CLI in dry-run and verify the output JSON is well-formed."""
        output = tmp_path / "out.json"
        result = subprocess.run(
            [sys.executable, "prompt_eval_harness.py",
             "--dry-run", "--output", str(output), "--group", "baseline"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert output.exists(), "Output file not created"

        with open(output) as f:
            data = json.load(f)

        # Schema checks
        assert "meta" in data
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 10  # baseline group has 10 cases
        assert data["meta"]["runs"] == 1

    def test_meta_fields_present(self, tmp_path):
        output = tmp_path / "out.json"
        subprocess.run(
            [sys.executable, "prompt_eval_harness.py",
             "--dry-run", "--output", str(output), "--group", "invariance"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent,
        )
        with open(output) as f:
            data = json.load(f)
        meta = data["meta"]
        for key in ("timestamp", "provider", "model", "temperature", "runs", "cases_total"):
            assert key in meta, f"Missing meta key: {key}"

    def test_result_entries_have_pass_field(self, tmp_path):
        output = tmp_path / "out.json"
        subprocess.run(
            [sys.executable, "prompt_eval_harness.py",
             "--dry-run", "--output", str(output)],
            capture_output=True, text=True,
            cwd=Path(__file__).parent,
        )
        with open(output) as f:
            data = json.load(f)
        for r in data["results"]:
            assert "pass" in r
            assert "id" in r
            assert "scorer" in r


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════

class TestCLI:
    def _run_cli(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "prompt_eval_harness.py", "--dry-run", *args],
            capture_output=True, text=True,
            cwd=cwd or Path(__file__).parent,
        )

    def test_dry_run_exits_zero(self):
        r = self._run_cli()
        assert r.returncode == 0

    def test_group_flag_filters(self):
        r = self._run_cli("--group", "invariance")
        assert "INV-001a" in r.stdout
        assert "SYN-001" not in r.stdout

    def test_tag_flag_filters(self):
        r = self._run_cli("--tag", "mcq")
        assert "PERT-001" in r.stdout
        assert "SYN-001" not in r.stdout

    def test_invalid_group_exits_nonzero(self):
        r = self._run_cli("--group", "nonexistent_group_xyz")
        assert r.returncode != 0

    def test_help_flag(self):
        r = subprocess.run(
            [sys.executable, "prompt_eval_harness.py", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent,
        )
        assert r.returncode == 0
        assert "usage" in r.stdout.lower()

    def test_output_contains_summary_header(self):
        r = self._run_cli()
        assert "RESULTS" in r.stdout

    def test_invariance_section_shown(self):
        r = self._run_cli("--group", "invariance")
        assert "INVARIANCE PAIRS" in r.stdout

    def test_perturbation_section_shown(self):
        r = self._run_cli("--group", "perturbation")
        assert "PERTURBATION PAIRS" in r.stdout


# ══════════════════════════════════════════════════════════════════════════════
# 7. SCORER REGISTRY COMPLETENESS
# ══════════════════════════════════════════════════════════════════════════════

class TestScorerRegistry:
    def test_all_case_scorers_registered(self):
        """Every scorer referenced in test_cases.json must exist in SCORERS."""
        cases = load_cases()
        used = {c["scoring"] for c in cases}
        missing = used - set(SCORERS.keys())
        assert missing == set(), f"Unregistered scorers in test_cases.json: {missing}"

    def test_scorers_are_callable(self):
        for name, fn in SCORERS.items():
            assert callable(fn), f"Scorer '{name}' is not callable"


# ══════════════════════════════════════════════════════════════════════════════
# 8. EDGE CASES & REGRESSION GUARDS
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Regression guards for known real-world flakiness patterns."""

    def test_rw002_float_rounding_gotcha(self):
        """round(2.675, 2) → 2.67 due to IEEE 754, NOT 2.68."""
        case = {"pattern": r"2\.67"}
        # Correct answer
        assert score_regex("2.67", case)["pass"] is True
        # Wrong answer a model might give
        assert score_regex("2.68", case)["pass"] is False

    def test_rw003_json_with_preamble_fails(self):
        """Model adds prose before JSON — score_json_set should fail."""
        case = {"expected_set": ["red", "green", "blue"]}
        bad_response = 'The three primary colors are: ["red", "green", "blue"]'
        r = score_json_set(bad_response, case)
        # Leading text makes JSON.parse fail → score False (strict format check)
        assert r["pass"] is False

    def test_invariance_typo_prompt_still_scores(self):
        """Typo-laden prompt maps to same expected answer."""
        case = {"expected": "Paris", "scoring": "exact_ci"}
        # Model should see past typos and say Paris
        r = score_exact_ci("Paris", case)
        assert r["pass"] is True

    def test_perturbation_position_c_anchor(self):
        """Simulate a model that always picks C regardless of content."""
        # PERT-001b has the correct answer at position A, not C
        case = {"expected": "A", "scoring": "exact_ci"}
        biased_response = "C"  # position-anchored model
        r = score_exact_ci(biased_response, case)
        assert r["pass"] is False, "Position-biased model should fail shuffled MCQ"

    def test_empty_expected_set(self):
        """Edge: empty expected_set → vacuously passes."""
        r = score_set_match("anything", {"expected_set": []})
        assert r["pass"] is True

    def test_unicode_in_response(self):
        """Responses with unicode shouldn't crash scorers."""
        r = score_exact_ci("Páris", {"expected": "Paris"})
        # Won't match — different chars — but should not raise
        assert isinstance(r["pass"], bool)
