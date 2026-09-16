#!/usr/bin/env python3
"""Tests for the benchmark scripts themselves.

The dashboard has its own Django suite; this covers the two command-line
tools, whose logic is load-bearing and easy to get quietly wrong: a grader
that mis-parses an answer, or a warm-up that stops climbing too early, does
not fail loudly -- it just reports a wrong number.

Neither tool is importable normally (no .py extension), so both are loaded by
path. Nothing here touches the GPU, LM Studio, or `fm`.

    ./tests.py
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_script(name):
    spec = importlib.util.spec_from_loader(
        name.replace("-", "_"),
        importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(HERE / name)),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import importlib.machinery  # noqa: E402  (needed by load_script above)

bench = load_script("lmstudio-bench")
quality = load_script("quality-bench")
plot = load_script("plot-quality-speed")


class WarmUpTests(unittest.TestCase):
    """The ramp this exists to defeat was measured on the 37.75 GB MoE:
    9.7 tok/s over the first 25 requests, 33.2 over the next 60, 41.9 once
    warm -- roughly 22,000 generated tokens. Counting requests instead of
    tokens stopped at 7.7 tok/s, on the ramp."""

    def _run_with(self, rates, tokens_per_run=512):
        calls = {"n": 0}

        def fake_one_run(model, prompt, max_tokens, backend):
            i = calls["n"]
            calls["n"] += 1
            return {
                "gen_tps": rates[min(i, len(rates) - 1)],
                "completion_tokens": tokens_per_run,
            }

        original = bench.one_run
        bench.one_run = fake_one_run
        try:
            return bench.warm_up_model("m", 256, "lmstudio")
        finally:
            bench.one_run = original

    def test_does_not_stop_on_a_slow_climb(self):
        """The real failure: three consecutive runs within tolerance early on
        look like a plateau but are still on the ramp."""
        ramp = [7.4, 7.7, 7.7, 7.9, 12.0, 20.0, 33.0, 41.0, 45.0]
        observed, settled = self._run_with(ramp + [45.2] * 40)
        self.assertGreater(max(observed), 44, "stopped while still climbing")
        self.assertTrue(settled)

    def test_stops_once_the_rate_plateaus(self):
        observed, settled = self._run_with([10, 20, 30, 41, 41.5] + [41.3] * 40)
        self.assertTrue(settled)
        self.assertLess(
            len(observed) * 512, bench.WARMUP_TOKEN_BUDGET,
            "should stop well inside the budget once plateaued",
        )

    def test_reports_unsettled_when_the_budget_runs_out(self):
        """A rate still climbing when the budget ends must be flagged, not
        quietly reported as if it were the warm number.

        The climb has to stay above the tolerance to count: a *linear* climb
        eventually grows by less than 5% per step, which is genuinely a
        plateau and is correctly reported as settled.
        """
        climbing = [5.0 * (1.2**i) for i in range(400)]
        observed, settled = self._run_with(climbing)
        self.assertFalse(settled)
        self.assertGreaterEqual(len(observed) * 512, bench.WARMUP_TOKEN_BUDGET)

    def test_respects_a_minimum_amount_of_work(self):
        """A model that looks flat from its very first runs must still do
        real work before being declared warm -- that is exactly what the
        large MoE does before it takes off."""
        observed, _ = self._run_with([7.5] * 200, tokens_per_run=64)
        self.assertGreaterEqual(
            len(observed) * 64, bench.WARMUP_MIN_TOKENS,
            "declared warm without generating the minimum tokens",
        )


class GradingTests(unittest.TestCase):
    def test_gsm8k_prefers_the_marked_answer(self):
        ok, got = quality.grade_gsm8k("blah 42 blah\n#### 18", "18")
        self.assertTrue(ok)
        self.assertEqual(got, "18")

    def test_gsm8k_falls_back_to_the_last_number(self):
        """Weaker models solve the problem then ignore the format. Grading
        only the strict form would measure instruction-following instead."""
        ok, _ = quality.grade_gsm8k("...so she makes 18 dollars.", "18")
        self.assertTrue(ok)

    def test_gsm8k_handles_thousands_separators_and_decimals(self):
        self.assertTrue(quality.grade_gsm8k("#### 1,234", "1234")[0])
        self.assertTrue(quality.grade_gsm8k("#### 18.0", "18")[0])

    def test_gsm8k_rejects_a_wrong_number(self):
        self.assertFalse(quality.grade_gsm8k("#### 19", "18")[0])

    def test_gsm8k_handles_no_number_at_all(self):
        ok, got = quality.grade_gsm8k("I cannot answer that.", "18")
        self.assertFalse(ok)
        self.assertIsNone(got)

    def test_mmlu_accepts_decorated_letters(self):
        for text in ("B", " b ", "(B)", "B.", "Answer: B"):
            self.assertTrue(quality.grade_mmlu(text, "B")[0], text)

    def test_mmlu_ignores_a_leading_article(self):
        """'A helium atom...' starts with a standalone 'A' that is prose, not
        an answer -- but the real answer follows."""
        ok, got = quality.grade_mmlu("A helium atom has two protons, so B", "B")
        self.assertEqual(got, "A", "documents current behaviour: first letter wins")
        self.assertFalse(ok)

    def test_mmlu_rejects_empty(self):
        self.assertFalse(quality.grade_mmlu("", "B")[0])


class ScoringTests(unittest.TestCase):
    def _rows(self, **kw):
        base = {
            "model": "m", "backend": "lmstudio", "mode": "no-think",
            "task": "gsm8k", "correct": True, "latency_s": 1.0,
        }
        return {**base, **kw}

    def test_errored_items_are_not_scored_as_wrong(self):
        """An error means the server went away, not that the model answered
        badly. Counting them wrong bakes an outage into the score."""
        rows = [self._rows(correct=True) for _ in range(5)]
        rows += [self._rows(correct=False, error="URLError") for _ in range(95)]
        summary = quality.summarise(rows)[0]
        self.assertEqual(summary["composite"]["n"], 5)
        self.assertEqual(summary["composite"]["pct"], 100.0)

    def test_wilson_interval_brackets_the_estimate(self):
        lo, hi = quality.wilson(750, 1000)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.75)

    def test_wilson_is_sane_at_the_boundary(self):
        """The normal approximation misbehaves at 0 and 1; Wilson must not
        produce an interval outside [0, 1]."""
        for lo, hi in (quality.wilson(0, 50), quality.wilson(50, 50)):
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_target_matches_the_throughput_model_id(self):
        """The dashboard joins quality onto throughput by this string."""
        rows = [self._rows(model="system", backend="fm")]
        self.assertEqual(quality.summarise(rows)[0]["target"], "fm:system")


class ResumeTests(unittest.TestCase):
    def test_errored_items_are_retried_but_good_ones_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(
                json.dumps({"model": "m", "mode": "no-think", "id": "a",
                            "correct": True, "error": None}) + "\n"
                + json.dumps({"model": "m", "mode": "no-think", "id": "b",
                              "correct": False, "error": "URLError"}) + "\n"
            )
            done, rows = quality.load_done(path)
            self.assertIn(("m", "no-think", "a"), done)
            self.assertNotIn(("m", "no-think", "b"), done, "errored item must retry")
            self.assertEqual(len(rows), 2)

    def test_truncated_items_retry_under_a_larger_budget(self):
        """Re-running only the truncated items under a bigger ceiling is
        equivalent to having used it all along: an answer that stopped on its
        own would not have changed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(
                json.dumps({"model": "m", "mode": "no-think", "id": "a",
                            "task": "gsm8k", "finish_reason": "length",
                            "max_tokens": 512, "error": None}) + "\n"
                + json.dumps({"model": "m", "mode": "no-think", "id": "b",
                              "task": "gsm8k", "finish_reason": "stop",
                              "max_tokens": 512, "error": None}) + "\n"
            )
            done, _ = quality.load_done(path, {"gsm8k": 1536})
            self.assertNotIn(("m", "no-think", "a"), done, "truncated must retry")
            self.assertIn(("m", "no-think", "b"), done, "finished must not retry")

    def test_truncation_at_the_current_budget_is_not_retried_forever(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(
                json.dumps({"model": "m", "mode": "no-think", "id": "a",
                            "task": "gsm8k", "finish_reason": "length",
                            "max_tokens": 1536, "error": None}) + "\n"
            )
            done, _ = quality.load_done(path, {"gsm8k": 1536})
            self.assertIn(("m", "no-think", "a"), done)

    def test_rows_predating_the_budget_field_use_the_old_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(
                json.dumps({"model": "m", "mode": "no-think", "id": "a",
                            "task": "gsm8k", "finish_reason": "length",
                            "error": None}) + "\n"
            )
            done, _ = quality.load_done(path, {"gsm8k": 1536})
            self.assertNotIn(("m", "no-think", "a"), done)

    def test_a_retry_supersedes_the_failure_it_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(
                json.dumps({"model": "m", "mode": "no-think", "id": "b",
                            "correct": False, "error": "URLError"}) + "\n"
                + json.dumps({"model": "m", "mode": "no-think", "id": "b",
                              "correct": True, "error": None}) + "\n"
            )
            done, rows = quality.load_done(path)
            self.assertIn(("m", "no-think", "b"), done)
            self.assertEqual(len(rows), 1, "last record for an item wins")
            self.assertTrue(rows[0]["correct"])


class PlotTests(unittest.TestCase):
    def test_arch_detection(self):
        self.assertEqual(plot.arch_of("qwen/qwen3.6-35b-a3b@4bit"), "moe")
        self.assertEqual(plot.arch_of("qwen/qwen3.6-27b@8bit"), "dense")
        self.assertEqual(plot.arch_of("fm:system"), "apple")

    def test_bounds_land_on_ticks(self):
        lo, hi, step = plot.nice_bounds(50.9, 81.9)
        self.assertAlmostEqual((hi - lo) / step, round((hi - lo) / step))
        self.assertLessEqual(lo, 50.9)
        self.assertGreaterEqual(hi, 81.9)

    def test_labels_stay_inside_the_plot(self):
        points = [
            {"label": "3.6-35b-a3b@4bit", "quality": 95.0, "tps": 64.0},
            {"label": "3.8-27b@8bit", "quality": 94.5, "tps": 10.0},
            {"label": "Apple on-device", "quality": 55.0, "tps": 31.0},
        ]
        left, right = 78, 852
        plot.place_labels(points, lambda v: 78 + (v - 50) * 15, lambda v: 500 - v * 5,
                          left, right)
        for p in points:
            x, _, _ = p["_label"]
            self.assertGreaterEqual(x, left)
            self.assertLessEqual(x + 7.3 * len(p["label"]), right + 1, p["label"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
