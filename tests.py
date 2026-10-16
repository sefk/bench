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
lmsmon = load_script("lmsmon")


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


    def test_code_is_scored_but_kept_out_of_the_composite(self):
        """Folding code into the composite would move every composite score
        recorded before the code task existed."""
        rows = [self._rows(task="gsm8k", correct=True) for _ in range(4)]
        rows += [self._rows(task="code", correct=False) for _ in range(6)]
        summary = quality.summarise(rows)[0]
        self.assertEqual(summary["composite"]["n"], 4)
        self.assertEqual(summary["composite"]["pct"], 100.0)
        self.assertEqual(summary["code"]["n"], 6)
        self.assertEqual(summary["code"]["pct"], 0.0)

    def test_code_only_run_has_no_composite(self):
        summary = quality.summarise([self._rows(task="code")])[0]
        self.assertNotIn("composite", summary)
        self.assertEqual(summary["code"]["pct"], 100.0)


class CodeGradingTests(unittest.TestCase):
    """The pieces of the code grader that do not need a subprocess."""

    def test_extracts_the_last_fenced_block(self):
        text = "Sample:\n```\n3 4\n```\nSolution:\n```python\nprint(7)\n```\n"
        self.assertEqual(quality.extract_code(text), "print(7)\n")

    def test_extracts_an_unclosed_block_cut_off_by_the_budget(self):
        self.assertEqual(quality.extract_code("```python\nx = 1\ny ="), "x = 1\ny =")

    def test_no_fence_means_no_code(self):
        self.assertEqual(quality.extract_code("print(7)"), "")

    def test_outputs_ignore_whitespace_differences(self):
        self.assertTrue(quality.outputs_match("1 2\n3\n\n", "1  2\r\n3"))

    def test_outputs_compare_numbers_as_numbers(self):
        self.assertTrue(quality.outputs_match("0.3333333", "0.33333333333"))
        self.assertTrue(quality.outputs_match("2.50", "2.5"))
        self.assertFalse(quality.outputs_match("0.34", "0.33"))

    def test_outputs_reject_missing_or_extra_lines(self):
        self.assertFalse(quality.outputs_match("1\n2", "1"))
        self.assertFalse(quality.outputs_match("Yes", "No"))

    def test_outputs_honour_a_tighter_problem_tolerance(self):
        """abc392_d wants 1e-8; grading it at the 1e-6 default would pass an
        answer the judge rejects."""
        self.assertTrue(quality.outputs_match("1.0000001", "1.0"))
        self.assertFalse(quality.outputs_match("1.0000001", "1.0", 1e-8))

    def test_functional_results_compare_within_tolerance(self):
        """A binary search that should return 1.0 lands on 0.9999999999999999,
        and the judge accepts it; strict equality does not."""
        self.assertTrue(quality.values_match(0.9999999999999999, 1.0, 1e-5))
        self.assertTrue(quality.values_match([1.0, 2.5], (1.0000000001, 2.5), 1e-6))
        self.assertFalse(quality.values_match(1.01, 1.0, 1e-5))

    def test_booleans_are_never_compared_by_tolerance(self):
        """True == 1 in Python and LiveCodeBench accepts that, so it is kept.
        What a boolean must not get is float tolerance: near-1 is not True."""
        self.assertTrue(quality.values_match(True, 1, 1e-6))
        self.assertFalse(quality.values_match(True, 1.000001, 1e-5))
        self.assertFalse(quality.values_match([1, 2], [1, 2, 3], 1e-6))
        self.assertTrue(quality.values_match({"a": 1.0}, {"a": 1.0}, 1e-6))

    def test_nan_is_never_a_match(self):
        self.assertFalse(quality.outputs_match("nan", "nan0"))
        self.assertFalse(quality.outputs_match("nan", "1.0"))


class CodeToleranceParsingTests(unittest.TestCase):
    """The item builder reads each problem's stated precision."""

    def setUp(self):
        self.mk = load_script("quality/make-code-items.py")

    def test_atcoder_wording(self):
        q = ("Print the answer. Your answer is considered correct if the "
             "absolute or relative error from the true solution does not "
             "exceed 10^{-8}. Constraints - 2 \\leq N \\leq 100")
        self.assertEqual(self.mk.stated_tolerance(q), 1e-8)

    def test_leetcode_wording(self):
        q = "Answers within 10^-5 of the actual answer will be accepted."
        self.assertEqual(self.mk.stated_tolerance(q), 1e-5)

    def test_constraints_are_not_mistaken_for_precision(self):
        q = "1 <= squares.length <= 5 * 10^4\n0 <= x_i, y_i <= 10^9"
        self.assertIsNone(self.mk.stated_tolerance(q))

    def test_exact_problems_have_no_tolerance(self):
        self.assertIsNone(self.mk.stated_tolerance("Print the number of ways."))


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

    def test_score_records_the_budget_it_reflects(self):
        """After a 16k resume, finished 4k rows sit beside retried 16k ones;
        the score reflects the larger budget."""
        rows = [
            {"model": "m", "backend": "lmstudio", "mode": "no-think", "task": "code",
             "correct": True, "latency_s": 1.0, "max_tokens": budget}
            for budget in (4096, 16384)
        ]
        self.assertEqual(quality.summarise(rows)[0]["code"]["max_tokens"], 16384)

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


class ReasoningModeTests(unittest.TestCase):
    def test_qwen_defaults_to_no_think(self):
        mode, budgets = quality.resolve_mode("qwen/qwen3.6-27b@4bit")
        self.assertEqual(mode, "no-think")
        self.assertEqual(budgets["mmlu"], 8)

    def test_a_model_that_always_reasons_runs_at_low_effort(self):
        mode, budgets = quality.resolve_mode("openai/gpt-oss-20b@mxfp4")
        self.assertEqual(mode, "effort-low")
        # It reasons before the letter, so MMLU's 8 tokens would truncate all.
        self.assertGreater(budgets["mmlu"], 8)
        self.assertEqual(budgets["code"], quality.MAX_TOKENS["code"])

    def test_explicit_effort_and_think(self):
        self.assertEqual(
            quality.resolve_mode("openai/gpt-oss-20b@mxfp4", effort="high")[0],
            "effort-high",
        )
        self.assertEqual(
            quality.resolve_mode("openai/gpt-oss-20b@mxfp4", think=True)[0], "think"
        )

    def _payload_for(self, mode):
        sent = {}

        def fake_http(url, payload, timeout=3600):
            sent.update(payload)
            return json.dumps({
                "model": payload["model"],
                "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
                "usage": {},
            })

        real = quality.http_json
        quality.http_json = fake_http
        try:
            quality.ask_lmstudio("m", "q", 64, mode)
        finally:
            quality.http_json = real
        return sent

    def test_effort_mode_sends_reasoning_effort_and_no_prefill(self):
        sent = self._payload_for("effort-low")
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertEqual([m["role"] for m in sent["messages"]], ["user"])

    def test_no_think_sends_the_prefill_only(self):
        sent = self._payload_for("no-think")
        self.assertNotIn("reasoning_effort", sent)
        self.assertEqual(sent["messages"][-1]["content"], quality.NO_THINK_PREFILL)


class PlotTests(unittest.TestCase):
    def test_arch_detection(self):
        self.assertEqual(plot.arch_of("qwen/qwen3.6-35b-a3b@4bit"), "moe")
        self.assertEqual(plot.arch_of("qwen/qwen3.6-27b@8bit"), "dense")
        self.assertEqual(plot.arch_of("fm:system"), "apple")
        # MoE without the suffix, known from model-facts.json.
        self.assertEqual(plot.arch_of("openai/gpt-oss-20b@mxfp4"), "moe")

    def test_loads_across_date_directories(self):
        """A model benchmarked on a later date joins the chart beside the
        earlier ones, and a re-run supersedes its older numbers."""
        def entry(model, mode, pct):
            return {"model": model, "target": model, "mode": mode,
                    "composite": {"pct": pct, "ci95": [pct - 3, pct + 3]}}

        def speed(model, tps):
            return [{"model": model, "size": "4000tok", "gen_tps": tps}]

        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp, "2026-09-14"), Path(tmp, "2026-10-14")
            old.mkdir(); new.mkdir()
            (old / "quality.json").write_text(json.dumps([entry("q/a", "no-think", 88.0)]))
            (old / "a.json").write_text(json.dumps(speed("q/a", 60.0)))
            (new / "quality.json").write_text(json.dumps([entry("o/g", "effort-low", 87.0)]))
            (new / "g.json").write_text(json.dumps(speed("o/g", 78.0)))
            (new / "a.json").write_text(json.dumps(speed("q/a", 65.0)))

            points = {p["model"]: p for p in plot.load([new, old], "4000tok")}
            self.assertEqual(set(points), {"q/a", "o/g"})
            self.assertEqual(points["q/a"]["tps"], 65.0, "later date wins")
            self.assertTrue(points["o/g"]["label"].endswith("(low reasoning)"))

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


class LmsmonTests(unittest.TestCase):
    """Parsing in the live monitor, against event shapes captured from
    LM Studio 0.4's `lms ps --json` and `lms log stream --json --stats`."""

    INPUT = json.dumps({"timestamp": 1792113556449, "data": {
        "type": "llm.prediction.input", "input": "<|start|>user...",
        "modelPath": "openai/gpt-oss-20b", "modelIdentifier": "openai/gpt-oss-20b"}})
    OUTPUT = json.dumps({"timestamp": 1792113562471, "data": {
        "type": "llm.prediction.output", "output": "Hello, how are you?",
        "stats": {"stopReason": "eosFound", "tokensPerSecond": 40.95,
                  "timeToFirstTokenSec": 5.46, "totalTimeSec": 0.561,
                  "promptTokensCount": 73, "predictedTokensCount": 23},
        "modelIdentifier": "openai/gpt-oss-20b"}})

    def test_parse_ps(self):
        ps = json.dumps([{
            "identifier": "openai/gpt-oss-20b", "quantization": {"name": "MXFP4"},
            "sizeBytes": 12104314420, "contextLength": 131072, "status": "generating",
            "queued": 1, "parallel": 4, "ttlMs": 3600000, "lastUsedTime": 1_000_000,
        }, {
            "identifier": "qwen/qwen3.6-35b-a3b", "status": "idle", "queued": 0,
            "parallel": 4, "ttlMs": None, "lastUsedTime": 1_000_000,
        }])
        a, b = lmsmon.parse_ps(ps, now_ms=1_000_000 + 600_000)
        self.assertEqual(a["quant"], "MXFP4")
        self.assertEqual(a["ttl_s"], 3000.0)
        self.assertIsNone(b["ttl_s"], "no TTL means never unloads")
        self.assertEqual(lmsmon.parse_ps("", 0), [], "nothing loaded")
        self.assertEqual(lmsmon.fmt_ttl(a), "50m00s")
        self.assertEqual(lmsmon.fmt_ttl(b), "never")

    def test_ttl_while_busy(self):
        """Mid-request LM Studio drops lastUsedTime; that's busy, not 'never'."""
        ps = json.dumps([{"identifier": "m", "ttlMs": 3600000, "lastUsedTime": None}])
        (row,) = lmsmon.parse_ps(ps, now_ms=0)
        self.assertEqual(lmsmon.fmt_ttl(row), "in use")

    def test_ttl_never_negative(self):
        m = {"ttlMs": 1000, "lastUsedTime": 0}
        self.assertEqual(lmsmon.ttl_remaining(m, now_ms=5000), 0.0)

    def test_parse_events(self):
        self.assertIsNone(lmsmon.parse_event("Streaming logs from LM Studio"))
        self.assertIsNone(lmsmon.parse_event('{"data": {"type": "llm.other"}}'))
        start = lmsmon.parse_event(self.INPUT)
        self.assertEqual((start["kind"], start["model"]), ("input", "openai/gpt-oss-20b"))
        self.assertAlmostEqual(start["ts"], 1792113556.449)
        done = lmsmon.parse_event(self.OUTPUT)
        self.assertEqual(done["kind"], "output")
        self.assertEqual((done["prompt_tokens"], done["completion_tokens"]), (73, 23))
        self.assertEqual(done["stop"], "eosFound")

    def test_requests_pair_input_with_output(self):
        reqs = lmsmon.Requests()
        rid = reqs.start(lmsmon.parse_event(self.INPUT))
        self.assertEqual(reqs.in_flight(), 1)
        self.assertEqual(reqs.finish(lmsmon.parse_event(self.OUTPUT)), (rid, True))
        self.assertEqual(reqs.in_flight(), 0)
        self.assertEqual((reqs.count, reqs.prompt_tokens), (1, 73))

    def test_output_without_input_gets_own_row(self):
        """Started mid-request: the output still counts, under a fresh id."""
        reqs = lmsmon.Requests()
        rid, existed = reqs.finish(lmsmon.parse_event(self.OUTPUT))
        self.assertFalse(existed)
        self.assertEqual(reqs.start(lmsmon.parse_event(self.INPUT)), rid + 1)

    def test_sparkline(self):
        self.assertEqual(lmsmon.sparkline([]), "")
        self.assertEqual(lmsmon.sparkline([0, 50, 100]), "▁▅█")
        self.assertEqual(len(lmsmon.sparkline(range(1, 100), width=10)), 10)

    def test_accent_cycles_and_persists(self):
        names = list(lmsmon.ACCENTS)
        self.assertEqual(lmsmon.next_accent(names[0]), names[1])
        self.assertEqual(lmsmon.next_accent(names[-1]), names[0], "wraps")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "sub", "config.json")
            self.assertEqual(lmsmon.load_accent(path), names[0], "default")
            lmsmon.save_accent("amber", path)
            self.assertEqual(lmsmon.load_accent(path), "amber")
            path.write_text('{"accent": "plaid"}')
            self.assertEqual(lmsmon.load_accent(path), names[0], "unknown -> default")

    def test_formatting(self):
        self.assertEqual(lmsmon.fmt_bytes(12104314420), "11.3 GB")
        self.assertEqual(lmsmon.fmt_duration(59), "59s")
        self.assertEqual(lmsmon.fmt_duration(3000), "50m00s")
        self.assertEqual(lmsmon.fmt_duration(3700), "1h01m")


if __name__ == "__main__":
    unittest.main(verbosity=2)
