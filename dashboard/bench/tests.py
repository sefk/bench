import json
import tempfile
from pathlib import Path

from django.test import SimpleTestCase, override_settings

from .loader import build_meta, derive_dimensions, load_quality, load_rows


class DeriveDimensionsTests(SimpleTestCase):
    def test_moe_4bit(self):
        d = derive_dimensions("qwen/qwen3.6-35b-a3b@4bit")
        self.assertEqual(d["family"], "qwen")
        self.assertEqual(d["base"], "qwen3.6-35b-a3b")
        self.assertEqual(d["version"], "3.6")
        self.assertEqual(d["arch"], "moe")
        self.assertEqual(d["params"], "35b")
        self.assertEqual(d["active"], "3b")
        self.assertEqual(d["quant"], "4bit")
        self.assertEqual(d["runtime"], "mlx")
        self.assertEqual(d["variant"], "qwen3.6-35b-a3b@4bit")

    def test_dense_8bit(self):
        d = derive_dimensions("qwen/qwen3.6-27b@8bit")
        self.assertEqual(d["family"], "qwen")
        self.assertEqual(d["base"], "qwen3.6-27b")
        self.assertEqual(d["version"], "3.6")
        self.assertEqual(d["arch"], "dense")
        self.assertEqual(d["params"], "27b")
        self.assertEqual(d["active"], "27b")
        self.assertEqual(d["quant"], "8bit")
        self.assertEqual(d["runtime"], "mlx")

    def test_dense_gguf(self):
        d = derive_dimensions("qwen/qwen3.8-27b@q4_k_m")
        self.assertEqual(d["version"], "3.8")
        self.assertEqual(d["arch"], "dense")
        self.assertEqual(d["params"], "27b")
        self.assertEqual(d["quant"], "q4_k_m")
        self.assertEqual(d["runtime"], "gguf")

    def test_precision_spans_runtimes(self):
        """4bit (MLX) and q4_k_m (GGUF) are the same precision, not the same
        build. The dashboard must be able to say both things."""
        mlx = derive_dimensions("qwen/qwen3.8-27b@4bit")
        gguf = derive_dimensions("qwen/qwen3.8-27b@q4_k_m")
        self.assertEqual(mlx["precision"], gguf["precision"], "both are 4-bit")
        self.assertNotEqual(mlx["runtime"], gguf["runtime"])
        self.assertNotEqual(mlx["variant"], gguf["variant"])
        self.assertEqual(
            derive_dimensions("qwen/qwen3.6-27b@8bit")["precision"], "8-bit"
        )

    def test_apple_foundation_model(self):
        d = derive_dimensions("fm:system")
        self.assertEqual(d["family"], "apple")
        self.assertEqual(d["runtime"], "afm")
        self.assertEqual(d["variant"], "apple-foundation-model")
        # Apple publishes no parameter count or weight format; the loader must
        # not invent one.
        self.assertEqual(d["params"], "undisclosed")
        self.assertEqual(d["precision"], "undisclosed")

    def test_no_quant(self):
        d = derive_dimensions("foo/bar")
        self.assertEqual(d["family"], "foo")
        self.assertEqual(d["base"], "bar")
        self.assertEqual(d["version"], "unknown")
        self.assertEqual(d["arch"], "dense")
        self.assertEqual(d["params"], "unknown")
        self.assertEqual(d["active"], "unknown")
        self.assertEqual(d["quant"], "unknown")
        self.assertEqual(d["runtime"], "unknown")
        self.assertEqual(d["precision"], "unknown")
        self.assertEqual(d["variant"], "bar")


class LoaderTests(SimpleTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)
        date_dir = self.results_dir / "2026-08-03"
        date_dir.mkdir(parents=True)
        rows = [
            {
                "model": "qwen/qwen3.6-27b@4bit",
                "size": "1000tok",
                "sized": True,
                "prompt_tokens": 837,
                "completion_tokens": 255,
                "gen_tps": 18.4,
                "ttft": 9.79,
                "prefill_tps": 87.1,
                "runs": 3,
                "ttft_spread": 0.05,
                "watts": 43.9,
                "energy_wh": 0.5,
                "incremental_wh": 0.4,
                "tokens_per_wh": 1000.0,
            }
        ]
        (date_dir / "qwen3.6-27b-4bit.json").write_text(json.dumps(rows))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_rows(self):
        rows = load_rows(self.results_dir, force=True)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["date"], "2026-08-03")
        self.assertEqual(row["variant"], "qwen3.6-27b@4bit")
        self.assertEqual(row["arch"], "dense")

    def test_reload_on_new_file(self):
        rows = load_rows(self.results_dir, force=True)
        self.assertEqual(len(rows), 1)

        date_dir2 = self.results_dir / "2026-08-14"
        date_dir2.mkdir()
        (date_dir2 / "other.json").write_text(
            json.dumps(
                [
                    {
                        "model": "qwen/qwen3.8-27b@q4_k_m",
                        "size": "short",
                        "sized": False,
                        "prompt_tokens": 72,
                        "completion_tokens": 98,
                        "gen_tps": 11.5,
                        "ttft": 1.4,
                        "prefill_tps": 49.7,
                        "runs": 3,
                        "ttft_spread": 0.12,
                        "watts": None,
                        "energy_wh": None,
                        "incremental_wh": None,
                        "tokens_per_wh": None,
                    }
                ]
            )
        )
        # No force: the mtime-check cache invalidation should pick this up.
        rows = load_rows(self.results_dir)
        self.assertEqual(len(rows), 2)
        dates = {row["date"] for row in rows}
        self.assertEqual(dates, {"2026-08-03", "2026-08-14"})

    def test_build_meta(self):
        rows = load_rows(self.results_dir, force=True)
        meta = build_meta(rows)
        self.assertIn("version", meta)
        self.assertIn("3.6", meta["version"])
        self.assertTrue(any(m["key"] == "gen_tps" for m in meta["measures"]))


class QualityJoinTests(SimpleTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)
        date_dir = self.results_dir / "2026-09-14"
        date_dir.mkdir(parents=True)
        (date_dir / "qwen3.6-27b-4bit.json").write_text(
            json.dumps(
                [
                    {
                        "model": "qwen/qwen3.6-27b@4bit",
                        "size": "1000tok",
                        "sized": True,
                        "gen_tps": 18.4,
                        "ttft": 9.79,
                        "total_s": 23.5,
                        "prefill_tps": 87.1,
                        "runs": 3,
                        "ttft_spread": 0.05,
                    }
                ]
            )
        )
        (date_dir / "fm-system.json").write_text(
            json.dumps(
                [
                    {
                        "model": "fm:system",
                        "size": "1000tok",
                        "sized": True,
                        "gen_tps": 30.0,
                        "ttft": 0.9,
                        "total_s": 9.4,
                        "prefill_tps": 900.0,
                        "runs": 3,
                        "ttft_spread": 0.04,
                    }
                ]
            )
        )
        (date_dir / "quality.json").write_text(
            json.dumps(
                [
                    {
                        "model": "qwen/qwen3.6-27b@4bit",
                        "backend": "lmstudio",
                        "target": "qwen/qwen3.6-27b@4bit",
                        "mode": "no-think",
                        "gsm8k": {"pct": 80.0},
                        "mmlu": {"pct": 70.0},
                        "composite": {"pct": 73.3, "ci95": [70.0, 76.4], "n": 750},
                    },
                    {
                        "model": "system",
                        "backend": "fm",
                        "target": "fm:system",
                        "mode": "no-think",
                        "gsm8k": {"pct": 60.0},
                        "mmlu": {"pct": 50.0},
                        "composite": {"pct": 53.3, "ci95": [49.8, 56.9], "n": 750},
                    },
                    {
                        # A reasoning-mode run of the same model must not
                        # overwrite the comparable no-think score.
                        "model": "qwen/qwen3.6-27b@4bit",
                        "backend": "lmstudio",
                        "target": "qwen/qwen3.6-27b@4bit",
                        "mode": "think",
                        "composite": {"pct": 99.0, "ci95": [98.0, 99.5], "n": 750},
                    },
                ]
            )
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_quality_summary_is_not_loaded_as_a_row(self):
        rows = load_rows(self.results_dir, force=True)
        self.assertEqual(len(rows), 2, "quality.json must not become rows")

    def test_quality_joins_onto_throughput_rows(self):
        rows = {r["model"]: r for r in load_rows(self.results_dir, force=True)}
        self.assertAlmostEqual(rows["qwen/qwen3.6-27b@4bit"]["quality_pct"], 73.3)
        self.assertAlmostEqual(rows["qwen/qwen3.6-27b@4bit"]["quality_gsm8k"], 80.0)
        self.assertAlmostEqual(rows["fm:system"]["quality_pct"], 53.3)
        self.assertAlmostEqual(rows["fm:system"]["quality_mmlu"], 50.0)

    def test_think_mode_scores_are_excluded(self):
        scores = load_quality(self.results_dir)
        self.assertAlmostEqual(scores["qwen/qwen3.6-27b@4bit"]["quality_pct"], 73.3)

    def test_code_scores_merge_without_blanking_the_composite(self):
        """quality-code.json carries only the code task; joining it must add
        quality_code, not wipe the GSM8K/MMLU scores from quality.json."""
        (self.results_dir / "2026-09-14" / "quality-code.json").write_text(
            json.dumps(
                [
                    {
                        "model": "qwen/qwen3.6-27b@4bit",
                        "backend": "lmstudio",
                        "target": "qwen/qwen3.6-27b@4bit",
                        "mode": "no-think",
                        "code": {"pct": 40.0, "ci95": [33.0, 47.4], "n": 175},
                    }
                ]
            )
        )
        scores = load_quality(self.results_dir)["qwen/qwen3.6-27b@4bit"]
        self.assertAlmostEqual(scores["quality_code"], 40.0)
        self.assertAlmostEqual(scores["quality_code_ci_low"], 33.0)
        self.assertAlmostEqual(scores["quality_pct"], 73.3)
        self.assertAlmostEqual(scores["quality_gsm8k"], 80.0)

    def test_scatter_measures_available(self):
        meta = build_meta(load_rows(self.results_dir, force=True))
        keys = {m["key"] for m in meta["measures"]}
        self.assertIn("quality_pct", keys)
        self.assertIn("total_s", keys)


@override_settings()
class ViewTests(SimpleTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)
        date_dir = self.results_dir / "2026-08-03"
        date_dir.mkdir(parents=True)
        (date_dir / "sample.json").write_text(
            json.dumps(
                [
                    {
                        "model": "qwen/qwen3.6-27b@4bit",
                        "size": "1000tok",
                        "sized": True,
                        "prompt_tokens": 837,
                        "completion_tokens": 255,
                        "gen_tps": 18.4,
                        "ttft": 9.79,
                        "prefill_tps": 87.1,
                        "runs": 3,
                        "ttft_spread": 0.05,
                        "watts": None,
                        "energy_wh": None,
                        "incremental_wh": None,
                        "tokens_per_wh": None,
                    }
                ]
            )
        )
        self.override = override_settings(RESULTS_DIR=self.results_dir)
        self.override.enable()

    def tearDown(self):
        self.override.disable()
        self.tmpdir.cleanup()

    def test_index_ok(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_api_rows_ok(self):
        response = self.client.get("/api/rows/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["rows"]), 1)

    def test_api_meta_ok(self):
        response = self.client.get("/api/meta/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("version", data)
        self.assertIn("measures", data)
