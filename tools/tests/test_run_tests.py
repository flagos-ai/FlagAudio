"""CPU-free tests for the scheduler and its report formats."""

import csv
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import psum_html, psum_text, run_tests


class SchedulerTests(unittest.TestCase):
    def test_catalog_matches_operator_sources(self):
        catalog = run_tests.load_catalog(run_tests.ROOT / "conf/operators.yaml")
        ids = {op["id"] for op in catalog}
        source_ids = {path.stem for path in (run_tests.ROOT / "src/flag_audio/ops").glob("*.py")}
        source_ids.discard("__init__")
        self.assertEqual(ids, source_ids)
        for directory in ("tests", "benchmark"):
            marks = set()
            for source in (run_tests.ROOT / directory).glob("test_*.py"):
                marks.update(re.findall(
                    r"(?:@pytest\.mark\.|pytestmark\s*=\s*pytest\.mark\.)([A-Za-z_][A-Za-z_0-9]*)",
                    source.read_text(encoding="utf-8"),
                ))
            self.assertFalse(ids - marks, f"{directory} lacks markers for {ids - marks}")

    def test_scheduler_and_reports(self):
        commands = []
        environments = []

        def fake_run(command, **kwargs):
            commands.append(command)
            environments.append(kwargs["env"])
            output = Path(command[command.index("--output") + 1])
            if "tests/" in command:
                output.write_text(json.dumps({"test_gain": {"result": "passed"}}), encoding="utf-8")
            else:
                output.write_text(json.dumps({"gain": {"details": [{
                    "dtype": "torch.float32", "result": [{"shape_detail": [[1024]],
                    "latency_base": 2.0, "latency": 1.0, "speedup": 2.0}],
                }]}}), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"BLAS_VENDOR": "nvidia"}
        ), patch.object(
            run_tests.subprocess, "run", side_effect=fake_run
        ):
            output = Path(directory)
            self.assertEqual(run_tests.main(["--ops", "gain", "--output-dir", directory]), 0)
            summary = output / "summary.json"
            result = json.loads(summary.read_text(encoding="utf-8"))["operators"][0]
            self.assertEqual(result["accuracy"]["counts"]["passed"], 1)
            self.assertEqual(result["performance"]["measurements"][0]["speedup"], 2.0)
            self.assertEqual(len(commands), 2)
            self.assertIn("tests/", commands[0])
            self.assertIn("benchmark/", commands[1])
            self.assertNotIn("--ref", commands[0])

            psum_text.main([str(summary)])
            psum_html.main([str(summary)])
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["operator"], "gain")
            self.assertEqual(rows[0]["speedup"], "2.0")
            self.assertIn("gain", (output / "summary.html").read_text(encoding="utf-8"))

            with patch.dict(os.environ, {"BLAS_VENDOR": "ascend"}):
                self.assertEqual(run_tests.main([
                    "--ops", "gain", "--output-dir", directory,
                    "--device-env", "ASCEND_RT_VISIBLE_DEVICES", "--accuracy-only",
                ]), 0)
            self.assertEqual(environments[-1]["ASCEND_RT_VISIBLE_DEVICES"], "0")
            self.assertNotIn("--ref", commands[-1])

            with patch.dict(os.environ, {"BLAS_VENDOR": "iluvatar"}):
                self.assertEqual(run_tests.main([
                    "--ops", "gain", "--output-dir", directory, "--accuracy-only",
                ]), 0)
            self.assertEqual(commands[-1][-2:], ["--ref", "cpu"])

    def test_missing_pytest_reports_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            self.assertEqual(run_tests.accuracy_result(missing, 0)["status"], "FAIL")
            self.assertEqual(run_tests.performance_result(missing, 0, "gain")["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
