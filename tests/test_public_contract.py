import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csv_analytics_pipeline.ingestion import CSVIngestionError, ingest_csv
from csv_analytics_pipeline.models import FilterSpec, PipelineConfig
from csv_analytics_pipeline.pipeline import run_pipeline
from csv_analytics_pipeline.utils import apply_filters, parse_number


class PublicContractTests(unittest.TestCase):
    def test_duplicate_and_blank_headers_fail_before_data_loss(self):
        for raw in (b"amount,amount\n10,900\n", b"name, \nA,1\n", b" name,name \nA,B\n"):
            with self.subTest(raw=raw), self.assertRaises(CSVIngestionError):
                ingest_csv(raw, PipelineConfig())

    def test_identifiers_and_nonfinite_values_are_not_numbers(self):
        for value in ("ABC123", "order-42", "12 kg", "1e999", float("inf"), float("nan"), "1,2"):
            with self.subTest(value=value):
                self.assertIsNone(parse_number(value))
        for value, expected in (("$1,234.50", 1234.5), ("(24.5)", -24.5), ("25%", .25), ("1.2e3", 1200)):
            self.assertEqual(parse_number(value), expected)

    def test_stream_read_is_bounded(self):
        class GuardedStream(io.BytesIO):
            def read(self, size=-1):
                self.requested_size = size
                if size < 0:
                    raise AssertionError("Unbounded read")
                return super().read(size)
        stream = GuardedStream(b"name\n" + b"x" * 1000)
        with self.assertRaises(CSVIngestionError):
            ingest_csv(stream, PipelineConfig(max_file_size_bytes=10))
        self.assertEqual(stream.requested_size, 11)

    def test_typed_equality_and_not_equals(self):
        rows = [{"amount": 10.0, "enabled": True}, {"amount": 20.0, "enabled": False}]
        self.assertEqual(apply_filters(rows, [FilterSpec("amount", value="10")]), rows[:1])
        self.assertEqual(apply_filters(rows, [FilterSpec("enabled", value="yes")]), rows[:1])
        self.assertEqual(apply_filters(rows, [FilterSpec("amount", "not_equals", "10")]), rows[1:])
        with self.assertRaises(ValueError):
            apply_filters(rows, [FilterSpec("amount", "typo", "10")])

    def test_saved_success_matches_result_and_zero_charts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_pipeline(b"region,amount\nnorth,10\nsouth,20\nnorth,30\n", PipelineConfig(output_directory=out, chart_limit=0))
            self.assertTrue(result.success, result.errors)
            self.assertTrue(json.loads((out / "pipeline_summary.json").read_text())["success"])
            self.assertEqual(len(result.cleaned_rows), 3)
            self.assertEqual(list((out / "charts").glob("*.svg")), [])

    def test_unknown_column_fails_and_persists_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_pipeline(b"amount\n10\n20\n30\n", PipelineConfig(output_directory=out, slice_filters=[FilterSpec("typo", value="10")]))
            self.assertFalse(result.success)
            self.assertIn("Unknown filter column", result.errors[0])
            self.assertFalse(json.loads((out / "pipeline_summary.json").read_text())["success"])

    def test_cli_rejects_negative_limits(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / "run_pipeline.py"), "missing.csv", "--max-rows", "-1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
