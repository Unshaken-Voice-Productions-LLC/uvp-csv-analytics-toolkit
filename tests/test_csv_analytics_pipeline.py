from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from csv_analytics_pipeline.analysis import analyze_dataset
from csv_analytics_pipeline.cleaning import clean_dataset
from csv_analytics_pipeline.ingestion import CSVIngestionError, ingest_csv
from csv_analytics_pipeline.models import ColumnType, IngestedDataset, PipelineConfig
from csv_analytics_pipeline.pipeline import run_pipeline
from csv_analytics_pipeline.validation import build_quality_report, detect_outliers, infer_schema


class CSVAnalyticsPipelineTests(unittest.TestCase):
    def test_schema_detection_infers_numeric_date_and_categorical(self) -> None:
        rows = [
            {"id": "1", "city": "Austin", "date": "2026-01-01", "amount": "10.5"},
            {"id": "2", "city": "Boston", "date": "2026-01-02", "amount": "20.0"},
            {"id": "3", "city": "Austin", "date": "2026-01-03", "amount": "30"},
        ]
        profiles = infer_schema(rows, ["id", "city", "date", "amount"], sample_limit=10)
        detected = {profile.name: profile.detected_type for profile in profiles}
        self.assertEqual(detected["id"], ColumnType.NUMERIC)
        self.assertEqual(detected["city"], ColumnType.CATEGORICAL)
        self.assertEqual(detected["date"], ColumnType.DATE)
        self.assertEqual(detected["amount"], ColumnType.NUMERIC)

    def test_missing_value_handling_uses_configurable_strategies(self) -> None:
        rows = [
            {"region": "North", "amount": "10"},
            {"region": "north", "amount": "20"},
            {"region": None, "amount": None},
            {"region": "South", "amount": "30"},
        ]
        profiles = infer_schema(rows, ["region", "amount"], sample_limit=10)
        config = PipelineConfig(
            missing_value_overrides={"region": "mode", "amount": "mean"},
            categorical_case="lower",
        )
        cleaned_rows, report = clean_dataset(rows, profiles, config)
        self.assertEqual(report.filled_cells, 2)
        self.assertEqual(cleaned_rows[2]["region"], "north")
        self.assertAlmostEqual(cleaned_rows[2]["amount"], 20.0)
        self.assertGreaterEqual(len(report.transformations), 2)

    def test_outlier_detection_flags_extreme_value(self) -> None:
        rows = [
            {"value": "10"},
            {"value": "11"},
            {"value": "12"},
            {"value": "10"},
            {"value": "100"},
        ]
        dataset = IngestedDataset(
            source_name="sample.csv",
            encoding="utf-8",
            dialect="excel",
            headers=["value"],
            rows=rows,
        )
        quality = build_quality_report(dataset, PipelineConfig())
        self.assertTrue(any(outlier.column == "value" and outlier.value == 100.0 for outlier in quality.outliers))

    def test_empty_dataset_runs_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text("name,amount\n", encoding="utf-8")
            result = run_pipeline(csv_path, PipelineConfig(output_directory=Path(tmpdir) / "out"))
            self.assertTrue(result.success)
            self.assertEqual(result.quality.row_count, 0)
            self.assertIn("report_html", result.output_paths)

    def test_single_column_dataset_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "single.csv"
            csv_path.write_text("amount\n10\n20\n30\n", encoding="utf-8")
            result = run_pipeline(csv_path, PipelineConfig(output_directory=Path(tmpdir) / "out"))
            self.assertTrue(result.success)
            self.assertEqual(result.quality.column_count, 1)
            self.assertEqual(len(result.analysis.correlations), 0)

    def test_large_file_limit_blocks_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "large.csv"
            csv_path.write_text("name\n" + ("x" * 1024), encoding="utf-8")
            with self.assertRaises(CSVIngestionError):
                ingest_csv(csv_path, PipelineConfig(max_file_size_bytes=8))

    def test_invalid_input_extension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "not_csv.txt"
            txt_path.write_text("name\nvalue\n", encoding="utf-8")
            with self.assertRaises(CSVIngestionError):
                ingest_csv(txt_path, PipelineConfig())


if __name__ == "__main__":
    unittest.main()

