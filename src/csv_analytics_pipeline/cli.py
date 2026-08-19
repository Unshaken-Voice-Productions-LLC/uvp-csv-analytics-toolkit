from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .models import FilterSpec, PipelineConfig
from .pipeline import run_pipeline


def _parse_filter_spec(expression: str) -> FilterSpec:
    for operator in (">=", "<=", "!=", ">", "<", "~", "="):
        if operator in expression:
            left, right = expression.split(operator, 1)
            column = left.strip()
            raw_value = right.strip()
            if operator == "=" and ".." in raw_value:
                low, high = raw_value.split("..", 1)
                return FilterSpec(column=column, operation="between", value=low.strip(), value_max=high.strip())
            if operator == "=":
                return FilterSpec(column=column, operation="equals", value=raw_value)
            if operator == "!=":
                return FilterSpec(column=column, operation="not_equals", value=raw_value)
            if operator == "~":
                return FilterSpec(column=column, operation="contains", value=raw_value)
            if operator == ">=":
                return FilterSpec(column=column, operation="greater_equal", value=raw_value)
            if operator == ">":
                return FilterSpec(column=column, operation="greater_than", value=raw_value)
            if operator == "<=":
                return FilterSpec(column=column, operation="less_equal", value=raw_value)
            if operator == "<":
                return FilterSpec(column=column, operation="less_than", value=raw_value)
    raise argparse.ArgumentTypeError(
        "Filters must use one of these forms: column=value, column!=value, column~value, column>value, column>=value, "
        "column<value, column<=value, or column=low..high"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CSV analytics pipeline.")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to analyze.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated report artifacts.")
    parser.add_argument("--max-file-size-mb", type=int, default=None, help="Override the default file size limit in MB.")
    parser.add_argument("--max-rows", type=int, default=None, help="Override the soft row-processing limit.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Override the number of rows used for schema inference.")
    parser.add_argument("--target-column", type=str, default=None, help="Optional numeric target column for regression.")
    parser.add_argument("--feature-column", action="append", default=None, help="Optional feature columns for regression.")
    parser.add_argument("--filter", dest="filters", action="append", default=None, type=_parse_filter_spec, help="Slice expression like region=west or amount>=100.")
    parser.add_argument("--chart-limit", type=int, default=None, help="Maximum number of charts to generate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    defaults = PipelineConfig()

    config = PipelineConfig(
        output_directory=args.output_dir,
        max_file_size_bytes=(args.max_file_size_mb * 1024 * 1024) if args.max_file_size_mb else defaults.max_file_size_bytes,
        max_rows=args.max_rows or defaults.max_rows,
        sample_rows_for_inference=args.sample_rows or defaults.sample_rows_for_inference,
        target_column=args.target_column,
        feature_columns=args.feature_column,
        slice_filters=args.filters or [],
        chart_limit=args.chart_limit or defaults.chart_limit,
    )

    result = run_pipeline(args.csv_path, config=config)

    if result.success:
        print(f"Pipeline completed successfully for {result.source_name}.")
        print(f"Output directory: {result.output_paths.get('output_dir', config.output_directory or '')}")
        print(f"Report: {result.output_paths.get('report_html', '')}")
        if result.insights:
            print(result.insights.executive_summary)
        return 0

    print(f"Pipeline failed for {result.source_name}.", file=sys.stderr)
    for error in result.errors:
        print(f"- {error}", file=sys.stderr)
    if result.warnings:
        print("Warnings:", file=sys.stderr)
        for warning in result.warnings:
            print(f"- {warning}", file=sys.stderr)
    if result.output_paths.get("report_html"):
        print(f"Partial report: {result.output_paths['report_html']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
