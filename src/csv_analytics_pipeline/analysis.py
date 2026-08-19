from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .models import (
    AnalysisReport,
    ColumnProfile,
    ColumnType,
    CorrelationResult,
    IssueSeverity,
    ModelResult,
    PipelineConfig,
    SegmentSummary,
    TrendResult,
    ValidationIssue,
)
from .utils import fit_multiple_linear_regression, linear_regression, parse_date, parse_number, pearson_correlation, safe_mean, safe_median, safe_stddev


def _numeric_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        number = parse_number(row.get(column))
        if number is not None:
            values.append(number)
    return values


def compute_summary_statistics(rows: list[dict[str, Any]], profiles: list[ColumnProfile]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for profile in profiles:
        if profile.detected_type != ColumnType.NUMERIC:
            continue
        values = _numeric_values(rows, profile.name)
        if not values:
            continue
        ordered = sorted(values)
        q1_index = max(0, int((len(ordered) - 1) * 0.25))
        q3_index = max(0, int((len(ordered) - 1) * 0.75))
        summary[profile.name] = {
            "count": len(values),
            "mean": safe_mean(values) or 0.0,
            "median": safe_median(values) or 0.0,
            "stddev": safe_stddev(values) or 0.0,
            "min": min(values),
            "q1": ordered[q1_index],
            "q3": ordered[q3_index],
            "max": max(values),
        }
    return summary


def compute_correlations(rows: list[dict[str, Any]], profiles: list[ColumnProfile]) -> list[CorrelationResult]:
    numeric_columns = [profile.name for profile in profiles if profile.detected_type == ColumnType.NUMERIC]
    correlations: list[CorrelationResult] = []
    for left_index, left_column in enumerate(numeric_columns):
        for right_column in numeric_columns[left_index + 1 :]:
            paired = []
            for row in rows:
                left_value = parse_number(row.get(left_column))
                right_value = parse_number(row.get(right_column))
                if left_value is None or right_value is None:
                    continue
                paired.append((left_value, right_value))
            if len(paired) < 2:
                continue
            left_values = [pair[0] for pair in paired]
            right_values = [pair[1] for pair in paired]
            pair_count = len(paired)
            correlation = pearson_correlation(left_values, right_values)
            if correlation is None:
                continue
            correlations.append(
                CorrelationResult(
                    column_a=left_column,
                    column_b=right_column,
                    correlation=correlation,
                    pair_count=pair_count,
                )
            )
    correlations.sort(key=lambda item: abs(item.correlation), reverse=True)
    return correlations


def compute_segments(rows: list[dict[str, Any]], profiles: list[ColumnProfile]) -> list[SegmentSummary]:
    numeric_columns = [profile.name for profile in profiles if profile.detected_type == ColumnType.NUMERIC]
    reference_numeric = numeric_columns[0] if numeric_columns else None
    segments: list[SegmentSummary] = []
    total_rows = len(rows) or 1
    for profile in profiles:
        if profile.detected_type not in {ColumnType.CATEGORICAL, ColumnType.BOOLEAN}:
            continue
        non_missing_values = [row.get(profile.name) for row in rows if row.get(profile.name) is not None]
        counter = Counter(str(value) for value in non_missing_values)
        if len(counter) < 2:
            continue
        for value, count in counter.most_common(10):
            numeric_aggregates: dict[str, float] = {}
            if reference_numeric:
                numeric_values = [parse_number(row.get(reference_numeric)) for row in rows if str(row.get(profile.name)) == value]
                numeric_values = [number for number in numeric_values if number is not None]
                if numeric_values:
                    numeric_aggregates[f"mean_{reference_numeric}"] = safe_mean(numeric_values) or 0.0
                    numeric_aggregates[f"median_{reference_numeric}"] = safe_median(numeric_values) or 0.0
            segments.append(
                SegmentSummary(
                    dimension=profile.name,
                    value=value,
                    row_count=count,
                    share=count / total_rows,
                    numeric_aggregates=numeric_aggregates,
                )
            )
    return segments


def detect_trends(rows: list[dict[str, Any]], profiles: list[ColumnProfile], zscore_threshold: float) -> tuple[list[TrendResult], list[ValidationIssue]]:
    date_columns = [profile.name for profile in profiles if profile.detected_type == ColumnType.DATE]
    numeric_columns = [profile.name for profile in profiles if profile.detected_type == ColumnType.NUMERIC]
    trends: list[TrendResult] = []
    anomalies: list[ValidationIssue] = []
    if not date_columns or not numeric_columns:
        return trends, anomalies

    for date_column in date_columns:
        for numeric_column in numeric_columns[:3]:
            paired: list[tuple[int, datetime, float]] = []
            for row_number, row in enumerate(rows, start=2):
                date_value = parse_date(row.get(date_column))
                numeric_value = parse_number(row.get(numeric_column))
                if date_value is None or numeric_value is None:
                    continue
                paired.append((row_number, date_value, numeric_value))
            if len(paired) < 3:
                continue
            paired.sort(key=lambda item: item[1])
            xs = [item[1].toordinal() for item in paired]
            ys = [item[2] for item in paired]
            try:
                slope, intercept, r_squared = linear_regression(xs, ys)
            except ValueError:
                continue
            direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
            predictions = [intercept + slope * x for x in xs]
            residuals = [actual - predicted for actual, predicted in zip(ys, predictions)]
            residual_stddev = safe_stddev(residuals)
            anomaly_rows: list[int] = []
            if residual_stddev and residual_stddev > 0:
                for (row_number, _, actual), predicted, residual in zip(paired, predictions, residuals):
                    zscore = abs(residual / residual_stddev)
                    if zscore >= zscore_threshold:
                        anomaly_rows.append(row_number)
                        anomalies.append(
                            ValidationIssue(
                                severity=IssueSeverity.INFO,
                                code="trend_anomaly",
                                column=numeric_column,
                                row_number=row_number,
                                message=(
                                    f"Row {row_number} in '{numeric_column}' deviates from the fitted trend around '{date_column}'."
                                ),
                                details={
                                    "date_column": date_column,
                                    "zscore": zscore,
                                    "predicted": predicted,
                                    "actual": actual,
                                },
                            )
                        )
            trends.append(
                TrendResult(
                    date_column=date_column,
                    value_column=numeric_column,
                    slope=slope,
                    direction=direction,
                    r_squared=r_squared,
                    anomaly_row_numbers=anomaly_rows,
                )
            )
    return trends, anomalies


def _choose_model_columns(rows: list[dict[str, Any]], profiles: list[ColumnProfile], config: PipelineConfig) -> tuple[str | None, list[str]]:
    numeric_columns = [profile.name for profile in profiles if profile.detected_type == ColumnType.NUMERIC]
    if not numeric_columns:
        return None, []
    target = config.target_column if config.target_column in numeric_columns else numeric_columns[0]
    if config.feature_columns:
        features = [column for column in config.feature_columns if column != target and column in numeric_columns]
    else:
        features = [column for column in numeric_columns if column != target]
    if not features:
        return None, []
    return target, features[: min(4, len(features))]


def build_model(
    rows: list[dict[str, Any]],
    profiles: list[ColumnProfile],
    config: PipelineConfig,
) -> ModelResult | None:
    if not config.model_enabled or not rows:
        return None

    target, features = _choose_model_columns(rows, profiles, config)
    if not target or not features:
        return None

    regression = fit_multiple_linear_regression(rows, features, target)
    if regression is None:
        return None

    metrics = regression["metrics"]
    return ModelResult(
        model_type="regression",
        target_column=target,
        feature_columns=list(features),
        coefficients=dict(regression["coefficients"]),
        intercept=float(regression["intercept"]),
        metrics={key: float(value) if isinstance(value, (int, float)) and value is not None else value for key, value in metrics.items()},
    )


def analyze_dataset(
    rows: list[dict[str, Any]],
    profiles: list[ColumnProfile],
    config: PipelineConfig,
) -> AnalysisReport:
    summary_statistics = compute_summary_statistics(rows, profiles)
    correlations = compute_correlations(rows, profiles)
    segments = compute_segments(rows, profiles)
    trends, trend_anomalies = detect_trends(rows, profiles, config.outlier_zscore_threshold)
    model = build_model(rows, profiles, config)
    anomalies = list(trend_anomalies)
    return AnalysisReport(
        summary_statistics=summary_statistics,
        correlations=correlations,
        segments=segments,
        trends=trends,
        anomalies=anomalies,
        model=model,
    )
