from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .models import (
    AnalysisReport,
    ColumnProfile,
    ColumnType,
    DataQualityReport,
    IngestedDataset,
    IssueSeverity,
    OutlierRecord,
    PipelineConfig,
    ValidationIssue,
)
from .utils import is_missing, parse_boolean, parse_date, parse_number, percentiles, safe_mean, safe_median, safe_stddev


@dataclass(slots=True)
class SchemaInferenceResult:
    profiles: list[ColumnProfile]
    issues: list[ValidationIssue]


def _normalize_counter_value(value: Any) -> str:
    return " ".join(str(value).strip().split()).lower()


def _profile_type(values: list[Any]) -> tuple[ColumnType, float, int]:
    non_missing_values = [value for value in values if not is_missing(value)]
    if not non_missing_values:
        return ColumnType.TEXT, 0.0, 0

    bool_count = sum(1 for value in non_missing_values if parse_boolean(value) is not None)
    numeric_count = sum(1 for value in non_missing_values if parse_number(value) is not None)
    date_count = sum(1 for value in non_missing_values if parse_date(value) is not None)

    unique_count = len({_normalize_counter_value(value) for value in non_missing_values})
    avg_length = sum(len(str(value).strip()) for value in non_missing_values) / len(non_missing_values)

    bool_ratio = bool_count / len(non_missing_values)
    numeric_ratio = numeric_count / len(non_missing_values)
    date_ratio = date_count / len(non_missing_values)
    categorical_signal = unique_count <= max(20, int(len(non_missing_values) * 0.3)) or (
        unique_count <= max(50, int(len(non_missing_values) * 0.5)) and avg_length <= 40
    )

    if bool_ratio >= 0.9:
        return ColumnType.BOOLEAN, bool_ratio, bool_count
    if numeric_ratio >= 0.8 and numeric_count >= 3:
        return ColumnType.NUMERIC, numeric_ratio, numeric_count
    if date_ratio >= 0.8 and date_count >= 3:
        return ColumnType.DATE, date_ratio, date_count
    if categorical_signal:
        categorical_confidence = min(1.0, 1.0 - (unique_count / max(1, len(non_missing_values))) + 0.5)
        return ColumnType.CATEGORICAL, max(0.5, categorical_confidence), len(non_missing_values)
    return ColumnType.TEXT, max(numeric_ratio, date_ratio, bool_ratio), 0


def infer_schema(
    rows: list[dict[str, Any]],
    headers: list[str],
    sample_limit: int,
) -> list[ColumnProfile]:
    sampled_rows = rows[:sample_limit] if sample_limit > 0 else list(rows)
    profiles: list[ColumnProfile] = []
    for header in headers:
        column_values = [row.get(header) for row in sampled_rows]
        non_missing_values = [value for value in column_values if not is_missing(value)]
        detected_type, confidence, matched_count = _profile_type(column_values)
        numeric_values = [parse_number(value) for value in non_missing_values]
        numeric_values = [value for value in numeric_values if value is not None]
        date_values = [parse_date(value) for value in non_missing_values]
        date_values = [value for value in date_values if value is not None]
        distinct_count = len({_normalize_counter_value(value) for value in non_missing_values})
        top_values = Counter(_normalize_counter_value(value) for value in non_missing_values).most_common(5)

        invalid_count = 0
        if detected_type == ColumnType.NUMERIC:
            invalid_count = len(non_missing_values) - len(numeric_values)
        elif detected_type == ColumnType.DATE:
            invalid_count = len(non_missing_values) - len(date_values)
        elif detected_type == ColumnType.BOOLEAN:
            invalid_count = sum(1 for value in non_missing_values if parse_boolean(value) is None)
        elif detected_type == ColumnType.CATEGORICAL:
            invalid_count = 0
        else:
            invalid_count = 0

        profile = ColumnProfile(
            name=header,
            detected_type=detected_type,
            confidence=confidence,
            non_missing_count=len(non_missing_values),
            missing_count=len(column_values) - len(non_missing_values),
            distinct_count=distinct_count,
            invalid_count=invalid_count,
            sample_values=non_missing_values[:5],
            top_values=top_values,
        )

        if detected_type == ColumnType.NUMERIC and numeric_values:
            profile.numeric_min = min(numeric_values)
            profile.numeric_max = max(numeric_values)
            profile.numeric_mean = safe_mean(numeric_values)
            profile.numeric_median = safe_median(numeric_values)
            profile.numeric_stddev = safe_stddev(numeric_values)
        elif detected_type == ColumnType.DATE and date_values:
            profile.date_min = min(date_values).isoformat()
            profile.date_max = max(date_values).isoformat()

        profiles.append(profile)
    return profiles


def detect_outliers(
    rows: list[dict[str, Any]],
    profiles: list[ColumnProfile],
    iqr_multiplier: float,
    zscore_threshold: float,
) -> list[OutlierRecord]:
    outliers: list[OutlierRecord] = []
    numeric_profiles = [profile for profile in profiles if profile.detected_type == ColumnType.NUMERIC]
    for profile in numeric_profiles:
        values: list[tuple[int, float]] = []
        for row_index, row in enumerate(rows, start=2):
            number = parse_number(row.get(profile.name))
            if number is not None:
                values.append((row_index, number))
        if len(values) < 4:
            continue

        numeric_values = [value for _, value in values]
        q1 = percentiles(numeric_values, 0.25)
        q3 = percentiles(numeric_values, 0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mean_value = safe_mean(numeric_values)
        stddev_value = safe_stddev(numeric_values)

        for row_number, value in values:
            iqr_flag = value < lower or value > upper
            zscore = 0.0
            zscore_flag = False
            if mean_value is not None and stddev_value not in (None, 0):
                zscore = abs((value - mean_value) / stddev_value)
                zscore_flag = zscore >= zscore_threshold
            if iqr_flag or zscore_flag:
                method = "both" if iqr_flag and zscore_flag else "iqr" if iqr_flag else "zscore"
                outliers.append(
                    OutlierRecord(
                        column=profile.name,
                        row_number=row_number,
                        value=value,
                        method=method,
                        score=zscore if zscore_flag else abs(value - (mean_value or value)),
                        lower_bound=lower,
                        upper_bound=upper,
                    )
                )
    return outliers


def _count_missing_cells(rows: list[dict[str, Any]], headers: list[str]) -> int:
    total = 0
    for row in rows:
        for header in headers:
            if is_missing(row.get(header)):
                total += 1
    return total


def _count_duplicates(rows: list[dict[str, Any]], headers: list[str]) -> int:
    seen = Counter(tuple(row.get(header) for header in headers) for row in rows)
    return sum(count - 1 for count in seen.values() if count > 1)


def build_quality_report(dataset: IngestedDataset, config: PipelineConfig) -> DataQualityReport:
    profiles = infer_schema(dataset.rows, dataset.headers, config.sample_rows_for_inference)
    outliers = detect_outliers(dataset.rows, profiles, config.outlier_iqr_multiplier, config.outlier_zscore_threshold)
    missing_cells = _count_missing_cells(dataset.rows, dataset.headers)
    duplicate_rows = _count_duplicates(dataset.rows, dataset.headers)

    issues: list[ValidationIssue] = []
    if dataset.malformed_rows:
        issues.append(
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                code="malformed_rows",
                message=f"Detected {len(dataset.malformed_rows)} malformed row(s) during ingestion.",
                details={
                    "malformed_rows": [
                        {
                            "line_number": row.line_number,
                            "expected_fields": row.expected_fields,
                            "actual_fields": row.actual_fields,
                            "raw_preview": row.raw_preview,
                        }
                        for row in dataset.malformed_rows
                    ]
                },
            )
        )
    if duplicate_rows:
        issues.append(
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                code="duplicate_rows",
                message=f"Detected {duplicate_rows} duplicate record(s).",
                details={"duplicate_rows": duplicate_rows},
            )
        )

    for profile in profiles:
        if profile.missing_count:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.INFO if profile.missing_count / max(1, len(dataset.rows)) < config.quality_issue_threshold else IssueSeverity.WARNING,
                    code="missing_values",
                    column=profile.name,
                    message=(
                        f"Column '{profile.name}' contains {profile.missing_count} missing value(s) "
                        f"out of {max(1, len(dataset.rows))} row(s)."
                    ),
                    details={"missing_count": profile.missing_count, "missing_rate": profile.missing_count / max(1, len(dataset.rows))},
                )
            )
        if profile.invalid_count:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="inconsistent_values",
                    column=profile.name,
                    message=(
                        f"Column '{profile.name}' has {profile.invalid_count} value(s) that do not fit the inferred "
                        f"{profile.detected_type.value} schema."
                    ),
                    details={"invalid_count": profile.invalid_count, "detected_type": profile.detected_type.value},
                )
            )
        if profile.confidence < 0.65 and profile.non_missing_count:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.INFO,
                    code="low_schema_confidence",
                    column=profile.name,
                    message=f"Schema confidence for '{profile.name}' is only {profile.confidence:.2f}.",
                )
            )

    if outliers:
        outlier_counts = Counter(record.column for record in outliers)
        for column, count in outlier_counts.items():
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="outliers_detected",
                    column=column,
                    message=f"Detected {count} potential outlier(s) in '{column}'.",
                    details={"outlier_count": count},
                )
            )

    return DataQualityReport(
        row_count=len(dataset.rows),
        column_count=len(dataset.headers),
        malformed_row_count=len(dataset.malformed_rows),
        duplicate_row_count=duplicate_rows,
        missing_cell_count=missing_cells,
        profiles=profiles,
        issues=issues,
        outliers=outliers,
    )

