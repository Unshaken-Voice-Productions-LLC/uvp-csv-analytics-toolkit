from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .models import (
    CleaningReport,
    ColumnProfile,
    ColumnType,
    PipelineConfig,
    TransformationRecord,
)
from .utils import is_missing, normalize_text, parse_boolean, parse_date, parse_number


@dataclass(slots=True)
class _ColumnStrategy:
    name: str
    strategy: str
    replacement: Any = None


def _default_strategy(profile: ColumnProfile) -> str:
    if profile.detected_type == ColumnType.NUMERIC:
        return "median"
    if profile.detected_type in {ColumnType.CATEGORICAL, ColumnType.BOOLEAN}:
        return "mode"
    if profile.detected_type == ColumnType.DATE:
        return "mode"
    return "leave_as_null"


def _normalized_non_missing_value(value: Any, profile: ColumnProfile, config: PipelineConfig) -> Any:
    if is_missing(value):
        return None
    if profile.detected_type == ColumnType.NUMERIC:
        return parse_number(value)
    if profile.detected_type == ColumnType.DATE:
        parsed = parse_date(value)
        return parsed
    if profile.detected_type == ColumnType.BOOLEAN:
        return parse_boolean(value)
    if profile.detected_type == ColumnType.CATEGORICAL:
        return normalize_text(value, case=config.categorical_case, strip=config.strip_categoricals)
    return " ".join(str(value).split())


def _fallback_replacement(profile: ColumnProfile, config: PipelineConfig, strategy: str, rows: list[dict[str, Any]]) -> Any:
    column_values = [_normalized_non_missing_value(row.get(profile.name), profile, config) for row in rows]
    clean_values = [value for value in column_values if value is not None]
    if not clean_values:
        return config.fill_values.get(profile.name, None if strategy == "leave_as_null" else "")
    if strategy == "mean":
        numbers = [float(value) for value in clean_values if isinstance(value, (int, float))]
        return sum(numbers) / len(numbers) if numbers else None
    if strategy == "median":
        numbers = [float(value) for value in clean_values if isinstance(value, (int, float))]
        if numbers:
            ordered = sorted(numbers)
            mid = len(ordered) // 2
            return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        if profile.detected_type == ColumnType.DATE:
            ordered_dates = sorted(value for value in clean_values if isinstance(value, datetime))
            if ordered_dates:
                return ordered_dates[len(ordered_dates) // 2]
        return Counter(clean_values).most_common(1)[0][0]
    if strategy == "mode":
        return Counter(clean_values).most_common(1)[0][0]
    if strategy == "constant":
        if profile.name in config.fill_values:
            return config.fill_values[profile.name]
        if profile.detected_type == ColumnType.NUMERIC:
            return 0.0
        if profile.detected_type == ColumnType.DATE:
            return None
        if profile.detected_type == ColumnType.BOOLEAN:
            return False
        return ""
    return None


def _build_backfill_map(rows: list[dict[str, Any]], profile: ColumnProfile, config: PipelineConfig) -> list[Any]:
    values: list[Any] = [None] * len(rows)
    next_seen: Any = None
    for index in range(len(rows) - 1, -1, -1):
        value = _normalized_non_missing_value(rows[index].get(profile.name), profile, config)
        if value is not None:
            next_seen = value
        values[index] = next_seen
    return values


def clean_dataset(
    rows: list[dict[str, Any]],
    profiles: list[ColumnProfile],
    config: PipelineConfig,
) -> tuple[list[dict[str, Any]], CleaningReport]:
    profile_map = {profile.name: profile for profile in profiles}
    strategy_map: dict[str, _ColumnStrategy] = {}
    for profile in profiles:
        strategy_name = config.missing_value_overrides.get(profile.name, _default_strategy(profile))
        replacement = _fallback_replacement(profile, config, strategy_name, rows)
        strategy_map[profile.name] = _ColumnStrategy(name=profile.name, strategy=strategy_name, replacement=replacement)

    backfill_maps: dict[str, list[Any]] = {}
    for profile in profiles:
        if strategy_map[profile.name].strategy == "back_fill":
            backfill_maps[profile.name] = _build_backfill_map(rows, profile, config)

    last_seen: dict[str, Any] = {}
    transformations: list[TransformationRecord] = []
    cleaned_rows: list[dict[str, Any]] = []
    dropped_rows = 0
    filled_cells = 0
    standardized_numeric_cells = 0
    normalized_categorical_cells = 0

    for row_index, row in enumerate(rows, start=2):
        cleaned_row: dict[str, Any] = {}
        drop_row = False
        for column in row.keys():
            profile = profile_map.get(column)
            raw_value = row.get(column)
            strategy = strategy_map.get(column)

            if profile is None:
                cleaned_row[column] = raw_value
                continue

            if is_missing(raw_value):
                if strategy and strategy.strategy == "drop_row":
                    drop_row = True
                    transformations.append(
                        TransformationRecord(
                            row_number=row_index,
                            column=column,
                            operation="drop_row",
                            before=raw_value,
                            after=None,
                            reason="Configured drop_row strategy encountered a missing value.",
                        )
                    )
                    break
                if strategy and strategy.strategy == "forward_fill" and column in last_seen:
                    replacement = last_seen[column]
                elif strategy and strategy.strategy == "back_fill":
                    replacement = backfill_maps.get(column, [None] * len(rows))[row_index - 2]
                else:
                    replacement = strategy.replacement if strategy else None
                if replacement is not None or (strategy and strategy.strategy != "leave_as_null"):
                    cleaned_row[column] = replacement
                    filled_cells += 1
                    transformations.append(
                        TransformationRecord(
                            row_number=row_index,
                            column=column,
                            operation="fill_missing",
                            before=raw_value,
                            after=replacement,
                            reason=f"Applied {strategy.strategy if strategy else 'leave_as_null'} strategy.",
                        )
                    )
                else:
                    cleaned_row[column] = None
                continue

            cleaned_value = _normalized_non_missing_value(raw_value, profile, config)
            if profile.detected_type == ColumnType.NUMERIC and cleaned_value is not None:
                if cleaned_value != raw_value:
                    standardized_numeric_cells += 1
                    transformations.append(
                        TransformationRecord(
                            row_number=row_index,
                            column=column,
                            operation="standardize_numeric",
                            before=raw_value,
                            after=cleaned_value,
                            reason="Normalized numeric text into a numeric value.",
                        )
                    )
                cleaned_row[column] = cleaned_value
            elif profile.detected_type == ColumnType.CATEGORICAL and isinstance(cleaned_value, str):
                normalized = cleaned_value
                if normalized != raw_value:
                    normalized_categorical_cells += 1
                    transformations.append(
                        TransformationRecord(
                            row_number=row_index,
                            column=column,
                            operation="normalize_categorical",
                            before=raw_value,
                            after=normalized,
                            reason="Normalized categorical text for consistent grouping and reporting.",
                        )
                    )
                cleaned_row[column] = normalized
            elif profile.detected_type == ColumnType.DATE:
                cleaned_row[column] = cleaned_value
                if cleaned_value is not None and cleaned_value != raw_value:
                    transformations.append(
                        TransformationRecord(
                            row_number=row_index,
                            column=column,
                            operation="standardize_date",
                            before=raw_value,
                            after=cleaned_value,
                            reason="Converted date text into datetime objects.",
                        )
                    )
            elif profile.detected_type == ColumnType.BOOLEAN:
                cleaned_row[column] = cleaned_value
                if cleaned_value is not None and cleaned_value != raw_value:
                    transformations.append(
                        TransformationRecord(
                            row_number=row_index,
                            column=column,
                            operation="standardize_boolean",
                            before=raw_value,
                            after=cleaned_value,
                            reason="Converted boolean-like text into boolean values.",
                        )
                    )
            else:
                cleaned_row[column] = cleaned_value

            if strategy and strategy.strategy == "forward_fill":
                last_seen[column] = cleaned_row[column]
            elif cleaned_row[column] is not None:
                last_seen[column] = cleaned_row[column]

        if drop_row:
            dropped_rows += 1
            continue

        cleaned_rows.append(cleaned_row)

    report = CleaningReport(
        rows_in=len(rows),
        rows_out=len(cleaned_rows),
        dropped_rows=dropped_rows,
        filled_cells=filled_cells,
        standardized_numeric_cells=standardized_numeric_cells,
        normalized_categorical_cells=normalized_categorical_cells,
        transformations=transformations,
    )
    return cleaned_rows, report

