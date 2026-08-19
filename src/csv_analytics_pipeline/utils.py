from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import csv
import json
import math
import re
from statistics import mean as statistics_mean
from typing import Any, Iterable, Sequence

from .models import FilterSpec, stringify_scalar


MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "-", "--"}

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%b %d %Y",
    "%B %d %Y",
)

NUMERIC_CLEAN_RE = re.compile(r"[^0-9eE\.\-\+\(\),%]")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_TOKENS
    return False


def normalize_text(value: Any, case: str = "lower", strip: bool = True) -> Any:
    if value is None:
        return None
    text = str(value)
    if strip:
        text = " ".join(text.strip().split())
    if case == "lower":
        text = text.lower()
    elif case == "upper":
        text = text.upper()
    elif case == "title":
        text = text.title()
    return text


def parse_number(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    cleaned = NUMERIC_CLEAN_RE.sub("", text)
    cleaned = cleaned.replace(",", "")
    if cleaned in {"", ".", "+", "-", "+.", "-."}:
        return None
    try:
        number = float(cleaned)
        if negative:
            number = -number
        if percent:
            number /= 100.0
        return number
    except ValueError:
        return None


def parse_date(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_boolean(value: Any) -> bool | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def percentiles(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentiles requires at least one value")
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (pos - lower)


def safe_mean(values: Sequence[float]) -> float | None:
    return float(statistics_mean(values)) if values else None


def safe_median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def safe_stddev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = safe_mean(values)
    if avg is None:
        return None
    variance = sum((float(v) - avg) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    if mean_x is None or mean_y is None:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denominator = denominator_x * denominator_y
    if denominator == 0:
        return None
    return numerator / denominator


def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("linear_regression requires at least two paired values")
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    if mean_x is None or mean_y is None:
        raise ValueError("linear_regression requires numeric values")
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("linear_regression cannot fit constant x values")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    predictions = [intercept + slope * x for x in xs]
    total_ss = sum((y - mean_y) ** 2 for y in ys)
    residual_ss = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, predictions))
    r_squared = 1.0 if total_ss == 0 and residual_ss == 0 else (0.0 if total_ss == 0 else 1 - residual_ss / total_ss)
    return slope, intercept, r_squared


def build_design_matrix(rows: Sequence[dict[str, Any]], features: Sequence[str]) -> tuple[list[list[float]], list[int]]:
    matrix: list[list[float]] = []
    kept_rows: list[int] = []
    for index, row in enumerate(rows):
        vector = [1.0]
        valid = True
        for feature in features:
            number = parse_number(row.get(feature))
            if number is None:
                valid = False
                break
            vector.append(number)
        if valid:
            matrix.append(vector)
            kept_rows.append(index)
    return matrix, kept_rows


def transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def matrix_multiply(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    b_transposed = transpose(b)
    result: list[list[float]] = []
    for row in a:
        result_row = []
        for column in b_transposed:
            result_row.append(sum(x * y for x, y in zip(row, column)))
        result.append(result_row)
    return result


def solve_linear_system(matrix: Sequence[Sequence[float]], values: Sequence[float]) -> list[float]:
    augmented = [list(row) + [float(value)] for row, value in zip(matrix, values)]
    n = len(augmented)
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [current - factor * pivot_value for current, pivot_value in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def fit_multiple_linear_regression(rows: Sequence[dict[str, Any]], feature_columns: Sequence[str], target_column: str) -> dict[str, Any] | None:
    matrix: list[list[float]] = []
    target_values: list[float] = []
    for row in rows:
        target = parse_number(row.get(target_column))
        if target is None:
            continue
        vector = [1.0]
        valid = True
        for feature in feature_columns:
            number = parse_number(row.get(feature))
            if number is None:
                valid = False
                break
            vector.append(number)
        if not valid:
            continue
        matrix.append(vector)
        target_values.append(target)
    if len(matrix) < len(feature_columns) + 2 or len(target_values) != len(matrix):
        return None
    x_transposed = transpose(matrix)
    xtx = matrix_multiply(x_transposed, matrix)
    # Small ridge term to avoid singular matrices on narrow data.
    ridge = 1e-6
    for i in range(len(xtx)):
        xtx[i][i] += ridge
    xty = matrix_multiply(x_transposed, [[value] for value in target_values])
    coefficients = solve_linear_system(xtx, [row[0] for row in xty])
    intercept = coefficients[0]
    weights = coefficients[1:]
    predictions = []
    actuals = target_values
    for row in matrix:
        predictions.append(sum(weight * value for weight, value in zip(weights, row[1:])) + intercept)
    mean_actual = safe_mean(actuals) or 0.0
    ss_total = sum((actual - mean_actual) ** 2 for actual in actuals)
    ss_res = sum((actual - pred) ** 2 for actual, pred in zip(actuals, predictions))
    r_squared = 1.0 if ss_total == 0 and ss_res == 0 else (0.0 if ss_total == 0 else 1 - ss_res / ss_total)
    rmse = math.sqrt(ss_res / len(actuals)) if actuals else None
    mae = sum(abs(actual - pred) for actual, pred in zip(actuals, predictions)) / len(actuals) if actuals else None
    return {
        "intercept": intercept,
        "coefficients": dict(zip(feature_columns, weights)),
        "predictions": predictions,
        "metrics": {
            "r_squared": r_squared,
            "rmse": rmse,
            "mae": mae,
            "observations": len(actuals),
        },
    }


def apply_filters(rows: Sequence[dict[str, Any]], filters: Sequence[FilterSpec]) -> list[dict[str, Any]]:
    if not filters:
        return list(rows)

    def matches(row: dict[str, Any], spec: FilterSpec) -> bool:
        value = row.get(spec.column)
        if spec.operation == "equals":
            return value == spec.value
        if spec.operation == "not_equals":
            return value != spec.value
        if spec.operation == "contains":
            return spec.value is not None and str(spec.value).lower() in str(value).lower()
        if spec.operation == "greater_than":
            left = parse_number(value)
            right = parse_number(spec.value)
            return left is not None and right is not None and left > right
        if spec.operation == "greater_equal":
            left = parse_number(value)
            right = parse_number(spec.value)
            return left is not None and right is not None and left >= right
        if spec.operation == "less_than":
            left = parse_number(value)
            right = parse_number(spec.value)
            return left is not None and right is not None and left < right
        if spec.operation == "less_equal":
            left = parse_number(value)
            right = parse_number(spec.value)
            return left is not None and right is not None and left <= right
        if spec.operation == "between":
            left = parse_number(value)
            low = parse_number(spec.value)
            high = parse_number(spec.value_max)
            return left is not None and low is not None and high is not None and low <= left <= high
        if spec.operation == "in":
            allowed = spec.value if isinstance(spec.value, (list, tuple, set)) else [spec.value]
            return value in allowed
        return True

    filtered = list(rows)
    for spec in filters:
        filtered = [row for row in filtered if matches(row, spec)]
    return filtered


def slugify_filename(value: str, fallback: str = "chart") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or fallback


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date, Path)):
        return stringify_scalar(value)
    return stringify_scalar(value)


def dump_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def csv_dialect_name(dialect: csv.Dialect) -> str:
    return getattr(dialect, "name", dialect.__class__.__name__)
