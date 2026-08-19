from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    TEXT = "text"
    BOOLEAN = "boolean"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class FilterSpec:
    column: str
    operation: str = "equals"
    value: Any = None
    value_max: Any | None = None


@dataclass(slots=True)
class PipelineConfig:
    max_file_size_bytes: int = 25_000_000
    max_rows: int = 100_000
    sample_rows_for_inference: int = 5_000
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    categorical_case: str = "lower"
    strip_categoricals: bool = True
    encoding_candidates: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    quality_issue_threshold: float = 0.1
    chart_limit: int = 6
    model_enabled: bool = True
    target_column: str | None = None
    feature_columns: list[str] | None = None
    slice_filters: list[FilterSpec] = field(default_factory=list)
    missing_value_overrides: dict[str, str] = field(default_factory=dict)
    fill_values: dict[str, Any] = field(default_factory=dict)
    output_directory: Path | None = None
    log_level: str = "INFO"


@dataclass(slots=True)
class IngestionIssue:
    severity: IssueSeverity
    code: str
    message: str
    line_number: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MalformedRow:
    line_number: int
    expected_fields: int
    actual_fields: int
    raw_preview: str


@dataclass(slots=True)
class IngestedDataset:
    source_name: str
    encoding: str
    dialect: str
    headers: list[str]
    rows: list[dict[str, Any]]
    malformed_rows: list[MalformedRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False


@dataclass(slots=True)
class ColumnProfile:
    name: str
    detected_type: ColumnType
    confidence: float
    non_missing_count: int
    missing_count: int
    distinct_count: int
    invalid_count: int
    sample_values: list[Any] = field(default_factory=list)
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_mean: float | None = None
    numeric_median: float | None = None
    numeric_stddev: float | None = None
    date_min: str | None = None
    date_max: str | None = None


@dataclass(slots=True)
class ValidationIssue:
    severity: IssueSeverity
    code: str
    message: str
    column: str | None = None
    row_number: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutlierRecord:
    column: str
    row_number: int
    value: float
    method: str
    score: float
    lower_bound: float
    upper_bound: float


@dataclass(slots=True)
class DataQualityReport:
    row_count: int
    column_count: int
    malformed_row_count: int
    duplicate_row_count: int
    missing_cell_count: int
    profiles: list[ColumnProfile]
    issues: list[ValidationIssue]
    outliers: list[OutlierRecord]


@dataclass(slots=True)
class TransformationRecord:
    row_number: int
    column: str
    operation: str
    before: Any
    after: Any
    reason: str


@dataclass(slots=True)
class CleaningReport:
    rows_in: int
    rows_out: int
    dropped_rows: int
    filled_cells: int
    standardized_numeric_cells: int
    normalized_categorical_cells: int
    transformations: list[TransformationRecord]


@dataclass(slots=True)
class CorrelationResult:
    column_a: str
    column_b: str
    correlation: float
    pair_count: int


@dataclass(slots=True)
class SegmentSummary:
    dimension: str
    value: str
    row_count: int
    share: float
    numeric_aggregates: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TrendResult:
    date_column: str
    value_column: str
    slope: float
    direction: str
    r_squared: float
    anomaly_row_numbers: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ModelResult:
    model_type: str
    target_column: str
    feature_columns: list[str]
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisReport:
    summary_statistics: dict[str, dict[str, float]]
    correlations: list[CorrelationResult]
    segments: list[SegmentSummary]
    trends: list[TrendResult]
    anomalies: list[ValidationIssue]
    model: ModelResult | None


@dataclass(slots=True)
class ChartArtifact:
    title: str
    kind: str
    filename: str
    svg: str
    description: str


@dataclass(slots=True)
class VisualizationReport:
    charts: list[ChartArtifact]


@dataclass(slots=True)
class InsightReport:
    executive_summary: str
    key_drivers: list[str]
    risks: list[str]
    recommended_actions: list[str]


@dataclass(slots=True)
class StageTiming:
    stage: str
    duration_seconds: float
    status: str
    message: str | None = None


@dataclass(slots=True)
class PipelineResult:
    success: bool
    source_name: str
    ingestion: IngestedDataset | None
    quality: DataQualityReport | None
    cleaning: CleaningReport | None
    analysis: AnalysisReport | None
    visualization: VisualizationReport | None
    insights: InsightReport | None
    cleaned_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: list[StageTiming] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)


def stringify_scalar(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value

