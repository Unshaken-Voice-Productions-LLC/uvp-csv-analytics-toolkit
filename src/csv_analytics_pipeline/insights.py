from __future__ import annotations

from collections import Counter
from typing import Any

from .models import AnalysisReport, ColumnType, DataQualityReport, InsightReport, PipelineConfig


def _format_pair(column_a: str, column_b: str, correlation: float, pair_count: int) -> str:
    return f"{column_a} and {column_b} move together with correlation {correlation:.2f} across {pair_count} paired observations."


def _top_missing_profiles(quality: DataQualityReport) -> list[str]:
    profiles = sorted(quality.profiles, key=lambda profile: profile.missing_count, reverse=True)
    return [profile.name for profile in profiles if profile.missing_count > 0][:3]


def _top_outlier_columns(quality: DataQualityReport) -> list[str]:
    counter = Counter(record.column for record in quality.outliers)
    return [column for column, _ in counter.most_common(3)]


def build_insights(
    quality: DataQualityReport,
    analysis: AnalysisReport | None,
    config: PipelineConfig,
) -> InsightReport:
    if quality.row_count == 0:
        return InsightReport(
            executive_summary="The uploaded CSV contains no data rows, so no statistical insights can be generated.",
            key_drivers=[],
            risks=["The file is empty or only contains a header row."],
            recommended_actions=["Provide at least one populated data row and rerun the pipeline."],
        )

    analysis = analysis or AnalysisReport({}, [], [], [], [], None)
    key_drivers: list[str] = []
    risks: list[str] = []
    recommended_actions: list[str] = []

    strong_correlations = [item for item in analysis.correlations if abs(item.correlation) >= 0.7 and item.pair_count >= 3]
    for item in strong_correlations[:3]:
        key_drivers.append(_format_pair(item.column_a, item.column_b, item.correlation, item.pair_count))

    if analysis.trends:
        trend = analysis.trends[0]
        if abs(trend.slope) > 0:
            direction = "increasing" if trend.slope > 0 else "decreasing"
            key_drivers.append(
                f"{trend.value_column} shows a {direction} trend over {trend.date_column} with R² of {trend.r_squared:.2f}."
            )

    if analysis.model and analysis.model.metrics:
        r_squared = analysis.model.metrics.get("r_squared")
        if isinstance(r_squared, (int, float)):
            key_drivers.append(
                f"Basic regression on {analysis.model.target_column} explains {r_squared:.2f} of observed variance using "
                f"{len(analysis.model.feature_columns)} feature(s)."
            )

    missing_profiles = _top_missing_profiles(quality)
    if missing_profiles:
        risks.append(
            f"Missing values are concentrated in {', '.join(missing_profiles)}; imputation or source cleanup is likely needed."
        )

    if quality.duplicate_row_count:
        risks.append(f"The dataset contains {quality.duplicate_row_count} duplicate record(s).")

    if quality.outliers:
        outlier_columns = _top_outlier_columns(quality)
        risks.append(f"Potential outliers were found in {', '.join(outlier_columns)}.")

    low_confidence_columns = [profile.name for profile in quality.profiles if profile.confidence < 0.65 and profile.non_missing_count]
    if low_confidence_columns:
        risks.append(
            f"Schema confidence is weak for {', '.join(low_confidence_columns[:3])}, so those columns should be reviewed manually."
        )

    if quality.malformed_row_count:
        risks.append(f"{quality.malformed_row_count} malformed row(s) were repaired or truncated during ingestion.")

    if quality.row_count < 10:
        risks.append("The dataset is very small, so correlations and models may be unstable.")

    if quality.column_count == 1:
        risks.append("Single-column inputs limit correlation and segmentation analysis.")

    if not key_drivers:
        key_drivers.append("No strong correlations or trends crossed the reporting threshold, so the data appears relatively flat.")

    recommended_actions.extend(
        [
            "Review the top missing-value columns and choose a fill strategy that matches the business meaning of each field.",
            "Validate the outlier rows against source systems before treating them as true anomalies.",
            "Promote the strongest correlated dimensions into monitored KPIs or downstream feature engineering.",
        ]
    )
    if analysis.model and isinstance(analysis.model.metrics.get("r_squared"), (int, float)) and analysis.model.metrics.get("r_squared", 0) < 0.4:
        recommended_actions.append("Treat the current predictive model as exploratory only; more features or more rows may be needed.")
    if quality.row_count < 50:
        recommended_actions.append("Collect more history before relying on trend or regression output for decisions.")
    if quality.malformed_row_count or quality.duplicate_row_count:
        recommended_actions.append("Add source-side validation to prevent malformed or duplicate rows from entering the pipeline.")

    executive_summary = (
        f"Processed {quality.row_count:,} row(s) across {quality.column_count} column(s). "
        f"The quality report found {quality.missing_cell_count:,} missing cell(s), "
        f"{quality.duplicate_row_count:,} duplicate record(s), and {len(quality.outliers):,} potential outlier(s). "
        + (
            f"Strongest analytical signals include {key_drivers[0]}."
            if key_drivers
            else "No strong analytical signals crossed the configured thresholds."
        )
    )

    return InsightReport(
        executive_summary=executive_summary,
        key_drivers=key_drivers,
        risks=risks,
        recommended_actions=recommended_actions,
    )

