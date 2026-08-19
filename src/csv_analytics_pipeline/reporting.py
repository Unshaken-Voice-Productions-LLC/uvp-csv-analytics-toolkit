from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json

from .models import AnalysisReport, CleaningReport, DataQualityReport, InsightReport, PipelineResult, VisualizationReport
from .utils import dump_json, to_jsonable


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.2e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, dict):
        return escape(json.dumps(to_jsonable(value), ensure_ascii=False))
    return escape(str(value))


def _render_table(headers: list[str], rows: list[list[Any]], empty_message: str) -> str:
    if not rows:
        return f'<p class="empty">{escape(empty_message)}</p>'
    head_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_html = []
    for row in rows:
        body_html.append("<tr>" + "".join(f"<td>{_format_value(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body_html)}</tbody></table>"


def _render_list(items: list[str], empty_message: str) -> str:
    if not items:
        return f'<p class="empty">{escape(empty_message)}</p>'
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def build_html_report(result: PipelineResult) -> str:
    quality = result.quality
    cleaning = result.cleaning
    analysis = result.analysis
    visualization = result.visualization
    insights = result.insights

    rows = quality.row_count if quality else 0
    columns = quality.column_count if quality else 0
    missing = quality.missing_cell_count if quality else 0
    duplicates = quality.duplicate_row_count if quality else 0
    outliers = len(quality.outliers) if quality else 0
    malformed = quality.malformed_row_count if quality else 0
    charts = len(visualization.charts) if visualization else 0

    profile_rows = []
    if quality:
        for profile in quality.profiles:
            profile_rows.append(
                [
                    profile.name,
                    profile.detected_type.value,
                    f"{profile.confidence:.2f}",
                    profile.non_missing_count,
                    profile.missing_count,
                    profile.invalid_count,
                    profile.distinct_count,
                    ", ".join(str(value) for value in profile.sample_values[:3]),
                ]
            )

    issue_rows = []
    if quality:
        for issue in quality.issues[:20]:
            issue_rows.append(
                [
                    issue.severity.value,
                    issue.code,
                    issue.column or "",
                    issue.message,
                ]
            )

    stat_rows = []
    if analysis:
        for column, stats in analysis.summary_statistics.items():
            stat_rows.append(
                [
                    column,
                    stats.get("count"),
                    stats.get("mean"),
                    stats.get("median"),
                    stats.get("stddev"),
                    stats.get("min"),
                    stats.get("max"),
                ]
            )

    correlation_rows = []
    if analysis:
        for corr in analysis.correlations[:10]:
            correlation_rows.append([corr.column_a, corr.column_b, corr.correlation, corr.pair_count])

    trend_rows = []
    if analysis:
        for trend in analysis.trends[:10]:
            trend_rows.append([trend.date_column, trend.value_column, trend.slope, trend.direction, trend.r_squared, ", ".join(map(str, trend.anomaly_row_numbers))])

    model_rows = []
    if analysis and analysis.model:
        model_rows.append(["Model", analysis.model.model_type])
        model_rows.append(["Target", analysis.model.target_column])
        model_rows.append(["Features", ", ".join(analysis.model.feature_columns)])
        for key, value in analysis.model.metrics.items():
            model_rows.append([key, value])

    chart_html = []
    if visualization:
        for chart in visualization.charts:
            chart_html.append(
                f'<figure class="chart"><figcaption><strong>{escape(chart.title)}</strong><br/><span>{escape(chart.description)}</span></figcaption>{chart.svg}</figure>'
            )

    error_block = _render_list(result.errors, "No errors were recorded.")
    warning_block = _render_list(result.warnings, "No warnings were recorded.")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CSV Analytics Report - {escape(result.source_name)}</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #111827;
      --panel-alt: #172033;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #34d399;
      --border: #243041;
      --warn: #fbbf24;
      --danger: #f87171;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: linear-gradient(180deg, #08101d 0%, #0f172a 100%);
      color: var(--text);
      line-height: 1.45;
    }}
    header {{
      padding: 32px 40px 20px;
      border-bottom: 1px solid var(--border);
      background: rgba(8, 16, 29, 0.9);
      position: sticky;
      top: 0;
      backdrop-filter: blur(12px);
      z-index: 1;
    }}
    h1, h2, h3 {{ margin: 0 0 8px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; margin-top: 0; }}
    .subtle {{ color: var(--muted); }}
    main {{ padding: 24px 40px 40px; max-width: 1440px; margin: 0 auto; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      margin: 16px 0 24px;
    }}
    .card {{
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-alt) 100%);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .card .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    section {{
      margin-bottom: 24px;
      background: rgba(17, 24, 39, 0.62);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(17, 24, 39, 0.4);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
      font-size: 14px;
    }}
    th {{ background: rgba(56, 189, 248, 0.1); color: #dbeafe; position: sticky; top: 0; }}
    tr:nth-child(even) td {{ background: rgba(255, 255, 255, 0.02); }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin-bottom: 6px; }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
    figure.chart {{
      margin: 0;
      padding: 14px;
      background: #f8fafc;
      border-radius: 16px;
      color: #0f172a;
      overflow: hidden;
    }}
    figure.chart figcaption {{ margin-bottom: 10px; }}
    .empty {{
      color: var(--muted);
      padding: 6px 0;
    }}
    .tag {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.15);
      color: #bae6fd;
      font-size: 12px;
      margin-right: 6px;
    }}
    .note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 720px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CSV Analytics Report</h1>
    <div class="subtle">{escape(result.source_name)}</div>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="label">Rows</div><div class="value">{rows:,}</div></div>
      <div class="card"><div class="label">Columns</div><div class="value">{columns}</div></div>
      <div class="card"><div class="label">Missing Cells</div><div class="value">{missing:,}</div></div>
      <div class="card"><div class="label">Duplicates</div><div class="value">{duplicates:,}</div></div>
      <div class="card"><div class="label">Outliers</div><div class="value">{outliers:,}</div></div>
      <div class="card"><div class="label">Malformed Rows</div><div class="value">{malformed:,}</div></div>
      <div class="card"><div class="label">Charts</div><div class="value">{charts}</div></div>
    </div>

    <section>
      <h2>Executive Summary</h2>
      <p>{escape(insights.executive_summary if insights else "No insight summary available.")}</p>
      <div class="grid">
        <div>
          <h3>Key Drivers</h3>
          {_render_list(insights.key_drivers if insights else [], "No key drivers identified.")}
        </div>
        <div>
          <h3>Risks</h3>
          {_render_list(insights.risks if insights else [], "No explicit risks identified.")}
        </div>
      </div>
      <div class="note">Recommended next actions are captured in the insight JSON artifact as structured guidance.</div>
    </section>

    <section>
      <h2>Run Diagnostics</h2>
      <div class="grid">
        <div>
          <h3>Warnings</h3>
          {warning_block}
        </div>
        <div>
          <h3>Errors</h3>
          {error_block}
        </div>
      </div>
    </section>

    <section>
      <h2>Recommended Actions</h2>
      {_render_list(insights.recommended_actions if insights else [], "No recommended actions generated.")}
    </section>

    <section>
      <h2>Data Quality</h2>
      {_render_table(["Severity", "Code", "Column", "Message"], issue_rows, "No validation issues were raised.")}
    </section>

    <section>
      <h2>Schema Profiles</h2>
      {_render_table(["Column", "Type", "Confidence", "Non-missing", "Missing", "Invalid", "Distinct", "Sample Values"], profile_rows, "No schema profiles available.")}
    </section>

    <section>
      <h2>Summary Statistics</h2>
      {_render_table(["Column", "Count", "Mean", "Median", "Std Dev", "Min", "Max"], stat_rows, "No numeric columns were available for summary statistics.")}
    </section>

    <section>
      <h2>Correlations</h2>
      {_render_table(["Column A", "Column B", "Correlation", "Pairs"], correlation_rows, "No correlations met the minimum threshold.")}
    </section>

    <section>
      <h2>Trends</h2>
      {_render_table(["Date Column", "Value Column", "Slope", "Direction", "R²", "Anomalies"], trend_rows, "No date-based trends were found.")}
    </section>

    <section>
      <h2>Predictive Model</h2>
      {_render_table(["Metric / Property", "Value"], model_rows, "No predictive model was generated for this dataset.")}
    </section>

    <section>
      <h2>Visualizations</h2>
      <div class="grid">
        {''.join(chart_html) if chart_html else '<p class="empty">No charts were generated.</p>'}
      </div>
    </section>

    <section>
      <h2>Runtime</h2>
      {_render_table(["Stage", "Seconds", "Status", "Message"], [[timing.stage, timing.duration_seconds, timing.status, timing.message or ""] for timing in result.timings], "No runtime timings recorded.")}
    </section>
  </main>
</body>
</html>
"""


def write_report_bundle(result: PipelineResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    paths: dict[str, str] = {}

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    if result.visualization:
        for chart in result.visualization.charts:
            chart_path = charts_dir / chart.filename
            chart_path.write_text(chart.svg, encoding="utf-8")
            paths[f"chart:{chart.filename}"] = str(chart_path.resolve())

    summary_payload = {
        "success": result.success,
        "source_name": result.source_name,
        "errors": result.errors,
        "warnings": result.warnings,
        "timings": result.timings,
        "quality": result.quality,
        "cleaning": result.cleaning,
        "analysis": result.analysis,
        "insights": result.insights,
        "output_paths": paths,
    }

    report_path = output_dir / "report.html"
    report_path.write_text(build_html_report(result), encoding="utf-8")
    paths["report_html"] = str(report_path.resolve())

    dump_json(output_dir / "pipeline_summary.json", summary_payload)
    dump_json(output_dir / "quality_report.json", result.quality)
    dump_json(output_dir / "cleaning_report.json", result.cleaning)
    dump_json(output_dir / "analysis_report.json", result.analysis)
    dump_json(output_dir / "insights.json", result.insights)

    paths["pipeline_summary_json"] = str((output_dir / "pipeline_summary.json").resolve())
    paths["quality_json"] = str((output_dir / "quality_report.json").resolve())
    paths["cleaning_json"] = str((output_dir / "cleaning_report.json").resolve())
    paths["analysis_json"] = str((output_dir / "analysis_report.json").resolve())
    paths["insights_json"] = str((output_dir / "insights.json").resolve())
    return paths
