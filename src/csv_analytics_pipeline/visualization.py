from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
import math
from typing import Any

from .models import AnalysisReport, ChartArtifact, ColumnProfile, ColumnType, PipelineConfig, VisualizationReport
from .utils import parse_date, parse_number, slugify_filename


PALETTE = {
    "primary": "#1D4ED8",
    "secondary": "#0F766E",
    "accent": "#B45309",
    "grid": "#D1D5DB",
    "text": "#111827",
    "muted": "#6B7280",
    "background": "#FFFFFF",
    "negative": "#B91C1C",
}


def _svg_header(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PALETTE["background"]}" rx="14" ry="14"/>',
        f'<text x="24" y="34" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="{PALETTE["text"]}">{escape(title)}</text>',
    ] + ([f'<text x="24" y="56" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{PALETTE["muted"]}">{escape(subtitle)}</text>'] if subtitle else [])


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _histogram_bins(values: list[float]) -> list[tuple[float, float, int]]:
    if not values:
        return []
    if len(values) == 1 or min(values) == max(values):
        v = values[0]
        return [(v - 0.5, v + 0.5, len(values))]
    bins = min(12, max(3, int(math.ceil(math.log2(len(values))) + 1)))
    low, high = min(values), max(values)
    width = (high - low) / bins
    histogram = [0] * bins
    for value in values:
        if value == high:
            index = bins - 1
        else:
            index = int((value - low) / width) if width > 0 else 0
        histogram[min(max(index, 0), bins - 1)] += 1
    result: list[tuple[float, float, int]] = []
    for index, count in enumerate(histogram):
        start = low + index * width
        end = start + width
        result.append((start, end, count))
    return result


def _format_tick(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.2e}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _histogram_svg(values: list[float], title: str, subtitle: str = "") -> str:
    width, height = 720, 420
    left, right, top, bottom = 64, 24, 72, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    bins = _histogram_bins(values)
    if not bins:
        return _placeholder_svg(title, "No numeric values available for histogram.")
    counts = [count for _, _, count in bins]
    max_count = max(counts) if counts else 1
    lines = _svg_header(width, height, title, subtitle)
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
    for index, (start, end, count) in enumerate(bins):
        bar_w = plot_w / len(bins) - 10
        x = left + index * (plot_w / len(bins)) + 5
        bar_h = 0 if max_count == 0 else (count / max_count) * plot_h
        y = height - bottom - bar_h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{PALETTE["primary"]}" rx="6" ry="6"/>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="{PALETTE["text"]}">{count}</text>'
        )
        label = f"{_format_tick(start)}-{_format_tick(end)}"
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - bottom + 18}" text-anchor="middle" font-family="Arial" font-size="10" fill="{PALETTE["muted"]}">{escape(label)}</text>'
        )
    lines.append(
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="12" fill="{PALETTE["muted"]}">Value bins</text>'
    )
    lines.append(f'<text transform="translate(18 {height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="12" fill="{PALETTE["muted"]}">Count</text>')
    lines.extend(_svg_footer())
    return "".join(lines)


def _bar_chart_svg(counts: list[tuple[str, int]], title: str, subtitle: str = "") -> str:
    width, height = 720, 420
    left, right, top, bottom = 180, 30, 72, 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not counts:
        return _placeholder_svg(title, "No categorical values available for bar chart.")
    max_count = max(count for _, count in counts)
    rows = len(counts)
    row_height = plot_h / max(1, rows)
    lines = _svg_header(width, height, title, subtitle)
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
    for index, (label, count) in enumerate(counts):
        y = top + index * row_height + 8
        bar_w = 0 if max_count == 0 else (count / max_count) * plot_w
        lines.append(
            f'<text x="{left - 12}" y="{y + 18:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="{PALETTE["text"]}">{escape(_truncate(label, 24))}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{min(22, row_height - 6):.1f}" fill="{PALETTE["secondary"]}" rx="5" ry="5"/>'
        )
        lines.append(
            f'<text x="{left + bar_w + 8:.1f}" y="{y + 16:.1f}" font-family="Arial" font-size="11" fill="{PALETTE["muted"]}">{count}</text>'
        )
    lines.append(
        f'<text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="12" fill="{PALETTE["muted"]}">Record count</text>'
    )
    lines.extend(_svg_footer())
    return "".join(lines)


def _line_chart_svg(points: list[tuple[str, float]], title: str, subtitle: str = "") -> str:
    width, height = 760, 420
    left, right, top, bottom = 72, 28, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not points:
        return _placeholder_svg(title, "No time-series values available for line chart.")
    values = [value for _, value in points]
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 1
        max_value += 1
    step_x = plot_w / max(1, len(points) - 1)
    lines = _svg_header(width, height, title, subtitle)
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
    point_coords: list[tuple[float, float, str]] = []
    for index, (label, value) in enumerate(points):
        x = left + index * step_x
        ratio = (value - min_value) / (max_value - min_value)
        y = height - bottom - ratio * plot_h
        point_coords.append((x, y, label))
    if len(point_coords) > 1:
        path = " ".join(f"L{x:.1f},{y:.1f}" for x, y, _ in point_coords[1:])
        lines.append(f'<path d="M{point_coords[0][0]:.1f},{point_coords[0][1]:.1f} {path}" fill="none" stroke="{PALETTE["primary"]}" stroke-width="2.5"/>')
    for index, (x, y, label) in enumerate(point_coords):
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{PALETTE["accent"]}"/>')
        if len(point_coords) <= 12 or index % max(1, len(point_coords) // 8) == 0:
            lines.append(
                f'<text x="{x:.1f}" y="{height - bottom + 18}" text-anchor="middle" font-family="Arial" font-size="10" fill="{PALETTE["muted"]}">{escape(_truncate(label, 14))}</text>'
            )
    lines.append(
        f'<text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="12" fill="{PALETTE["muted"]}">Timeline</text>'
    )
    lines.extend(_svg_footer())
    return "".join(lines)


def _scatter_svg(points: list[tuple[float, float]], title: str, subtitle: str = "") -> str:
    width, height = 720, 420
    left, right, top, bottom = 72, 30, 70, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not points:
        return _placeholder_svg(title, "No paired numeric values available for scatter plot.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    lines = _svg_header(width, height, title, subtitle)
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
    for x_value, y_value in points:
        x = left + ((x_value - min_x) / (max_x - min_x)) * plot_w
        y = height - bottom - ((y_value - min_y) / (max_y - min_y)) * plot_h
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{PALETTE["secondary"]}" opacity="0.85"/>')
    lines.append(f'<text x="{left}" y="{height - 18}" font-family="Arial" font-size="12" fill="{PALETTE["muted"]}">{_format_tick(min_x)}</text>')
    lines.append(f'<text x="{width - right}" y="{height - 18}" text-anchor="end" font-family="Arial" font-size="12" fill="{PALETTE["muted"]}">{_format_tick(max_x)}</text>')
    lines.append(f'<text transform="translate(16 {height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="12" fill="{PALETTE["muted"]}">{_format_tick(max_y)} to {_format_tick(min_y)}</text>')
    lines.extend(_svg_footer())
    return "".join(lines)


def _placeholder_svg(title: str, message: str) -> str:
    width, height = 720, 300
    lines = _svg_header(width, height, title)
    lines.append(
        f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" fill="{PALETTE["muted"]}">{escape(message)}</text>'
    )
    lines.extend(_svg_footer())
    return "".join(lines)


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


def _numeric_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        number = parse_number(row.get(column))
        if number is not None:
            values.append(number)
    return values


def _categorical_counts(rows: list[dict[str, Any]], column: str) -> list[tuple[str, int]]:
    counter = Counter(str(row.get(column)) for row in rows if row.get(column) is not None)
    return counter.most_common(10)


def _trend_points(rows: list[dict[str, Any]], date_column: str, value_column: str) -> list[tuple[str, float]]:
    paired: list[tuple[str, float]] = []
    for row in rows:
        date_value = parse_date(row.get(date_column))
        numeric_value = parse_number(row.get(value_column))
        if date_value is None or numeric_value is None:
            continue
        paired.append((date_value.isoformat()[:10], numeric_value))
    paired.sort(key=lambda item: item[0])
    return paired


def _correlation_points(rows: list[dict[str, Any]], column_a: str, column_b: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        left = parse_number(row.get(column_a))
        right = parse_number(row.get(column_b))
        if left is None or right is None:
            continue
        points.append((left, right))
    return points


def build_visualization_report(
    rows: list[dict[str, Any]],
    profiles: list[ColumnProfile],
    analysis: AnalysisReport,
    config: PipelineConfig,
) -> VisualizationReport:
    charts: list[ChartArtifact] = []
    numeric_profiles = [profile for profile in profiles if profile.detected_type == ColumnType.NUMERIC]
    categorical_profiles = [profile for profile in profiles if profile.detected_type in {ColumnType.CATEGORICAL, ColumnType.BOOLEAN}]
    date_profiles = [profile for profile in profiles if profile.detected_type == ColumnType.DATE]

    for profile in numeric_profiles[: max(1, min(3, config.chart_limit))]:
        values = _numeric_values(rows, profile.name)
        if not values:
            continue
        average = sum(values) / len(values)
        skew_note = "Skewed distribution likely" if average and (max(values) / max(0.001, abs(average)) > 10) else ""
        charts.append(
            ChartArtifact(
                title=f"Distribution of {profile.name}",
                kind="histogram",
                filename=f"{slugify_filename(profile.name)}-histogram.svg",
                svg=_histogram_svg(values, f"Distribution of {profile.name}", skew_note),
                description=f"Histogram for numeric column '{profile.name}'.",
            )
        )

    for profile in categorical_profiles[: max(1, min(2, config.chart_limit - len(charts)))]:
        counts = _categorical_counts(rows, profile.name)
        if not counts:
            continue
        charts.append(
            ChartArtifact(
                title=f"Top values for {profile.name}",
                kind="bar",
                filename=f"{slugify_filename(profile.name)}-bar.svg",
                svg=_bar_chart_svg(counts, f"Top values for {profile.name}"),
                description=f"Category frequency chart for '{profile.name}'.",
            )
        )

    if analysis.trends and len(charts) < config.chart_limit:
        trend = analysis.trends[0]
        points = _trend_points(rows, trend.date_column, trend.value_column)
        if points:
            charts.append(
                ChartArtifact(
                    title=f"Trend: {trend.value_column} by {trend.date_column}",
                    kind="line",
                    filename=f"{slugify_filename(trend.value_column)}-trend.svg",
                    svg=_line_chart_svg(points, f"Trend: {trend.value_column} by {trend.date_column}", f"R² = {trend.r_squared:.2f}"),
                    description=f"Time-series trend for '{trend.value_column}' over '{trend.date_column}'.",
                )
            )

    if analysis.correlations and len(charts) < config.chart_limit:
        strongest = analysis.correlations[0]
        points = _correlation_points(rows, strongest.column_a, strongest.column_b)
        if points:
            charts.append(
                ChartArtifact(
                    title=f"Scatter: {strongest.column_a} vs {strongest.column_b}",
                    kind="scatter",
                    filename=f"{slugify_filename(strongest.column_a)}-vs-{slugify_filename(strongest.column_b)}.svg",
                    svg=_scatter_svg(points, f"{strongest.column_a} vs {strongest.column_b}", f"Correlation = {strongest.correlation:.2f}"),
                    description=f"Scatter plot for the strongest numeric correlation pair.",
                )
            )

    return VisualizationReport(charts=charts[: config.chart_limit])

