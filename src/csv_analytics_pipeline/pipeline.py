from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import logging
import time
from typing import Any, Callable

from .analysis import analyze_dataset
from .cleaning import clean_dataset
from .ingestion import CSVIngestionError, ingest_csv
from .insights import build_insights
from .models import PipelineConfig, PipelineResult, StageTiming
from .reporting import write_report_bundle
from .utils import apply_filters
from .validation import build_quality_report
from .visualization import build_visualization_report


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"csv-analytics-{stamp}"


def _configure_logger(output_dir: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("csv_analytics_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logger.level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger.level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass


def _record_stage(result: PipelineResult, stage_name: str, func: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    try:
        value = func()
        duration = time.perf_counter() - start
        result.timings.append(StageTiming(stage=stage_name, duration_seconds=duration, status="ok"))
        return value
    except Exception as exc:
        duration = time.perf_counter() - start
        result.timings.append(StageTiming(stage=stage_name, duration_seconds=duration, status="error", message=str(exc)))
        raise


def _derive_source_name(source: str | Path | bytes | Any, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    if isinstance(source, Path):
        return source.name
    if isinstance(source, str):
        path = Path(source)
        if path.exists():
            return path.name
        return "uploaded.csv"
    return "uploaded.csv"


def run_pipeline(
    source: str | Path | bytes | Any,
    config: PipelineConfig | None = None,
    source_name: str | None = None,
) -> PipelineResult:
    config = config or PipelineConfig()
    output_dir = (config.output_directory or _default_output_dir()).resolve()
    logger = _configure_logger(output_dir, config.log_level)
    inferred_source_name = _derive_source_name(source, source_name)
    result = PipelineResult(
        success=False,
        source_name=inferred_source_name,
        ingestion=None,
        quality=None,
        cleaning=None,
        analysis=None,
        visualization=None,
        insights=None,
    )

    logger.info("Starting CSV analytics pipeline for %s", inferred_source_name)

    try:
        dataset, ingest_issues = _record_stage(
            result,
            "ingestion",
            lambda: ingest_csv(source, config, source_name=source_name),
        )
        result.ingestion = dataset
        result.warnings.extend(dataset.warnings)
        result.warnings.extend(issue.message for issue in ingest_issues if issue.message)
        for issue in ingest_issues:
            if issue.severity.value == "error":
                result.errors.append(issue.message)

        quality = _record_stage(result, "validation", lambda: build_quality_report(dataset, config))
        result.quality = quality

        cleaned_rows, cleaning_report = _record_stage(
            result,
            "cleaning",
            lambda: clean_dataset(dataset.rows, quality.profiles, config),
        )
        result.cleaning = cleaning_report

        filtered_rows = apply_filters(cleaned_rows, config.slice_filters)
        if config.slice_filters:
            result.warnings.append(
                f"Applied {len(config.slice_filters)} slice filter(s) before analysis and visualization."
            )

        analysis = _record_stage(result, "analysis", lambda: analyze_dataset(filtered_rows, quality.profiles, config))
        result.analysis = analysis

        visualization = _record_stage(
            result,
            "visualization",
            lambda: build_visualization_report(filtered_rows, quality.profiles, analysis, config),
        )
        result.visualization = visualization

        insights = _record_stage(result, "insights", lambda: build_insights(quality, analysis, config))
        result.insights = insights

        result.output_paths = _record_stage(result, "reporting", lambda: write_report_bundle(result, output_dir))
        result.success = True
        logger.info("Pipeline completed successfully. Output written to %s", output_dir)
    except CSVIngestionError as exc:
        result.errors.append(str(exc))
        logger.error("Ingestion failed: %s", exc)
    except Exception as exc:
        result.errors.append(str(exc))
        logger.exception("Pipeline failed")
    finally:
        if not result.output_paths:
            try:
                result.output_paths = write_report_bundle(result, output_dir)
            except Exception as exc:
                result.warnings.append(f"Could not write report bundle: {exc}")
                logger.exception("Report bundle write failed")
        result.output_paths["output_dir"] = str(output_dir)
        _close_logger(logger)
    return result
