from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO, TextIOBase
import csv
from pathlib import Path
from typing import Any, BinaryIO

from .models import IngestedDataset, IngestionIssue, IssueSeverity, MalformedRow, PipelineConfig
from .utils import csv_dialect_name


class CSVIngestionError(ValueError):
    """Raised when the uploaded file cannot be accepted as a CSV."""


@dataclass(slots=True)
class EncodingDetectionResult:
    encoding: str
    text: str
    fallback_used: bool
    warning: str | None = None


def _read_source_bytes(source: str | Path | bytes | BinaryIO | TextIOBase, source_name: str | None) -> tuple[bytes, str]:
    if isinstance(source, bytes):
        return source, source_name or "uploaded.csv"
    if isinstance(source, Path):
        if not source.exists():
            raise CSVIngestionError(f"File not found: {source}")
        return source.read_bytes(), source.name
    if isinstance(source, str):
        path = Path(source)
        if path.exists():
            return path.read_bytes(), path.name
        raise CSVIngestionError(
            "CSV source must be a file path or bytes payload. "
            "If you are passing raw text, encode it to bytes first."
        )
    if isinstance(source, TextIOBase):
        payload = source.read()
        if isinstance(payload, str):
            return payload.encode("utf-8"), source_name or getattr(source, "name", "uploaded.csv")
        raise CSVIngestionError("Unexpected text stream payload.")
    if hasattr(source, "read"):
        payload = source.read()
        if isinstance(payload, str):
            return payload.encode("utf-8"), source_name or getattr(source, "name", "uploaded.csv")
        if isinstance(payload, bytes):
            return payload, source_name or getattr(source, "name", "uploaded.csv")
    raise CSVIngestionError("Unsupported CSV source type.")


def _validate_extension(source_name: str) -> None:
    suffix = Path(source_name).suffix.lower()
    if suffix != ".csv":
        raise CSVIngestionError(f"Unsupported file type '{suffix or '<none>'}'. Only .csv files are accepted.")


def _detect_encoding(raw_bytes: bytes, candidates: tuple[str, ...]) -> EncodingDetectionResult:
    bom_map = [
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ]
    for prefix, encoding in bom_map:
        if raw_bytes.startswith(prefix):
            return EncodingDetectionResult(encoding=encoding, text=raw_bytes.decode(encoding), fallback_used=False)

    for encoding in candidates:
        try:
            return EncodingDetectionResult(encoding=encoding, text=raw_bytes.decode(encoding), fallback_used=encoding != "utf-8")
        except UnicodeDecodeError:
            continue

    text = raw_bytes.decode("utf-8", errors="replace")
    warning = "Encoding fallback used: utf-8 with replacement characters."
    return EncodingDetectionResult(encoding="utf-8", text=text, fallback_used=True, warning=warning)


def _sniff_dialect(sample_text: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
    except csv.Error:
        return csv.get_dialect("excel")


def ingest_csv(
    source: str | Path | bytes | BinaryIO | TextIOBase,
    config: PipelineConfig,
    source_name: str | None = None,
) -> tuple[IngestedDataset, list[IngestionIssue]]:
    raw_bytes, derived_name = _read_source_bytes(source, source_name)
    effective_name = source_name or derived_name
    _validate_extension(effective_name)

    if len(raw_bytes) > config.max_file_size_bytes:
        raise CSVIngestionError(
            f"File '{effective_name}' is {len(raw_bytes):,} bytes, which exceeds the configured limit of "
            f"{config.max_file_size_bytes:,} bytes."
        )

    encoding_result = _detect_encoding(raw_bytes, config.encoding_candidates)
    if encoding_result.warning:
        warnings = [encoding_result.warning]
    else:
        warnings = []

    sample_text = encoding_result.text[:8_192]
    dialect = _sniff_dialect(sample_text)
    reader = csv.reader(StringIO(encoding_result.text), dialect)

    try:
        headers = next(reader)
    except StopIteration as exc:
        raise CSVIngestionError(f"CSV file '{effective_name}' is empty.") from exc

    headers = [header.strip() for header in headers]
    if not any(headers):
        raise CSVIngestionError(f"CSV file '{effective_name}' does not contain a usable header row.")

    rows: list[dict[str, Any]] = []
    malformed_rows: list[MalformedRow] = []
    row_limit_reached = False
    expected_fields = len(headers)

    for index, row in enumerate(reader, start=2):
        if config.max_rows and len(rows) >= config.max_rows:
            row_limit_reached = True
            warnings.append(
                f"Row limit reached at {config.max_rows:,} records. The input was truncated to keep processing bounded."
            )
            break

        if not row or all(not cell.strip() for cell in row):
            continue

        if len(row) != expected_fields:
            malformed_rows.append(
                MalformedRow(
                    line_number=index,
                    expected_fields=expected_fields,
                    actual_fields=len(row),
                    raw_preview=",".join(row[: min(len(row), 5)]),
                )
            )

        normalized_row: dict[str, Any] = {}
        for position, header in enumerate(headers):
            normalized_row[header] = row[position] if position < len(row) and row[position] != "" else None
        rows.append(normalized_row)

    if not rows:
        warnings.append("The CSV file contained a header row but no data rows.")

    dataset = IngestedDataset(
        source_name=effective_name,
        encoding=encoding_result.encoding,
        dialect=csv_dialect_name(dialect),
        headers=headers,
        rows=rows,
        malformed_rows=malformed_rows,
        warnings=warnings,
        truncated=row_limit_reached,
    )

    issues = [
        IngestionIssue(
            severity=IssueSeverity.WARNING if warning else IssueSeverity.INFO,
            code="encoding_warning" if encoding_result.fallback_used else "encoding_detected",
            message=warning or f"Detected encoding '{encoding_result.encoding}'.",
        )
        for warning in warnings
        if warning
    ]
    return dataset, issues

