# UVP CSV Analytics Toolkit

Analyze a local CSV and produce a data-quality report, exploratory statistics, SVG charts and a self-contained HTML report. The runtime uses Python's standard library and makes no network calls.

## Quick start

Requires Python 3.11 or newer. From a checkout:

```bash
python run_pipeline.py examples/synthetic_sales.csv --output-dir outputs/demo
```

Or install in a virtual environment:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install .
csv-analytics examples/synthetic_sales.csv --output-dir outputs/demo
```

Open `outputs/demo/report.html`. JSON reports and `charts/*.svg` are saved alongside it. All example rows are fictional.

## Filters and limits

```bash
csv-analytics examples/synthetic_sales.csv --filter 'amount>=100' --chart-limit 3
csv-analytics examples/synthetic_sales.csv --filter 'amount=150' --chart-limit 0
```

Quote filter expressions so the shell does not interpret comparison operators. Filters operate on cleaned values; categories are lowercase by default. Numeric and boolean equality are type-aware. Use `python run_pipeline.py --help` for all options. Size, row and inference-sample limits must be positive; zero charts disables chart output. Headers must be nonblank and unique after trimming. Malformed rows are recorded in the quality report; extra cells are discarded and absent cells become null.

## Python API

```python
from pathlib import Path
from csv_analytics_pipeline import PipelineConfig, run_pipeline
result = run_pipeline(Path('examples/synthetic_sales.csv'), PipelineConfig(output_directory=Path('outputs/api')))
if not result.success:
    raise RuntimeError(result.errors)
print(result.output_paths['report_html'])
```

## Interpretation and privacy

- Schema types and profile counts use an inference sample (default first 5,000 rows). Overall row/missing/duplicate counts cover all ingested rows. Review mixed or sparse columns before relying on results.
- Missing numeric values default to median filling; categories and dates default to mode filling. Inspect `cleaning_report.json` and its before/after transformation log. A filled value is an estimate.
- Analysis uses cleaned, filtered rows. Correlations do not establish causes. Regression metrics are measured on the same data used to fit the model; no held-out validation or forecasting guarantee is provided.
- Dates with mixed timezone-aware and naive timestamps should be normalized before ingestion. Currency parsing accepts $, £, €, ¥, grouped thousands, percentages and accounting parentheses; locale-specific decimal commas and units are not supported.
- Processing is in memory, with a 25 MB / 100,000-row default cap. Wide datasets can consume substantially more memory and CPU. This is an exploratory local tool, not an untrusted multi-tenant upload service.
- Reports and logs can contain source values, categories, file paths and transformation details. Local processing does not anonymize data. Review outputs before sharing; they are excluded from Git by default.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Tests and a pinned secret-scanning workflow run on pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [LICENSE](LICENSE). The source is public; an open-source license has not yet been granted.
