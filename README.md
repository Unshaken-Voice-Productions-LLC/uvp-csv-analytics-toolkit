# CSV Analytics Pipeline

Modular, standard-library-first CSV analytics pipeline with:

- file ingestion and encoding checks
- schema inference and data quality reporting
- configurable cleaning and transformation logging
- summary analysis, correlations, segmentation, trend detection, and basic regression
- auto-generated SVG visualizations
- structured insight generation and HTML/JSON reporting

## Usage

```bash
python run_pipeline.py path\to\data.csv --output-dir outputs\run-001
```

## Architecture

- `ingestion.py` handles file validation, decoding, and malformed row capture.
- `validation.py` infers schema and builds the quality report.
- `cleaning.py` applies configurable missing-value and normalization strategies.
- `analysis.py` computes statistics, correlations, segments, trends, and a basic predictive model.
- `visualization.py` generates SVG charts and HTML-friendly artifacts.
- `insights.py` converts the analytical output into executive-level takeaways and next actions.
- `pipeline.py` orchestrates the stages and writes report artifacts.

The implementation intentionally avoids coupling the core logic to any UI so it can be reused in a CLI, web service, or scheduled batch job.
