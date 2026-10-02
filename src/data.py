"""Load and validate the Spotter freight-rate assessment data.

Run from the project root with ``python -m src.data``.
Source CSVs are never modified. Missing values are not imputed here;
fit imputation and other learned preprocessing on each training split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "posted_rate"
ID_COLUMN = "load_id"
FEATURE_COLUMNS = [
    "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "date",
    "market_index", "quote_signal",
]
NUMERIC_COLUMNS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "market_index", "quote_signal",
]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
DECEMBER_COLUMNS = [
    "pickup", "delivery", "distance", "equipment", "weight", "date",
    "predicted_rate",
]
FILE_NAMES = {
    "development": ("train-test.csv", "train_test.csv"),
    "validation": ("validation.csv",),
    "template": (
        "validation-predictions-template.csv", "validation_predictions_template.csv",
    ),
    "december": ("december-chart-inputs.csv", "december_chart_inputs.csv"),
}


class DataValidationError(ValueError):
    """A required input is missing or violates the assessment contract."""


@dataclass
class ProjectData:
    development: pd.DataFrame
    validation: pd.DataFrame
    template: pd.DataFrame
    december: pd.DataFrame
    quality_report: pd.DataFrame


def find_file(project_dir: Path, dataset: str) -> Path:
    """Accept root-level files or files in data/, with either naming style."""
    for folder in (project_dir, project_dir / "data"):
        for name in FILE_NAMES[dataset]:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    names = ", ".join(FILE_NAMES[dataset])
    raise FileNotFoundError(f"Missing {dataset} CSV in {project_dir} or data/: {names}")


def read_csv(path: Path, columns: list[str], *, exact: bool = False) -> pd.DataFrame:
    """Read identifiers as text and reject missing or unexpected required headers."""
    try:
        frame = pd.read_csv(path, dtype={ID_COLUMN: "string"})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not read {path.name}: {exc}") from exc
    missing = set(columns) - set(frame.columns)
    if missing:
        raise DataValidationError(f"{path.name}: missing columns {sorted(missing)}")
    if exact and list(frame.columns) != columns:
        raise DataValidationError(f"{path.name}: columns must be exactly {columns} in order")
    if frame.empty:
        raise DataValidationError(f"{path.name}: no data rows")
    return frame.copy()


def validate_ids(frame: pd.DataFrame, label: str) -> None:
    """Normalize surrounding whitespace and reject missing or duplicate IDs."""
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("string").str.strip().replace("", pd.NA)
    if frame[ID_COLUMN].isna().any():
        raise DataValidationError(f"{label}: missing load_id values")
    duplicates = frame[ID_COLUMN].duplicated()
    if duplicates.any():
        examples = frame.loc[duplicates, ID_COLUMN].head(3).tolist()
        raise DataValidationError(f"{label}: duplicate load_id values, e.g. {examples}")


def clean_frame(
    source: pd.DataFrame, label: str, issues: list[dict], *, labeled: bool = False
) -> pd.DataFrame:
    """Apply fixed parsing rules without learning anything from the dataset."""
    frame = source.copy()

    def record(column: str, issue: str, mask: pd.Series) -> None:
        count = int(mask.fillna(False).sum())
        if count:
            issues.append({"dataset": label, "column": column, "issue": issue, "rows": count})

    if ID_COLUMN in frame:
        validate_ids(frame, label)
    for column in CATEGORICAL_COLUMNS:
        original = frame[column].astype("string")
        stripped = original.str.strip()
        record(column, "surrounding whitespace removed", original.ne(stripped))
        frame[column] = stripped.replace("", pd.NA)
        record(column, "missing category", frame[column].isna())

    numeric = [c for c in NUMERIC_COLUMNS if c in frame]
    if labeled:
        numeric.append(TARGET_COLUMN)
    for column in numeric:
        original = frame[column]
        values = pd.to_numeric(original, errors="coerce").astype(float)
        record(column, "nonnumeric value converted to missing", original.notna() & values.isna())
        record(column, "non-finite value converted to missing", np.isinf(values))
        frame[column] = values.replace([np.inf, -np.inf], np.nan)

    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    if frame["date"].isna().any():
        raise DataValidationError(f"{label}: missing or invalid dates; expected YYYY-MM-DD")
    if not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise DataValidationError(f"{label}: dates must contain calendar days, without times")
    if frame["distance"].isna().any() or frame["distance"].le(0).any():
        raise DataValidationError(f"{label}: distance must be finite and positive for every load")
    if labeled and (frame[TARGET_COLUMN].isna().any() or frame[TARGET_COLUMN].le(0).any()):
        raise DataValidationError(f"{label}: posted_rate must be finite and positive for every load")

    bad_weight = frame["weight"].le(0)
    record("weight", "nonpositive weight converted to missing", bad_weight)
    frame.loc[bad_weight, "weight"] = np.nan
    for column, limit in [("pickup_lat", 90), ("delivery_lat", 90),
                          ("pickup_lon", 180), ("delivery_lon", 180)]:
        if column in frame:
            invalid = frame[column].abs().gt(limit)
            record(column, "coordinate outside valid range converted to missing", invalid)
            frame.loc[invalid, column] = np.nan

    for column in numeric:
        record(column, "missing after cleaning", frame[column].isna())
    if all(c in frame for c in FEATURE_COLUMNS):
        record("all features", "repeated feature row retained", frame.duplicated(subset=FEATURE_COLUMNS))
    return frame.reset_index(drop=True)


def validate_december(frame: pd.DataFrame) -> None:
    """Verify the fixed scenario without requiring predictions to be filled."""
    dates = set(pd.date_range("2025-12-01", "2025-12-31"))
    if len(frame) != 31 or not frame["date"].is_unique or set(frame["date"]) != dates:
        raise DataValidationError("December: expected one row per day of December 2025")
    for column, expected in {"pickup": "Lexington", "delivery": "Fort Wayne",
                             "equipment": "Dry Van"}.items():
        if not frame[column].eq(expected).fillna(False).all():
            raise DataValidationError(f"December: {column} must be {expected!r} on every row")
    for column, expected in {"distance": 360.0, "weight": 32000.0}.items():
        if not np.isclose(frame[column], expected).all():
            raise DataValidationError(f"December: {column} must be {expected:g} on every row")


def load_project_data(project_dir: str | Path | None = None) -> ProjectData:
    """Load all four inputs, preserving row order and blank prediction fields.

    Default root is the parent of src/ when installed in the Spotter project.
    Pass project_dir explicitly when importing this file from another location.
    The template and December predicted_rate columns are placeholders; output
    validation belongs to the supplied score.py after predictions are generated.
    """
    root = (Path(project_dir).expanduser().resolve() if project_dir is not None
            else Path(__file__).resolve().parents[1])
    issues: list[dict] = []
    development = clean_frame(
        read_csv(find_file(root, "development"), [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]),
        "development", issues, labeled=True,
    )
    validation = clean_frame(
        read_csv(find_file(root, "validation"), [ID_COLUMN, *FEATURE_COLUMNS]),
        "validation", issues,
    )
    if TARGET_COLUMN in validation:
        raise DataValidationError("validation.csv must be unlabeled; found posted_rate")
    template = read_csv(find_file(root, "template"), [ID_COLUMN, "predicted_rate"], exact=True)
    validate_ids(template, "template")
    expected = {f"TE-{i:06d}" for i in range(1, 12001)}
    for name, frame in [("validation", validation), ("template", template)]:
        ids = set(frame[ID_COLUMN])
        if len(frame) != 12000 or ids != expected:
            raise DataValidationError(
                f"{name}: expected 12,000 IDs TE-000001 through TE-012000; "
                f"missing={len(expected - ids)}, extra={len(ids - expected)}"
            )
    if set(development[ID_COLUMN]) & set(validation[ID_COLUMN]):
        raise DataValidationError("Development and validation load_id values overlap")
    december = clean_frame(
        read_csv(find_file(root, "december"), DECEMBER_COLUMNS, exact=True),
        "december", issues,
    )
    validate_december(december)
    report = pd.DataFrame(issues, columns=["dataset", "column", "issue", "rows"])
    return ProjectData(development, validation, template.reset_index(drop=True), december, report)


def split_development_by_date(
    development: pd.DataFrame, holdout_start: str | pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return earlier rows and a holdout, defaulting to the last labeled month.

    This is a split proposal, not an automatic model-selection decision.
    Review coverage first, and use earlier folds for tuning.
    """
    if not pd.api.types.is_datetime64_any_dtype(development["date"]):
        raise DataValidationError("Parse dates with load_project_data before splitting")
    if development.empty or development["date"].isna().any():
        raise DataValidationError("Cannot split empty development data or missing dates")
    cutoff = (development["date"].max().to_period("M").start_time
              if holdout_start is None else pd.Timestamp(holdout_start))
    if pd.isna(cutoff) or cutoff.tzinfo is not None:
        raise DataValidationError("holdout_start must be a valid timezone-free date")
    earlier = development.loc[development["date"] < cutoff].copy()
    holdout = development.loc[development["date"] >= cutoff].copy()
    if earlier.empty or holdout.empty:
        raise DataValidationError("The split must leave at least one row on each side")
    return earlier, holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, help="Folder containing the assignment CSVs")
    args = parser.parse_args()
    data = load_project_data(args.project_dir)
    for label in ("development", "validation", "december"):
        frame = getattr(data, label)
        print(f"{label}: {len(frame):,} rows, "
              f"{frame.date.min().date()} through {frame.date.max().date()}")
    print("Validated 12,000 submission IDs and 31 fixed December inputs.")
    print("Cleaning report (issue counts may overlap):")
    print(data.quality_report.to_string(index=False) if not data.quality_report.empty else "No issues found.")


if __name__ == "__main__":
    main()
