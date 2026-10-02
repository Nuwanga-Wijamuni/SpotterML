"""Generate the 12,000-load Spotter submission using the saved main model.

Run from the project root: python -m src.predict
The output contains exactly load_id,predicted_rate in template row order.
Rates are total dollars rounded to cents. Source inputs are never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data import DataValidationError, FEATURE_COLUMNS, TARGET_COLUMN, clean_frame, find_file, read_csv, validate_ids
from .train import MODEL_FILENAMES, load_model, predict_model


EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{i:06d}" for i in range(1, EXPECTED_ROWS + 1)}
SUBMISSION_COLUMNS = ["load_id", "predicted_rate"]


def validate_submission(frame: pd.DataFrame) -> None:
    """Check the assessment's required schema, identifiers and positive rates."""
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise DataValidationError("Submission must contain exactly load_id,predicted_rate, in order")
    if len(frame) != EXPECTED_ROWS or frame.load_id.isna().any() or not frame.load_id.is_unique:
        raise DataValidationError("Submission requires exactly 12,000 unique nonmissing load IDs")
    if set(frame.load_id.astype(str)) != EXPECTED_IDS:
        raise DataValidationError("Submission IDs must be TE-000001 through TE-012000")
    rates = pd.to_numeric(frame.predicted_rate, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise DataValidationError("Submission rates must be finite and positive")


def protect_output(output: Path, protected_paths: list[Path]) -> None:
    """Prevent command-line output choices from replacing source inputs/models."""
    for source in protected_paths:
        same_path = output.resolve() == source.resolve()
        same_file = output.exists() and source.exists() and os.path.samefile(output, source)
        if same_path or same_file:
            raise DataValidationError(f"Output must be separate from input/model: {source}")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Verify the CSV round trip before replacing an existing output file."""
    if path.suffix.lower() != ".csv":
        raise DataValidationError("Prediction output must be a .csv file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False, float_format="%.2f")
        saved = pd.read_csv(temporary, dtype={"load_id": "string"})
        if list(saved.columns) != list(frame.columns) or len(saved) != len(frame):
            raise DataValidationError("Saved CSV row count or column order changed")
        for column in frame:
            if pd.api.types.is_numeric_dtype(frame[column]):
                expected = frame[column].to_numpy(dtype=float)
                actual = pd.to_numeric(saved[column], errors="coerce").to_numpy(dtype=float)
                matches = np.allclose(expected, actual, rtol=0, atol=1e-8, equal_nan=True)
            else:
                expected = frame[column].astype("string").fillna("").reset_index(drop=True)
                actual = saved[column].astype("string").fillna("").reset_index(drop=True)
                matches = expected.equals(actual)
            if not matches:
                raise DataValidationError(f"Saved CSV values changed unexpectedly in {column}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_prediction_inputs(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the final prediction features and template, without training."""
    validation_path = find_file(project, "validation")
    validation = read_csv(validation_path, ["load_id", *FEATURE_COLUMNS])
    if TARGET_COLUMN in validation:
        raise DataValidationError("validation.csv should be unlabeled; found posted_rate")
    validation = clean_frame(validation, "validation", [])
    template = read_csv(find_file(project, "template"), SUBMISSION_COLUMNS, exact=True)
    validate_ids(template, "template")
    for label, frame in [("validation", validation), ("template", template)]:
        if len(frame) != EXPECTED_ROWS or set(frame.load_id.astype(str)) != EXPECTED_IDS:
            raise DataValidationError(f"{label}: expected the 12,000 assessment IDs")
    return validation, template


def make_submission(validation: pd.DataFrame, template: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, int]:
    """Match predictions by ID, so differing CSV row orders cannot misalign them."""
    if bundle.get("role") != "main":
        raise DataValidationError("Use main_model.joblib for final validation predictions")
    validation, template = validation.copy(), template.copy()
    validate_ids(validation, "validation")
    validate_ids(template, "template")
    if list(template.columns) != SUBMISSION_COLUMNS:
        raise DataValidationError("Prediction template has unexpected columns")
    if len(validation) != EXPECTED_ROWS or set(validation.load_id.astype(str)) != EXPECTED_IDS:
        raise DataValidationError("Validation input IDs do not match the assessment")
    if len(template) != EXPECTED_ROWS or set(template.load_id.astype(str)) != EXPECTED_IDS:
        raise DataValidationError("Template IDs do not match the assessment")
    rates, clipped = predict_model(bundle, validation)
    by_id = pd.Series(rates, index=validation.load_id)
    result = pd.DataFrame({"load_id": template.load_id.to_numpy(),
                           "predicted_rate": np.round(by_id.reindex(template.load_id).to_numpy(), 2)})
    validate_submission(result)
    return result, clipped


def run_prediction(project: Path, model_path: Path, output: Path) -> pd.DataFrame:
    protected = [model_path, *(project / "models" / name for name in MODEL_FILENAMES.values())]
    for dataset in ("development", "validation", "template", "december"):
        try:
            protected.append(find_file(project, dataset))
        except FileNotFoundError:
            pass
    protect_output(output, protected)
    validation, template = load_prediction_inputs(project)
    bundle = load_model(model_path)
    result, clipped = make_submission(validation, template, bundle)
    save_csv(result, output)
    validate_submission(pd.read_csv(output, dtype={"load_id": "string"}))
    print(f"Loaded main model: {bundle['experiment']}")
    print(f"Predicted and validated {len(result):,} loads in template row order.")
    print(f"Predicted rate range: ${result.predicted_rate.min():,.2f} to ${result.predicted_rate.max():,.2f}")
    print(f"Predictions adjusted to the saved positive-rate floor: {clipped}")
    print(f"Saved: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, help="Default: PROJECT/models/main_model.joblib")
    parser.add_argument("--output", type=Path, help="Default: PROJECT/outputs/validation_predictions.csv")
    args = parser.parse_args()
    project = args.project_dir.expanduser().resolve()
    model = (args.model or project / "models" / MODEL_FILENAMES["main"]).expanduser().resolve()
    output = (args.output or project / "outputs/validation_predictions.csv").expanduser().resolve()
    try:
        run_prediction(project, model, output)
    except (DataValidationError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
