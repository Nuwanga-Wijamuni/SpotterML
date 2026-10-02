"""Predict the 31 fixed December loads with the separate scenario model.

From the project root: python -m src.predict_december
Only predicted_rate is filled; the fixed inputs and their row order are kept.
Output rates are total dollars rounded to cents. The source CSV is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DECEMBER_COLUMNS, DataValidationError, clean_frame, find_file, read_csv, validate_december
from .predict import protect_output, save_csv
from .train import MODEL_FILENAMES, load_model, predict_model


def make_december_predictions(source: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, int]:
    """Use only scenario-available features and preserve all six fixed inputs."""
    if bundle.get("role") != "december_scenario":
        raise DataValidationError("Use december_model.joblib for the December scenario")
    if any(bundle["feature_config"].values()):
        raise DataValidationError("December model must exclude market, quote and geographic features")
    if list(source.columns) != DECEMBER_COLUMNS:
        raise DataValidationError("December input must retain its original seven columns in order")
    inputs = clean_frame(source, "december", [])
    validate_december(inputs)
    rates, clipped = predict_model(bundle, inputs)
    # Copy the original frame so no fixed input or source ordering is changed.
    result = source.copy().reset_index(drop=True)
    result["predicted_rate"] = np.round(rates, 2)
    values = result.predicted_rate.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataValidationError("December rates must be finite and positive after rounding")
    return result, clipped


def run_december_prediction(project: Path, model_path: Path, output: Path) -> pd.DataFrame:
    protected = [model_path, *(project / "models" / name for name in MODEL_FILENAMES.values())]
    for dataset in ("development", "validation", "template", "december"):
        try:
            protected.append(find_file(project, dataset))
        except FileNotFoundError:
            pass
    protect_output(output, protected)
    source = read_csv(find_file(project, "december"), DECEMBER_COLUMNS, exact=True)
    bundle = load_model(model_path)
    result, clipped = make_december_predictions(source, bundle)
    save_csv(result, output)
    saved = pd.read_csv(output)
    parsed = saved.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], format="%Y-%m-%d", errors="coerce")
    validate_december(parsed)
    print(f"Loaded December model: {bundle['experiment']}")
    print("Predicted and validated 31 fixed December loads.")
    print(f"Predicted rate range: ${result.predicted_rate.min():,.2f} to ${result.predicted_rate.max():,.2f}")
    print(f"Predictions adjusted to the saved positive-rate floor: {clipped}")
    print(f"Saved: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, help="Default: PROJECT/models/december_model.joblib")
    parser.add_argument("--output", type=Path, help="Default: PROJECT/outputs/december_predictions.csv")
    args = parser.parse_args()
    project = args.project_dir.expanduser().resolve()
    model = (args.model or project / "models" / MODEL_FILENAMES["december_scenario"]).expanduser().resolve()
    output = (args.output or project / "outputs/december_predictions.csv").expanduser().resolve()
    try:
        run_december_prediction(project, model, output)
    except (DataValidationError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
