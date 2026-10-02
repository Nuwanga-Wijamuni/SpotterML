"""Train the frozen Spotter model selections and save prediction bundles.

From the Spotter project root, run: python -m src.train

The experiment JSON selects the main model and the separate December model.
October evaluates the frozen choices; it does not tune their parameters.
Final models are then fitted on all labeled development rows. Existing matching
holdout results are reused on subsequent runs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from . import data as data_module
from . import features as features_module
from .data import DataValidationError, TARGET_COLUMN, find_file, load_project_data, split_development_by_date
from .features import FeatureConfig, create_features, get_feature_columns, make_preprocessor


MODEL_FORMAT_VERSION = 1
ROLES = ("main", "december_scenario")
MODEL_FILENAMES = {"main": "main_model.joblib", "december_scenario": "december_model.joblib"}
PACKAGE_NAMES = ("numpy", "pandas", "scikit-learn", "scipy", "catboost", "joblib", "matplotlib")
REQUIRED_PARAMETERS = (
    "seed", "catboost_iterations", "catboost_depth", "catboost_learning_rate",
    "catboost_threads", "catboost_l2_leaf_reg", "ridge_alpha", "minimum_total_rate",
)


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DataValidationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{path.name}: expected a JSON object")
    return value


def write_json(path: Path, value: dict) -> None:
    """Replace a report atomically so an interrupted write cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_selection(path: Path) -> dict:
    """Validate notebook settings without changing its selected parameters."""
    selection = read_json(path)
    for role in ROLES:
        spec = selection.get(role, {})
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise DataValidationError(f"Selection JSON is missing the {role} model specification")
        if spec.get("algorithm") not in ("baseline", "ridge", "catboost"):
            raise DataValidationError(f"{role}: unsupported model algorithm")
        if spec.get("target_space") not in ("total_rate", "rate_per_mile"):
            raise DataValidationError(f"{role}: unsupported target_space")
        config = spec.get("config", {})
        if not isinstance(config, dict) or set(config) != set(FeatureConfig.__dataclass_fields__):
            raise DataValidationError(f"{role}: feature configuration is missing or invalid")
        if not all(isinstance(flag, bool) for flag in config.values()):
            raise DataValidationError(f"{role}: feature configuration values must be booleans")
    if any(selection["december_scenario"]["config"].values()):
        raise DataValidationError("December model must exclude market index, quote signal and geography")
    parameters = selection.get("hyperparameters", {})
    if not isinstance(parameters, dict):
        raise DataValidationError("hyperparameters must be a JSON object")
    for key in REQUIRED_PARAMETERS:
        value = parameters.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise DataValidationError(f"Missing or invalid hyperparameter: {key}")
        if key == "seed":
            if value < 0 or int(value) != value:
                raise DataValidationError("seed must be a nonnegative integer")
        elif value <= 0:
            raise DataValidationError(f"{key} must be positive")
    for key in ("catboost_iterations", "catboost_depth", "catboost_threads"):
        if int(parameters[key]) != parameters[key]:
            raise DataValidationError(f"{key} must be an integer")
    if "holdout_start" not in selection:
        raise DataValidationError("Selection JSON must include holdout_start")
    try:
        cutoff = pd.Timestamp(selection["holdout_start"])
        folds = [pd.Period(month, freq="M") for month in selection.get("selection_folds", [])]
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Invalid holdout or fold dates in selection JSON") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise DataValidationError("holdout_start must be a valid timezone-free calendar day")
    if not folds or any((month + 1).start_time > cutoff for month in folds):
        raise DataValidationError("Model-selection folds must finish before the holdout starts")
    return selection


def fit_model(training: pd.DataFrame, spec: dict, parameters: dict) -> dict:
    """Return a serializable dictionary containing a fitted model and its schema."""
    if training.empty or TARGET_COLUMN not in training:
        raise DataValidationError("Training requires nonempty labeled development data")
    config = FeatureConfig(**spec["config"])
    X_train = create_features(training, config)
    target = pd.to_numeric(training[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise DataValidationError("Training targets must be finite and positive")
    if spec["target_space"] == "rate_per_mile":
        target = target / X_train.distance.to_numpy(dtype=float)

    algorithm = spec["algorithm"]
    if algorithm == "baseline":
        rates = pd.DataFrame({"equipment": X_train.equipment, "target": target}, index=X_train.index)
        estimator = {"medians": rates.groupby("equipment").target.median().to_dict(),
                     "fallback": float(np.median(target))}
    elif algorithm == "ridge":
        estimator = Pipeline([
            ("preprocessing", make_preprocessor(config)),
            ("regression", Ridge(alpha=parameters["ridge_alpha"], solver="lsqr", tol=1e-5)),
        ])
        estimator.fit(X_train, target)
    elif algorithm == "catboost":
        _, categorical = get_feature_columns(config)
        estimator = CatBoostRegressor(
            iterations=int(parameters["catboost_iterations"]),
            depth=int(parameters["catboost_depth"]),
            learning_rate=float(parameters["catboost_learning_rate"]),
            loss_function="RMSE", random_seed=int(parameters["seed"]),
            thread_count=int(parameters["catboost_threads"]), verbose=False,
            allow_writing_files=False, l2_leaf_reg=float(parameters["catboost_l2_leaf_reg"]),
        )
        estimator.fit(X_train, target, cat_features=categorical)
    else:
        raise DataValidationError(f"Unsupported algorithm: {algorithm}")

    return {
        "format_version": MODEL_FORMAT_VERSION, "experiment": spec["name"],
        "algorithm": algorithm, "target_space": spec["target_space"],
        "feature_config": deepcopy(spec["config"]), "feature_columns": list(X_train.columns),
        "minimum_total_rate": float(parameters["minimum_total_rate"]), "estimator": estimator,
        "hyperparameters": deepcopy(parameters), "versions": package_versions(),
        "python_version": platform.python_version(), "training_rows": len(training),
        "training_first_date": str(training.date.min().date()),
        "training_last_date": str(training.date.max().date()),
    }


def predict_model(bundle: dict, rows: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Return positive TOTAL-dollar predictions and the number floor-adjusted.

    Call this function from the later prediction scripts to apply the saved
    feature configuration, preprocessing, target-space conversion and floor.
    """
    if bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise DataValidationError("Unsupported saved model format")
    X = create_features(rows, FeatureConfig(**bundle["feature_config"]))
    if list(X.columns) != bundle["feature_columns"]:
        raise DataValidationError("Prediction feature columns differ from the fitted model")
    if bundle["algorithm"] == "baseline":
        model = bundle["estimator"]
        raw = X.equipment.map(model["medians"]).fillna(model["fallback"]).to_numpy(dtype=float)
    else:
        raw = np.asarray(bundle["estimator"].predict(X), dtype=float)
    if bundle["target_space"] == "rate_per_mile":
        raw = raw * X.distance.to_numpy(dtype=float)
    if raw.shape != (len(rows),) or not np.isfinite(raw).all():
        raise DataValidationError("Model produced invalid prediction values or row count")
    floor = bundle["minimum_total_rate"]
    return np.maximum(raw, floor), int((raw < floor).sum())


def load_model(path: str | Path) -> dict:
    """Load a bundle produced by this script for use in prediction scripts."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise DataValidationError(f"Invalid model bundle: {path}")
    return bundle


def save_model(bundle: dict, path: Path, check_rows: pd.DataFrame) -> None:
    """Check serialized prediction equivalence before replacing the model file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        loaded = load_model(temporary)
        expected, _ = predict_model(bundle, check_rows)
        actual, _ = predict_model(loaded, check_rows)
        if not np.allclose(expected, actual, rtol=1e-10, atol=1e-8):
            raise DataValidationError(f"Saved model predictions changed: {path.name}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_metrics(truth: np.ndarray, predictions: np.ndarray, clipped: int) -> dict:
    error = predictions - truth
    return {"rows": len(error), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean()),
            "median_absolute_error": float(np.median(np.abs(error))),
            "clipped_predictions": clipped}


def validate_holdout_metrics(metrics: dict, selection: dict, expected_rows: int) -> None:
    """Reject incomplete, stale or non-finite holdout reports."""
    if not isinstance(metrics, dict):
        raise DataValidationError("Holdout report is missing its metrics")
    for role in ROLES:
        item = metrics.get(role, {})
        if not isinstance(item, dict) or item.get("experiment") != selection[role]["name"]:
            raise DataValidationError("Holdout metrics do not match the frozen model choices")
        if item.get("rows") != expected_rows:
            raise DataValidationError("Holdout metrics have a different evaluation row count")
        for metric in ("mae", "rmse", "bias"):
            value = item.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise DataValidationError(f"Holdout {role}: invalid {metric}")
        if item["mae"] < 0 or item["rmse"] + 1e-7 < item["mae"]:
            raise DataValidationError(f"Holdout {role}: inconsistent MAE or RMSE")


def evaluate_frozen_models(earlier: pd.DataFrame, holdout: pd.DataFrame, selection: dict) -> tuple[dict, pd.DataFrame]:
    """Fit on earlier rows and measure the fixed selections on later rows."""
    metrics, tables = {}, []
    for role in ROLES:
        spec = selection[role]
        print(f"Evaluating {role}: {spec['name']} on {len(holdout):,} holdout loads...", flush=True)
        fitted = fit_model(earlier, spec, selection["hyperparameters"])
        predictions, clipped = predict_model(fitted, holdout)
        metrics[role] = {"experiment": spec["name"],
                         **evaluation_metrics(holdout[TARGET_COLUMN].to_numpy(dtype=float), predictions, clipped)}
        table = holdout[["load_id", "date", "pickup", "delivery", "equipment", "distance", "weight"]].copy()
        table["role"], table["experiment"] = role, spec["name"]
        table["actual_rate"], table["predicted_rate"] = holdout[TARGET_COLUMN].to_numpy(), predictions
        table["absolute_error"] = np.abs(predictions - table.actual_rate)
        table["squared_error"] = np.square(predictions - table.actual_rate)
        tables.append(table)
    return metrics, pd.concat(tables, ignore_index=True)


def save_segment_metrics(predictions: pd.DataFrame, path: Path) -> None:
    """Record holdout equipment, distance and missing-weight error breakdowns."""
    frame = predictions.copy()
    frame["distance_band"] = pd.cut(frame.distance, [0, 250, 750, 1500, np.inf],
                                    labels=["0-250", "250-750", "750-1500", "1500+"])
    frame["weight_missing"] = frame.weight.isna()
    tables = []
    for column in ("equipment", "distance_band", "weight_missing"):
        grouped = frame.groupby(["role", column], observed=True, dropna=False).agg(
            rows=("absolute_error", "size"), mae=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean")).reset_index()
        grouped["rmse"] = np.sqrt(grouped.pop("mean_squared_error"))
        grouped = grouped.rename(columns={column: "segment"})
        grouped["segment"] = grouped.segment.astype(str)
        grouped["segment_feature"] = column
        tables.append(grouped)
    pd.concat(tables, ignore_index=True).to_csv(path, index=False)


def run_training(project_dir: Path, selection_path: Path, models_dir: Path, output_dir: Path) -> dict:
    selection = load_selection(selection_path)
    data = load_project_data(project_dir)
    earlier, holdout = split_development_by_date(data.development, selection["holdout_start"])
    if earlier.date.max() >= holdout.date.min():
        raise DataValidationError("Training and holdout dates overlap")
    versions = package_versions()
    signature_payload = {
        "models": {role: selection[role] for role in ROLES},
        "hyperparameters": selection["hyperparameters"], "holdout_start": selection["holdout_start"],
        "development_sha256": sha256_file(find_file(project_dir, "development")),
        "data_code_sha256": sha256_file(Path(data_module.__file__)),
        "features_code_sha256": sha256_file(Path(features_module.__file__)),
        "training_code_sha256": sha256_file(Path(__file__)),
        "versions": versions,
    }
    fingerprint = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    holdout_path = output_dir / "holdout_metrics.json"
    generated_at = datetime.now(timezone.utc).isoformat()

    if holdout_path.exists():
        holdout_report = read_json(holdout_path)
        if holdout_report.get("fingerprint") != fingerprint:
            raise DataValidationError(
                "Existing holdout report uses different data, code, versions or settings. "
                "Keep the frozen results; changing settings after seeing them makes October selection data."
            )
        print("Reusing the matching saved holdout metrics.", flush=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if selection.get("final_holdout_evaluated", False):
            # The notebook may have already assessed the frozen choices.
            previous = read_json(selection_path.parent / "final_holdout_metrics.json")
            validate_holdout_metrics(previous, selection, len(holdout))
            metrics, origin = previous, "reused_notebook_report"
            print("Reusing the notebook's October metrics; no repeated holdout evaluation.", flush=True)
        else:
            metrics, predictions = evaluate_frozen_models(earlier, holdout, selection)
            predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
            save_segment_metrics(predictions, output_dir / "holdout_segments.csv")
            origin = "evaluated_frozen_selection"
        holdout_report = {"fingerprint": fingerprint, "evaluated_at_utc": generated_at,
                          "origin": origin, "signature": signature_payload,
                          "training_rows": len(earlier), "holdout_rows": len(holdout),
                          "holdout_first_date": str(holdout.date.min().date()),
                          "holdout_last_date": str(holdout.date.max().date()), "metrics": metrics}
        write_json(holdout_path, holdout_report)

    validate_holdout_metrics(holdout_report.get("metrics"), selection, len(holdout))
    for role in ROLES:
        result = holdout_report["metrics"][role]
        print(f"Holdout {role}: MAE=${result['mae']:,.2f}, RMSE=${result['rmse']:,.2f}", flush=True)
    print("Refitting the frozen selections on all labeled development data.", flush=True)
    artifacts = {}
    for role in ROLES:
        started = time.perf_counter()
        spec = selection[role]
        print(f"Training final {role}: {spec['name']} on {len(data.development):,} rows...", flush=True)
        bundle = fit_model(data.development, spec, selection["hyperparameters"])
        bundle.update({"role": role, "trained_at_utc": generated_at,
                       "selection_fingerprint": fingerprint,
                       "holdout_metrics": holdout_report["metrics"][role]})
        path = models_dir / MODEL_FILENAMES[role]
        check_rows = data.validation.head(32) if role == "main" else data.december
        save_model(bundle, path, check_rows)
        artifacts[role] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                           "experiment": spec["name"], "training_rows": bundle["training_rows"],
                           "feature_config": bundle["feature_config"],
                           "feature_columns": bundle["feature_columns"],
                           "fit_and_save_seconds": time.perf_counter() - started}
        print(f"Saved and verified: {path}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    data.quality_report.to_csv(output_dir / "data_quality_report.csv", index=False)
    (output_dir / "model_requirements.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
    report = {"trained_at_utc": generated_at, "selection_path": str(selection_path.resolve()),
              "fingerprint": fingerprint, "holdout_report": str(holdout_path.resolve()),
              "final_training_rows": len(data.development), "versions": versions,
              "python_version": platform.python_version(), "artifacts": artifacts,
              "selection": selection,
              "signal_availability": selection.get("signal_availability", "Unverified"),
              "scope": "Local holdout metrics; hidden validation metrics are unavailable."}
    write_json(output_dir / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selection", type=Path, help="Selected configuration JSON from the experiments notebook")
    parser.add_argument("--models-dir", type=Path, help="Default: PROJECT/models")
    parser.add_argument("--output-dir", type=Path, help="Default: PROJECT/outputs/training")
    args = parser.parse_args()
    project = args.project_dir.expanduser().resolve()
    selection = (args.selection or project / "outputs/model_experiments/selected_configuration.json").expanduser().resolve()
    models = (args.models_dir or project / "models").expanduser().resolve()
    output = (args.output_dir or project / "outputs/training").expanduser().resolve()
    try:
        run_training(project, selection, models, output)
    except (DataValidationError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print("Training complete. Models are ready for the prediction scripts.")


if __name__ == "__main__":
    main()
