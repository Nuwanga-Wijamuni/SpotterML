"""Build freight-rate model inputs and training-only preprocessing.

Run ``python -m src.features`` from the Spotter project root to inspect features.
Feature construction is deterministic. Call make_preprocessor().fit() only on
training features, then transform the holdout and final inputs with that same
fitted object. No target, ID, or prediction columns are used as features.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment", "route"]
REFERENCE_DATE = pd.Timestamp("2025-01-01")
COORDINATE_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
BASE_NUMERIC_FEATURES = [
    "distance", "log_distance", "weight", "weight_missing",
    "day_of_week", "month", "day_of_year", "is_weekend",
    "days_since_reference", "weekday_sin", "weekday_cos",
    "year_sin", "year_cos",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Use the same configuration for fitting and prediction.

    market_index and quote_signal are provisional inputs: their prediction-time
    availability must be checked during experiments. Setting both to False,
    with geography disabled, creates a December-compatible feature set.
    Geography is disabled by default pending coordinate-quality review.
    """

    include_market_index: bool = True
    include_quote_signal: bool = True
    include_geography: bool = False


def get_feature_columns(config: FeatureConfig | None = None) -> tuple[list[str], list[str]]:
    """Return a stable (numeric, categorical) schema for the chosen inputs."""
    config = config or FeatureConfig()
    numeric = BASE_NUMERIC_FEATURES.copy()
    if config.include_market_index:
        numeric += ["market_index", "market_index_missing"]
    if config.include_quote_signal:
        numeric += ["quote_signal", "quote_signal_missing"]
    if config.include_geography:
        numeric += COORDINATE_COLUMNS
        numeric += [f"{name}_missing" for name in COORDINATE_COLUMNS]
        numeric += ["great_circle_miles", "road_to_air_ratio", "geography_missing"]
    return numeric, CATEGORICAL_FEATURES.copy()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Represent absent optional inputs and invalid numbers as NaN."""
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _category(values: pd.Series) -> pd.Series:
    """Normalize text and distinguish missing values from literal city names."""
    text = values.astype("string").str.strip().replace("", pd.NA)
    # Prefix real values so the missing token cannot collide with a real label.
    return ("value:" + text).fillna("missing:").astype(str)


def create_features(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Return a new feature frame while preserving input row order and index.

    Missing optional signals stay NaN, with explicit indicators. This permits a
    stable schema for December but does not establish reduced-input accuracy;
    validate that scenario or fit a model with both signals disabled.
    The caller must use data.load_project_data for full dataset-level validation.
    """
    config = config or FeatureConfig()
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot create features; missing columns: {sorted(missing)}")
    if not frame.index.is_unique:
        raise ValueError("Feature input index must be unique; reset_index(drop=True) first")

    result = pd.DataFrame(index=frame.index)
    for column in ["pickup", "delivery", "equipment"]:
        result[column] = _category(frame[column])
    # JSON pairs avoid ambiguous routes when city labels contain a separator.
    result["route"] = [json.dumps([pickup, delivery], ensure_ascii=False)
                       for pickup, delivery in zip(result.pickup, result.delivery)]

    result["distance"] = _numeric(frame, "distance")
    if result.distance.isna().any() or result.distance.le(0).any():
        raise ValueError("Distance must be finite and positive before feature creation")
    result["log_distance"] = np.log1p(result.distance)
    weight = _numeric(frame, "weight")
    result["weight"] = weight.where(weight > 0)
    result["weight_missing"] = result.weight.isna().astype(float)

    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("Date must be valid and nonmissing before feature creation")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Dates must be timezone-free calendar days")
    result["day_of_week"] = dates.dt.dayofweek.astype(float)
    result["month"] = dates.dt.month.astype(float)
    result["day_of_year"] = dates.dt.dayofyear.astype(float)
    result["is_weekend"] = dates.dt.dayofweek.ge(5).astype(float)
    # Fixed origin ensures dates never change meaning across batches or splits.
    result["days_since_reference"] = (dates - REFERENCE_DATE).dt.days.astype(float)
    weekday_angle = 2 * np.pi * dates.dt.dayofweek / 7
    year_length = np.where(dates.dt.is_leap_year, 366, 365)
    year_angle = 2 * np.pi * (dates.dt.dayofyear - 1) / year_length
    result["weekday_sin"], result["weekday_cos"] = np.sin(weekday_angle), np.cos(weekday_angle)
    result["year_sin"], result["year_cos"] = np.sin(year_angle), np.cos(year_angle)

    for column, enabled in [("market_index", config.include_market_index),
                            ("quote_signal", config.include_quote_signal)]:
        if enabled:
            result[column] = _numeric(frame, column)
            result[f"{column}_missing"] = result[column].isna().astype(float)

    if config.include_geography:
        for column in COORDINATE_COLUMNS:
            values = _numeric(frame, column)
            limit = 90 if column.endswith("lat") else 180
            result[column] = values.where(values.abs() <= limit)
            result[f"{column}_missing"] = result[column].isna().astype(float)
        lat1, lon1, lat2, lon2 = [np.radians(result[c]) for c in COORDINATE_COLUMNS]
        haversine = (np.sin((lat2 - lat1) / 2) ** 2
                     + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        result["great_circle_miles"] = 3958.7613 * 2 * np.arcsin(np.sqrt(haversine.clip(0, 1)))
        # Coincident/near-coincident coordinates cannot yield a useful ratio.
        denominator = result.great_circle_miles.where(result.great_circle_miles > 1)
        result["road_to_air_ratio"] = result.distance / denominator
        result["geography_missing"] = result[COORDINATE_COLUMNS].isna().any(axis=1).astype(float)

    numeric, categorical = get_feature_columns(config)
    result = result[categorical + numeric].copy()
    result[numeric] = result[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    return result


def make_preprocessor(config: FeatureConfig | None = None, *, scale_numeric: bool = True):
    """Create an UNFITTED scikit-learn transformer for create_features output.

    Example:
        X_train = create_features(training_rows, config)
        preprocessing = make_preprocessor(config)
        transformed_train = preprocessing.fit_transform(X_train)
        transformed_holdout = preprocessing.transform(create_features(holdout_rows, config))

    Use in a model Pipeline so every validation fold fits its own preprocessing.
    All-empty numeric training columns are retained and filled with zero by
    SimpleImputer; the explicit missing indicators still identify missing inputs.
    Unknown categorical values are safely handled without fitting on final data.
    """
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError(
            'Install preprocessing dependency: python -m pip install "scikit-learn>=1.3,<2"'
        )
    # Preserve dependency compatibility tracebacks instead of misreporting them
    # as a missing scikit-learn installation.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = get_feature_columns(config)
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
        ],
        remainder="drop",
    )


def main() -> None:
    from .data import load_project_data, split_development_by_date

    parser = argparse.ArgumentParser(description="Inspect freight-rate model features")
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--scenario-only", action="store_true", help="Exclude market and quote signals")
    parser.add_argument("--without-quote-signal", action="store_true")
    parser.add_argument("--include-geography", action="store_true", help="Use only after coordinate review")
    args = parser.parse_args()
    if args.scenario_only and args.include_geography:
        parser.error("December scenario inputs do not contain geography")
    config = FeatureConfig(
        include_market_index=not args.scenario_only,
        include_quote_signal=not (args.scenario_only or args.without_quote_signal),
        include_geography=args.include_geography,
    )
    data = load_project_data(args.project_dir)
    earlier, holdout = split_development_by_date(data.development)
    numeric, categorical = get_feature_columns(config)
    print(f"Features: {len(numeric)} numeric, {len(categorical)} categorical")
    columns = None
    for label, frame in [("training", earlier), ("holdout", holdout),
                         ("validation", data.validation), ("december", data.december)]:
        built = create_features(frame, config)
        if columns is None:
            columns = list(built.columns)
        assert list(built.columns) == columns, "Feature columns differ across datasets"
        print(f"{label}: {len(built):,} rows, {len(built.columns)} features")
        missing = built[numeric].isna().sum()
        if missing.gt(0).any():
            print("  Missing numeric values:", missing[missing > 0].to_dict())
    print("No preprocessing was fitted. Fit on training rows inside model validation.")
    print("Model inputs exclude load_id, posted_rate, and predicted_rate.")


if __name__ == "__main__":
    main()
