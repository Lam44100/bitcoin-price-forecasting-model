"""Validation-only selection, fixed-parameter rolling tests, and full-history refits."""
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .config import MODELS, ExperimentConfig, experiment_key
from .data import Snapshot, canonical_csv, fingerprint, validate_daily
from .models import candidates, create_model


@dataclass
class ExperimentResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    forecasts: pd.DataFrame
    tuning: pd.DataFrame
    manifest: dict
    directory: Path | None = None


def split_bounds(n):
    return int(n * 0.70), int(n * 0.85), n


def evaluation_origins(start, end, horizon=7):
    # First origin is the last available training day. Targets are all in [start,end).
    return list(range(start - 1, end - horizon, horizon))


def evaluate_origins(model, frame, origins, name):
    rows = []
    for origin in origins:
        result = model.forecast(frame.iloc[:origin + 1])
        for h, (date, value) in enumerate(zip(result.dates, result.prices), 1):
            if date != frame.Datetime.iloc[origin + h]:
                raise ValueError("Forecast timestamp does not match its held-out target.")
            rows.append({"model": name, "origin": frame.Datetime.iloc[origin], "target_date": date,
                         "horizon": h, "actual": frame.Close.iloc[origin + h], "prediction": value})
    return pd.DataFrame(rows)


def calculate_metrics(predictions):
    rows = []
    for name, group in predictions.groupby("model", sort=False):
        for horizon in ["Overall", *sorted(group.horizon.unique())]:
            subset = group if horizon == "Overall" else group[group.horizon == horizon]
            error = subset.prediction.to_numpy() - subset.actual.to_numpy()
            rows.append({"model": name, "horizon": str(horizon), "MAE": float(np.abs(error).mean()),
                         "RMSE": float(np.sqrt(np.square(error).mean())), "observations": len(error)})
    metrics = pd.DataFrame(rows)
    baseline = metrics[metrics.model == "Last price"].set_index("horizon")
    for metric in ["MAE", "RMSE"]:
        denominators = metrics.horizon.map(baseline[metric]).replace(0, np.nan)
        metrics[f"{metric}_improvement_pct"] = 100 * (1 - metrics[metric] / denominators)
    return metrics


def run_experiment(snapshot: Snapshot, config=ExperimentConfig(), progress=None):
    started = time.perf_counter()
    frame = validate_daily(snapshot.frame)
    digest = fingerprint(frame)
    if digest != snapshot.metadata.get("fingerprint", digest):
        raise ValueError("Input data no longer matches the snapshot fingerprint.")
    train_end, validation_end, end = split_bounds(len(frame))
    validation_origins = evaluation_origins(train_end, validation_end, config.horizon)
    test_origins = evaluation_origins(validation_end, end, config.horizon)
    if not validation_origins or not test_origins:
        raise ValueError("Not enough validation or test observations for a full seven-day target.")
    names = [name for name in MODELS if name == "Last price" or name in config.models]
    predictions, forecasts, tuning, selected, durations, model_warnings = [], [], [], {}, {}, {}
    notify = progress or (lambda message: None)
    for name in names:
        model_started = time.perf_counter()
        best_params, best_score = None, float("inf")
        for params in candidates(name):
            notify(f"Selecting {name}: {params or 'baseline'} on validation data")
            try:
                model = create_model(name, config, params).fit(frame.iloc[:train_end])
                validation = evaluate_origins(model, frame, validation_origins, name)
                score = float(np.abs(validation.prediction - validation.actual).mean())
                tuning.append({"model": name, "parameters": json.dumps(params), "validation_MAE": score, "error": ""})
                if score < best_score:
                    best_score, best_params = score, params
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                tuning.append({"model": name, "parameters": json.dumps(params), "validation_MAE": np.nan, "error": str(exc)})
        if best_params is None:
            failures = [r["error"] for r in tuning if r["model"] == name]
            raise ValueError(f"All {name} candidates failed: {'; '.join(failures)}")
        selected[name] = best_params
        notify(f"Evaluating {name} on untouched test targets")
        model = create_model(name, config, best_params).fit(frame.iloc[:validation_end])
        model_warnings[name] = {"test_fit": getattr(model, "fit_warnings", [])}
        predictions.append(evaluate_origins(model, frame, test_origins, name))
        notify(f"Refitting {name} on full history for the future outlook")
        final_model = create_model(name, config, best_params).fit(frame)
        model_warnings[name]["final_fit"] = getattr(final_model, "fit_warnings", [])
        future = final_model.forecast(frame)
        for i, (date, price) in enumerate(zip(future.dates, future.prices)):
            forecasts.append({"model": name, "origin": frame.Datetime.iloc[-1], "target_date": date,
                              "horizon": i + 1, "prediction": price,
                              "lower_95": future.lower[i] if future.lower is not None else np.nan,
                              "upper_95": future.upper[i] if future.upper is not None else np.nan})
        durations[name] = round(time.perf_counter() - model_started, 3)
    predictions = pd.concat(predictions, ignore_index=True)
    packages = {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "statsmodels", "tensorflow", "streamlit", "yfinance", "plotly"]}
    manifest = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "experiment_key": experiment_key(digest, config), "config": config.as_dict(),
        "data": {**snapshot.metadata, "fingerprint": digest, "rows": len(frame)},
        "split": {"train_end": frame.Datetime.iloc[train_end - 1].isoformat(),
                  "validation_start": frame.Datetime.iloc[train_end].isoformat(),
                  "validation_end": frame.Datetime.iloc[validation_end - 1].isoformat(),
                  "test_start": frame.Datetime.iloc[validation_end].isoformat(), "test_end": frame.Datetime.iloc[-1].isoformat(),
                  "validation_origins": len(validation_origins), "test_origins": len(test_origins),
                  "unused_test_tail_days": len(frame) - 1 - (test_origins[-1] + config.horizon)},
        "selection_metric": "Mean absolute error across validation origins and horizons",
        "protocol": "70/15/15 chronological; non-overlapping seven-day target blocks; fixed test parameters; full-history outlook refit",
        "selected_parameters": selected, "model_warnings": model_warnings,
        "duration_seconds": round(time.perf_counter() - started, 3), "model_duration_seconds": durations,
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
    }
    return ExperimentResult(predictions, calculate_metrics(predictions), pd.DataFrame(forecasts), pd.DataFrame(tuning), manifest)


def save_result(result, snapshot, root):
    root = Path(root)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / f"{stamp}-{result.manifest['experiment_key'][:10]}"
    directory.mkdir(parents=True, exist_ok=False)
    for name in ["predictions", "metrics", "forecasts", "tuning"]:
        getattr(result, name).to_csv(directory / f"{name}.csv", index=False)
    (directory / "data.csv").write_text(canonical_csv(snapshot.frame), encoding="utf-8")
    (directory / "data.json").write_text(json.dumps(snapshot.metadata, indent=2), encoding="utf-8")
    # Written last: a directory without a manifest is an incomplete run.
    (directory / "manifest.json").write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")
    result.directory = directory
    return directory


def load_result(directory, expected_key=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported saved-result schema.")
    if expected_key and manifest["experiment_key"] != expected_key:
        raise ValueError("Saved result does not match the selected data and configuration.")
    tables = {name: pd.read_csv(directory / f"{name}.csv") for name in ["predictions", "metrics", "forecasts", "tuning"]}
    for name in ["predictions", "forecasts"]:
        for column in ["origin", "target_date"]:
            tables[name][column] = pd.to_datetime(tables[name][column], utc=True)
    return ExperimentResult(**tables, manifest=manifest, directory=directory)
