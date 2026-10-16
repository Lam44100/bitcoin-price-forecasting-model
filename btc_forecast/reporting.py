"""Export a source-backed Markdown report and publication-independent PNG figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .charts import COLORS
from .evaluation import load_result


def write_report(run_directory, output_directory):
    result = load_result(run_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    overall = result.metrics[result.metrics.horizon == "Overall"]
    with plt.rc_context({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        bars = ax.barh(overall.model, overall.MAE, color=[COLORS[x] for x in overall.model])
        ax.bar_label(bars, fmt="$%.0f", padding=5)
        ax.set_xlim(0, overall.MAE.max() * 1.18)
        ax.invert_yaxis()
        ax.set(xlabel="Mean absolute error (USD) — lower is better", title="Held-out seven-day forecasts compared with persistence")
        fig.savefig(output / "benchmark-mae.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        rows = result.metrics[result.metrics.horizon != "Overall"]
        for name, group in rows.groupby("model", sort=False):
            ax.plot(group.horizon.astype(int), group.MAE, marker="o", label=name, color=COLORS[name])
        ax.set(xlabel="Horizon (days)", ylabel="MAE (USD)", title="Error by forecast horizon", xticks=range(1, 8))
        ax.legend(frameon=False)
        fig.savefig(output / "error-by-horizon.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        baseline = result.predictions[result.predictions.model == "Last price"]
        ax.plot(baseline.target_date, baseline.actual, label="Observed", color="#0f172a", linewidth=1.8)
        for name, group in result.predictions.groupby("model", sort=False):
            ax.plot(group.target_date, group.prediction, label=name, color=COLORS[name], alpha=.8, linewidth=1)
        ax.set(ylabel="Closing price (USD)", title="Held-out observations and forecast blocks")
        ax.legend(frameon=False, ncol=3)
        fig.savefig(output / "held-out-predictions.png")
        plt.close(fig)
    manifest = result.manifest
    split, data = manifest["split"], manifest["data"]
    lines = ["# Measured Bitcoin forecasting benchmark", "",
        f"Generated from run `{Path(run_directory).name}` at {manifest['created_at']}.", "",
        f"Source: **{data['source']}**. {data['rows']:,} completed UTC daily candles, "
        f"{data.get('start', '')[:10]} through {data.get('end', '')[:10]}. Retrieved {data.get('fetched_at', 'unknown')}.", "",
        f"Data SHA-256: `{data['fingerprint']}`.", "",
        f"Training ends {split['train_end'][:10]}; validation {split['validation_start'][:10]}–{split['validation_end'][:10]}; "
        f"test partition {split['test_start'][:10]}–{split['test_end'][:10]}.", "",
        f"The test contains {split['test_origins']} origins, seven days apart. "
        f"Scored targets run from {baseline.target_date.min():%Y-%m-%d} through {baseline.target_date.max():%Y-%m-%d}; "
        f"{split['unused_test_tail_days']} trailing day(s) are excluded because the final block is incomplete.", "",
        "| Model | MAE (USD) | RMSE (USD) | MAE improvement vs baseline |",
        "|---|---:|---:|---:|"]
    for row in overall.itertuples():
        lines.append(f"| {row.model} | {row.MAE:,.2f} | {row.RMSE:,.2f} | {row.MAE_improvement_pct:+.2f}% |")
    lines += ["", "Positive improvement means lower error. Negative means worse; no trading returns are inferred.", "",
        "![Overall held-out MAE](benchmark-mae.png)", "", "![Error by horizon](error-by-horizon.png)", "",
        "![Observed vs predicted closes](held-out-predictions.png)", "", "## Selected settings", "",
        "```json", json.dumps(manifest["selected_parameters"], indent=2), "```", "",
        f"Seed: {manifest['config']['seed']}; LSTM epochs: {manifest['config']['epochs']}; "
        f"RF trees: {manifest['config']['trees']}; runtime: {manifest['duration_seconds']:.1f} seconds on {manifest['platform']}.", "",
        "Selection minimizes validation MAE over all seven horizons. Test models are refitted on training plus validation, "
        "then evaluated with fixed parameters. Forecast inputs advance to each origin; ARIMA filters new observations without "
        "re-estimating coefficients. Future outlooks are separate full-history refits and are not included in these scores.", "",
        "## Interpretation and limits", "",
        "Read model performance relative to the last-price baseline. If ARIMA selects (0,1,0), it is a random walk without drift "
        "and produces the same point forecast as persistence. Random Forest price regression cannot extrapolate beyond its "
        "training target range; regime changes are a plausible contributor to weak results. LSTM performance depends on "
        "representation, capacity, and training duration. These explanations are hypotheses, not causal findings from this experiment.", "",
        "This is a single chronological holdout and a small, predefined validation search. Absolute USD errors vary with the price "
        "regime. No claim is made about profitability, calibrated uncertainty for RF/LSTM, or generalization to other assets. "
        "Future changes should be evaluated on new data or a preregistered walk-forward design rather than repeatedly optimizing this test set.", "",
        "## Reproduction", "",
        "Run `python -m btc_forecast.cli evaluate --data <saved-run>/data.csv` with the configuration recorded in "
        "`manifest.json`. Defaults reproduce the standard 12-epoch, 200-tree, seed-42 study. A later download changes the dataset "
        "and is a new experiment. Exact snapshots and full outputs are retained locally under ignored `artifacts/runs/`; "
        "share the run directory separately when exact reproduction is needed.", "",
        "Package versions and per-model durations are recorded in the run manifest. CPU runs with the pinned environment should "
        "agree within small floating-point tolerances; cross-platform TensorFlow training may differ.", ""]
    (output / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    result.metrics.to_csv(output / "metrics.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output / "benchmark.md"
