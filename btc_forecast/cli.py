"""Run with python -m btc_forecast.cli --help."""
import argparse
from pathlib import Path

from .config import MODELS, ExperimentConfig
from .data import DEFAULT_CACHE, ROOT, DataError, fetch_daily, load_snapshot
from .evaluation import load_result, run_experiment, save_result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Daily BTC-USD forecasting benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download and validate five years of daily Yahoo data")
    fetch.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    evaluate = sub.add_parser("evaluate", help="Select, test, refit and save a benchmark")
    evaluate.add_argument("--data", type=Path, help="Snapshot CSV; defaults to latest cache")
    evaluate.add_argument("--output", type=Path, default=ROOT / "artifacts" / "runs")
    evaluate.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    evaluate.add_argument("--epochs", type=int, default=12)
    evaluate.add_argument("--trees", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=42)
    show = sub.add_parser("show", help="Print metrics from a saved run")
    show.add_argument("directory", type=Path)
    report = sub.add_parser("report", help="Export a measured Markdown report and PNG charts")
    report.add_argument("directory", type=Path)
    report.add_argument("--output", type=Path, default=ROOT / "docs" / "benchmark")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            snapshot = fetch_daily(args.cache)
            print(f"Saved {len(snapshot.frame)} completed daily candles to {snapshot.path}")
            print(f"SHA-256: {snapshot.metadata['fingerprint']}")
        elif args.command == "evaluate":
            snapshot = load_snapshot(args.data)
            config = ExperimentConfig(epochs=args.epochs, trees=args.trees, seed=args.seed, models=tuple(args.models))
            result = run_experiment(snapshot, config, progress=print)
            directory = save_result(result, snapshot, args.output)
            print(result.metrics[result.metrics.horizon == "Overall"].to_string(index=False))
            print(f"Saved reproducible run: {directory}")
        elif args.command == "report":
            from .reporting import write_report
            print(write_report(args.directory, args.output))
        else:
            print(load_result(args.directory).metrics.to_string(index=False))
    except (DataError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
