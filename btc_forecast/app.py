"""Streamlit presentation; model and evaluation modules have no UI dependency."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from .charts import forecast_chart, history_chart, horizon_chart, test_chart
from .config import ExperimentConfig, experiment_key
from .data import DEFAULT_CACHE, ROOT, DataError, fetch_daily, load_snapshot, save_snapshot, validate_daily
from .evaluation import load_result, run_experiment, save_result

RUNS = ROOT / "artifacts" / "runs"


def sync_result_state(state, key):
    changed = state.get("result_key") is not None and state.get("result_key") != key
    if changed:
        state.pop("result", None)
        state.pop("result_key", None)
    return changed


def matching_run(key):
    for path in sorted(RUNS.glob("*/manifest.json"), reverse=True):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("experiment_key") == key:
                return path.parent
        except (OSError, ValueError):
            continue
    return None


def main():
    st.set_page_config(page_title="Bitcoin Forecasting Lab", page_icon="₿", layout="wide")
    st.markdown("""<style>
    .block-container {padding-top:2rem;max-width:1400px;}
    [data-testid="stMetric"] {background:#f1f5f9;border-radius:10px;padding:16px;}
    [data-testid="stMetricLabel"] {color:#475569;}
    </style>""", unsafe_allow_html=True)
    st.caption("TIME SERIES RESEARCH  /  BTC-USD  /  DAILY UTC")
    st.title("Bitcoin Forecasting Lab")
    st.write("Comparing statistical and machine learning forecasts against a simple last-price baseline.")

    with st.sidebar:
        st.header("Experiment settings")
        st.caption("Daily closing prices · seven-day horizon")
        if st.button("Refresh daily data", use_container_width=True):
            try:
                with st.spinner("Downloading completed daily candles…"):
                    st.session_state.snapshot = fetch_daily()
                st.success("Snapshot refreshed and saved.")
            except (DataError, OSError) as exc:
                st.error(str(exc))
                if "snapshot" in st.session_state:
                    st.warning("Refresh failed. The previous snapshot remains selected; its dates are shown below.")
        with st.expander("Load or import a snapshot"):
            available = sorted(DEFAULT_CACHE.glob("btc-usd-*.csv"), reverse=True)
            if available:
                filename = st.selectbox("Saved snapshot", [p.name for p in available])
                if st.button("Load snapshot"):
                    try:
                        st.session_state.snapshot = load_snapshot(DEFAULT_CACHE / filename)
                    except DataError as exc:
                        st.error(str(exc))
            uploaded = st.file_uploader("Daily OHLCV CSV", type=["csv"])
            st.caption("Columns: Datetime, Open, High, Low, Close, Volume. Use midnight UTC timestamps.")
            if st.button("Import CSV", disabled=uploaded is None):
                try:
                    frame = validate_daily(pd.read_csv(uploaded))
                    st.session_state.snapshot = save_snapshot(frame, source="User-imported CSV (provenance not verified)")
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        models = st.multiselect("Models", ["Random Forest", "ARIMA", "LSTM"], default=["Random Forest", "ARIMA", "LSTM"])
        st.caption("The last-price baseline is always included.")
        epochs = st.slider("LSTM epochs", 1, 24, 12)
        trees = st.selectbox("Random Forest trees", [50, 100, 200, 300], index=2)
        seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, step=1)
        config = ExperimentConfig(epochs=epochs, trees=trees, seed=int(seed), models=tuple(["Last price", *models]))
        st.caption("Candidate grid: RF depth 6/12, LSTM units 32/64, four non-seasonal ARIMA orders. Selection uses validation MAE.")

    if "snapshot" not in st.session_state:
        try:
            st.session_state.snapshot = load_snapshot(os.environ.get("BTC_SNAPSHOT_PATH"))
        except DataError as exc:
            st.info(str(exc))
            st.markdown("**Start with real observations.** Refresh daily data in the sidebar, or import a saved snapshot. Nothing is downloaded automatically.")
            st.caption("Synthetic data is used only in automated tests and is never substituted into this dashboard.")
            return
    snapshot = st.session_state.snapshot
    frame = snapshot.frame
    key = experiment_key(snapshot.metadata["fingerprint"], config)
    if sync_result_state(st.session_state, key):
        st.info("Data or model settings changed. Run the benchmark again or load matching saved results.")
    with st.sidebar:
        st.divider()
        st.caption(f"Source: {snapshot.metadata['source']}")
        st.caption(f"Retrieved: {snapshot.metadata.get('fetched_at') or 'unknown'}")
        st.caption(f"SHA-256: {snapshot.metadata['fingerprint'][:16]}…")
        saved = matching_run(key)
        if st.button("Load matching saved run", disabled=saved is None, use_container_width=True):
            try:
                st.session_state.result = load_result(saved, key)
                st.session_state.result_key = key
            except (OSError, ValueError) as exc:
                st.error(f"Cannot load saved run: {exc}")
        run = st.button("Run benchmark", type="primary", use_container_width=True)
        st.caption("LSTM training can take several minutes. Saved results can be reopened without retraining.")
    if run:
        st.session_state.pop("result", None)
        st.session_state.pop("result_key", None)
        status = st.empty()
        try:
            with st.spinner("Running validation, test evaluation, and final refits…"):
                result = run_experiment(snapshot, config, progress=status.info)
                save_result(result, snapshot, RUNS)
            st.session_state.result = result
            st.session_state.result_key = key
            status.success("Benchmark completed. Results and the exact input snapshot are saved locally.")
        except Exception as exc:
            status.error(f"Benchmark failed: {exc}. No partial result is shown as a completed experiment.")

    last = frame.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last completed close", f"${last.Close:,.0f}")
    c2.metric("Daily close change", f"{100 * (last.Close / frame.Close.iloc[-2] - 1):+.2f}%")
    c3.metric("Completed candles", f"{len(frame):,}")
    c4.metric("Last candle · UTC", last.Datetime.strftime("%d %b %Y"))
    st.caption(f"Snapshot: {frame.Datetime.iloc[0]:%d %b %Y} – {last.Datetime:%d %b %Y}. Prices are daily closes, not live quotes.")
    if last.Datetime < pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1):
        st.info("Historical snapshot selected. The outlook starts after this snapshot's final candle, not after today.")
    overview, comparison, forecast, methodology = st.tabs(["Overview", "Model Comparison", "Forecast", "Methodology"])
    result = st.session_state.get("result")
    with overview:
        overlays = st.multiselect("Chart overlays", ["20-day average", "Bollinger bands"], default=["20-day average"])
        st.plotly_chart(history_chart(frame, overlays), use_container_width=True)
        with st.expander("Inspect daily observations"):
            st.dataframe(frame.tail(100), use_container_width=True, hide_index=True)
        st.download_button("Download selected snapshot", frame.to_csv(index=False), "btc-daily-snapshot.csv", "text/csv")
    with comparison:
        if result is None:
            st.info("Run a benchmark or load saved results to compare held-out performance.")
        else:
            manifest = result.manifest
            split = manifest["split"]
            st.subheader("Does complexity beat the baseline?")
            st.caption(f"Test period: {split['test_start'][:10]} – {split['test_end'][:10]} · {split['test_origins']} origins · runtime {manifest['duration_seconds']:,.1f}s")
            overall = result.metrics[result.metrics.horizon == "Overall"][["model", "MAE", "RMSE", "MAE_improvement_pct"]]
            overall = overall.rename(columns={"model": "Model", "MAE": "MAE (USD)", "RMSE": "RMSE (USD)", "MAE_improvement_pct": "MAE improvement (%)"})
            st.dataframe(overall, use_container_width=True, hide_index=True, column_config={
                "MAE (USD)": st.column_config.NumberColumn(format="$%.2f"),
                "RMSE (USD)": st.column_config.NumberColumn(format="$%.2f"),
                "MAE improvement (%)": st.column_config.NumberColumn(format="%.2f%%"),
            })
            st.caption("MAE and RMSE are in USD. Positive improvement means lower error than the baseline; negative means worse. Undefined percentages indicate a zero-error baseline.")
            st.plotly_chart(horizon_chart(result.metrics), use_container_width=True)
            name = st.selectbox("Inspect held-out predictions", list(result.predictions.model.unique()))
            st.plotly_chart(test_chart(result.predictions, name), use_container_width=True)
            st.download_button("Download test predictions", result.predictions.to_csv(index=False), "test-predictions.csv", "text/csv")
            st.download_button("Download metrics", result.metrics.to_csv(index=False), "metrics.csv", "text/csv")
            with st.expander("Validation selection and run details"):
                st.dataframe(result.tuning, use_container_width=True, hide_index=True)
                st.json(manifest)
    with forecast:
        if result is None:
            st.info("Complete a benchmark to generate a separate full-history forecast.")
        else:
            st.subheader("Seven days beyond the snapshot")
            st.caption("These forecasts come from separate full-history refits. Their future accuracy is unknown.")
            names = st.multiselect("Show forecasts", list(result.forecasts.model.unique()), default=list(result.forecasts.model.unique()))
            st.plotly_chart(forecast_chart(frame, result.forecasts, names), use_container_width=True)
            table = result.forecasts.pivot(index="target_date", columns="model", values="prediction").reset_index()
            st.dataframe(table, use_container_width=True, hide_index=True, column_config={
                "target_date": st.column_config.DatetimeColumn("Date · UTC", format="YYYY-MM-DD"),
                **{name: st.column_config.NumberColumn(format="$%.2f") for name in result.forecasts.model.unique()},
            })
            st.caption("Only ARIMA reports a model-based 95% prediction interval. Its assumptions may not describe cryptocurrency price shocks.")
            st.download_button("Download future forecasts", result.forecasts.to_csv(index=False), "future-forecasts.csv", "text/csv")
    with methodology:
        st.subheader("A forecast is only as credible as its evaluation")
        st.markdown("""
1. **Chronological split:** first 70% training, next 15% validation, final 15% test.
2. **Causal inputs:** trailing indicators and a 60-day lookback; indicator warm-up rows are dropped. Training targets cannot cross split boundaries. LSTM scalers see training history only.
3. **Validation-only selection:** choose candidate settings by seven-horizon MAE, then refit on training plus validation before testing.
4. **Matched evaluation:** all models predict identical, non-overlapping seven-day target blocks. Parameters stay fixed; newly observed history becomes available at each origin. ARIMA updates its state without refitting coefficients. Incomplete final blocks are excluded.
5. **Separate outlook:** after evaluation, refit the selected configurations on all completed observations and forecast the next seven dates.

**Interpretation:** the last-price baseline is a serious comparator. A complex model may fail to beat it. USD errors depend on the price regime; one chronological holdout does not establish generalization across all markets. These are forecasting experiments, not validated trading signals.
        """)
        st.caption("Reproducibility: every saved run includes exact input data, timestamps, seed, selected settings, package versions, and a SHA-256 fingerprint. Presentation controls do not trigger training.")


if __name__ == "__main__":
    main()
