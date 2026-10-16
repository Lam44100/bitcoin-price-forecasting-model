# Portfolio walkthrough

The demo uses the fixed real Yahoo snapshot and saved benchmark. It does not require retraining or a live data request. The video is a captioned walkthrough assembled from captured application screens; it has no voice narration.

## Open the same run

1. Start the app with `python -m streamlit run BTCPredictionModel.py`.
2. Under **Load or import a snapshot**, select the snapshot whose fingerprint begins `72977dc46866d33c`.
3. Keep all three models selected, with 12 LSTM epochs, 200 RF trees, and seed 42.
4. Click **Load matching saved run**. The complete local run folder and snapshot must be present. For a fresh clone, fetch and evaluate a new snapshot first; its metrics will differ.

## Two-and-a-half-minute story

| Time | Screen | Point to explain |
|---|---|---|
| 0:00–0:25 | Overview | The question is whether statistical and ML models beat a last-price forecast. Prices are completed daily closes, not live quotes. |
| 0:25–0:50 | Overview / source | The snapshot covers 1,825 real UTC daily candles. It is timestamped and fingerprinted; cached inputs make the demonstration repeatable. |
| 0:50–1:15 | Model Comparison | All models forecast identical seven-day blocks. Baseline MAE is $2,482; ARIMA matches it, while RF and LSTM are worse. |
| 1:15–1:40 | Comparison / errors | Inspect error by horizon and actual versus predicted prices. A more complex model is not automatically more useful. |
| 1:40–2:05 | Forecast | Future forecasts are separate full-history refits. Only ARIMA has a native model-based prediction interval; its coverage is not established here. |
| 2:05–2:30 | Methodology | Explain chronological splits, training-only scalers, validation-only selection, fixed test parameters, and the limits of one holdout. |

The README and benchmark report provide the evidence behind each statement. Do not describe the experiment as a profitable trading strategy or claim that it beats the baseline.

## Rebuild the video

Capture the corresponding application views into `docs/screenshots/`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe scripts\build_walkthrough.py
```

The script adds timed captions to the saved screens and produces a 150-second MP4 plus an SRT transcript under `artifacts/demo/`. Video files are ignored by Git; upload the finished clip to your chosen portfolio platform when ready.
