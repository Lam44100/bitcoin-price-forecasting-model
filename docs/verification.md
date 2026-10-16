# Verification record

Verified locally on **16 October 2026**, using Windows and 64-bit Python 3.11.5.

| Check | Result |
|---|---|
| Install from `requirements.txt` into a new isolated venv | Passed |
| `pip check` in the new environment | Passed; no broken requirements |
| Full unittest suite in the new environment | 14 tests passed |
| Actual Yahoo daily-data retrieval through the implemented adapter | Passed; 1,825 completed candles saved |
| Full baseline / RF / ARIMA / LSTM benchmark | Passed; validation selection, test evaluation and final refits completed |
| Same snapshot benchmark in the fresh environment | Passed; matched origins and targets; predictions within $0.10 absolute tolerance |
| AppTest empty-cache, successful-run, rerun and invalidation flows | Passed |
| Browser inspection | Overview, comparison and future forecast charts rendered; saved results loaded without training |
| Final presentation changes | All four AppTest tests rerun successfully after chart/table changes |
| Captioned walkthrough | H.264 MP4, 1440 × 1080, 150 seconds; sampled frames checked for readable, correctly timed captions |

The largest per-prediction difference between the reference and fresh-environment runs was **$0.0234375**, in the LSTM held-out predictions. Its future predictions differed by at most **$0.0078125**. Baseline and ARIMA were identical; RF differences were below $0.00000000003. See the [machine-readable reproduction comparison](benchmark/reproduction-check.json).

Reference run: `20261016T041636624217Z-c47b2dc512`.
Fresh-environment run: `20261016T072424943313Z-c47b2dc512`.

## Environment-specific observations

- An initial restricted-network request could not resolve Yahoo; a direct public endpoint check returned HTTP 429. The actual yfinance adapter later succeeded outside the restricted environment. Cached snapshots avoid depending on a new request during a demo.
- The sandbox blocked atomic replacement of a pointer file in its temporary directory. The same snapshot round-trip test and the full suite passed outside that restriction; the application's timestamped cache also saved successfully.
- TensorFlow 2.15 emits upstream deprecation warnings on Python 3.11. They did not prevent training or deterministic-operation checks.
- Port 8501 already served another Streamlit process. The updated application was inspected on **127.0.0.1:8502** without stopping the existing process.

The GitHub Actions workflow is supplied but has not been run on GitHub. Other operating systems and GPU training were not verified. The tolerance check establishes local reproducibility for this saved snapshot, not bit-identical behavior on arbitrary hardware.
