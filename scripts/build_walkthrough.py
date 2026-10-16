"""Build a 150-second captioned walkthrough from genuine captured app screens."""
from pathlib import Path
import subprocess

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "docs" / "screenshots"
OUTPUT = ROOT / "artifacts" / "demo"

SCENES = [
    ("overview.png", [
        "Bitcoin Forecasting: Comparing Statistical and Machine Learning Models",
        "The question: can ARIMA, Random Forest or LSTM beat the last observed price?",
        "The dashboard uses completed daily UTC candles. These are not live quotes."]),
    ("overview.png", [
        "This fixed Yahoo snapshot contains 1,825 daily observations, from October 2021 to October 2026.",
        "The source, retrieval time and data fingerprint identify exactly what was used.",
        "Saved runs reopen offline. Changing model settings invalidates results; chart controls do not retrain."]),
    ("comparison.png", [
        "Identical seven-day forecast blocks make the comparison fair: 39 origins and 273 held-out targets.",
        "Last-price MAE is $2,482.47. ARIMA selected a random walk and matched the baseline.",
        "Random Forest and LSTM had higher errors on this holdout. Complexity did not guarantee improvement."]),
    ("held-out.png", [
        "Inspect predictions against observations instead of relying only on aggregate scores.",
        "Error by horizon shows how the task changes from one day ahead to seven days ahead.",
        "One chronological holdout does not establish performance across every market regime."]),
    ("forecast.png", [
        "The future outlook is a separate refit using all completed observations in this snapshot.",
        "Every forecast starts after the final observed candle. Historical snapshots keep their original dates.",
        "Only ARIMA shows a native model-based 95% prediction interval. Future accuracy remains unknown."]),
    ("methodology.png", [
        "Chronological 70/15/15 splits; causal indicators; training-only scalers; no labels crossing boundaries.",
        "Validation selects settings. Test parameters stay fixed, and every model receives the same new history.",
        "The result is a reproducible experiment with honest limitations, not a validated trading strategy."]),
]


def timestamp(seconds):
    milliseconds = int(round(seconds * 1000))
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"00:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines, captions = [], []
    for index, (filename, text) in enumerate(SCENES):
        path = SCREENS / filename
        if not path.exists():
            raise SystemExit(f"Missing genuine dashboard capture: {path}")
        # ffconcat resolves these paths relative to artifacts/demo/.
        lines.extend([f"file '../../docs/screenshots/{filename}'", "duration 25"])
        for j, caption in enumerate(text):
            start, end = index * 25 + j * 25 / 3, index * 25 + (j + 1) * 25 / 3
            captions.extend([str(len(captions) // 4 + 1), f"{timestamp(start)} --> {timestamp(end)}", caption, ""])
    lines.append(f"file '../../docs/screenshots/{SCENES[-1][0]}'")
    (OUTPUT / "screens.ffconcat").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "walkthrough.srt").write_text("\n".join(captions), encoding="utf-8")
    movie = OUTPUT / "bitcoin-forecasting-walkthrough.mp4"
    video_filter = ("fps=12,scale=1440:960:force_original_aspect_ratio=decrease,"
        "pad=1440:1080:(ow-iw)/2:0:color=0x0f172a,"
        "subtitles=walkthrough.srt:force_style='Fontname=Arial,Fontsize=8,PrimaryColour=&HFFFFFF,"
        "Outline=1,Shadow=0,MarginV=4,MarginL=15,MarginR=15,Alignment=2'")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "concat", "-safe", "0", "-i", "screens.ffconcat", "-vf", video_filter,
        "-r", "12", "-t", "150", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie)], cwd=OUTPUT, check=True)
    print(movie)


if __name__ == "__main__":
    main()
