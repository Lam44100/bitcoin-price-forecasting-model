"""Charts share the same timestamped tables used by metrics and CSV downloads."""
import plotly.graph_objects as go

COLORS = {"Last price": "#94a3b8", "Random Forest": "#0d9488", "ARIMA": "#d97706", "LSTM": "#6366f1"}


def style(fig, title, yaxis="Closing price · USD"):
    fig.update_layout(title=title, template="plotly_white", height=480, hovermode="x unified",
                      margin=dict(l=20, r=20, t=60, b=110), yaxis_title=yaxis,
                      legend=dict(orientation="h", y=-0.20, yanchor="top", traceorder="normal"),
                      font=dict(family="Arial", color="#334155"))
    return fig


def history_chart(frame, overlays):
    recent = frame.tail(365)
    fig = go.Figure(go.Scatter(x=recent.Datetime, y=recent.Close, name="BTC-USD", line=dict(color="#0f172a", width=2)))
    if "20-day average" in overlays:
        fig.add_trace(go.Scatter(x=recent.Datetime, y=frame.Close.rolling(20).mean().tail(365), name="20-day average", line=dict(color="#0d9488")))
    if "Bollinger bands" in overlays:
        ma, spread = frame.Close.rolling(20).mean(), 2 * frame.Close.rolling(20).std()
        fig.add_trace(go.Scatter(x=recent.Datetime, y=(ma - spread).tail(365), name="Lower band", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=recent.Datetime, y=(ma + spread).tail(365), name="Bollinger bands", line=dict(width=0), fill="tonexty", fillcolor="rgba(13,148,136,0.12)"))
    return style(fig, "Daily closing prices · latest 365 observations")


def horizon_chart(metrics):
    rows = metrics[metrics.horizon != "Overall"].copy()
    rows["horizon"] = rows.horizon.astype(int)
    fig = go.Figure()
    for name, group in rows.groupby("model", sort=False):
        fig.add_trace(go.Scatter(x=group.horizon, y=group.MAE, name=name,
                                mode="lines+markers", line=dict(color=COLORS[name])))
    fig.update_xaxes(title="Forecast horizon · days", dtick=1)
    return style(fig, "How error changes with forecast horizon", "MAE · USD")


def test_chart(predictions, model):
    rows = predictions[predictions.model == model].sort_values("target_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.target_date, y=rows.actual, name="Observed", line=dict(color="#0f172a")))
    fig.add_trace(go.Scatter(x=rows.target_date, y=rows.prediction, name=model, line=dict(color=COLORS[model], dash="dash")))
    return style(fig, f"Held-out observations vs {model} · seven-day forecast blocks")


def forecast_chart(frame, forecasts, models):
    recent = frame.tail(60)
    fig = go.Figure(go.Scatter(x=recent.Datetime, y=recent.Close, name="Observed", line=dict(color="#0f172a")))
    for name in models:
        rows = forecasts[forecasts.model == name]
        if name == "ARIMA" and rows.lower_95.notna().all():
            fig.add_trace(go.Scatter(x=rows.target_date, y=rows.lower_95, mode="lines", line=dict(width=0), showlegend=False, name="ARIMA lower 95%"))
            fig.add_trace(go.Scatter(x=rows.target_date, y=rows.upper_95, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(217,119,6,0.13)", name="ARIMA 95% prediction interval"))
        fig.add_trace(go.Scatter(x=rows.target_date, y=rows.prediction, name=name, mode="lines+markers", line=dict(color=COLORS[name], dash="dash")))
    return style(fig, "Seven-day outlook · fitted on all completed observations")
