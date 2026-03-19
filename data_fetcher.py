"""
Real-time weather data fetcher for Astana, Kazakhstan.

This module is responsible for:
  1. Fetching current weather from OpenWeatherMap API (live source).
  2. Loading the last N hours from the local historical CSV (offline fallback).
  3. Applying feature engineering (cyclic temporal encodings) matching the
     training pipeline in src/preprocessing.py.
  4. Normalising and packaging data into the exact tensor shape [1, 24, 10]
     expected by both WeatherLSTM and HybridWeatherModel.

Usage:
    tensor, source = fetch_live_sequence(api_key="YOUR_KEY")
    # tensor: np.ndarray of shape [1, 24, 10]  or None on failure
    # source: human-readable string describing where data came from
"""

import os
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASTANA_LAT: float = 51.1694
ASTANA_LON: float = 71.4491

# Exact feature order used during training (must match preprocessing.py)
FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "dew_point",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",
]

N_FEATURES = len(FEATURE_COLUMNS)   # 10
SEQUENCE_LENGTH = 24                # hours


# ---------------------------------------------------------------------------
# Raw API helpers
# ---------------------------------------------------------------------------

def fetch_current_weather(api_key: str, timeout: int = 10) -> dict:
    """
    Fetch the current weather observation for Astana from OpenWeatherMap.

    Args:
        api_key: OpenWeatherMap API key (free tier is sufficient).
        timeout: HTTP request timeout in seconds.

    Returns:
        Raw JSON response dict from the /weather endpoint.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
        requests.Timeout: If the request exceeds `timeout` seconds.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": ASTANA_LAT,
        "lon": ASTANA_LON,
        "appid": api_key,
        "units": "metric",
    }
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_owm_response(data: dict) -> dict:
    """
    Parse an OpenWeatherMap /weather JSON response into a flat observation dict.

    Dew point is estimated via the Magnus approximation formula
    (±0.5 °C error for typical humidity levels):
        T_d ≈ T − (100 − RH) / 5

    Args:
        data: Raw JSON dict returned by `fetch_current_weather`.

    Returns:
        Dict with keys: time (datetime), temperature, humidity, pressure,
        wind_speed (km/h), dew_point (°C).
    """
    temp     = data["main"]["temp"]
    humidity = float(data["main"]["humidity"])
    pressure = float(data["main"]["pressure"])
    wind_ms  = data["wind"].get("speed", 0.0)
    wind_kmh = wind_ms * 3.6                      # m/s → km/h
    dew_pt   = temp - ((100.0 - humidity) / 5.0)  # Magnus approximation
    dt       = datetime.utcfromtimestamp(data["dt"])

    return {
        "time":        dt,
        "temperature": temp,
        "humidity":    humidity,
        "pressure":    pressure,
        "wind_speed":  wind_kmh,
        "dew_point":   dew_pt,
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic temporal encodings to a DataFrame that has a 'time' column.

    Cyclic (sin/cos) encoding preserves the circular nature of time
    (e.g. hour 23 is close to hour 0), which plain integer encoding breaks.

    Adds columns: hour_sin, hour_cos, month_sin, month_cos, day_of_week.

    Args:
        df: DataFrame with a 'time' column of dtype datetime64.

    Returns:
        Same DataFrame with the five new feature columns appended in-place.
    """
    df["hour_sin"]    = np.sin(2 * np.pi * df["time"].dt.hour   / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["time"].dt.hour   / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["time"].dt.month  / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["time"].dt.month  / 12)
    df["day_of_week"] = df["time"].dt.dayofweek.astype(float)
    return df


def add_cyclic_features_from_dt(obs: dict, dt: datetime) -> dict:
    """
    Add cyclic temporal features to a single observation dict.

    Args:
        obs: Flat observation dict (must already have the raw weather fields).
        dt:  The datetime of the observation.

    Returns:
        The same dict with hour_sin, hour_cos, month_sin, month_cos,
        day_of_week added.
    """
    obs["hour_sin"]    = np.sin(2 * np.pi * dt.hour  / 24)
    obs["hour_cos"]    = np.cos(2 * np.pi * dt.hour  / 24)
    obs["month_sin"]   = np.sin(2 * np.pi * dt.month / 12)
    obs["month_cos"]   = np.cos(2 * np.pi * dt.month / 12)
    obs["day_of_week"] = float(dt.weekday())
    return obs


# ---------------------------------------------------------------------------
# Open-Meteo (free, no API key required)
# ---------------------------------------------------------------------------

def fetch_openmeteo_data(
    past_days: int = 2,
    forecast_days: int = 2,
    timeout: int = 15,
) -> Optional[pd.DataFrame]:
    """
    Fetch hourly weather data for Astana from Open-Meteo.

    Completely free and requires no API key. Returns past observations
    merged with upcoming forecast in the same schema used by the training
    pipeline (wind speed in km/h, pressure in hPa).

    Args:
        past_days:     Days of historical data to include (max 92).
        forecast_days: Days of forecast to include (max 16).
        timeout:       HTTP request timeout in seconds.

    Returns:
        DataFrame with columns: time, temperature, humidity, pressure,
        wind_speed, dew_point — or None on failure.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        ASTANA_LAT,
        "longitude":       ASTANA_LON,
        "hourly":          (
            "temperature_2m,relativehumidity_2m,"
            "surface_pressure,windspeed_10m,dewpoint_2m"
        ),
        "past_days":       past_days,
        "forecast_days":   forecast_days,
        "timezone":        "UTC",          # match CSV (UTC)
        "wind_speed_unit": "kmh",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame({
        "time":        pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "humidity":    h["relativehumidity_2m"],
        "pressure":    h["surface_pressure"],
        "wind_speed":  h["windspeed_10m"],
        "dew_point":   h["dewpoint_2m"],
    })
    return df.dropna().reset_index(drop=True)


def fetch_openmeteo_forecast_temps(
    hours: int = 24,
    timeout: int = 15,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Return Open-Meteo's own hourly temperature forecast for the next `hours`.

    Used in the Streamlit app to compare the hybrid model's autoregressive
    predictions against Open-Meteo's professional NWP forecast — useful for
    thesis evaluation (how close is the model to a state-of-the-art service?).

    Args:
        hours:   Number of future hours to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple (timestamps: pd.Series, temperatures: pd.Series), or
        (None, None) on failure.
    """
    try:
        df = fetch_openmeteo_data(past_days=0, forecast_days=3, timeout=timeout)
        if df is None or len(df) == 0:
            return None, None
        now_utc = pd.Timestamp.utcnow().replace(tzinfo=None)
        future = df[df["time"] > now_utc].head(hours)
        if len(future) == 0:
            return None, None
        return (
            future["time"].reset_index(drop=True),
            future["temperature"].reset_index(drop=True),
        )
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# CSV fallback
# ---------------------------------------------------------------------------

def load_recent_from_csv(
    csv_path: str = "data/raw/astana_historical.csv",
    n_rows: int = 48,
) -> Optional[pd.DataFrame]:
    """
    Load the most recent `n_rows` observations from the historical CSV.

    Args:
        csv_path: Path to the Meteostat CSV file.
        n_rows:   How many rows to load from the end of the file.

    Returns:
        DataFrame sorted by time, or None if the file does not exist.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values("time").tail(n_rows).reset_index(drop=True)
    return df


def get_recent_temperatures(
    csv_path: str = "data/raw/astana_historical.csv",
    n_hours: int = 24,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Return the last `n_hours` of actual observed temperatures.

    Priority:
      1. Open-Meteo past observations (always current, free).
      2. Historical CSV (offline fallback).

    Used by the Streamlit Forecast page to draw the solid 'actual' line
    on the historical + forecast chart.

    Args:
        csv_path: Path to the historical CSV file (fallback).
        n_hours:  Number of hours to retrieve.

    Returns:
        Tuple (timestamps: pd.Series, temperatures: pd.Series), or
        (None, None) if no data source is available.
    """
    # ── Try Open-Meteo first ─────────────────────────────────────────────
    try:
        # forecast_days=1 ensures today's hours are included in the response
        om_df = fetch_openmeteo_data(past_days=2, forecast_days=1)
        if om_df is not None and len(om_df) > 0:
            now_utc = pd.Timestamp.utcnow().replace(tzinfo=None)
            hist = om_df[om_df["time"] <= now_utc].tail(n_hours).reset_index(drop=True)
            if len(hist) >= n_hours // 2:          # at least half the window
                return hist["time"], hist["temperature"]
    except Exception:
        pass

    # ── Fallback: CSV ───────────────────────────────────────────────────
    df = load_recent_from_csv(csv_path, n_rows=n_hours)
    if df is None or len(df) == 0:
        return None, None
    return (
        df["time"].reset_index(drop=True),
        df["temperature"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------

def _normalise_sequence(
    raw: np.ndarray,
    scaler_path: str,
    seq_len: int = SEQUENCE_LENGTH,
) -> Optional[np.ndarray]:
    """
    Normalise a raw [seq_len, n_features] array using the saved MinMaxScaler.

    Args:
        raw:         Array of shape [seq_len, N_FEATURES] in physical units.
        scaler_path: Path to scaler.pkl saved during training.
        seq_len:     Sequence length expected by the model.

    Returns:
        Normalised array of shape [1, seq_len, N_FEATURES], or None on error.
    """
    if not os.path.exists(scaler_path):
        return None
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    normalised = scaler.transform(raw[-seq_len:])
    return normalised.reshape(1, seq_len, N_FEATURES)


def build_tensor_from_df(
    df: pd.DataFrame,
    scaler_path: str = "data/processed/scaler.pkl",
    seq_len: int = SEQUENCE_LENGTH,
) -> Optional[np.ndarray]:
    """
    Convert a DataFrame of weather observations into a model input tensor.

    Applies feature engineering if the cyclic columns are absent, then
    normalises using the training scaler.

    Args:
        df:          DataFrame with at least the columns in FEATURE_COLUMNS
                     (except the cyclic ones, which are auto-computed from 'time').
        scaler_path: Path to the MinMaxScaler pickle saved during training.
        seq_len:     Number of time-steps required by the model (24).

    Returns:
        np.ndarray of shape [1, seq_len, N_FEATURES], ready for torch.FloatTensor,
        or None if data is insufficient or scaler is missing.
    """
    if "hour_sin" not in df.columns:
        df = add_cyclic_features(df.copy())

    df = df.dropna(subset=FEATURE_COLUMNS)
    if len(df) < seq_len:
        return None

    raw = df.tail(seq_len)[FEATURE_COLUMNS].values.astype(np.float32)
    return _normalise_sequence(raw, scaler_path, seq_len)


# ---------------------------------------------------------------------------
# Autoregressive forecast helpers
# ---------------------------------------------------------------------------

def compute_next_cyclic_features(base_time: datetime, step: int) -> dict:
    """
    Compute normalised cyclic temporal features for a future time-step.

    Instead of re-running the scaler (which would require reconstructing all
    10 features), we exploit the fact that cyclic (sin/cos) features always
    lie in [−1, 1], so MinMaxScaler maps them to [(x+1)/2].
    day_of_week ∈ [0, 6] is mapped to x/6.

    Args:
        base_time: The datetime of the last known observation.
        step:      Number of hours ahead (1-indexed).

    Returns:
        Dict with normalised values for hour_sin, hour_cos, month_sin,
        month_cos, day_of_week at the target future time.
    """
    future_dt = base_time + timedelta(hours=step)
    h_sin = np.sin(2 * np.pi * future_dt.hour  / 24)
    h_cos = np.cos(2 * np.pi * future_dt.hour  / 24)
    m_sin = np.sin(2 * np.pi * future_dt.month / 12)
    m_cos = np.cos(2 * np.pi * future_dt.month / 12)
    dow   = float(future_dt.weekday())

    return {
        "hour_sin":    (h_sin + 1) / 2,   # MinMaxScaler: (x - (-1)) / (1-(-1))
        "hour_cos":    (h_cos + 1) / 2,
        "month_sin":   (m_sin + 1) / 2,
        "month_cos":   (m_cos + 1) / 2,
        "day_of_week": dow / 6,            # MinMaxScaler: x / 6
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_live_sequence(
    api_key: Optional[str] = None,
    csv_path: str = "data/raw/astana_historical.csv",
    scaler_path: str = "data/processed/scaler.pkl",
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Build a ready-to-use model input tensor for the current moment in Astana.

    Priority order:
      1. OpenWeatherMap live observation merged with 23 rows from historical CSV.
         Provides the most up-to-date sequence (requires valid `api_key`).
      2. Last `sequence_length` rows from the historical CSV (offline fallback).
      3. Returns (None, error_message) if no data source is available.

    Args:
        api_key:         OpenWeatherMap API key.  Pass None to use CSV only.
        csv_path:        Path to the historical CSV file.
        scaler_path:     Path to the MinMaxScaler pickle from training.
        sequence_length: Number of time-steps (default 24).

    Returns:
        Tuple (tensor, source_label):
          - tensor: np.ndarray [1, sequence_length, N_FEATURES] or None.
          - source_label: Human-readable string for display in the UI.
    """
    hist_df = load_recent_from_csv(csv_path, n_rows=sequence_length + 24)

    # ---- Attempt Open-Meteo (primary: free, no key, always current) ----
    try:
        # forecast_days=1 ensures current-day hours are included
        om_df = fetch_openmeteo_data(past_days=2, forecast_days=1)
        if om_df is not None:
            now_utc = pd.Timestamp.utcnow().replace(tzinfo=None)
            om_hist = om_df[om_df["time"] <= now_utc]
            tensor = build_tensor_from_df(om_hist, scaler_path, sequence_length)
            if tensor is not None:
                return tensor, "Open-Meteo (live, free)"
    except Exception as _om_exc:
        pass   # fall through to OWM / CSV

    # ---- Attempt OWM (requires valid paid key) -------------------------
    if api_key:
        try:
            raw_json = fetch_current_weather(api_key)
            current  = parse_owm_response(raw_json)
            current  = add_cyclic_features_from_dt(current, current["time"])

            if hist_df is not None and len(hist_df) >= sequence_length - 1:
                hist_slice = add_cyclic_features(hist_df.copy())
                hist_slice = hist_slice.dropna(subset=FEATURE_COLUMNS)
                # Take the 23 most recent historical rows + 1 live observation
                hist_part = hist_slice.tail(sequence_length - 1)[FEATURE_COLUMNS].values
                live_row  = np.array([[current[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
                combined  = np.vstack([hist_part, live_row]).astype(np.float32)
            else:
                # No historical data: repeat the single live observation
                live_row = np.array([[current[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
                combined = np.repeat(live_row, sequence_length, axis=0)

            tensor = _normalise_sequence(combined, scaler_path, sequence_length)
            source = "OpenWeatherMap (live) + historical CSV"
            return tensor, source

        except Exception as exc:
            # Fall through to CSV-only mode
            _source_err = f"historical CSV (API error: {exc})"
    else:
        _source_err = "historical CSV (no API key)"

    # ---- Fallback: CSV only -------------------------------------------
    if hist_df is None:
        return None, "no data available — CSV not found"

    tensor = build_tensor_from_df(hist_df, scaler_path, sequence_length)
    return tensor, _source_err


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 data_fetcher self-test (CSV fallback) …")
    tensor, source = fetch_live_sequence(api_key=None)
    if tensor is not None:
        print(f"✅ Tensor shape : {tensor.shape}")
        print(f"   Source       : {source}")
        print(f"   Value range  : [{tensor.min():.3f}, {tensor.max():.3f}]")
    else:
        print(f"⚠️  No tensor produced. Source: {source}")

    print("\n📊 Recent temperatures:")
    times, temps = get_recent_temperatures()
    if times is not None:
        for t, temp in zip(times.tail(5), temps.tail(5)):
            print(f"   {t}  →  {temp:.1f} °C")
