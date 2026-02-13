"""
Calibration Tool — Computes bias corrections and error distributions.

Fetches historical forecast vs actual data to determine:
  1. Per-city bias correction (are models systematically high or low?)
  2. Real forecast error standard deviation (replacing guesses)
  3. Error distribution shape (Gaussian vs t-distribution fit)

Data sources:
  - Open-Meteo Historical API: past model forecasts
  - NWS Climate Reports: actual observed highs (via api.weather.gov)

Usage:
    python calibrate.py                # run full calibration
    python calibrate.py --days 30      # calibrate with 30 days of data
    python calibrate.py --city NYC     # calibrate just one city
"""

import sys
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import requests
import numpy as np
from scipy import stats

from config import CITIES, MODEL_WEIGHTS

NWS_HEADERS = {"User-Agent": "KalshiWeatherBot/2.0 (contact@example.com)"}
CALIBRATION_FILE = Path(__file__).parent / "data" / "calibration.json"


def fetch_nws_observed_highs(city_key: str, start_date: str, end_date: str) -> dict:
    """
    Fetch actual observed daily high temperatures from NWS grid data.

    NWS gridpoint data includes recent observations in the maxTemperature field.
    For historical data beyond the grid cache, we fall back to Open-Meteo archive.

    Returns dict of {date_str: actual_high_f}.
    """
    city = CITIES[city_key]
    office, gx, gy = city["nws_grid"]

    try:
        url = f"https://api.weather.gov/gridpoints/{office}/{gx},{gy}"
        resp = requests.get(url, headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        max_temps = data["properties"]["maxTemperature"]["values"]
        result = {}
        for v in max_temps:
            date_str = v["validTime"][:10]
            if start_date <= date_str <= end_date:
                temp_c = v["value"]
                if temp_c is not None:
                    result[date_str] = round(temp_c * 9.0 / 5.0 + 32.0, 1)

        return result

    except Exception as e:
        print(f"  [WARN] NWS observed data failed for {city_key}: {e}")
        return {}


def fetch_open_meteo_historical(city_key: str, start_date: str, end_date: str) -> dict:
    """
    Fetch historical actual temperatures from Open-Meteo archive API.

    This gives us the real observed temperatures for past dates.
    Returns dict of {date_str: actual_high_f}.
    """
    city = CITIES[city_key]
    lat, lon = city["lat"], city["lon"]

    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])

        result = {}
        for d, t in zip(dates, max_temps):
            if t is not None:
                result[d] = round(t, 1)

        return result

    except Exception as e:
        print(f"  [WARN] Open-Meteo archive failed for {city_key}: {e}")
        return {}


def fetch_model_hindcasts(city_key: str, target_date: str) -> dict:
    """
    Fetch what models WOULD HAVE predicted for a past date.

    Uses Open-Meteo's forecast API with the current model state.
    Note: This is an approximation — true hindcasts would need archived
    model runs. For now, we use recent forecast behavior as a proxy.

    Returns dict of {model_name: predicted_high_f}.
    """
    from weather import get_all_model_highs
    return get_all_model_highs(city_key, target_date)


def compute_city_calibration(city_key: str, n_days: int = 14) -> dict:
    """
    Compute calibration statistics for one city by comparing
    ensemble forecasts against ACTUAL observed temperatures.

    Two data sources for forecasts:
      1. Stored model_forecasts from DB (logged by scanner during operation)
      2. Current model runs as a proxy (fallback for days without stored data)

    Returns dict with:
      - bias: mean(ensemble - actual), positive = models too warm
      - std: standard deviation of ensemble errors
      - mad: median absolute deviation (robust)
      - n_samples: number of data points
      - errors: list of (ensemble - actual) values
      - model_errors: per-model error statistics
      - fitted_df: Student's t degrees of freedom fitted to error distribution
    """
    from weather import get_all_model_highs, compute_weighted_ensemble
    import db

    city_name = CITIES[city_key]["name"]
    print(f"\n  Calibrating {city_key} ({city_name})...")

    # Get actual observed highs for recent days
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=n_days + 1)).strftime("%Y-%m-%d")

    # Try NWS first (recent days), fall back to Open-Meteo archive (longer history)
    actuals = fetch_nws_observed_highs(city_key, start_date, end_date)
    if len(actuals) < 3:
        print(f"    NWS returned {len(actuals)} days, trying Open-Meteo archive...")
        om_actuals = fetch_open_meteo_historical(city_key, start_date, end_date)
        # Merge: prefer NWS where available (official observations)
        for d, t in om_actuals.items():
            if d not in actuals:
                actuals[d] = t

    if not actuals:
        print(f"    No historical data available for {city_key}")
        return {"bias": 0.0, "std": 3.0, "mad": 2.0, "n_samples": 0,
                "errors": [], "model_errors": {}, "fitted_df": 8.0}

    print(f"    Got {len(actuals)} days of actual observed highs")

    # Compare ensemble forecasts against actual highs
    ensemble_errors = []
    model_errors_by_name = {}  # {model_name: [errors]}

    for date_str, actual_high in sorted(actuals.items()):
        # Try stored forecasts from DB first (most accurate — what we actually used)
        stored = db.get_forecasts_for_date(city_key, date_str)
        if stored:
            model_highs = {f["model_name"]: f["forecast_high_f"] for f in stored}
        else:
            # Fallback: fetch current model runs as proxy
            # This is imperfect for past dates but better than nothing
            model_highs = get_all_model_highs(city_key, date_str)

        if not model_highs:
            continue

        # Compute weighted ensemble mean for this date
        w_mean, w_std, n = compute_weighted_ensemble(model_highs)
        if w_mean is None:
            continue

        # Ensemble error (positive = forecast too warm)
        ens_error = w_mean - actual_high
        ensemble_errors.append(ens_error)

        # Per-model errors
        for model_name, temp in model_highs.items():
            err = temp - actual_high
            model_errors_by_name.setdefault(model_name, []).append(err)

        source = "DB" if stored else "live"
        print(f"    {date_str}: actual={actual_high:.1f}°F, "
              f"ensemble={w_mean:.1f}°F, error={ens_error:+.1f}°F "
              f"[{source}, n={n}]")

    if not ensemble_errors:
        print(f"    No forecast-vs-actual comparisons available")
        return {"bias": 0.0, "std": 3.0, "mad": 2.0, "n_samples": 0,
                "errors": [], "model_errors": {}, "fitted_df": 8.0}

    # Aggregate ensemble statistics
    errors_arr = np.array(ensemble_errors)
    bias = float(np.mean(errors_arr))
    std = float(np.std(errors_arr, ddof=1)) if len(errors_arr) > 1 else 3.0
    mad = float(np.median(np.abs(errors_arr)))

    print(f"\n    Ensemble vs Actuals ({len(ensemble_errors)} days):")
    print(f"      Bias (ensemble - actual): {bias:+.1f}°F")
    print(f"      Std:  {std:.1f}°F")
    print(f"      MAD:  {mad:.1f}°F")

    # Per-model summary
    print(f"\n    Per-model accuracy:")
    model_stats = {}
    for model_name, errs in sorted(model_errors_by_name.items(),
                                    key=lambda x: abs(np.mean(x[1]))):
        m_bias = float(np.mean(errs))
        m_mae = float(np.mean(np.abs(errs)))
        m_n = len(errs)
        model_stats[model_name] = {"bias": m_bias, "mae": m_mae, "n": m_n}
        print(f"      {model_name:20s}: bias={m_bias:+.1f}°F, "
              f"MAE={m_mae:.1f}°F, n={m_n}")

    # Fit Student's t-distribution to error distribution (Phase 7)
    fitted_df = 8.0  # default moderate fat tails
    if len(ensemble_errors) >= 20:
        try:
            df_fit, _, _ = stats.t.fit(errors_arr)
            fitted_df = float(max(3.0, min(50.0, df_fit)))
            print(f"\n    Fitted t-distribution df: {fitted_df:.1f}")
        except Exception:
            pass

    result = {
        "bias": bias,
        "std": std,
        "mad": mad,
        "n_samples": len(ensemble_errors),
        "errors": [float(e) for e in ensemble_errors],
        "model_errors": model_stats,
        "fitted_df": fitted_df,
    }

    return result


def run_calibration(n_days: int = 14, city_filter: str = None):
    """Run calibration for all (or selected) cities."""
    print("=" * 70)
    print("CALIBRATION — Model Error Analysis")
    print(f"  Period: {n_days} days")
    print("=" * 70)

    calibration = {}

    for city_key in CITIES:
        if city_filter and city_key != city_filter:
            continue

        cal = compute_city_calibration(city_key, n_days)
        calibration[city_key] = cal

    # Summary
    print(f"\n{'='*70}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'City':<6} {'Bias':>8} {'Std':>8} {'MAD':>8} {'Samples':>8}")
    print("-" * 40)

    for city_key, cal in calibration.items():
        print(f"{city_key:<6} {cal['bias']:>+7.1f}°F {cal['std']:>7.1f}°F "
              f"{cal['mad']:>7.1f}°F {cal['n_samples']:>7d}")

    # Save calibration data
    CALIBRATION_FILE.parent.mkdir(exist_ok=True)
    existing = {}
    if CALIBRATION_FILE.exists():
        try:
            existing = json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            pass

    existing["last_updated"] = datetime.now().isoformat()
    existing["cities"] = {}
    for city_key, cal in calibration.items():
        existing["cities"][city_key] = {
            "bias": cal["bias"],
            "std": cal["std"],
            "mad": cal["mad"],
            "n_samples": cal["n_samples"],
            "fitted_df": cal.get("fitted_df", 8.0),
            "model_errors": cal.get("model_errors", {}),
        }

    CALIBRATION_FILE.write_text(json.dumps(existing, indent=2))
    print(f"\nCalibration saved to {CALIBRATION_FILE}")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print("Update config.py bias_correction values:")
    for city_key, cal in calibration.items():
        bias = cal["bias"]
        if abs(bias) > 0.5:
            # Bias correction goes in opposite direction:
            # if models are too cold (bias negative), add positive correction
            correction = -bias
            print(f"  {city_key}: bias_correction = {correction:+.1f}  "
                  f"(models are {abs(bias):.1f}°F too {'cold' if bias < 0 else 'warm'})")
        else:
            print(f"  {city_key}: bias_correction = 0.0  (OK, <0.5°F bias)")

    return calibration


if __name__ == "__main__":
    n_days = 14
    city_filter = None

    for i, arg in enumerate(sys.argv):
        if arg == "--days" and i + 1 < len(sys.argv):
            n_days = int(sys.argv[i + 1])
        if arg == "--city" and i + 1 < len(sys.argv):
            city_filter = sys.argv[i + 1].upper()

    run_calibration(n_days, city_filter)
