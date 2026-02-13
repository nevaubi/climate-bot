"""
Weather data ingestion and probability calculation — v3.

Data sources (8 models + observations):
  1. NWS Official Forecast (api.weather.gov) — bias-corrected blend of 40+ models
  2. HRRR (NOAA High-Resolution Rapid Refresh, via Open-Meteo) — best US short-range
  3. GFS (NOAA, via Open-Meteo)
  4. ECMWF IFS (European, via Open-Meteo)
  5. JMA (Japanese, via Open-Meteo)
  6. GEM (Canadian, via Open-Meteo)
  7. DWD ICON (German, via Open-Meteo)
  8. Meteo-France ARPEGE (French, via Open-Meteo)
  + METAR observations from aviationweather.gov

Key improvements over v2:
  - HRRR model (3km resolution, hourly updates, best for same-day US temps)
  - Lead-time-dependent model weights (HRRR weighted high for same-day, ECMWF for 2+ days)
  - METAR temperature history tracking (not just latest reading)
  - Improved diurnal model using forecast-vs-observed delta
  - Time-varying observation/forecast blending for intraday updates
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import numpy as np
from scipy import stats

from config import CITIES, MODEL_WEIGHTS, get_model_weights
import db


# =======================================================================
# NWS Official Forecast
# =======================================================================

NWS_HEADERS = {"User-Agent": "KalshiWeatherBot/3.0 (contact@example.com)"}


def fetch_nws_forecast(city_key: str) -> dict:
    """
    Fetch official NWS maxTemperature forecast for a city.

    Uses the gridpoint API which returns multi-day max temps.
    Returns dict of {date_str: temp_f} for available forecast days.
    """
    city = CITIES[city_key]
    office, gx, gy = city["nws_grid"]
    url = f"https://api.weather.gov/gridpoints/{office}/{gx},{gy}"

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=NWS_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            max_temps = data.get("properties", {}).get("maxTemperature", {})
            values = max_temps.get("values", [])

            result = {}
            for v in values:
                valid_time = v.get("validTime", "")
                temp_c = v.get("value")
                if temp_c is None:
                    continue

                date_str = valid_time[:10]
                temp_f = temp_c * 9.0 / 5.0 + 32.0
                result[date_str] = round(temp_f, 1)

            return result

        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as retry_err:
            if attempt < 2:
                import time
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"  [WARN] NWS forecast failed for {city_key} after 3 retries: {retry_err}")
            return {}
        except Exception as e:
            print(f"  [WARN] NWS forecast failed for {city_key}: {e}")
            return {}

    return {}


# =======================================================================
# Open-Meteo Multi-Model Forecasts (now includes HRRR)
# =======================================================================

OPEN_METEO_MODELS = [
    {
        "name": "hrrr",
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "ncep_hrrr_conus"},
    },
    {
        "name": "gfs_seamless",
        "url": "https://api.open-meteo.com/v1/gfs",
        "params": {"models": "gfs_seamless"},
    },
    {
        "name": "ecmwf_ifs025",
        "url": "https://api.open-meteo.com/v1/ecmwf",
        "params": {},
    },
    {
        "name": "jma_seamless",
        "url": "https://api.open-meteo.com/v1/jma",
        "params": {"models": "jma_seamless"},
    },
    {
        "name": "gem_seamless",
        "url": "https://api.open-meteo.com/v1/gem",
        "params": {"models": "gem_seamless"},
    },
    {
        "name": "dwd_icon",
        "url": "https://api.open-meteo.com/v1/dwd-icon",
        "params": {},
    },
    {
        "name": "arpege_seamless",
        "url": "https://api.open-meteo.com/v1/meteofrance",
        "params": {"models": "arpege_seamless"},
    },
]


def fetch_open_meteo_forecasts(city_key: str) -> dict:
    """
    Fetch temperature forecasts from all Open-Meteo models for a city.

    Returns dict keyed by model name with hourly temperature data.
    """
    city = CITIES[city_key]
    lat, lon = city["lat"], city["lon"]
    results = {}

    for model_cfg in OPEN_METEO_MODELS:
        name = model_cfg["name"]
        url = model_cfg["url"]

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "forecast_days": 3,
            **model_cfg["params"],
        }

        try:
            resp = None
            for attempt in range(3):
                try:
                    resp = requests.get(url, params=params, timeout=12)
                    resp.raise_for_status()
                    break
                except (requests.exceptions.SSLError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout) as retry_err:
                    if attempt < 2:
                        import time
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise retry_err

            if resp is None:
                continue

            data = resp.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])

            results[name] = {"times": times, "temps": temps}

        except Exception as e:
            print(f"  [WARN] Open-Meteo {name} failed for {city_key}: {e}")
            continue

    return results


def extract_daily_high(forecast_data: dict, target_date: str) -> dict:
    """
    Extract the forecasted daily high temperature for a specific date
    from each model's hourly data.

    Returns dict of {model_name: forecasted_high_temp_f}.
    """
    highs = {}

    for model_name, model_data in forecast_data.items():
        times = model_data["times"]
        temps = model_data["temps"]

        day_temps = []
        for t, temp in zip(times, temps):
            if temp is None:
                continue
            if t.startswith(target_date):
                day_temps.append(temp)

        if day_temps:
            highs[model_name] = max(day_temps)

    return highs


# =======================================================================
# Combined Forecast (NWS + Open-Meteo Ensemble)
# =======================================================================

def get_all_model_highs(city_key: str, target_date: str) -> dict:
    """
    Fetch forecasted daily high from ALL sources (NWS + 7 Open-Meteo models).

    Returns dict of {model_name: temp_f} for all models that returned data.
    """
    all_highs = {}

    # 1. NWS Official
    nws_data = fetch_nws_forecast(city_key)
    if target_date in nws_data:
        all_highs["nws_official"] = nws_data[target_date]

    # 2. Open-Meteo models (includes HRRR)
    om_data = fetch_open_meteo_forecasts(city_key)
    om_highs = extract_daily_high(om_data, target_date)
    all_highs.update(om_highs)

    return all_highs


def compute_weighted_ensemble(model_highs: dict,
                              weights_override: dict = None) -> tuple:
    """
    Compute weighted ensemble mean and standard deviation.

    Args:
        model_highs: dict of {model_name: temp_f}
        weights_override: optional dict of {model_name: weight} to use
            instead of default MODEL_WEIGHTS. Used for lead-time-dependent
            weighting.

    Returns (weighted_mean, weighted_std, n_models).
    """
    if not model_highs:
        return None, None, 0

    weight_source = weights_override or MODEL_WEIGHTS

    temps = []
    weights = []

    for model_name, temp in model_highs.items():
        w = weight_source.get(model_name, 1.0)
        temps.append(temp)
        weights.append(w)

    temps = np.array(temps)
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize

    weighted_mean = np.average(temps, weights=weights)

    # Weighted standard deviation
    variance = np.average((temps - weighted_mean) ** 2, weights=weights)
    weighted_std = math.sqrt(variance) if variance > 0 else 0.0

    return float(weighted_mean), float(weighted_std), len(temps)


# =======================================================================
# Model Correlation Groups (Phase 5)
# =======================================================================

# Models in the same group share initialization data and/or physical schemes.
# We count effective independent information sources to avoid overconfidence.
MODEL_CORRELATION_GROUPS = {
    "nws": ["nws_official"],          # Independent (bias-corrected super-ensemble)
    "hrrr": ["hrrr"],                 # Semi-independent (US mesoscale, 3km)
    "global_ops": ["gfs_seamless", "ecmwf_ifs025"],  # Major operational models
    "euro_regional": ["dwd_icon", "arpege_seamless"],  # European regional (Day 1: ARPEGE best, DWD decent)
    "low_skill": ["jma_seamless", "gem_seamless"],     # Day 1: JMA MAE 4.7F, GEM MAE 3.1F — isolated
}

# Maximum fraction of total weight any correlation group can contribute
MAX_GROUP_WEIGHT_FRAC = 0.35


def _count_effective_groups(model_highs: dict) -> int:
    """Count how many independent model groups have data."""
    groups_present = 0
    for members in MODEL_CORRELATION_GROUPS.values():
        if any(m in model_highs for m in members):
            groups_present += 1
    return groups_present


# =======================================================================
# Probability Calculation (Student's t-distribution)
# =======================================================================

def calculate_bracket_probabilities(
    model_highs: dict,
    brackets: list,
    base_error_std: float = 2.5,
    city_key: str = None,
    weights_override: dict = None,
) -> dict:
    """
    Calculate probability distribution across temperature brackets
    using weighted ensemble and Student's t-distribution.

    Improvements over v3:
      - Correlation-aware spread inflation (correlated models don't reduce uncertainty)
      - Group weight capping (prevents 4 regional models drowning NWS/HRRR)
      - Dynamic t-distribution df from calibration (default df=8 for fat tails)
      - Higher uncertainty floor (2.0F, reflecting real MOS error levels)
    """
    if not model_highs:
        n = len(brackets)
        return {b[2]: 1.0 / n for b in brackets}

    # --- Group weight capping (Phase 5B) ---
    weight_source = (weights_override or MODEL_WEIGHTS).copy()
    # Cap any correlation group at MAX_GROUP_WEIGHT_FRAC of total
    total_raw = sum(weight_source.get(m, 1.0) for m in model_highs)
    if total_raw > 0:
        for group_name, members in MODEL_CORRELATION_GROUPS.items():
            group_models = [m for m in members if m in model_highs]
            if not group_models:
                continue
            group_weight = sum(weight_source.get(m, 1.0) for m in group_models)
            group_frac = group_weight / total_raw
            if group_frac > MAX_GROUP_WEIGHT_FRAC:
                scale = (MAX_GROUP_WEIGHT_FRAC * total_raw) / group_weight
                for m in group_models:
                    weight_source[m] = weight_source.get(m, 1.0) * scale

    # Weighted ensemble with capped weights
    ensemble_mean, ensemble_spread, n_models = compute_weighted_ensemble(
        model_highs, weight_source
    )

    # Apply station bias correction — scaled by Open-Meteo fraction
    bias = 0.0
    if city_key and city_key in CITIES:
        raw_bias = CITIES[city_key].get("bias_correction", 0.0)
        if raw_bias != 0.0 and model_highs:
            om_weight = sum(weight_source.get(k, 1.0)
                           for k in model_highs if k != "nws_official")
            total_weight = sum(weight_source.get(k, 1.0)
                              for k in model_highs)
            om_fraction = om_weight / total_weight if total_weight > 0 else 0.0
            bias = raw_bias * om_fraction
    corrected_mean = ensemble_mean + bias

    # --- Correlation-aware uncertainty (Phase 5A) ---
    n_effective = _count_effective_groups(model_highs)

    # With correlated models, raw ensemble spread underestimates true uncertainty.
    # Inflate spread based on how few independent groups we actually have.
    if n_effective <= 2:
        spread_inflation = 1.5
    elif n_effective <= 3:
        spread_inflation = 1.2
    else:
        spread_inflation = 1.0

    effective_spread = ensemble_spread * spread_inflation

    total_std = math.sqrt(effective_spread ** 2 + base_error_std ** 2)

    # Floor: don't go below 2.0F (real MOS errors are 2-3F even same-day)
    total_std = max(total_std, 2.0)

    # Inflate uncertainty when running on very few models (degraded data)
    if n_models <= 1:
        total_std *= 1.8
    elif n_models <= 3:
        total_std *= 1.3

    # --- Student's t-distribution with dynamic df (Phase 7) ---
    # Default df=8 gives meaningful fat tails for weather surprises.
    # Calibration can override with fitted df per city.
    df = 8
    if city_key and city_key in CITIES:
        df = CITIES[city_key].get("fitted_df", 8)
    dist = stats.t(df=df, loc=corrected_mean, scale=total_std)

    probs = {}
    for low, high, label in brackets:
        p_below_high = dist.cdf(high)
        p_below_low = dist.cdf(low)
        probs[label] = max(0.0, p_below_high - p_below_low)

    # Normalize to sum to 1.0
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


# =======================================================================
# METAR Observations
# =======================================================================

METAR_URL = "https://aviationweather.gov/api/data/metar"


def _parse_cloud_cover(obs: dict) -> float:
    """
    Parse cloud cover fraction (0.0 = clear, 1.0 = overcast) from METAR JSON.

    Cloud layers: SKC/CLR=0, FEW=0.2, SCT=0.4, BKN=0.7, OVC=1.0
    Uses the highest-coverage layer reported.
    """
    COVER_MAP = {"SKC": 0.0, "CLR": 0.0, "FEW": 0.2, "SCT": 0.4, "BKN": 0.7, "OVC": 1.0}

    # Try structured cloud data first
    clouds = obs.get("clouds", [])
    if clouds:
        max_cover = 0.0
        for layer in clouds:
            cover = layer.get("cover", "")
            max_cover = max(max_cover, COVER_MAP.get(cover, 0.0))
        return max_cover

    # Fallback: parse from raw METAR string
    raw = obs.get("rawOb", "")
    if raw:
        import re
        max_cover = 0.0
        for code in COVER_MAP:
            if re.search(rf'\b{code}\d{{3}}\b', raw) or f' {code} ' in f' {raw} ':
                max_cover = max(max_cover, COVER_MAP[code])
        return max_cover

    return 0.0  # Unknown = assume clear (conservative for heating)


def _parse_wind_speed(obs: dict) -> float:
    """Parse wind speed in knots from METAR JSON. Returns 0 if unavailable."""
    wspd = obs.get("wspd")
    if wspd is not None:
        return float(wspd)

    # Fallback: parse from raw METAR
    raw = obs.get("rawOb", "")
    if raw:
        import re
        m = re.search(r'(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT', raw)
        if m:
            return float(m.group(2))
    return 0.0


def fetch_metar(station_id: str) -> Optional[dict]:
    """
    Fetch latest METAR observation for a station.

    Returns dict with temp_f, temp_c, observation_time, raw_metar,
    cloud_cover (0-1), and wind_kt, or None if fetch fails.
    """
    try:
        params = {
            "ids": station_id,
            "format": "json",
            "hours": 2,
        }
        resp = requests.get(METAR_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return None

        obs = data[0] if isinstance(data, list) else data

        temp_c = obs.get("temp")
        if temp_c is None:
            return None

        temp_f = temp_c * 9.0 / 5.0 + 32.0

        return {
            "temp_f": round(temp_f, 1),
            "temp_c": temp_c,
            "observation_time": obs.get("reportTime", ""),
            "raw_metar": obs.get("rawOb", ""),
            "cloud_cover": _parse_cloud_cover(obs),
            "wind_kt": _parse_wind_speed(obs),
        }
    except Exception as e:
        print(f"  [WARN] METAR fetch failed for {station_id}: {e}")
        return None


def fetch_all_observations() -> dict:
    """Fetch latest METAR for all configured cities."""
    results = {}
    for city_key, city_info in CITIES.items():
        obs = fetch_metar(city_info["station_id"])
        if obs:
            results[city_key] = obs
    return results


def get_observed_high_today(city_key: str) -> Optional[float]:
    """
    Get the highest observed temperature today from stored observations.
    Returns the max temp_f or None if no observations today.
    """
    conn = db.get_conn()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT MAX(temp_f) as max_temp FROM observations
        WHERE city = ? AND timestamp LIKE ?
    """, (city_key, f"{today}%")).fetchone()
    conn.close()

    if rows and rows["max_temp"] is not None:
        return rows["max_temp"]
    return None


def get_metar_temperature_trajectory(city_key: str, hours: int = 6) -> list:
    """
    Get recent METAR temperature readings for trajectory analysis.

    Returns list of (hours_ago, temp_f) tuples sorted by time.
    """
    observations = db.get_recent_observations(city_key, hours)
    if not observations:
        return []

    now = datetime.now(timezone.utc)
    trajectory = []
    for obs in observations:
        try:
            obs_time = datetime.fromisoformat(obs["timestamp"])
            hours_ago = (now - obs_time).total_seconds() / 3600.0
            if obs["temp_f"] is not None:
                trajectory.append((hours_ago, obs["temp_f"]))
        except (ValueError, TypeError):
            continue

    return sorted(trajectory, key=lambda x: x[0], reverse=True)


# =======================================================================
# Intraday Probability Update (Improved Diurnal Model)
# =======================================================================

def estimate_remaining_temp_rise(current_temp_f: float, hours_remaining: float,
                                 current_hour: float,
                                 forecast_high: float = None,
                                 cloud_cover: float = 0.0) -> tuple:
    """
    Estimate how much more the temperature could rise given the time of day.

    Improved model that uses the forecast high as a reference point.
    If the current temp is already near or above the forecast high,
    the distribution tightens significantly.

    Cloud cover dampens solar heating:
      - OVC (1.0): reduces heating potential by ~40%
      - BKN (0.7): reduces heating by ~25%
      - SCT (0.4): reduces heating by ~10%
      - CLR (0.0): no reduction

    Returns (expected_additional_rise, uncertainty_std).
    """
    if hours_remaining <= 0:
        return 0.0, 0.3

    # Diurnal heating rate by hour (relative, peaks at solar noon)
    if current_hour < 9:
        remaining_heating_frac = 0.85
    elif current_hour < 11:
        remaining_heating_frac = 0.55
    elif current_hour < 13:
        remaining_heating_frac = 0.30
    elif current_hour < 15:
        remaining_heating_frac = 0.10
    else:
        remaining_heating_frac = 0.02

    # Phase 10: Cloud cover reduces solar heating potential
    # Scale: OVC -> 40% reduction, BKN -> 25%, SCT -> 10%
    cloud_damping = 1.0 - (0.4 * cloud_cover)
    remaining_heating_frac *= cloud_damping

    # If we have a forecast high, use it as reference
    if forecast_high is not None:
        gap = forecast_high - current_temp_f

        if gap <= 0:
            # Already at or above forecast high — very little upside
            # Temperature rarely exceeds the forecast by more than a few degrees
            expected_rise = max(0.0, 1.0 * remaining_heating_frac)
            uncertainty = max(0.5, 1.5 * remaining_heating_frac)
        elif gap <= 3.0:
            # Close to forecast high — likely to reach it
            expected_rise = gap * 0.7 * remaining_heating_frac
            uncertainty = max(0.5, gap * 0.4)
        else:
            # Still well below forecast — follow forecast guidance
            expected_rise = gap * 0.5 * remaining_heating_frac
            uncertainty = max(0.8, gap * 0.3)
    else:
        # No forecast reference — use generic model
        max_expected_rise = 15.0 * remaining_heating_frac
        expected_rise = max_expected_rise * 0.4
        uncertainty = max(0.5, max_expected_rise * 0.3)

    return expected_rise, uncertainty


def get_observation_weight(current_hour: float) -> float:
    """
    Get the weight to give observations vs forecast for intraday blending.

    Earlier in the day, trust the forecast more.
    Later in the day, trust observations more.
    """
    if current_hour < 10:
        return 0.0   # too early, forecast only
    elif current_hour < 12:
        return 0.20   # light observation weight
    elif current_hour < 14:
        return 0.50   # equal blend
    elif current_hour < 15:
        return 0.80   # strong observation weight
    else:
        return 0.95   # near-lock observation weight


def update_probabilities_with_observation(
    prior_probs: dict,
    brackets: list,
    observed_high_so_far: float,
    hours_remaining: float,
    forecast_high: float = None,
    cloud_cover: float = 0.0,
) -> dict:
    """
    Update bracket probabilities given the observed high temperature so far.

    Uses diurnal cycle model to estimate remaining temperature rise
    potential. Eliminates brackets that are already impossible.
    Accepts forecast_high for better diurnal model estimates and
    cloud_cover (0-1) to dampen heating predictions under overcast.
    """
    now = datetime.now()
    current_hour = now.hour + now.minute / 60.0

    expected_rise, uncertainty = estimate_remaining_temp_rise(
        observed_high_so_far, hours_remaining, current_hour,
        forecast_high=forecast_high,
        cloud_cover=cloud_cover,
    )

    projected_mean = observed_high_so_far + expected_rise
    projected_std = max(uncertainty, 0.5)

    # Use t-distribution for consistency (df=20, close to Gaussian)
    df = 20
    dist = stats.t(df=df, loc=projected_mean, scale=projected_std)

    updated = {}
    for low, high, label in brackets:
        if high <= observed_high_so_far:
            # This bracket is already impossible
            updated[label] = 0.0
        else:
            effective_low = max(low, observed_high_so_far)
            p = dist.cdf(high) - dist.cdf(effective_low)
            updated[label] = max(0.0, p)

    # Normalize
    total = sum(updated.values())
    if total > 0:
        updated = {k: v / total for k, v in updated.items()}

    # Blend with prior probabilities based on time of day
    obs_weight = get_observation_weight(current_hour)
    if obs_weight < 1.0 and prior_probs:
        blended = {}
        for label in updated:
            obs_p = updated.get(label, 0.0)
            prior_p = prior_probs.get(label, 0.0)
            blended[label] = obs_weight * obs_p + (1.0 - obs_weight) * prior_p
        # Re-normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    return updated


# =======================================================================
# Full Forecast Report
# =======================================================================

def get_full_forecast_report(target_date: str = None) -> dict:
    """
    Pull forecasts from all sources for all cities.

    Returns dict of {city_key: {
        "model_highs": {model: temp},
        "weighted_mean": float,
        "weighted_std": float,
        "n_models": int,
        "target_date": str,
    }}
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    report = {}

    for city_key in CITIES:
        print(f"  Fetching forecasts for {city_key}...")
        model_highs = get_all_model_highs(city_key, target_date)

        w_mean, w_std, n = compute_weighted_ensemble(model_highs)

        report[city_key] = {
            "model_highs": model_highs,
            "weighted_mean": w_mean,
            "weighted_std": w_std,
            "n_models": n,
            "target_date": target_date,
        }

        # Log each model forecast to DB
        for model_name, high_temp in model_highs.items():
            db.log_forecast(city_key, target_date, model_name, high_temp, {})

    return report


# =======================================================================
# CLI Test
# =======================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEATHER DATA TEST — v3 (8 models + NWS + HRRR)")
    print("=" * 70)

    target = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    for date_label, date_str in [("Today", target), ("Tomorrow", tomorrow)]:
        print(f"\n{'='*70}")
        print(f"  {date_label}: {date_str}")
        print(f"{'='*70}")

        # Compute lead days for weight selection
        today_dt = datetime.now().date()
        target_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        lead_days = (target_dt - today_dt).days
        weights = get_model_weights(lead_days)

        report = get_full_forecast_report(date_str)
        for city_key, data in report.items():
            city_name = CITIES[city_key]["name"]
            print(f"\n  {city_key} ({city_name}):")

            for model, temp in sorted(data["model_highs"].items()):
                weight = weights.get(model, 1.0)
                w_tag = f" [weight={weight}x]" if weight != 1.0 else ""
                print(f"    {model:20s}: {temp:.1f}F{w_tag}")

            if data["weighted_mean"] is not None:
                print(f"    {'':20s}  --------------------")
                print(f"    {'Weighted Mean':20s}: {data['weighted_mean']:.1f}F "
                      f"(spread: {data['weighted_std']:.1f}F, "
                      f"n={data['n_models']})")

    print(f"\n{'='*70}")
    print("  Current METAR Observations")
    print(f"{'='*70}")
    obs = fetch_all_observations()
    for city_key, o in obs.items():
        print(f"  {city_key}: {o['temp_f']:.1f}F "
              f"(observed at {o['observation_time']})")

    print(f"\n{'='*70}")
