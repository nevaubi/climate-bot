"""
Configuration for Kalshi Weather Trading Bot v3.

City/station mappings match Kalshi's weather market resolution sources.
NWS grid coordinates are used for official forecast lookups.
Includes lead-time-dependent model weights, correlation groups,
and dynamic calibration loading.
"""

import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Kalshi API ---
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# --- Trading ---
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"
STARTING_BANKROLL = 150.00  # USD (actual Kalshi balance as of 2026-02-13)
EDGE_THRESHOLD = 0.10  # minimum edge to trade (after fees)
KELLY_FRACTION = 0.15  # 15% Kelly (conservative for early live trading)
MAX_POSITION_PER_CITY = 0.20  # max 20% of bankroll on one city
MAX_POSITION_PER_BRACKET = 0.12  # max 12% of bankroll on one bracket
MAX_TOTAL_EXPOSURE = 0.40  # max 40% of bankroll deployed across ALL trades
MAX_CONTRACTS_PER_TRADE = 30  # hard cap on contracts per trade
SCAN_INTERVAL_SECONDS = 45  # how often to poll markets

# --- Spread & Liquidity Filters ---
MAX_SPREAD = 0.15  # skip brackets with bid-ask spread > 15 cents
MIN_VOLUME = 5  # skip brackets with fewer than 5 contracts traded

# --- Same-Day Cutoff ---
SAME_DAY_CUTOFF_HOUR = 15  # 3pm — switch to observation-dominated mode (no longer hard stop)
LATE_DAY_EDGE_MULTIPLIER = 1.5  # require 1.5x normal edge after cutoff for safety

# --- Maker Orders ---
USE_MAKER_ORDERS = True  # use maker orders when spread allows (1.75% fee vs 7% taker)
MAKER_MIN_SPREAD = 0.04  # minimum spread to attempt maker order (4 cents)
MAKER_PRICE_OFFSET = 0.01  # place limit 1 cent inside spread

# --- Drawdown Protection ---
MAX_DAILY_LOSS = 0.10  # stop trading if daily losses exceed 10% of bankroll
MAX_DRAWDOWN = 0.25  # stop trading if drawdown from peak exceeds 25%

# --- Position Exit ---
EXIT_EDGE_FLIP_THRESHOLD = 0.40  # exit YES position if model_prob drops below this
EXIT_EDGE_DECAY_THRESHOLD = 0.03  # exit if remaining edge drops below 3%

# --- Database ---
DB_PATH = Path(__file__).parent / "data" / "bot.db"

# --- Logging ---
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# --- Calibration File ---
CALIBRATION_FILE = Path(__file__).parent / "data" / "calibration.json"

# --- Weather Cities ---
# nws_grid: (office, gridX, gridY) for api.weather.gov lookups
# station_id: ICAO station for METAR observations
# bias_correction: additive offset applied to raw model ensemble mean
#   Derived from calibrate.py output (2026-02-10 run).
#   Positive = models underpredict → shift mean upward.
#   Re-run calibrate.py weekly to update these.
# These are fallback values; dynamic values from calibration.json take priority.
CITIES = {
    "NYC": {
        "name": "New York City",
        "ticker_prefix": "KXHIGHNY",
        "station_id": "KNYC",
        "nws_station": "NYC",
        "nws_grid": ("OKX", 34, 38),
        "lat": 40.7828,
        "lon": -73.9653,
        "bias_correction": 0.0,    # Day 1: models already +2.2F hot, old +2.0 made it worse
        "base_error_std": 2.0,     # Raised: Day 1 ensemble error was 4.3F (after old correction)
    },
    "CHI": {
        "name": "Chicago",
        "ticker_prefix": "KXHIGHCHI",
        "station_id": "KMDW",
        "nws_station": "CHI",
        "nws_grid": ("LOT", 72, 69),
        "lat": 41.7861,
        "lon": -87.7522,
        "bias_correction": 1.3,    # Day 1: models underpredicted by 1.8F, old +0.8 helped but not enough
        "base_error_std": 2.0,
    },
    "MIA": {
        "name": "Miami",
        "ticker_prefix": "KXHIGHMIA",
        "station_id": "KMIA",
        "nws_station": "MIA",
        "nws_grid": ("MFL", 106, 51),
        "lat": 25.7959,
        "lon": -80.2870,
        "bias_correction": 0.0,
        "base_error_std": 1.8,
    },
    "AUS": {
        "name": "Austin",
        "ticker_prefix": "KXHIGHAUS",
        "station_id": "KAUS",
        "nws_station": "AUS",
        "nws_grid": ("EWX", 159, 88),
        "lat": 30.1945,
        "lon": -97.6699,
        "bias_correction": 1.5,    # Day 1: raw models near-perfect (+0.1F), old +3.1 way too high
        "base_error_std": 2.2,
    },
    "LA": {
        "name": "Los Angeles",
        "ticker_prefix": "KXHIGHLA",
        "station_id": "KLAX",
        "nws_station": "LAX",
        "nws_grid": ("LOX", 148, 41),
        "lat": 33.9425,
        "lon": -118.4081,
        "bias_correction": 2.3,
        "base_error_std": 1.5,
    },
    "PHI": {
        "name": "Philadelphia",
        "ticker_prefix": "KXHIGHPHI",
        "station_id": "KPHL",
        "nws_station": "PHI",
        "nws_grid": ("PHI", 48, 72),
        "lat": 39.8721,
        "lon": -75.2411,
        "bias_correction": 1.4,
        "base_error_std": 1.5,
    },
    "DC": {
        "name": "Washington DC",
        "ticker_prefix": "KXHIGHDC",
        "station_id": "KDCA",
        "nws_station": "DCA",
        "nws_grid": ("LWX", 97, 69),
        "lat": 38.8512,
        "lon": -77.0402,
        "bias_correction": 1.3,
        "base_error_std": 1.8,
    },
}

# --- Correlation Groups ---
# Cities in the same group share weather patterns (same fronts, same air masses).
# Apply group-level exposure limits to avoid concentrated bets.
CORRELATION_GROUPS = {
    "northeast": ["NYC", "PHI", "DC"],
    "midwest": ["CHI"],
    "south": ["AUS", "MIA"],
    "west": ["LA"],
}
MAX_POSITION_PER_GROUP = 0.30  # max 30% of bankroll in correlated cities

# --- Model Weights (Default / Static) ---
# These are fallback weights. Use get_model_weights(lead_days) for
# lead-time-dependent weights that better reflect each model's skill.
# Updated 2026-02-12 based on Day 1 accuracy data (3 cities):
#   ARPEGE: MAE 1.0F (best), ECMWF: 1.1F, GFS/HRRR: 1.5F, NWS: 1.7F
#   DWD: 2.7F, GEM: 3.1F (hot bias), JMA: 4.7F (cold bias)
MODEL_WEIGHTS = {
    "nws_official": 2.0,
    "hrrr": 1.5,  # NOAA HRRR: best US short-range model (3km, hourly updates)
    "gfs_seamless": 1.0,
    "ecmwf_ifs025": 1.5,   # Day 1: MAE 1.1F, best global model
    "jma_seamless": 0.3,   # Day 1: MAE 4.7F, severe cold bias — heavily penalized
    "gem_seamless": 0.4,   # Day 1: MAE 3.1F, hot bias — reduced
    "dwd_icon": 0.7,       # Day 1: MAE 2.7F, mixed bias — modest reduction
    "arpege_seamless": 1.3, # Day 1: MAE 1.0F, near-unbiased — promoted
}

# --- Lead-Time-Dependent Model Weights ---
# Different models excel at different forecast horizons.
# HRRR: exceptional 0-18hr, good 18-48hr, degrades beyond that.
# ECMWF: best medium-range (2-7 day), good all horizons.
# GFS: solid all-around, slightly behind ECMWF at longer ranges.
# NWS: always high because it's already a bias-corrected super-ensemble.
LEAD_TIME_WEIGHTS = {
    # Same-day (lead_days=0): HRRR shines, NWS strong, others less reliable
    # JMA/GEM heavily penalized based on Day 1 performance
    0: {
        "nws_official": 2.0,
        "hrrr": 2.0,
        "gfs_seamless": 0.8,
        "ecmwf_ifs025": 1.2,   # ECMWF good even same-day
        "jma_seamless": 0.2,   # Terrible Day 1 accuracy (MAE 4.7F)
        "gem_seamless": 0.3,   # Poor Day 1 accuracy (MAE 3.1F)
        "dwd_icon": 0.5,       # Below average (MAE 2.7F)
        "arpege_seamless": 0.8, # Best Day 1 model — promoted
    },
    # Next-day (lead_days=1): HRRR still good, ECMWF catches up
    1: {
        "nws_official": 2.0,
        "hrrr": 1.5,
        "gfs_seamless": 1.0,
        "ecmwf_ifs025": 1.5,   # Best global model, Day 1 confirmed
        "jma_seamless": 0.3,   # Heavily penalized
        "gem_seamless": 0.4,   # Penalized
        "dwd_icon": 0.7,       # Modest reduction
        "arpege_seamless": 1.3, # Promoted based on accuracy
    },
    # 2+ days out: ECMWF leads, HRRR less relevant
    2: {
        "nws_official": 2.0,
        "hrrr": 0.5,
        "gfs_seamless": 1.0,
        "ecmwf_ifs025": 1.8,   # ECMWF strongest at range
        "jma_seamless": 0.3,   # Still penalized
        "gem_seamless": 0.4,   # Still penalized
        "dwd_icon": 0.7,       # Modest reduction
        "arpege_seamless": 1.3, # Promoted
    },
}


def _load_optimized_weights():
    """Load optimized weights from auto_optimize.py output, if available."""
    opt_path = Path(__file__).parent / "data" / "optimized_weights.json"
    if not opt_path.exists():
        return None
    try:
        return json.loads(opt_path.read_text())
    except (json.JSONDecodeError, KeyError):
        return None


def get_model_weights(lead_days: int = 1) -> dict:
    """
    Get model weights appropriate for the forecast lead time.

    Loads from optimized_weights.json if available (auto_optimize output),
    otherwise falls back to hardcoded LEAD_TIME_WEIGHTS.

    Args:
        lead_days: 0=same-day, 1=next-day, 2+=extended range

    Returns:
        dict of {model_name: weight}
    """
    # Try loading optimized weights first
    opt = _load_optimized_weights()
    if opt and "model_weights" in opt:
        base = dict(opt["model_weights"])
        # Apply lead-time scaling factors
        factors = opt.get("lead_time_factors", {})
        if lead_days <= 0:
            ref = LEAD_TIME_WEIGHTS[0]
        elif lead_days == 1:
            ref = LEAD_TIME_WEIGHTS[1]
        else:
            ref = LEAD_TIME_WEIGHTS[2]
        # Scale each model's optimized weight by lead-time ratio
        result = {}
        for model in base:
            if model in ref and model in MODEL_WEIGHTS and MODEL_WEIGHTS[model] > 0:
                lt_ratio = ref[model] / MODEL_WEIGHTS[model]
                result[model] = round(base[model] * lt_ratio, 2)
            else:
                result[model] = base[model]
        return result

    # Fallback to hardcoded lead-time weights
    if lead_days <= 0:
        return LEAD_TIME_WEIGHTS[0].copy()
    elif lead_days == 1:
        return LEAD_TIME_WEIGHTS[1].copy()
    else:
        return LEAD_TIME_WEIGHTS[2].copy()


def load_dynamic_calibration():
    """
    Load per-city bias and error std from calibration.json.

    Updates CITIES dict with latest calibration values.
    Falls back to hardcoded values if file is missing or invalid.
    """
    if not CALIBRATION_FILE.exists():
        return

    try:
        data = json.loads(CALIBRATION_FILE.read_text())
        cities_cal = data.get("cities", {})
        for city_key, cal in cities_cal.items():
            if city_key in CITIES:
                if "bias" in cal and cal.get("n_samples", 0) >= 14:
                    # Calibration bias is (ensemble - actual), so correction is -bias
                    CITIES[city_key]["bias_correction"] = round(-cal["bias"], 1)
                if "std" in cal and cal.get("n_samples", 0) >= 14:
                    CITIES[city_key]["base_error_std"] = round(
                        max(cal["std"], 1.5), 1
                    )
                if "fitted_df" in cal and cal.get("n_samples", 0) >= 20:
                    CITIES[city_key]["fitted_df"] = round(cal["fitted_df"], 1)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass  # fall back to hardcoded values


# Load dynamic calibration on import
load_dynamic_calibration()


# --- Kalshi Fee Calculation ---
def kalshi_taker_fee(price: float, contracts: int) -> float:
    """Calculate Kalshi taker fee. Price is in dollars (0.0 to 1.0)."""
    raw = 0.07 * contracts * price * (1.0 - price)
    return math.ceil(raw * 100) / 100


def kalshi_maker_fee(price: float, contracts: int) -> float:
    """Calculate Kalshi maker fee. Price is in dollars (0.0 to 1.0)."""
    raw = 0.0175 * contracts * price * (1.0 - price)
    return math.ceil(raw * 100) / 100
