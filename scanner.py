"""
Main scanner loop — v3.

Improvements over v2:
  - Edge calculation uses ask price (not mid-price) for accurate cost basis
  - Spread and liquidity filters (skip wide-spread / low-volume brackets)
  - Fixed Kelly criterion formula (direct binary contract Kelly)
  - Lead-time-dependent model weights (HRRR for same-day, ECMWF for 2+ days)
  - HRRR model in ensemble (8 models total)
  - Correlation-aware position limits (NE cities share weather)
  - Drawdown protection (daily loss limit + max drawdown circuit breaker)
  - Earlier METAR integration (10am instead of 1pm, time-varying blend)
  - Position exit logic (sell when edge flips or decays below threshold)
  - Forecast staleness protection (tighter cache for same-day markets)
  - Proper trade side tracking (yes/no) in database

Usage:
    python scanner.py              # run continuous scanner
    python scanner.py --once       # run one scan cycle and exit
"""

import sys
import re
import time
import math
import signal
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

# Kalshi markets settle on Eastern Time
EASTERN = ZoneInfo("America/New_York")


def now_eastern() -> datetime:
    """Get current time in US Eastern (handles DST automatically)."""
    return datetime.now(EASTERN)


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)

from config import (
    CITIES, PAPER_MODE, EDGE_THRESHOLD, KELLY_FRACTION,
    MAX_POSITION_PER_CITY, MAX_POSITION_PER_BRACKET,
    MAX_TOTAL_EXPOSURE, MAX_CONTRACTS_PER_TRADE,
    SCAN_INTERVAL_SECONDS, SAME_DAY_CUTOFF_HOUR,
    LATE_DAY_EDGE_MULTIPLIER,
    USE_MAKER_ORDERS, MAKER_MIN_SPREAD, MAKER_PRICE_OFFSET,
    MAX_SPREAD, MIN_VOLUME,
    MAX_DAILY_LOSS, MAX_DRAWDOWN, MAX_POSITION_PER_GROUP,
    CORRELATION_GROUPS,
    EXIT_EDGE_FLIP_THRESHOLD, EXIT_EDGE_DECAY_THRESHOLD,
    kalshi_taker_fee, kalshi_maker_fee, MODEL_WEIGHTS, get_model_weights,
)
from kalshi_client import KalshiClient
from weather import (
    get_all_model_highs, compute_weighted_ensemble,
    calculate_bracket_probabilities,
    fetch_metar, get_observed_high_today,
    update_probabilities_with_observation,
)
import db


# Graceful shutdown
RUNNING = True


def handle_signal(signum, frame):
    global RUNNING
    print("\n[SCANNER] Shutting down gracefully...")
    RUNNING = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# =======================================================================
# Bracket Parsing
# =======================================================================

def parse_brackets_from_markets(markets: list) -> list:
    """
    Parse Kalshi market data into bracket dicts using structured fields.

    Uses floor_strike, cap_strike, and strike_type for reliable bounds.
    Applies half-integer boundaries for continuous probability calculation
    since NWS reports integer temperatures.
    """
    brackets = []

    for m in markets:
        ticker = m.get("ticker", "")
        subtitle = (m.get("subtitle") or "").strip()
        strike_type = m.get("strike_type", "")
        floor_strike = m.get("floor_strike")
        cap_strike = m.get("cap_strike")

        # Determine bounds from structured strike data
        if strike_type == "greater" and floor_strike is not None:
            low = float(floor_strike) + 0.5
            high = float("inf")
            label = subtitle or f">{floor_strike}"
        elif strike_type == "less" and cap_strike is not None:
            low = float("-inf")
            high = float(cap_strike) - 0.5
            label = subtitle or f"<{cap_strike}"
        elif strike_type == "between" and floor_strike is not None and cap_strike is not None:
            low = float(floor_strike) - 0.5
            high = float(cap_strike) + 0.5
            label = subtitle or f"{floor_strike} to {cap_strike}"
        else:
            low, high = parse_temp_bounds(subtitle or ticker)
            label = subtitle or ticker

        # Price data
        yes_bid = m.get("yes_bid", 0) / 100.0 if m.get("yes_bid") else None
        yes_ask = m.get("yes_ask", 0) / 100.0 if m.get("yes_ask") else None
        no_bid = m.get("no_bid", 0) / 100.0 if m.get("no_bid") else None
        no_ask = m.get("no_ask", 0) / 100.0 if m.get("no_ask") else None

        # Midprice estimate (used for display, NOT for edge calculation)
        if yes_bid and yes_ask:
            mid_price = (yes_bid + yes_ask) / 2.0
        elif yes_bid:
            mid_price = yes_bid
        elif m.get("last_price"):
            mid_price = m["last_price"] / 100.0
        else:
            mid_price = 0.5

        brackets.append({
            "low": low,
            "high": high,
            "label": label,
            "ticker": ticker,
            "strike_type": strike_type,
            "floor_strike": floor_strike,
            "cap_strike": cap_strike,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "mid_price": mid_price,
            "volume": m.get("volume", 0),
            "open_interest": m.get("open_interest", 0),
            "status": m.get("status", ""),
        })

    brackets.sort(key=lambda b: b["low"])
    return brackets


def parse_temp_bounds(title: str) -> tuple:
    """Fallback: parse temperature bounds from bracket title text."""
    title_clean = title.replace("\u00b0F", "").replace("\u00b0", "").replace("F", "").strip()

    m = re.search(r'(-?\d+)\s*or\s*(?:lower|below|less)', title_clean, re.IGNORECASE)
    if m:
        return (float("-inf"), float(m.group(1)))

    m = re.search(r'(-?\d+)\s*or\s*(?:higher|above|more|greater)', title_clean, re.IGNORECASE)
    if m:
        return (float(m.group(1)), float("inf"))

    m = re.search(r'(-?\d+)\s*(?:to|through|-|\u2013)\s*(-?\d+)', title_clean, re.IGNORECASE)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    nums = re.findall(r'-?\d+', title_clean)
    if len(nums) >= 2:
        return (float(nums[0]), float(nums[-1]))
    elif len(nums) == 1:
        return (float(nums[0]), float(nums[0]))

    return (float("-inf"), float("inf"))


# =======================================================================
# Date Extraction
# =======================================================================

def extract_date_from_event_ticker(event_ticker: str) -> str:
    """Extract date from Kalshi event ticker (e.g., KXHIGHNY-26FEB11 -> 2026-02-11)."""
    m = re.search(r'-(\d{2})([A-Z]{3})(\d{1,2})$', event_ticker)
    if not m:
        return None

    year_short = int(m.group(1))
    month_str = m.group(2)
    day = int(m.group(3))

    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = months.get(month_str, 1)
    year = 2000 + year_short

    return f"{year}-{month:02d}-{day:02d}"


def group_markets_by_date(markets: list) -> dict:
    """Group markets by their event date."""
    groups = {}
    for m in markets:
        event_ticker = m.get("event_ticker", "")
        date_str = extract_date_from_event_ticker(event_ticker)
        if date_str:
            groups.setdefault(date_str, []).append(m)
    return groups


def estimate_hours_remaining(target_date: str = None) -> float:
    """Estimate hours until daily high is locked in (~4pm Eastern)."""
    now = now_eastern()
    today_str = now.strftime("%Y-%m-%d")

    if target_date and target_date > today_str:
        return 24.0

    peak_hour = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now >= peak_hour:
        return 0.0
    return max(0.0, (peak_hour - now).total_seconds() / 3600.0)


def compute_lead_days(market_date: str) -> int:
    """Compute lead days from today to market date."""
    today = now_eastern().date()
    try:
        target = datetime.strptime(market_date, "%Y-%m-%d").date()
        return max(0, (target - today).days)
    except ValueError:
        return 1


# =======================================================================
# Position Sizing (Fixed Kelly Criterion for Binary Contracts)
# =======================================================================

def kelly_size(model_prob: float, exec_price: float, bankroll: float,
               kelly_fraction: float = KELLY_FRACTION,
               max_bet_frac: float = MAX_POSITION_PER_BRACKET) -> float:
    """
    Calculate position size using fractional Kelly criterion for binary contracts.

    For a binary contract paying $1 on win:
      - Cost per contract = exec_price
      - Profit on win = (1.0 - exec_price)
      - Loss on lose = exec_price

    Kelly fraction = (p * profit - q * loss) / profit
    where p = model_prob, q = 1 - model_prob

    Returns dollar amount to risk.
    """
    if model_prob <= 0 or model_prob >= 1 or exec_price <= 0 or exec_price >= 1:
        return 0.0

    profit = 1.0 - exec_price  # win payout minus cost
    loss = exec_price           # cost lost on losing trade

    p = model_prob
    q = 1.0 - model_prob

    # Kelly: f* = (p * profit - q * loss) / profit
    # Simplified for binary: f* = (p * (1 - exec_price) - (1 - p) * exec_price) / (1 - exec_price)
    # = (p - exec_price) / (1 - exec_price)
    full_kelly = (p - exec_price) / (1.0 - exec_price)
    full_kelly = max(0.0, full_kelly)

    fractional = full_kelly * kelly_fraction
    max_bet = bankroll * max_bet_frac

    bet_size = min(fractional * bankroll, max_bet)
    return max(0.0, round(bet_size, 2))


# =======================================================================
# Opportunity Detection (Buy YES + Buy NO) — Fixed Edge Calculation
# =======================================================================

def find_opportunities(brackets: list, model_probs: dict,
                       bankroll: float, existing_tickers: set = None,
                       edge_threshold: float = None) -> list:
    """
    Compare model probabilities to market prices and identify trades.

    Supports both taker and maker orders (Phase 4). Maker orders
    have lower fees (1.75% vs 7%) but may not fill immediately.

    Returns list of opportunity dicts sorted by absolute edge (descending).
    """
    threshold = edge_threshold if edge_threshold is not None else EDGE_THRESHOLD
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95

    if existing_tickers is None:
        existing_tickers = set()

    opps = []

    for bracket in brackets:
        label = bracket["label"]
        ticker = bracket["ticker"]
        yes_bid = bracket.get("yes_bid")
        yes_ask = bracket.get("yes_ask")
        no_bid = bracket.get("no_bid")
        no_ask = bracket.get("no_ask")

        # Skip if no real bid-ask (illiquid)
        if yes_bid is None and yes_ask is None:
            continue

        # Skip if we already have a position on this ticker
        if ticker in existing_tickers:
            continue

        # Spread filter: skip if bid-ask spread is too wide
        if yes_bid is not None and yes_ask is not None:
            spread = yes_ask - yes_bid
            if spread > MAX_SPREAD:
                continue

        # Volume filter: skip if too few contracts traded
        if bracket.get("volume", 0) < MIN_VOLUME:
            continue

        model_prob = model_probs.get(label, 0.0)

        # Compute spread for maker order eligibility
        has_spread = yes_bid is not None and yes_ask is not None
        spread = (yes_ask - yes_bid) if has_spread else 0.0

        # --- Buy YES opportunity ---
        yes_maker_found = False
        if yes_ask is not None and MIN_PRICE <= yes_ask <= MAX_PRICE:
            # Try maker order first (lower fees) if spread is wide enough
            if USE_MAKER_ORDERS and has_spread and spread >= MAKER_MIN_SPREAD:
                maker_price = round(yes_bid + MAKER_PRICE_OFFSET, 2)
                maker_fee = kalshi_maker_fee(maker_price, 1)
                maker_edge = model_prob - maker_price - maker_fee
                if maker_edge >= threshold:
                    bet = kelly_size(model_prob, maker_price, bankroll)
                    if bet >= 1.0:
                        contracts = min(max(1, int(bet / maker_price)),
                                        MAX_CONTRACTS_PER_TRADE)
                        cost = round(contracts * maker_price, 2)
                        fee = kalshi_maker_fee(maker_price, contracts)
                        opps.append({
                            "side": "yes", "action": "buy",
                            "order_type": "maker",
                            "ticker": ticker, "label": label,
                            "bracket_low": bracket["low"],
                            "bracket_high": bracket["high"],
                            "exec_price": maker_price,
                            "mid_price": bracket["mid_price"],
                            "model_prob": model_prob,
                            "edge": maker_edge,
                            "contracts": contracts, "cost": cost,
                            "fee": round(fee, 2),
                            "potential_profit": round(contracts * 1.0 - cost - fee, 2),
                            "yes_bid": yes_bid, "yes_ask": yes_ask,
                        })
                        yes_maker_found = True  # Skip taker but still evaluate NO side

            # Taker order (standard) — skip if maker already found for YES
            if not yes_maker_found:
                fee_per_contract = kalshi_taker_fee(yes_ask, 1)
                net_edge = model_prob - yes_ask - fee_per_contract

                if net_edge >= threshold:
                    bet = kelly_size(model_prob, yes_ask, bankroll)
                    if bet >= 1.0:
                        contracts = min(max(1, int(bet / yes_ask)),
                                        MAX_CONTRACTS_PER_TRADE)
                        cost = round(contracts * yes_ask, 2)
                        fee = kalshi_taker_fee(yes_ask, contracts)
                        opps.append({
                            "side": "yes", "action": "buy",
                            "order_type": "taker",
                            "ticker": ticker, "label": label,
                            "bracket_low": bracket["low"],
                            "bracket_high": bracket["high"],
                            "exec_price": yes_ask,
                            "mid_price": bracket["mid_price"],
                            "model_prob": model_prob,
                            "edge": net_edge,
                            "contracts": contracts, "cost": cost,
                            "fee": round(fee, 2),
                            "potential_profit": round(contracts * 1.0 - cost - fee, 2),
                            "yes_bid": yes_bid, "yes_ask": yes_ask,
                        })

        # --- Buy NO opportunity (model says bracket overpriced) ---
        no_model_prob = 1.0 - model_prob

        actual_no_ask = no_ask
        if actual_no_ask is None and yes_bid is not None:
            actual_no_ask = round(1.0 - yes_bid, 2)

        if actual_no_ask is not None and MIN_PRICE <= actual_no_ask <= MAX_PRICE:
            # Try maker order for NO side
            no_maker_found = False
            no_bid = bracket.get("no_bid")
            if no_bid is None and yes_ask is not None:
                no_bid = round(1.0 - yes_ask, 2)
            no_spread = (actual_no_ask - no_bid) if no_bid is not None else 0.0

            if USE_MAKER_ORDERS and no_bid is not None and no_spread >= MAKER_MIN_SPREAD:
                maker_price = round(no_bid + MAKER_PRICE_OFFSET, 2)
                maker_fee = kalshi_maker_fee(maker_price, 1)
                maker_edge = no_model_prob - maker_price - maker_fee
                if maker_edge >= threshold:
                    bet = kelly_size(no_model_prob, maker_price, bankroll)
                    if bet >= 1.0:
                        contracts = min(max(1, int(bet / maker_price)),
                                        MAX_CONTRACTS_PER_TRADE)
                        cost = round(contracts * maker_price, 2)
                        fee = kalshi_maker_fee(maker_price, contracts)
                        opps.append({
                            "side": "no", "action": "buy",
                            "order_type": "maker",
                            "ticker": ticker, "label": label,
                            "bracket_low": bracket["low"],
                            "bracket_high": bracket["high"],
                            "exec_price": maker_price,
                            "mid_price": bracket["mid_price"],
                            "model_prob": model_prob,
                            "edge": maker_edge,
                            "contracts": contracts, "cost": cost,
                            "fee": round(fee, 2),
                            "potential_profit": round(contracts * 1.0 - cost - fee, 2),
                            "yes_bid": yes_bid, "yes_ask": yes_ask,
                        })
                        no_maker_found = True

            # Taker order for NO — skip if maker already found
            if not no_maker_found:
                fee_per_contract = kalshi_taker_fee(actual_no_ask, 1)
                net_edge = no_model_prob - actual_no_ask - fee_per_contract

                if net_edge >= threshold:
                    bet = kelly_size(no_model_prob, actual_no_ask, bankroll)
                    if bet >= 1.0:
                        contracts = min(max(1, int(bet / actual_no_ask)),
                                        MAX_CONTRACTS_PER_TRADE)
                        cost = round(contracts * actual_no_ask, 2)
                        fee = kalshi_taker_fee(actual_no_ask, contracts)
                        opps.append({
                            "side": "no", "action": "buy",
                            "order_type": "taker",
                            "ticker": ticker, "label": label,
                            "bracket_low": bracket["low"],
                            "bracket_high": bracket["high"],
                            "exec_price": actual_no_ask,
                            "mid_price": bracket["mid_price"],
                            "model_prob": model_prob,
                            "edge": net_edge,
                            "contracts": contracts, "cost": cost,
                            "fee": round(fee, 2),
                            "potential_profit": round(contracts * 1.0 - cost - fee, 2),
                            "yes_bid": yes_bid, "yes_ask": yes_ask,
                        })

    opps.sort(key=lambda x: x["edge"], reverse=True)
    return opps


# =======================================================================
# Orderbook Depth Analysis (Phase 8)
# =======================================================================

def check_orderbook_depth(client: KalshiClient, ticker: str,
                          side: str, target_price: float,
                          desired_contracts: int) -> int:
    """
    Check orderbook depth and return the number of contracts available
    at or near the target price.

    Returns adjusted contract count (may be less than desired if
    insufficient liquidity). Returns 0 if orderbook fetch fails.
    """
    try:
        ob = client.get_orderbook(ticker, depth=10)
        orderbook = ob.get("orderbook", ob)

        if side == "yes":
            asks = orderbook.get("yes", [])
            # Count contracts available at or below target_price + 1 cent tolerance
            available = 0
            price_limit = target_price + 0.015  # 1.5 cent tolerance
            for level in asks:
                level_price = level[0] / 100.0 if level[0] > 1 else level[0]
                level_qty = level[1] if len(level) > 1 else 0
                if level_price <= price_limit:
                    available += level_qty
        else:
            asks = orderbook.get("no", [])
            available = 0
            price_limit = target_price + 0.015
            for level in asks:
                level_price = level[0] / 100.0 if level[0] > 1 else level[0]
                level_qty = level[1] if len(level) > 1 else 0
                if level_price <= price_limit:
                    available += level_qty

        if available <= 0:
            return desired_contracts  # No depth data, proceed with desired amount

        return min(desired_contracts, available)

    except Exception:
        # Don't let orderbook check failures block trading
        return desired_contracts


# =======================================================================
# Trade Execution
# =======================================================================

def execute_trade(client: KalshiClient, opp: dict, city_key: str,
                  market_date: str, bankroll: float) -> bool:
    """Execute a trade (paper or live). Returns True if successful."""
    side_label = opp["side"].upper()

    if PAPER_MODE:
        trade_id = db.log_trade(
            city=city_key,
            market_date=market_date,
            ticker=opp["ticker"],
            bracket=opp["label"],
            side=opp["side"],
            contracts=opp["contracts"],
            price=opp["exec_price"],
            cost=opp["cost"],
            fee=opp["fee"],
            model_prob=opp["model_prob"],
            market_prob=opp["mid_price"],
            edge=opp["edge"],
            paper_mode=True,
            notes=f"Paper trade. BUY {side_label} edge={opp['edge']:.3f} ({opp.get('order_type', 'taker')})",
            bracket_low=opp.get("bracket_low"),
            bracket_high=opp.get("bracket_high"),
        )
        order_tag = opp.get("order_type", "taker").upper()
        print(f"    [PAPER/{order_tag}] #{trade_id}: BUY {opp['contracts']}x {side_label} "
              f"{opp['ticker']} @ ${opp['exec_price']:.2f} "
              f"(edge={opp['edge']:.3f}, cost=${opp['cost']:.2f}, fee=${opp['fee']:.2f})")
        return True
    else:
        try:
            price_cents = int(round(opp["exec_price"] * 100))

            order_params = {
                "ticker": opp["ticker"],
                "side": opp["side"],
                "type": "limit",
                "count": opp["contracts"],
                "action": "buy",
            }
            if opp["side"] == "yes":
                order_params["yes_price"] = price_cents
            else:
                order_params["no_price"] = price_cents

            result = client.place_order(**order_params)

            order_id = result.get("order", {}).get("order_id", "unknown")
            trade_id = db.log_trade(
                city=city_key,
                market_date=market_date,
                ticker=opp["ticker"],
                bracket=opp["label"],
                side=opp["side"],
                contracts=opp["contracts"],
                price=opp["exec_price"],
                cost=opp["cost"],
                fee=opp["fee"],
                model_prob=opp["model_prob"],
                market_prob=opp["mid_price"],
                edge=opp["edge"],
                paper_mode=False,
                notes=f"Live order {order_id}. BUY {side_label}",
                bracket_low=opp.get("bracket_low"),
                bracket_high=opp.get("bracket_high"),
            )
            print(f"    [LIVE] #{trade_id} (order {order_id}): "
                  f"BUY {opp['contracts']}x {side_label} {opp['ticker']} "
                  f"@ ${opp['exec_price']:.2f}")
            return True

        except Exception as e:
            print(f"    [ERROR] Order failed for {opp['ticker']}: {e}")
            return False


# =======================================================================
# Position Tracking & Risk Management
# =======================================================================

def get_existing_positions() -> set:
    """Get set of tickers we already have open trades on."""
    conn = db.get_conn()
    rows = conn.execute("""
        SELECT DISTINCT ticker FROM trades
        WHERE resolved = 0
    """).fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def get_total_exposure() -> float:
    """Get total dollar cost of all unresolved trades."""
    conn = db.get_conn()
    row = conn.execute("""
        SELECT COALESCE(SUM(cost), 0) as total FROM trades
        WHERE resolved = 0
    """).fetchone()
    conn.close()
    return row["total"]


def get_city_group(city_key: str) -> str:
    """Get the correlation group name for a city."""
    for group_name, cities in CORRELATION_GROUPS.items():
        if city_key in cities:
            return group_name
    return city_key  # standalone if not in a group


def check_drawdown_limits(bankroll: float) -> tuple:
    """
    Check if drawdown limits have been breached.

    Returns (is_ok, reason_string).
    """
    # Check daily loss limit
    today_str = datetime.now().strftime("%Y-%m-%d")
    daily_pnl = db.get_daily_pnl(today_str)
    daily_loss_limit = bankroll * MAX_DAILY_LOSS

    if daily_pnl < 0 and abs(daily_pnl) > daily_loss_limit:
        return False, (f"Daily loss ${abs(daily_pnl):.2f} exceeds "
                       f"limit ${daily_loss_limit:.2f} ({MAX_DAILY_LOSS:.0%})")

    # Check max drawdown from peak
    peak = db.get_peak_bankroll()
    if peak > 0:
        drawdown = (peak - bankroll) / peak
        if drawdown > MAX_DRAWDOWN:
            return False, (f"Drawdown {drawdown:.1%} exceeds "
                           f"limit {MAX_DRAWDOWN:.0%} "
                           f"(peak=${peak:.2f}, current=${bankroll:.2f})")

    return True, ""


# =======================================================================
# Position Exit Logic
# =======================================================================

def check_position_exits(client: KalshiClient, bankroll: float,
                         forecast_cache: dict) -> int:
    """
    Check open positions and exit when edge has flipped or decayed.

    Uses CURRENT model probabilities (recalculated from cached ensemble),
    not the stale entry-time probability.

    Returns number of positions exited.
    """
    unresolved = db.get_unresolved_trades()
    if not unresolved:
        return 0

    exits = 0
    today_str = now_eastern().strftime("%Y-%m-%d")

    for trade in unresolved:
        ticker = trade["ticker"]
        city_key = trade["city"]
        market_date = trade["market_date"]
        trade_side = trade.get("side", "yes")

        # Only check exits for markets that haven't expired
        if market_date < today_str:
            continue

        # Get current model data from forecast cache
        cache_key = f"{city_key}_{market_date}"
        if cache_key not in forecast_cache:
            continue

        cached = forecast_cache[cache_key]
        model_highs = cached.get("model_highs", {})
        if not model_highs:
            continue

        try:
            # Get current market price from Kalshi
            market_data = client.get_market(ticker)
            market = market_data.get("market", market_data)
            status = market.get("status", "")

            if status != "open":
                continue

            current_yes_bid = market.get("yes_bid", 0)
            if current_yes_bid:
                current_yes_bid = current_yes_bid / 100.0
            else:
                continue

            # Recalculate current model probability using stored bracket bounds
            bracket_low = trade.get("bracket_low")
            bracket_high = trade.get("bracket_high")
            bracket_label = trade["bracket"]

            city_info = CITIES.get(city_key, {})
            lead_days = compute_lead_days(market_date)
            lead_weights = get_model_weights(lead_days)
            base_error = city_info.get("base_error_std", 2.0)

            if bracket_low is not None and bracket_high is not None:
                # We have stored bounds — recalculate accurately
                bracket_tuples = [(bracket_low, bracket_high, bracket_label)]
                current_probs = calculate_bracket_probabilities(
                    model_highs, bracket_tuples,
                    base_error_std=base_error,
                    city_key=city_key,
                    weights_override=lead_weights,
                )
                current_model_prob = current_probs.get(bracket_label, 0.5)
            else:
                # Fallback: estimate from ensemble mean shift vs entry
                original_model_prob = trade.get("model_prob", 0.5)
                w_mean, w_std, _ = compute_weighted_ensemble(model_highs, lead_weights)
                if w_mean is None:
                    continue
                # Use original prob as approximation (legacy trades without bounds)
                current_model_prob = original_model_prob

            entry_price = trade["price"]

            # Determine current exit value and remaining edge
            if trade_side == "yes":
                current_value = current_yes_bid
                remaining_edge = current_model_prob - current_value
            else:
                current_yes_ask = market.get("yes_ask", 0)
                if current_yes_ask:
                    current_value = 1.0 - current_yes_ask / 100.0
                else:
                    continue
                remaining_edge = (1.0 - current_model_prob) - current_value

            # --- Exit conditions ---
            should_exit = False
            exit_reason = ""

            # 1. Edge flip: model now disagrees with our position direction
            if trade_side == "yes" and current_model_prob < EXIT_EDGE_FLIP_THRESHOLD:
                should_exit = True
                exit_reason = f"Model prob dropped to {current_model_prob:.3f}"
            elif trade_side == "no" and current_model_prob > (1.0 - EXIT_EDGE_FLIP_THRESHOLD):
                should_exit = True
                exit_reason = f"Model prob rose to {current_model_prob:.3f}"

            # 2. Edge decay: remaining edge too small, lock in profit
            if not should_exit and remaining_edge < EXIT_EDGE_DECAY_THRESHOLD:
                if current_value > entry_price:
                    should_exit = True
                    exit_reason = (f"Edge decayed to {remaining_edge:.3f}, "
                                   f"locking profit")

            if should_exit:
                exit_pnl = round((current_value - entry_price) * trade["contracts"], 2)
                exit_payout = round(current_value * trade["contracts"], 2)

                if PAPER_MODE:
                    db.resolve_trade(
                        trade["id"],
                        outcome=f"EXIT: {exit_reason}",
                        payout=exit_payout,
                        pnl=exit_pnl,
                    )
                    print(f"  [EXIT] {ticker} ({trade_side}): {exit_reason}, "
                          f"P&L=${exit_pnl:+.2f}")
                    exits += 1
                else:
                    # Live mode: place sell order
                    try:
                        price_cents = int(round(current_value * 100))
                        order_params = {
                            "ticker": ticker,
                            "side": trade_side,
                            "type": "limit",
                            "count": trade["contracts"],
                            "action": "sell",
                        }
                        if trade_side == "yes":
                            order_params["yes_price"] = price_cents
                        else:
                            order_params["no_price"] = price_cents

                        result = client.place_order(**order_params)
                        order_id = result.get("order", {}).get("order_id", "unknown")

                        db.resolve_trade(
                            trade["id"],
                            outcome=f"EXIT: {exit_reason}",
                            payout=exit_payout,
                            pnl=exit_pnl,
                        )
                        print(f"  [EXIT] {ticker} ({trade_side}): {exit_reason}, "
                              f"order={order_id}, P&L=${exit_pnl:+.2f}")
                        exits += 1
                    except Exception as e:
                        print(f"  [EXIT FAIL] {ticker}: {e}")

        except Exception as e:
            # Don't let exit check failures crash the scanner
            continue

    return exits


# =======================================================================
# Main Scan Cycle
# =======================================================================

def run_scan_cycle(client: KalshiClient, bankroll: float,
                   forecast_cache: dict = None) -> dict:
    """Run one full scan cycle across all cities."""
    now_et = now_eastern()
    timestamp = now_et.strftime("%H:%M:%S")
    today_str = now_et.strftime("%Y-%m-%d")
    current_hour = now_et.hour

    print(f"\n[{timestamp}] Scan cycle -- bankroll=${bankroll:.2f}")
    print(f"  Mode: {'PAPER' if PAPER_MODE else 'LIVE'} | "
          f"Threshold: {EDGE_THRESHOLD:.0%} | Kelly: {KELLY_FRACTION}")

    if forecast_cache is None:
        forecast_cache = {}

    # --- Drawdown check ---
    drawdown_ok, drawdown_reason = check_drawdown_limits(bankroll)
    if not drawdown_ok:
        print(f"  [CIRCUIT BREAKER] {drawdown_reason}")
        print(f"  Skipping scan cycle. Trading paused until limits reset.")
        return forecast_cache

    existing_positions = get_existing_positions()
    current_exposure = get_total_exposure()
    max_exposure = bankroll * MAX_TOTAL_EXPOSURE

    if existing_positions:
        print(f"  Open positions: {len(existing_positions)} ticker(s), "
              f"exposure=${current_exposure:.2f} / ${max_exposure:.2f}")

    # --- Position exit check ---
    exits = check_position_exits(client, bankroll, forecast_cache)
    if exits > 0:
        print(f"  Position exits: {exits}")

    total_opportunities = 0
    total_trades = 0
    total_spent = 0.0

    for city_key, city_info in CITIES.items():
        try:
            # 1. Get Kalshi markets
            markets = client.get_weather_markets_for_city(
                city_info["ticker_prefix"]
            )
            if not markets:
                print(f"  {city_key}: No open markets")
                continue

            # 2. Group by date
            date_groups = group_markets_by_date(markets)
            if not date_groups:
                continue

            for market_date, date_markets in sorted(date_groups.items()):
                # --- SAME-DAY LATE MODE (Phase 6) ---
                # After cutoff hour, switch to observation-dominated mode
                # instead of stopping entirely. Require stronger edge for safety.
                late_day_mode = False
                active_edge_threshold = EDGE_THRESHOLD
                if market_date == today_str and current_hour >= SAME_DAY_CUTOFF_HOUR:
                    obs_high = get_observed_high_today(city_key)
                    if obs_high is None:
                        print(f"  {city_key} [{market_date}]: SKIPPED "
                              f"(past {SAME_DAY_CUTOFF_HOUR}:00, no observations)")
                        continue
                    late_day_mode = True
                    active_edge_threshold = EDGE_THRESHOLD * LATE_DAY_EDGE_MULTIPLIER

                brackets = parse_brackets_from_markets(date_markets)
                if not brackets:
                    continue

                # Skip markets with no liquidity
                has_any_bid = any(b.get("yes_bid") is not None for b in brackets)
                if not has_any_bid:
                    print(f"  {city_key} [{market_date}]: SKIPPED (no liquidity)")
                    continue

                hours_remaining = estimate_hours_remaining(market_date)
                lead_days = compute_lead_days(market_date)

                # Log snapshots
                for b in brackets:
                    db.log_market_snapshot(
                        city=city_key,
                        market_date=market_date,
                        ticker=b["ticker"],
                        bracket=b["label"],
                        yes_price=b.get("yes_bid", 0) or 0,
                        no_price=1.0 - (b.get("yes_bid", 0) or 0),
                        yes_volume=b.get("volume", 0),
                    )

                # 3. Get forecasts (cache with staleness protection)
                cache_key = f"{city_key}_{market_date}"
                now = utc_now()

                # Tighter cache for same-day markets (15 min vs 30 min)
                cache_ttl = 900 if market_date == today_str else 1800

                if cache_key not in forecast_cache or \
                   (now - forecast_cache[cache_key]["fetched_at"]).total_seconds() > cache_ttl:
                    model_highs = get_all_model_highs(city_key, market_date)
                    forecast_cache[cache_key] = {
                        "model_highs": model_highs,
                        "fetched_at": now,
                    }

                    # Phase 1A: Log forecasts to DB for accuracy tracking
                    if model_highs:
                        for model_name, high_temp in model_highs.items():
                            db.log_forecast(
                                city_key, market_date, model_name, high_temp, {}
                            )
                else:
                    model_highs = forecast_cache[cache_key]["model_highs"]

                if not model_highs:
                    print(f"  {city_key} [{market_date}]: No forecast data")
                    continue

                # 4. Calculate probabilities with lead-time-dependent weights
                bracket_tuples = [(b["low"], b["high"], b["label"]) for b in brackets]

                # Get lead-time-dependent model weights
                lead_weights = get_model_weights(lead_days)

                # Per-city forecast uncertainty with lead-time scaling
                city_base_error = city_info.get("base_error_std", 2.0)
                if market_date == today_str:
                    base_error = city_base_error * 0.85
                elif lead_days == 1:
                    base_error = city_base_error
                else:
                    base_error = city_base_error * 1.4

                model_probs = calculate_bracket_probabilities(
                    model_highs, bracket_tuples,
                    base_error_std=base_error,
                    city_key=city_key,
                    weights_override=lead_weights,
                )

                # 5. Intraday observation update — now starts at 10am
                intraday_cutoff_hour = 10
                if (market_date == today_str
                        and hours_remaining < 12
                        and current_hour >= intraday_cutoff_hour):
                    obs = fetch_metar(city_info["station_id"])
                    if obs:
                        db.log_observation(
                            city_key, city_info["station_id"],
                            obs["temp_f"], obs["temp_c"],
                            obs["observation_time"], obs.get("raw_metar", ""),
                        )
                        observed_high = get_observed_high_today(city_key)
                        if observed_high is None:
                            observed_high = obs["temp_f"]

                        # Get ensemble high for improved diurnal model
                        w_mean, _, _ = compute_weighted_ensemble(
                            model_highs, lead_weights
                        )
                        forecast_high = None
                        if w_mean is not None:
                            bias = city_info.get("bias_correction", 0.0)
                            forecast_high = w_mean + bias

                        if hours_remaining > 0:
                            model_probs = update_probabilities_with_observation(
                                model_probs, bracket_tuples,
                                observed_high, hours_remaining,
                                forecast_high=forecast_high,
                                cloud_cover=obs.get("cloud_cover", 0.0),
                            )

                # 6. Display bracket comparison
                w_mean, w_std, n_models = compute_weighted_ensemble(
                    model_highs, lead_weights
                )
                nws_temp = model_highs.get("nws_official", None)
                hrrr_temp = model_highs.get("hrrr", None)
                bias = city_info.get("bias_correction", 0.0)
                corrected_mean = w_mean + bias

                nws_str = f", NWS={nws_temp:.0f}F" if nws_temp else ""
                hrrr_str = f", HRRR={hrrr_temp:.0f}F" if hrrr_temp else ""
                bias_str = f", bias={bias:+.1f}" if bias != 0 else ""
                err_str = f", err={base_error:.1f}"
                lead_str = f", lead={lead_days}d"

                print(f"  {city_key} [{market_date}]: {len(brackets)} brackets, "
                      f"raw={w_mean:.1f}F, corrected={corrected_mean:.1f}F "
                      f"(+/-{w_std:.1f}), n={n_models}{nws_str}{hrrr_str}"
                      f"{bias_str}{err_str}{lead_str}")

                for b in brackets:
                    lbl = b["label"][:22].ljust(22)
                    low_s = f"{b['low']:.0f}" if b['low'] != float('-inf') else "-inf"
                    high_s = f"{b['high']:.0f}" if b['high'] != float('inf') else "+inf"
                    bounds = f"[{low_s},{high_s}]".ljust(12)
                    mp = b["mid_price"]
                    mp_str = f"${mp:.2f}" if mp else "  -- "
                    model_p = model_probs.get(b["label"], 0.0)

                    # Show edge vs ask price (not mid)
                    yes_edge_str = ""
                    no_edge_str = ""
                    if b.get("yes_ask") and model_p > b["yes_ask"]:
                        net_e = model_p - b["yes_ask"] - kalshi_taker_fee(b["yes_ask"], 1)
                        if net_e >= EDGE_THRESHOLD:
                            yes_edge_str = f" <-- BUY YES (net edge={net_e:.0%})"
                    if b.get("yes_bid") and (1-model_p) > (1-b["yes_bid"]):
                        no_ask_est = round(1.0 - b["yes_bid"], 2)
                        net_e = (1-model_p) - no_ask_est - kalshi_taker_fee(no_ask_est, 1)
                        if net_e >= EDGE_THRESHOLD:
                            no_edge_str = f" <-- BUY NO (net edge={net_e:.0%})"

                    flag = yes_edge_str or no_edge_str

                    bid_s = f"b${b['yes_bid']:.2f}" if b.get('yes_bid') else "b---"
                    ask_s = f"a${b['yes_ask']:.2f}" if b.get('yes_ask') else "a---"
                    sprd = ""
                    if b.get("yes_bid") and b.get("yes_ask"):
                        sprd = f" s={b['yes_ask']-b['yes_bid']:.2f}"

                    print(f"    {lbl} {bounds} mkt={mp_str} mod={model_p:.3f} "
                          f"({bid_s}/{ask_s}{sprd}){flag}")

                # 7. Find and execute opportunities
                opps = find_opportunities(
                    brackets, model_probs, bankroll, existing_positions,
                    edge_threshold=active_edge_threshold,
                )
                total_opportunities += len(opps)

                if opps:
                    print(f"    >>> {len(opps)} tradeable opportunity(ies):")
                    city_budget = bankroll * MAX_POSITION_PER_CITY
                    city_spent = 0.0

                    # Correlation group exposure check
                    group_name = get_city_group(city_key)
                    group_cities = CORRELATION_GROUPS.get(group_name, [city_key])
                    group_exposure = db.get_group_exposure(group_cities)
                    group_budget = bankroll * MAX_POSITION_PER_GROUP

                    for opp in opps:
                        # Check total exposure cap
                        if current_exposure + total_spent + opp["cost"] > max_exposure:
                            print(f"    [SKIP] {opp['ticker']}: "
                                  f"total exposure would exceed "
                                  f"${max_exposure:.2f} ({MAX_TOTAL_EXPOSURE:.0%} of bankroll)")
                            continue

                        # Check per-city cap
                        if city_spent + opp["cost"] > city_budget:
                            print(f"    [SKIP] {opp['ticker']}: exceeds city budget")
                            continue

                        # Check correlation group cap
                        if group_exposure + city_spent + opp["cost"] > group_budget:
                            print(f"    [SKIP] {opp['ticker']}: exceeds "
                                  f"{group_name} group budget "
                                  f"(${group_exposure:.2f}/${group_budget:.2f})")
                            continue

                        # Phase 8: Orderbook depth check (live mode only)
                        if not PAPER_MODE and opp.get("order_type") == "taker":
                            avail = check_orderbook_depth(
                                client, opp["ticker"], opp["side"],
                                opp["exec_price"], opp["contracts"],
                            )
                            if avail < opp["contracts"]:
                                if avail < 1:
                                    print(f"    [SKIP] {opp['ticker']}: no liquidity at target price")
                                    continue
                                old_ct = opp["contracts"]
                                opp["contracts"] = avail
                                opp["cost"] = round(avail * opp["exec_price"], 2)
                                print(f"    [DEPTH] {opp['ticker']}: reduced {old_ct} -> {avail} contracts")

                        success = execute_trade(
                            client, opp, city_key, market_date, bankroll
                        )
                        if success:
                            total_trades += 1
                            city_spent += opp["cost"]
                            total_spent += opp["cost"]
                            bankroll -= opp["cost"]
                            existing_positions.add(opp["ticker"])

        except Exception as e:
            print(f"  {city_key}: ERROR -- {e}")
            traceback.print_exc()

    print(f"\n  Summary: {total_opportunities} opps, "
          f"{total_trades} trades (${total_spent:.2f} deployed), "
          f"bankroll=${bankroll:.2f}")

    return forecast_cache


# =======================================================================
# Main
# =======================================================================

def main():
    print("=" * 70)
    print("KALSHI WEATHER TRADING BOT -- v3")
    print(f"  Mode: {'PAPER TRADING' if PAPER_MODE else '*** LIVE TRADING ***'}")
    print(f"  Edge threshold: {EDGE_THRESHOLD:.0%} (net of fees)")
    print(f"  Kelly fraction: {KELLY_FRACTION}")
    print(f"  Max total exposure: {MAX_TOTAL_EXPOSURE:.0%}")
    print(f"  Max spread: ${MAX_SPREAD:.2f} | Min volume: {MIN_VOLUME}")
    print(f"  Max drawdown: {MAX_DRAWDOWN:.0%} | Max daily loss: {MAX_DAILY_LOSS:.0%}")
    print(f"  Same-day cutoff: {SAME_DAY_CUTOFF_HOUR}:00")
    print(f"  Models: {len(MODEL_WEIGHTS)} ({', '.join(MODEL_WEIGHTS.keys())})")
    print(f"  Correlation groups: {list(CORRELATION_GROUPS.keys())}")
    print(f"  Scan interval: {SCAN_INTERVAL_SECONDS}s")
    print("=" * 70)

    # Initialize Kalshi client
    client = KalshiClient()
    print("[OK] Kalshi client initialized")

    # Get live balance
    try:
        bal = client.get_balance()
        balance_usd = bal.get("balance", 0) / 100.0
        print(f"[OK] Kalshi balance: ${balance_usd:.2f}")
    except Exception as e:
        print(f"[WARN] Could not fetch balance: {e}")
        balance_usd = 0.0

    # Get bankroll from DB or initialize
    bankroll = db.get_bankroll()
    if bankroll <= 0:
        from config import STARTING_BANKROLL
        bankroll = STARTING_BANKROLL
        db.update_bankroll(bankroll, "Initial bankroll")
        print(f"[OK] Initialized bankroll: ${bankroll:.2f}")
    else:
        print(f"[OK] Bankroll from DB: ${bankroll:.2f}")

    # Show city parameters and risk limits
    print(f"\n  City parameters:")
    for ck, ci in CITIES.items():
        b = ci.get("bias_correction", 0.0)
        e = ci.get("base_error_std", 2.0)
        group = get_city_group(ck)
        b_str = f"bias={b:+.1f}F" if b != 0 else "bias=none"
        print(f"    {ck}: {b_str}, err_std={e:.1f}F, group={group}")

    # Show drawdown status
    peak = db.get_peak_bankroll()
    if peak > 0:
        dd = (peak - bankroll) / peak
        print(f"\n  Peak bankroll: ${peak:.2f}, drawdown: {dd:.1%}")

    # Single scan mode
    if "--once" in sys.argv:
        print("\nRunning single scan cycle...\n")
        run_scan_cycle(client, bankroll)
        return

    # Continuous scanning
    print(f"\nStarting continuous scan (Ctrl+C to stop)...\n")
    forecast_cache = {}

    while RUNNING:
        try:
            forecast_cache = run_scan_cycle(client, bankroll, forecast_cache)
        except Exception as e:
            print(f"\n[ERROR] Scan cycle failed: {e}")
            traceback.print_exc()

        if RUNNING:
            print(f"\n  Next scan in {SCAN_INTERVAL_SECONDS}s...")
            for _ in range(SCAN_INTERVAL_SECONDS):
                if not RUNNING:
                    break
                time.sleep(1)

    print("\n[SCANNER] Stopped.")


if __name__ == "__main__":
    main()
