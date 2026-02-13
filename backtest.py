"""
Backtesting Framework — Simulates scanner decisions on historical data.

Loads stored forecasts, market snapshots, and actual outcomes from the DB
to evaluate how the strategy would have performed.

Usage:
    python backtest.py                # backtest all available data
    python backtest.py --days 7       # backtest last 7 days
    python backtest.py --city NYC     # backtest one city only
"""

import sys
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
from scipy import stats

import db
from config import (
    CITIES, EDGE_THRESHOLD, KELLY_FRACTION, MAX_CONTRACTS_PER_TRADE,
    MAX_SPREAD, MIN_VOLUME, kalshi_taker_fee, kalshi_maker_fee,
    get_model_weights,
)
from weather import calculate_bracket_probabilities, compute_weighted_ensemble


def load_historical_data(days: int = 30, city_filter: str = None):
    """
    Load historical forecasts, snapshots, and trade outcomes from DB.

    Returns list of dicts, one per (city, date) pair with forecasts and market data.
    """
    conn = db.get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    # Get all distinct city/date combos with forecasts
    city_clause = f"AND city = '{city_filter}'" if city_filter else ""
    dates = conn.execute(f"""
        SELECT DISTINCT city, forecast_date
        FROM model_forecasts
        WHERE forecast_date >= ? {city_clause}
        ORDER BY forecast_date, city
    """, (cutoff,)).fetchall()

    scenarios = []
    for row in dates:
        city_key = row["city"]
        market_date = row["forecast_date"]

        # Get forecasts for this city/date
        forecasts = conn.execute("""
            SELECT model_name, forecast_high_f
            FROM model_forecasts
            WHERE city = ? AND forecast_date = ?
            ORDER BY timestamp DESC
        """, (city_key, market_date)).fetchall()

        # Deduplicate (latest per model)
        model_highs = {}
        for f in forecasts:
            if f["model_name"] not in model_highs:
                model_highs[f["model_name"]] = f["forecast_high_f"]

        if len(model_highs) < 3:
            continue  # Need at least 3 models for useful ensemble

        # Get market snapshots for this city/date
        snapshots = conn.execute("""
            SELECT ticker, bracket, yes_price, no_price, yes_volume
            FROM market_snapshots
            WHERE city = ? AND market_date = ?
            ORDER BY timestamp DESC
        """, (city_key, market_date)).fetchall()

        # Deduplicate snapshots (latest per ticker)
        seen = set()
        brackets = []
        for s in snapshots:
            if s["ticker"] not in seen:
                seen.add(s["ticker"])
                brackets.append(dict(s))

        if not brackets:
            continue

        # Get actual outcome (from resolved trades or model_accuracy)
        actual_row = conn.execute("""
            SELECT actual_high_f FROM model_accuracy
            WHERE city = ? AND market_date = ?
            LIMIT 1
        """, (city_key, market_date)).fetchone()

        actual_high = actual_row["actual_high_f"] if actual_row else None

        scenarios.append({
            "city": city_key,
            "date": market_date,
            "model_highs": model_highs,
            "brackets": brackets,
            "actual_high": actual_high,
        })

    conn.close()
    return scenarios


def simulate_trades(scenarios: list, edge_threshold: float = None):
    """
    Simulate trade decisions on historical data.

    Returns list of simulated trade results.
    """
    if edge_threshold is None:
        edge_threshold = EDGE_THRESHOLD

    results = []

    for scenario in scenarios:
        city_key = scenario["city"]
        market_date = scenario["date"]
        model_highs = scenario["model_highs"]
        actual_high = scenario["actual_high"]
        city_info = CITIES.get(city_key, {})

        # Calculate lead days (assume 0 for historical)
        lead_days = 0
        lead_weights = get_model_weights(lead_days)

        base_error = city_info.get("base_error_std", 2.0)

        # Build bracket tuples from snapshot data
        bracket_tuples = []
        bracket_prices = {}
        for b in scenario["brackets"]:
            label = b["bracket"]
            yes_price = b.get("yes_price", 0)
            volume = b.get("yes_volume", 0)

            # Skip illiquid brackets
            if yes_price <= 0 or volume < MIN_VOLUME:
                continue

            # Parse bounds from label (simplified)
            try:
                parts = label.split(" to ")
                if len(parts) == 2:
                    low = float(parts[0].replace("°", "").strip()) - 0.5
                    high = float(parts[1].replace("°", "").strip()) + 0.5
                elif "or below" in label.lower() or "or lower" in label.lower():
                    import re
                    nums = re.findall(r'-?\d+', label)
                    if nums:
                        high = float(nums[0])
                        low = float("-inf")
                    else:
                        continue
                elif "or above" in label.lower() or "or higher" in label.lower():
                    import re
                    nums = re.findall(r'-?\d+', label)
                    if nums:
                        low = float(nums[0])
                        high = float("inf")
                    else:
                        continue
                else:
                    continue
            except (ValueError, IndexError):
                continue

            bracket_tuples.append((low, high, label))
            bracket_prices[label] = {
                "yes_price": yes_price,
                "volume": volume,
                "ticker": b.get("ticker", ""),
            }

        if not bracket_tuples:
            continue

        # Calculate model probabilities
        model_probs = calculate_bracket_probabilities(
            model_highs, bracket_tuples,
            base_error_std=base_error,
            city_key=city_key,
            weights_override=lead_weights,
        )

        # Find opportunities
        for label, model_prob in model_probs.items():
            if label not in bracket_prices:
                continue

            bp = bracket_prices[label]
            yes_ask = bp["yes_price"]

            # YES opportunity
            if 0.05 <= yes_ask <= 0.95:
                fee = kalshi_taker_fee(yes_ask, 1)
                net_edge = model_prob - yes_ask - fee
                if net_edge >= edge_threshold:
                    # Determine actual outcome
                    won = None
                    if actual_high is not None:
                        low, high, _ = next(
                            (t for t in bracket_tuples if t[2] == label), (None, None, None)
                        )
                        if low is not None:
                            won = low <= actual_high <= high

                    results.append({
                        "city": city_key,
                        "date": market_date,
                        "side": "yes",
                        "label": label,
                        "exec_price": yes_ask,
                        "model_prob": model_prob,
                        "edge": net_edge,
                        "won": won,
                        "pnl": (1.0 - yes_ask - fee) if won else (-yes_ask - fee) if won is not None else None,
                    })

            # NO opportunity
            no_model_prob = 1.0 - model_prob
            no_ask = round(1.0 - yes_ask, 2) if yes_ask else None
            if no_ask and 0.05 <= no_ask <= 0.95:
                fee = kalshi_taker_fee(no_ask, 1)
                net_edge = no_model_prob - no_ask - fee
                if net_edge >= edge_threshold:
                    won = None
                    if actual_high is not None:
                        low, high, _ = next(
                            (t for t in bracket_tuples if t[2] == label), (None, None, None)
                        )
                        if low is not None:
                            # NO wins when actual is NOT in bracket
                            won = not (low <= actual_high <= high)

                    results.append({
                        "city": city_key,
                        "date": market_date,
                        "side": "no",
                        "label": label,
                        "exec_price": no_ask,
                        "model_prob": no_model_prob,
                        "edge": net_edge,
                        "won": won,
                        "pnl": (1.0 - no_ask - fee) if won else (-no_ask - fee) if won is not None else None,
                    })

    return results


def report_results(results: list):
    """Print backtest summary report."""
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    if not results:
        print("No simulated trades. Need more historical data.")
        print("The bot needs to run for several days to accumulate forecast data.")
        return

    total = len(results)
    resolved = [r for r in results if r["won"] is not None]
    unresolved = total - len(resolved)

    print(f"\nTotal simulated trades: {total}")
    print(f"  With known outcomes: {len(resolved)}")
    print(f"  Unresolved: {unresolved}")

    if not resolved:
        print("\nNo resolved outcomes yet — actuals not available.")
        print("Run resolve.py after market settlement to fill in actuals.")

        # Still show opportunity analysis
        print(f"\n--- Opportunity Analysis ---")
        edges = [r["edge"] for r in results]
        print(f"  Mean predicted edge: {np.mean(edges):.1%}")
        print(f"  Median predicted edge: {np.median(edges):.1%}")
        print(f"  Edge range: {min(edges):.1%} to {max(edges):.1%}")

        by_city = defaultdict(list)
        for r in results:
            by_city[r["city"]].append(r)
        print(f"\n{'City':<6} {'Trades':>7} {'Avg Edge':>9} {'YES':>5} {'NO':>5}")
        print("-" * 40)
        for city in sorted(by_city.keys()):
            trades = by_city[city]
            yes_ct = sum(1 for t in trades if t["side"] == "yes")
            no_ct = sum(1 for t in trades if t["side"] == "no")
            avg_edge = np.mean([t["edge"] for t in trades])
            print(f"{city:<6} {len(trades):>7} {avg_edge:>8.1%} {yes_ct:>5} {no_ct:>5}")

        return

    # Win rate
    wins = sum(1 for r in resolved if r["won"])
    losses = len(resolved) - wins
    win_rate = wins / len(resolved) * 100

    # P&L
    pnls = [r["pnl"] for r in resolved if r["pnl"] is not None]
    total_pnl = sum(pnls) if pnls else 0
    avg_pnl = np.mean(pnls) if pnls else 0

    print(f"\n--- Performance ---")
    print(f"  Win rate: {wins}/{len(resolved)} ({win_rate:.1f}%)")
    print(f"  Total P&L (per contract): ${total_pnl:+.2f}")
    print(f"  Average P&L per trade: ${avg_pnl:+.4f}")

    if pnls:
        # Sharpe-like ratio (simple)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
            print(f"  Sharpe ratio (per trade): {sharpe:.2f}")

        # Max drawdown (cumulative P&L)
        cum_pnl = np.cumsum(pnls)
        peak_pnl = np.maximum.accumulate(cum_pnl)
        drawdowns = peak_pnl - cum_pnl
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
        print(f"  Max drawdown: ${max_dd:.2f}")

    # Edge bucket analysis
    print(f"\n--- Edge vs Outcome ---")
    print(f"{'Edge Range':<12} {'N':>5} {'WR%':>6} {'Avg P&L':>9}")
    print("-" * 35)
    buckets = [
        ("10-15%", 0.10, 0.15),
        ("15-20%", 0.15, 0.20),
        ("20-30%", 0.20, 0.30),
        ("30%+",   0.30, 1.00),
    ]
    for label, lo, hi in buckets:
        bucket_trades = [r for r in resolved if lo <= r["edge"] < hi]
        if not bucket_trades:
            continue
        bwins = sum(1 for t in bucket_trades if t["won"])
        bwr = bwins / len(bucket_trades) * 100
        bpnl = np.mean([t["pnl"] for t in bucket_trades if t["pnl"] is not None])
        print(f"{label:<12} {len(bucket_trades):>5} {bwr:>5.0f}% ${bpnl:>+8.4f}")

    # Per city
    print(f"\n--- By City ---")
    print(f"{'City':<6} {'N':>5} {'WR%':>6} {'P&L':>9}")
    print("-" * 30)
    by_city = defaultdict(list)
    for r in resolved:
        by_city[r["city"]].append(r)
    for city in sorted(by_city.keys()):
        trades = by_city[city]
        cwins = sum(1 for t in trades if t["won"])
        cwr = cwins / len(trades) * 100
        cpnl = sum(t["pnl"] for t in trades if t["pnl"] is not None)
        print(f"{city:<6} {len(trades):>5} {cwr:>5.0f}% ${cpnl:>+8.2f}")


def main():
    days = 30
    city_filter = None

    if "--days" in sys.argv:
        idx = sys.argv.index("--days")
        if idx + 1 < len(sys.argv):
            days = int(sys.argv[idx + 1])

    if "--city" in sys.argv:
        idx = sys.argv.index("--city")
        if idx + 1 < len(sys.argv):
            city_filter = sys.argv[idx + 1].upper()

    print(f"Loading historical data ({days} days)...")
    scenarios = load_historical_data(days, city_filter)
    print(f"  Found {len(scenarios)} city-date scenarios")

    if not scenarios:
        print("\nNo historical data available for backtesting.")
        print("The bot needs to run for at least a few days to accumulate data.")
        print("Forecasts are logged each scan cycle (Phase 1A).")
        return

    print(f"\nSimulating trades...")
    results = simulate_trades(scenarios)

    report_results(results)


if __name__ == "__main__":
    main()
