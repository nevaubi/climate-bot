"""
Trade resolution script — v3.1 (settlement-aware).

Checks Kalshi for settled markets and resolves open trades.
Calculates side-aware P&L (YES wins pay out on "yes" result,
NO wins pay out on "no" result) and updates bankroll.
Logs model accuracy data for auto-calibration.

IMPORTANT: Markets have status='closed' before settlement and
status='settled' after. The 'result' field ('yes'/'no') is only
populated after settlement (typically 10am ET next day).
Don't resolve on 'closed' — wait for 'settled' or use actual temps.

Usage:
    python resolve.py               # resolve settled trades
    python resolve.py --force       # also resolve 'closed' using actual temps
"""

import sys
import re
from datetime import datetime, timedelta
from kalshi_client import KalshiClient
import db


def _determine_result_from_actual(trade: dict, actual_high: float) -> str:
    """
    For paper trades or when API result is empty, determine the bracket
    outcome from the actual observed high temperature.

    Returns 'yes' if actual falls in bracket, 'no' otherwise, or '' if
    we can't determine.
    """
    bracket_low = trade.get("bracket_low")
    bracket_high = trade.get("bracket_high")

    if bracket_low is not None and bracket_high is not None:
        if bracket_low <= actual_high <= bracket_high:
            return "yes"
        else:
            return "no"

    # Fallback: parse bracket label
    label = trade.get("bracket", "")
    nums = re.findall(r'-?\d+\.?\d*', label)

    if "or below" in label.lower() or "or lower" in label.lower():
        if nums:
            return "yes" if actual_high <= float(nums[0]) else "no"
    elif "or above" in label.lower() or "or higher" in label.lower():
        if nums:
            return "yes" if actual_high >= float(nums[0]) else "no"
    elif len(nums) >= 2:
        low, high = float(nums[0]), float(nums[1])
        return "yes" if low <= actual_high <= high else "no"

    return ""


def _get_actual_high(city_key: str, market_date: str) -> float:
    """Get actual observed high from multiple sources."""
    # Try NWS gridpoint data
    try:
        from weather import fetch_nws_forecast
        nws_data = fetch_nws_forecast(city_key)
        if market_date in nws_data:
            return nws_data[market_date]
    except Exception:
        pass

    # Try Open-Meteo archive
    try:
        from calibrate import fetch_open_meteo_historical
        archive = fetch_open_meteo_historical(city_key, market_date, market_date)
        if market_date in archive:
            return archive[market_date]
    except Exception:
        pass

    # Try stored observations
    try:
        from weather import get_observed_high_today
        conn = db.get_conn()
        row = conn.execute("""
            SELECT MAX(temp_f) as max_temp FROM observations
            WHERE city = ? AND timestamp LIKE ?
        """, (city_key, f"{market_date}%")).fetchone()
        conn.close()
        if row and row["max_temp"] is not None:
            return row["max_temp"]
    except Exception:
        pass

    return None


def resolve_trades():
    print("=" * 60)
    print("TRADE RESOLUTION — v3.1 (settlement-aware)")
    print("=" * 60)

    force_mode = "--force" in sys.argv

    client = KalshiClient()
    unresolved = db.get_unresolved_trades()

    if not unresolved:
        print("No unresolved trades found.")
        return

    print(f"Found {len(unresolved)} unresolved trade(s).")
    if force_mode:
        print("  --force: will resolve 'closed' markets using actual temperatures")
    print()

    total_pnl = 0.0
    resolved_count = 0
    resolved_dates = set()  # track dates for accuracy logging

    for trade in unresolved:
        ticker = trade["ticker"]
        trade_side = trade.get("side", "yes")  # yes or no
        print(f"Checking {ticker} ({trade['city']} {trade['market_date']}, "
              f"side={trade_side})...")

        try:
            market_data = client.get_market(ticker)
            market = market_data.get("market", market_data)
            status = market.get("status", "")
            result = market.get("result", "")

            # Only resolve fully settled markets (result field populated)
            if status in ("settled", "finalized") and result in ("yes", "no"):
                pass  # Good to go
            elif status == "closed" and not result:
                if force_mode or trade.get("paper_mode"):
                    # Try to determine result from actual temperature
                    actual_high = _get_actual_high(trade["city"], trade["market_date"])
                    if actual_high is not None:
                        result = _determine_result_from_actual(trade, actual_high)
                        if result:
                            print(f"  Resolved from actual temp ({actual_high:.1f}F): "
                                  f"bracket={result}")
                        else:
                            print(f"  Closed but can't determine result, skipping.")
                            continue
                    else:
                        print(f"  Closed, no result yet and no actual temp available. Skipping.")
                        continue
                else:
                    print(f"  Closed but not settled yet (result empty). "
                          f"Try again after 10am ET or use --force.")
                    continue
            elif status not in ("settled", "finalized", "closed"):
                print(f"  Still open (status={status}), skipping.")
                continue
            else:
                print(f"  Status={status}, result={result!r}, skipping.")
                continue

            # Determine outcome — side-aware P&L
            contracts = trade["contracts"]
            cost = trade["cost"]
            fee = trade["fee"]

            if trade_side == "yes":
                # Bought YES contracts
                if result == "yes":
                    payout = contracts * 1.0
                    pnl = payout - cost - fee
                    outcome = "WIN"
                elif result == "no":
                    payout = 0.0
                    pnl = -cost - fee
                    outcome = "LOSS"
                else:
                    payout = cost
                    pnl = 0.0
                    outcome = f"VOID ({result})"
            elif trade_side == "no":
                # Bought NO contracts
                if result == "no":
                    payout = contracts * 1.0
                    pnl = payout - cost - fee
                    outcome = "WIN"
                elif result == "yes":
                    payout = 0.0
                    pnl = -cost - fee
                    outcome = "LOSS"
                else:
                    payout = cost
                    pnl = 0.0
                    outcome = f"VOID ({result})"
            else:
                # Unknown side — treat as YES for backwards compatibility
                if result == "yes":
                    payout = contracts * 1.0
                    pnl = payout - cost - fee
                    outcome = "WIN"
                elif result == "no":
                    payout = 0.0
                    pnl = -cost - fee
                    outcome = "LOSS"
                else:
                    payout = cost
                    pnl = 0.0
                    outcome = f"VOID ({result})"

            db.resolve_trade(trade["id"], outcome, payout, pnl)
            total_pnl += pnl
            resolved_count += 1
            resolved_dates.add((trade["city"], trade["market_date"]))

            prefix = "PAPER " if trade["paper_mode"] else ""
            print(f"  {prefix}{outcome} (side={trade_side}, result={result}): "
                  f"payout=${payout:.2f}, cost=${cost:.2f}, fee=${fee:.2f}, "
                  f"P&L=${pnl:+.2f}")

        except Exception as e:
            print(f"  Error checking {ticker}: {e}")

    # Update bankroll
    if resolved_count > 0:
        bankroll = db.get_bankroll()
        new_bankroll = bankroll + total_pnl
        db.update_bankroll(
            new_bankroll,
            f"Resolved {resolved_count} trades, P&L=${total_pnl:+.2f}"
        )
        print(f"\n{'=' * 40}")
        print(f"Resolved: {resolved_count} trade(s)")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"Bankroll: ${bankroll:.2f} -> ${new_bankroll:.2f}")
        print(f"{'=' * 40}")

        # Log model accuracy for resolved dates
        log_accuracy_for_dates(resolved_dates)
    else:
        print("\nNo trades resolved this run.")


def log_accuracy_for_dates(resolved_dates: set):
    """
    For each resolved (city, date), compare stored model forecasts
    against the actual result and log to model_accuracy table.

    This data feeds into the auto-calibration system.
    """
    if not resolved_dates:
        return

    print(f"\nLogging model accuracy for {len(resolved_dates)} city-date(s)...")

    for city_key, market_date in resolved_dates:
        # Get stored forecasts for this city/date
        forecasts = db.get_forecasts_for_date(city_key, market_date)
        if not forecasts:
            continue

        # Try to get actual observed high from the archive
        # For now, use NWS gridpoint data which includes recent observations
        from weather import fetch_nws_forecast
        nws_data = fetch_nws_forecast(city_key)
        actual_high = nws_data.get(market_date)

        if actual_high is None:
            # Try Open-Meteo archive as fallback
            try:
                from calibrate import fetch_open_meteo_historical
                archive = fetch_open_meteo_historical(
                    city_key, market_date, market_date
                )
                actual_high = archive.get(market_date)
            except Exception:
                pass

        if actual_high is None:
            print(f"  {city_key} [{market_date}]: No actual high available")
            continue

        # Log accuracy for each model
        logged = 0
        for forecast in forecasts:
            model_name = forecast["model_name"]
            forecast_high = forecast["forecast_high_f"]
            db.log_model_accuracy(
                city_key, market_date, model_name,
                forecast_high, actual_high
            )
            logged += 1

        print(f"  {city_key} [{market_date}]: actual={actual_high:.1f}F, "
              f"logged {logged} model comparisons")


def show_stats():
    """Print cumulative trading statistics."""
    conn = db.get_conn()

    total = conn.execute(
        "SELECT COUNT(*) as n FROM trades WHERE resolved = 1"
    ).fetchone()["n"]

    if total == 0:
        print("\nNo resolved trades yet.")
        conn.close()
        return

    wins = conn.execute(
        "SELECT COUNT(*) as n FROM trades WHERE resolved = 1 AND outcome = 'WIN'"
    ).fetchone()["n"]

    total_pnl = conn.execute(
        "SELECT SUM(pnl) as s FROM trades WHERE resolved = 1"
    ).fetchone()["s"] or 0.0

    avg_edge = conn.execute(
        "SELECT AVG(edge) as e FROM trades WHERE resolved = 1"
    ).fetchone()["e"] or 0.0

    total_cost = conn.execute(
        "SELECT SUM(cost) as c FROM trades WHERE resolved = 1"
    ).fetchone()["c"] or 0.0

    # Side breakdown
    yes_trades = conn.execute(
        "SELECT COUNT(*) as n FROM trades WHERE resolved = 1 AND side = 'yes'"
    ).fetchone()["n"]
    no_trades = conn.execute(
        "SELECT COUNT(*) as n FROM trades WHERE resolved = 1 AND side = 'no'"
    ).fetchone()["n"]

    yes_wins = conn.execute(
        "SELECT COUNT(*) as n FROM trades WHERE resolved = 1 AND side = 'yes' AND outcome = 'WIN'"
    ).fetchone()["n"]
    no_wins = conn.execute(
        "SELECT COUNT(*) as n FROM trades WHERE resolved = 1 AND side = 'no' AND outcome = 'WIN'"
    ).fetchone()["n"]

    conn.close()

    win_rate = wins / total * 100 if total > 0 else 0
    roi = total_pnl / total_cost * 100 if total_cost > 0 else 0

    print(f"\n--- Cumulative Stats ---")
    print(f"Total trades: {total}")
    print(f"Win rate: {win_rate:.1f}% ({wins}/{total})")
    print(f"Total P&L: ${total_pnl:+.2f}")
    print(f"ROI: {roi:+.1f}%")
    print(f"Avg edge at entry: {avg_edge:.3f}")

    if yes_trades > 0:
        yes_wr = yes_wins / yes_trades * 100
        print(f"  YES trades: {yes_trades} ({yes_wr:.0f}% win rate)")
    if no_trades > 0:
        no_wr = no_wins / no_trades * 100
        print(f"  NO trades:  {no_trades} ({no_wr:.0f}% win rate)")

    print(f"Current bankroll: ${db.get_bankroll():.2f}")

    # Show model accuracy if available
    accuracy_stats = db.get_model_accuracy_stats(30)
    if accuracy_stats:
        print(f"\n--- Model Accuracy (30 day) ---")
        for city, models in sorted(accuracy_stats.items()):
            for model, stats in sorted(models.items()):
                if stats["n_samples"] >= 3:
                    print(f"  {city}/{model}: "
                          f"bias={stats['bias']:+.1f}F, "
                          f"n={stats['n_samples']}")


if __name__ == "__main__":
    resolve_trades()
    show_stats()
