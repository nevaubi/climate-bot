"""
Performance analytics and reporting — v3.

Provides summary reports for trading performance, model accuracy,
and city-level breakdowns.

Usage:
    python analytics.py                # full report
    python analytics.py --models       # model accuracy only
    python analytics.py --city NYC     # single city report
"""

import sys
from datetime import datetime, timedelta
import db
from config import CITIES, MODEL_WEIGHTS


def show_daily_report(date_str: str = None):
    """Show today's trading activity and P&L."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    trades = db.get_todays_trades(date_str)

    print(f"\n{'='*60}")
    print(f"DAILY REPORT: {date_str}")
    print(f"{'='*60}")

    if not trades:
        print("No trades for this date.")
        return

    total_cost = 0
    total_pnl = 0
    wins = 0
    losses = 0
    pending = 0

    for t in trades:
        side = t.get("side", "?")
        resolved = t.get("resolved", 0)
        outcome = t.get("outcome", "PENDING")
        pnl = t.get("pnl", 0) or 0

        status = outcome if resolved else "OPEN"
        pnl_str = f"${pnl:+.2f}" if resolved else "---"

        print(f"  #{t['id']} {t['city']} {t['ticker'][:25]} "
              f"{side.upper()} {t['contracts']}x @ ${t['price']:.2f} "
              f"edge={t['edge']:.3f} -> {status} {pnl_str}")

        total_cost += t["cost"]
        if resolved:
            total_pnl += pnl
            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
        else:
            pending += 1

    print(f"\n  Total: {len(trades)} trades, ${total_cost:.2f} deployed")
    if wins + losses > 0:
        wr = wins / (wins + losses) * 100
        print(f"  Resolved: {wins}W / {losses}L ({wr:.0f}%), P&L=${total_pnl:+.2f}")
    if pending > 0:
        print(f"  Pending: {pending} trade(s)")
    print(f"  Bankroll: ${db.get_bankroll():.2f}")


def show_city_performance(days: int = 30):
    """Show per-city P&L and win rate."""
    conn = db.get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"CITY PERFORMANCE (last {days} days)")
    print(f"{'='*60}")
    print(f"{'City':<6} {'Trades':>7} {'Wins':>5} {'WR%':>6} "
          f"{'P&L':>9} {'Avg Edge':>9} {'ROI%':>7}")
    print("-" * 55)

    for city_key in CITIES:
        rows = conn.execute("""
            SELECT COUNT(*) as n,
                   SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as total_pnl,
                   AVG(edge) as avg_edge,
                   SUM(cost) as total_cost
            FROM trades
            WHERE city = ? AND resolved = 1 AND market_date >= ?
        """, (city_key, cutoff)).fetchone()

        n = rows["n"]
        if n == 0:
            continue

        wins = rows["wins"] or 0
        wr = wins / n * 100 if n > 0 else 0
        pnl = rows["total_pnl"] or 0
        avg_edge = rows["avg_edge"] or 0
        cost = rows["total_cost"] or 0
        roi = pnl / cost * 100 if cost > 0 else 0

        print(f"{city_key:<6} {n:>7} {wins:>5} {wr:>5.0f}% "
              f"${pnl:>+8.2f} {avg_edge:>8.3f} {roi:>+6.1f}%")

    conn.close()


def show_model_accuracy(days: int = 30):
    """Show per-model forecast accuracy."""
    stats = db.get_model_accuracy_stats(days)

    print(f"\n{'='*60}")
    print(f"MODEL ACCURACY (last {days} days)")
    print(f"{'='*60}")

    if not stats:
        print("No accuracy data available yet.")
        print("Run resolve.py after trades settle to start collecting data.")
        return

    print(f"{'City':<6} {'Model':<22} {'Bias':>8} {'Samples':>8} {'Weight':>7}")
    print("-" * 55)

    for city_key in sorted(stats.keys()):
        models = stats[city_key]
        for model_name in sorted(models.keys()):
            s = models[model_name]
            weight = MODEL_WEIGHTS.get(model_name, 1.0)
            print(f"{city_key:<6} {model_name:<22} {s['bias']:>+7.1f}F "
                  f"{s['n_samples']:>7} {weight:>6.1f}x")


def show_edge_analysis(days: int = 30):
    """Show predicted edge vs realized outcome."""
    conn = db.get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"EDGE ANALYSIS (last {days} days)")
    print(f"{'='*60}")

    # Edge bucket analysis
    buckets = [
        ("10-15%", 0.10, 0.15),
        ("15-20%", 0.15, 0.20),
        ("20-25%", 0.20, 0.25),
        ("25-30%", 0.25, 0.30),
        ("30%+",   0.30, 1.00),
    ]

    print(f"{'Edge Range':<12} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Avg P&L':>9}")
    print("-" * 45)

    for label, low, high in buckets:
        row = conn.execute("""
            SELECT COUNT(*) as n,
                   SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                   AVG(pnl) as avg_pnl
            FROM trades
            WHERE resolved = 1 AND edge >= ? AND edge < ? AND market_date >= ?
        """, (low, high, cutoff)).fetchone()

        n = row["n"]
        if n == 0:
            continue

        wins = row["wins"] or 0
        wr = wins / n * 100 if n > 0 else 0
        avg_pnl = row["avg_pnl"] or 0

        print(f"{label:<12} {n:>7} {wins:>5} {wr:>5.0f}% ${avg_pnl:>+8.2f}")

    conn.close()


def show_forecast_accuracy_report(days: int = 30):
    """Show forecast vs actual accuracy from stored data."""
    conn = db.get_conn()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"FORECAST ACCURACY (last {days} days)")
    print(f"{'='*60}")

    # Per-model accuracy
    rows = conn.execute("""
        SELECT model_name,
               COUNT(*) as n,
               AVG(error_f) as mean_error,
               AVG(ABS(error_f)) as mae,
               AVG(error_f * error_f) as mse
        FROM model_accuracy
        WHERE market_date >= ?
        GROUP BY model_name
        ORDER BY mae ASC
    """, (cutoff,)).fetchall()

    if not rows:
        print("No forecast accuracy data yet. Run resolve.py after trades settle.")
        conn.close()
        return

    print(f"{'Model':<22} {'N':>5} {'Bias':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 55)

    for r in rows:
        rmse = (r["mse"] ** 0.5) if r["mse"] else 0
        print(f"{r['model_name']:<22} {r['n']:>5} {r['mean_error']:>+7.1f}F "
              f"{r['mae']:>7.1f}F {rmse:>7.1f}F")

    # Per-city ensemble accuracy
    print(f"\n{'City':<6} {'N':>5} {'Ens Bias':>9} {'Ens MAE':>9}")
    print("-" * 35)

    for city_key in sorted(CITIES.keys()):
        city_rows = conn.execute("""
            SELECT AVG(error_f) as bias, AVG(ABS(error_f)) as mae, COUNT(DISTINCT market_date) as n
            FROM model_accuracy
            WHERE city = ? AND market_date >= ? AND model_name = 'ensemble'
        """, (city_key, cutoff)).fetchone()

        if city_rows and city_rows["n"] and city_rows["n"] > 0:
            print(f"{city_key:<6} {city_rows['n']:>5} {city_rows['bias']:>+8.1f}F "
                  f"{city_rows['mae']:>8.1f}F")

    conn.close()


def show_full_report(days: int = 30):
    """Show complete analytics report."""
    print("=" * 60)
    print(f"KALSHI WEATHER BOT — ANALYTICS REPORT")
    print(f"Bankroll: ${db.get_bankroll():.2f}")
    peak = db.get_peak_bankroll()
    if peak > 0:
        bankroll = db.get_bankroll()
        dd = (peak - bankroll) / peak
        print(f"Peak: ${peak:.2f} | Drawdown: {dd:.1%}")
    print("=" * 60)

    show_daily_report()
    show_city_performance(days)
    show_edge_analysis(days)
    show_model_accuracy(days)
    show_forecast_accuracy_report(days)


if __name__ == "__main__":
    days = 30

    if "--models" in sys.argv:
        show_model_accuracy(days)
    elif "--city" in sys.argv:
        idx = sys.argv.index("--city")
        if idx + 1 < len(sys.argv):
            city = sys.argv[idx + 1].upper()
            show_daily_report()
    else:
        show_full_report(days)
