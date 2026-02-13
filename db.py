"""
SQLite database setup and query helpers — v3.

Tables:
  - trades: every trade (paper or live) with entry, side, outcome, P&L
  - model_forecasts: weather model predictions per city/day
  - market_snapshots: periodic snapshots of Kalshi market prices
  - observations: METAR temperature observations from NWS stations
  - bankroll: bankroll tracking over time
  - model_accuracy: per-model forecast vs actual comparison for calibration
"""

import sqlite3
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from config import DB_PATH


def _utcnow_iso() -> str:
    """Get current UTC time as ISO string (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            city TEXT NOT NULL,
            market_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            bracket TEXT NOT NULL,
            side TEXT NOT NULL DEFAULT 'yes',
            contracts INTEGER NOT NULL,
            price REAL NOT NULL,
            cost REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            model_prob REAL,
            market_prob REAL,
            edge REAL,
            paper_mode INTEGER NOT NULL DEFAULT 1,
            outcome TEXT,
            payout REAL,
            pnl REAL,
            resolved INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS model_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            city TEXT NOT NULL,
            forecast_date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            forecast_high_f REAL NOT NULL,
            bracket_probs TEXT,
            raw_data TEXT
        );

        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            city TEXT NOT NULL,
            market_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            bracket TEXT NOT NULL,
            yes_price REAL,
            no_price REAL,
            yes_volume REAL,
            no_volume REAL
        );

        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            city TEXT NOT NULL,
            station_id TEXT NOT NULL,
            temp_f REAL,
            temp_c REAL,
            observation_time TEXT,
            raw_metar TEXT
        );

        CREATE TABLE IF NOT EXISTS bankroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            balance REAL NOT NULL,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS model_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            city TEXT NOT NULL,
            market_date TEXT NOT NULL,
            model_name TEXT NOT NULL,
            forecast_high_f REAL NOT NULL,
            actual_high_f REAL NOT NULL,
            error_f REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_city_date
            ON trades(city, market_date);
        CREATE INDEX IF NOT EXISTS idx_trades_ticker
            ON trades(ticker, resolved);
        CREATE INDEX IF NOT EXISTS idx_snapshots_city_date
            ON market_snapshots(city, market_date);
        CREATE INDEX IF NOT EXISTS idx_forecasts_city_date
            ON model_forecasts(city, forecast_date);
        CREATE INDEX IF NOT EXISTS idx_observations_city
            ON observations(city, timestamp);
        CREATE INDEX IF NOT EXISTS idx_model_accuracy_city
            ON model_accuracy(city, market_date);
    """)

    # Schema migration: add bracket bounds columns if missing
    try:
        conn.execute("SELECT bracket_low FROM trades LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE trades ADD COLUMN bracket_low REAL")
        conn.execute("ALTER TABLE trades ADD COLUMN bracket_high REAL")

    # Fix legacy trades that used side='buy' instead of 'yes'/'no'
    conn.execute("""
        UPDATE trades SET side = 'yes'
        WHERE side = 'buy' AND resolved = 1 AND pnl = 0
    """)

    conn.commit()
    conn.close()


def log_trade(city: str, market_date: str, ticker: str, bracket: str,
              side: str, contracts: int, price: float, cost: float, fee: float,
              model_prob: float, market_prob: float, edge: float,
              paper_mode: bool, notes: str = "",
              bracket_low: float = None, bracket_high: float = None) -> int:
    """Log a trade. side should be 'yes' or 'no'."""
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO trades (timestamp, city, market_date, ticker, bracket,
            side, contracts, price, cost, fee, model_prob, market_prob, edge,
            paper_mode, notes, bracket_low, bracket_high)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        _utcnow_iso(), city, market_date, ticker, bracket,
        side, contracts, price, cost, fee, model_prob, market_prob, edge,
        1 if paper_mode else 0, notes, bracket_low, bracket_high
    ))
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def log_market_snapshot(city: str, market_date: str, ticker: str,
                        bracket: str, yes_price: float, no_price: float,
                        yes_volume: float = 0, no_volume: float = 0):
    conn = get_conn()
    conn.execute("""
        INSERT INTO market_snapshots (timestamp, city, market_date, ticker,
            bracket, yes_price, no_price, yes_volume, no_volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        _utcnow_iso(), city, market_date, ticker,
        bracket, yes_price, no_price, yes_volume, no_volume
    ))
    conn.commit()
    conn.close()


def log_forecast(city: str, forecast_date: str, model_name: str,
                 forecast_high_f: float, bracket_probs: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO model_forecasts (timestamp, city, forecast_date,
            model_name, forecast_high_f, bracket_probs)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        _utcnow_iso(), city, forecast_date,
        model_name, forecast_high_f, json.dumps(bracket_probs)
    ))
    conn.commit()
    conn.close()


def log_observation(city: str, station_id: str, temp_f: float,
                    temp_c: float, observation_time: str, raw_metar: str = ""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO observations (timestamp, city, station_id, temp_f,
            temp_c, observation_time, raw_metar)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        _utcnow_iso(), city, station_id, temp_f,
        temp_c, observation_time, raw_metar
    ))
    conn.commit()
    conn.close()


def log_model_accuracy(city: str, market_date: str, model_name: str,
                       forecast_high_f: float, actual_high_f: float):
    """Log a forecast-vs-actual comparison for calibration tracking."""
    error_f = forecast_high_f - actual_high_f
    conn = get_conn()
    conn.execute("""
        INSERT INTO model_accuracy (timestamp, city, market_date,
            model_name, forecast_high_f, actual_high_f, error_f)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        _utcnow_iso(), city, market_date,
        model_name, forecast_high_f, actual_high_f, error_f
    ))
    conn.commit()
    conn.close()


def get_todays_trades(market_date: str) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE market_date = ? ORDER BY timestamp",
        (market_date,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_unresolved_trades() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE resolved = 0 ORDER BY timestamp"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_trade(trade_id: int, outcome: str, payout: float, pnl: float):
    conn = get_conn()
    conn.execute("""
        UPDATE trades SET outcome = ?, payout = ?, pnl = ?, resolved = 1
        WHERE id = ?
    """, (outcome, payout, pnl, trade_id))
    conn.commit()
    conn.close()


def get_bankroll() -> float:
    conn = get_conn()
    row = conn.execute(
        "SELECT balance FROM bankroll ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return row["balance"]
    return 0.0


def update_bankroll(balance: float, note: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO bankroll (timestamp, balance, note) VALUES (?, ?, ?)",
        (_utcnow_iso(), balance, note)
    )
    conn.commit()
    conn.close()


def get_peak_bankroll() -> float:
    """Get the highest bankroll balance ever recorded."""
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(balance) as peak FROM bankroll"
    ).fetchone()
    conn.close()
    if row and row["peak"] is not None:
        return row["peak"]
    return 0.0


def get_daily_pnl(date_str: str) -> float:
    """Get total realized P&L for trades resolved on a given date."""
    conn = get_conn()
    row = conn.execute("""
        SELECT COALESCE(SUM(pnl), 0) as daily_pnl FROM trades
        WHERE resolved = 1 AND market_date = ?
    """, (date_str,)).fetchone()
    conn.close()
    return row["daily_pnl"]


def get_group_exposure(city_keys: list) -> float:
    """Get total dollar cost of unresolved trades for a group of cities."""
    if not city_keys:
        return 0.0
    conn = get_conn()
    placeholders = ",".join("?" for _ in city_keys)
    row = conn.execute(f"""
        SELECT COALESCE(SUM(cost), 0) as total FROM trades
        WHERE resolved = 0 AND city IN ({placeholders})
    """, city_keys).fetchone()
    conn.close()
    return row["total"]


def get_recent_observations(city_key: str, hours: int = 6) -> list:
    """Get recent METAR observations for a city within the last N hours."""
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = conn.execute("""
        SELECT * FROM observations
        WHERE city = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    """, (city_key, cutoff)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_forecasts_for_date(city_key: str, market_date: str) -> list:
    """Get stored model forecasts for a city and date."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT DISTINCT model_name, forecast_high_f FROM model_forecasts
        WHERE city = ? AND forecast_date = ?
        ORDER BY timestamp DESC
    """, (city_key, market_date)).fetchall()
    conn.close()
    # Return latest forecast per model
    seen = set()
    result = []
    for r in rows:
        if r["model_name"] not in seen:
            seen.add(r["model_name"])
            result.append(dict(r))
    return result


def get_model_accuracy_stats(days: int = 30) -> dict:
    """
    Get per-model and per-city accuracy statistics from recent data.

    Returns dict of {city: {model: {bias, std, n_samples}}}
    """
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT city, model_name, AVG(error_f) as bias,
               COUNT(*) as n_samples
        FROM model_accuracy
        WHERE market_date >= ?
        GROUP BY city, model_name
    """, (cutoff,)).fetchall()
    conn.close()

    stats = {}
    for r in rows:
        city = r["city"]
        if city not in stats:
            stats[city] = {}
        stats[city][r["model_name"]] = {
            "bias": r["bias"],
            "n_samples": r["n_samples"],
        }
    return stats


# Initialize on import
init_db()
