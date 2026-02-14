"""
Modal deployment for Kalshi Weather Trading Bot.

Schedules:
  - Scanner: every 60s during market hours (6am-midnight ET)
  - Resolver: daily at 11:00 AM ET (after Kalshi settles at ~10am)
  - Calibrator: daily at 11:30 AM ET (after resolver updates accuracy)
  - Optimizer: daily at 11:45 AM ET (auto-adjust weights from accuracy data)

Usage:
  modal deploy modal_app.py          # Deploy to Modal cloud
  modal run modal_app.py::scan_once  # Test single scan
  modal run modal_app.py::resolve    # Test resolve
"""

import modal

app = modal.App("climate-bot")

# Modal image with all dependencies + local bot code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "cryptography>=42.0.0",
    )
    .add_local_file("config.py", "/app/config.py")
    .add_local_file("db.py", "/app/db.py")
    .add_local_file("kalshi_client.py", "/app/kalshi_client.py")
    .add_local_file("scanner.py", "/app/scanner.py")
    .add_local_file("weather.py", "/app/weather.py")
    .add_local_file("resolve.py", "/app/resolve.py")
    .add_local_file("calibrate.py", "/app/calibrate.py")
    .add_local_file("analytics.py", "/app/analytics.py")
    .add_local_file("backtest.py", "/app/backtest.py")
    .add_local_file("auto_optimize.py", "/app/auto_optimize.py")
    .add_local_file("data/calibration.json", "/app/data/calibration.json")
)

# Persistent volume for database and calibration data
volume = modal.Volume.from_name("climate-bot-data", create_if_missing=True)

# Secret for API keys
secret = modal.Secret.from_name("climate-bot-secrets")


def _setup():
    """Common setup: ensure data directory exists, write key file, copy DB from volume."""
    import os
    import base64
    import shutil
    from pathlib import Path

    os.chdir("/app")

    # Ensure data directory exists
    Path("/app/data").mkdir(exist_ok=True)
    Path("/app/logs").mkdir(exist_ok=True)

    # Write private key from base64 env var (Modal stores it as a secret)
    key_b64 = os.environ.get("KALSHI_PRIVATE_KEY_B64", "")
    key_path = "/app/kalshi-key.pem"
    if key_b64 and not os.path.exists(key_path):
        key_data = base64.b64decode(key_b64)
        with open(key_path, "wb") as f:
            f.write(key_data)
        os.chmod(key_path, 0o600)

    # Override paths for Modal environment (Git Bash may mangle /app/ paths in secrets)
    os.environ["KALSHI_PRIVATE_KEY_PATH"] = key_path
    os.environ.setdefault("PAPER_MODE", "false")

    # Copy DB from persistent volume if it exists
    vol_db = "/data/bot.db"
    local_db = "/app/data/bot.db"
    if os.path.exists(vol_db) and not os.path.exists(local_db):
        shutil.copy2(vol_db, local_db)

    # Copy calibration and optimized weights from volume if they exist
    for filename in ["calibration.json", "optimized_weights.json"]:
        vol_path = f"/data/{filename}"
        local_path = f"/app/data/{filename}"
        if os.path.exists(vol_path):
            shutil.copy2(vol_path, local_path)


def _persist():
    """Save DB, calibration, and optimized weights back to persistent volume."""
    import shutil
    from pathlib import Path

    Path("/data").mkdir(exist_ok=True)

    files_to_persist = [
        ("/app/data/bot.db", "/data/bot.db"),
        ("/app/data/calibration.json", "/data/calibration.json"),
        ("/app/data/optimized_weights.json", "/data/optimized_weights.json"),
    ]

    for src, dst in files_to_persist:
        if Path(src).exists():
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[PERSIST] ERROR copying {src} -> {dst}: {e}")


def _should_scan():
    """Check if we're within market hours (6am - midnight ET)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    return 6 <= now_et.hour <= 23


def _send_webhook(message: str):
    """Send alert to webhook (Slack or Discord)."""
    import os
    import requests

    url = os.environ.get("WEBHOOK_URL", "")
    if not url:
        return

    try:
        if "discord" in url:
            requests.post(url, json={"content": message}, timeout=10)
        else:
            # Slack format
            requests.post(url, json={"text": message}, timeout=10)
    except Exception as e:
        print(f"[WEBHOOK] Failed to send: {e}")


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[secret],
    schedule=modal.Cron("* * * * *"),
    timeout=120,
)
def scan_cycle():
    """Run one scan cycle. Scheduled every minute; _should_scan() gates on ET hours."""
    if not _should_scan():
        return

    _setup()

    import sys
    sys.path.insert(0, "/app")

    from kalshi_client import KalshiClient
    from scanner import run_scan_cycle
    import db

    client = KalshiClient()

    # Sync bankroll with actual Kalshi balance (detects deposits/withdrawals)
    try:
        bal = client.get_balance()
        balance_usd = bal.get("balance", 0) / 100.0
        portfolio_usd = bal.get("portfolio_value", 0) / 100.0
        bankroll = balance_usd + portfolio_usd
        if bankroll > 0:
            db.update_bankroll(bankroll, f"Synced from Kalshi (cash=${balance_usd:.2f} + positions=${portfolio_usd:.2f})")
    except Exception as e:
        print(f"[WARN] Could not sync balance: {e}")
        bankroll = db.get_bankroll()
        if bankroll <= 0:
            from config import STARTING_BANKROLL
            bankroll = STARTING_BANKROLL
            db.update_bankroll(bankroll, "Initial bankroll (Modal)")

    run_scan_cycle(client, bankroll)

    _persist()
    volume.commit()


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[secret],
    schedule=modal.Cron("0 16 * * *"),  # 11:00 AM ET = 16:00 UTC
    timeout=300,
)
def resolve():
    """Resolve yesterday's trades. Runs daily at 11am ET."""
    _setup()

    import sys
    sys.path.insert(0, "/app")

    from resolve import resolve_trades, show_stats

    resolve_trades()
    show_stats()

    _persist()
    volume.commit()

    # Send daily P&L webhook
    import db
    bankroll = db.get_bankroll()
    _send_webhook(f"[Climate Bot] Daily resolve complete. Bankroll: ${bankroll:.2f}")


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[secret],
    schedule=modal.Cron("30 16 * * *"),  # 11:30 AM ET = 16:30 UTC
    timeout=300,
)
def calibrate():
    """Recalibrate bias corrections. Runs daily at 11:30am ET."""
    _setup()

    import sys
    sys.path.insert(0, "/app")

    from calibrate import run_calibration

    run_calibration()

    _persist()
    volume.commit()


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[secret],
    schedule=modal.Cron("45 16 * * *"),  # 11:45 AM ET = 16:45 UTC
    timeout=300,
)
def optimize():
    """Auto-optimize model weights from accumulated accuracy data. Daily at 11:45am ET."""
    _setup()

    import sys
    sys.path.insert(0, "/app")

    from auto_optimize import run_optimization

    result = run_optimization()

    _persist()
    volume.commit()

    if result:
        _send_webhook(f"[Climate Bot] Auto-optimization: {result}")


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[secret],
    timeout=120,
)
def scan_once():
    """Manual trigger: run a single scan cycle."""
    _setup()

    import sys
    sys.path.insert(0, "/app")

    from kalshi_client import KalshiClient
    from scanner import run_scan_cycle
    import db

    client = KalshiClient()

    # Sync bankroll with actual Kalshi balance
    try:
        bal = client.get_balance()
        balance_usd = bal.get("balance", 0) / 100.0
        portfolio_usd = bal.get("portfolio_value", 0) / 100.0
        bankroll = balance_usd + portfolio_usd
        if bankroll > 0:
            db.update_bankroll(bankroll, f"Synced from Kalshi (cash=${balance_usd:.2f} + positions=${portfolio_usd:.2f})")
    except Exception as e:
        print(f"[WARN] Could not sync balance: {e}")
        bankroll = db.get_bankroll()
        if bankroll <= 0:
            from config import STARTING_BANKROLL
            bankroll = STARTING_BANKROLL
            db.update_bankroll(bankroll, "Initial bankroll (Modal)")

    run_scan_cycle(client, bankroll)

    _persist()
    volume.commit()
