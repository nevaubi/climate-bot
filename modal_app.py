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

# Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "cryptography>=42.0.0",
    )
)

# Persistent volume for database and calibration data
volume = modal.Volume.from_name("climate-bot-data", create_if_missing=True)

# Secret for API keys
secret = modal.Secret.from_name("climate-bot-secrets")

# Mount the bot code
code_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/app",
    condition=lambda path: (
        path.endswith(".py")
        and not path.startswith("test_")
        and not path.startswith("debug_")
        and not path.startswith("diagnose_")
        and not path.startswith("modal_")
        and ".claude" not in path
    ),
)


def _setup():
    """Common setup: ensure data directory exists, copy DB from volume."""
    import os
    import shutil
    from pathlib import Path

    os.chdir("/app")

    # Ensure data directory exists
    Path("/app/data").mkdir(exist_ok=True)
    Path("/app/logs").mkdir(exist_ok=True)

    # Copy DB from persistent volume if it exists
    vol_db = "/data/bot.db"
    local_db = "/app/data/bot.db"
    if os.path.exists(vol_db) and not os.path.exists(local_db):
        shutil.copy2(vol_db, local_db)

    # Copy calibration from volume
    vol_cal = "/data/calibration.json"
    local_cal = "/app/data/calibration.json"
    if os.path.exists(vol_cal) and not os.path.exists(local_cal):
        shutil.copy2(vol_cal, local_cal)


def _persist():
    """Save DB and calibration back to persistent volume."""
    import shutil
    from pathlib import Path

    Path("/data").mkdir(exist_ok=True)

    local_db = "/app/data/bot.db"
    if Path(local_db).exists():
        shutil.copy2(local_db, "/data/bot.db")

    local_cal = "/app/data/calibration.json"
    if Path(local_cal).exists():
        shutil.copy2(local_cal, "/data/calibration.json")


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
    mounts=[code_mount],
    volumes={"/data": volume},
    secrets=[secret],
    schedule=modal.Cron("* 6-23 * * *"),  # Every minute, 6am-11pm ET (UTC adjusted by Modal)
    timeout=120,
)
def scan_cycle():
    """Run one scan cycle. Scheduled every minute during market hours."""
    if not _should_scan():
        return

    _setup()

    import sys
    sys.path.insert(0, "/app")

    from kalshi_client import KalshiClient
    from scanner import run_scan_cycle
    import db

    client = KalshiClient()
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
    mounts=[code_mount],
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
    mounts=[code_mount],
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
    mounts=[code_mount],
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
    mounts=[code_mount],
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
    bankroll = db.get_bankroll()
    if bankroll <= 0:
        from config import STARTING_BANKROLL
        bankroll = STARTING_BANKROLL
        db.update_bankroll(bankroll, "Initial bankroll (Modal)")

    run_scan_cycle(client, bankroll)

    _persist()
    volume.commit()
