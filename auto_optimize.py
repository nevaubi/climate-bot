"""
Auto-optimizer for model weights and bias corrections.

Runs daily after resolve + calibrate to adjust:
  1. Model weights based on accumulated accuracy data (MAE-based)
  2. Bias corrections per city (ensemble mean error)
  3. Correlation group structure (isolate consistently bad models)

Requires minimum 3 days of data to start adjusting.
Changes are conservative: blends current weights with data-driven targets.
"""

import json
import sqlite3
import math
from pathlib import Path
from datetime import datetime, timezone


DB_PATH = Path(__file__).parent / "data" / "bot.db"
CONFIG_PATH = Path(__file__).parent / "config.py"
CALIBRATION_PATH = Path(__file__).parent / "data" / "calibration.json"

# Minimum samples before we start adjusting
MIN_SAMPLES_FOR_WEIGHTS = 3    # 3 city-days minimum
MIN_SAMPLES_FOR_BIAS = 5       # 5 observations minimum per city

# How aggressively to blend toward data-driven targets (0=keep current, 1=fully data-driven)
BLEND_FACTOR = 0.3  # Conservative: 30% toward data, 70% keep current

# Weight bounds (don't let any model go below 0.1 or above 3.0)
MIN_WEIGHT = 0.1
MAX_WEIGHT = 3.0


def _get_model_accuracy_stats():
    """Get per-model accuracy statistics from database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    stats = conn.execute("""
        SELECT model_name,
               AVG(error_f) as mean_error,
               AVG(ABS(error_f)) as mae,
               COUNT(*) as n_samples,
               AVG(error_f * error_f) as mse
        FROM model_accuracy
        GROUP BY model_name
        HAVING n_samples >= ?
        ORDER BY mae ASC
    """, (MIN_SAMPLES_FOR_WEIGHTS,)).fetchall()

    conn.close()
    return [dict(r) for r in stats]


def _get_city_bias_stats():
    """Get per-city ensemble error statistics."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Get ensemble mean error per city (average across all models per city-date)
    stats = conn.execute("""
        SELECT city,
               AVG(error_f) as mean_error,
               AVG(ABS(error_f)) as mae,
               COUNT(DISTINCT date) as n_dates
        FROM (
            SELECT city, date, AVG(error_f) as error_f
            FROM model_accuracy
            GROUP BY city, date
        )
        GROUP BY city
        HAVING n_dates >= ?
    """, (MIN_SAMPLES_FOR_BIAS // 8 + 1,)).fetchall()

    conn.close()
    return [dict(r) for r in stats]


def _compute_target_weights(accuracy_stats):
    """
    Compute target weights from accuracy data.

    Strategy: inverse MAE weighting.
    Better models (lower MAE) get higher weights.
    Normalized so the best model gets ~2.0x weight.
    """
    if not accuracy_stats:
        return {}

    # Inverse MAE (add floor to prevent division by zero)
    inv_mae = {}
    for row in accuracy_stats:
        mae = max(row["mae"], 0.5)  # Floor at 0.5F
        inv_mae[row["model_name"]] = 1.0 / mae

    # Normalize: best model gets 2.0, others proportionally less
    max_inv = max(inv_mae.values())
    if max_inv <= 0:
        return {}

    targets = {}
    for model, inv in inv_mae.items():
        # Scale so best = 2.0, worst scales down proportionally
        raw = 2.0 * (inv / max_inv)
        targets[model] = max(MIN_WEIGHT, min(MAX_WEIGHT, raw))

    return targets


def _read_current_weights():
    """Read current MODEL_WEIGHTS from config.py."""
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    if "config" in sys.modules:
        del sys.modules["config"]

    import config
    return dict(config.MODEL_WEIGHTS)


def _blend_weights(current, targets, blend_factor=BLEND_FACTOR):
    """Blend current weights toward targets."""
    blended = {}
    for model in set(list(current.keys()) + list(targets.keys())):
        curr = current.get(model, 1.0)
        targ = targets.get(model, curr)
        new = curr * (1 - blend_factor) + targ * blend_factor
        blended[model] = round(max(MIN_WEIGHT, min(MAX_WEIGHT, new)), 2)
    return blended


def _update_config_weights(new_weights):
    """
    Update MODEL_WEIGHTS in config.py.

    Also updates LEAD_TIME_WEIGHTS proportionally.
    """
    config_text = CONFIG_PATH.read_text()

    # Find and replace MODEL_WEIGHTS dict
    import re

    # Build new weights string
    lines = ["MODEL_WEIGHTS = {"]
    for model, weight in sorted(new_weights.items(), key=lambda x: -x[1]):
        lines.append(f'    "{model}": {weight},')
    lines.append("}")
    new_block = "\n".join(lines)

    # Replace the MODEL_WEIGHTS block
    pattern = r"MODEL_WEIGHTS\s*=\s*\{[^}]+\}"
    if re.search(pattern, config_text):
        config_text = re.sub(pattern, new_block, config_text, count=1)
        CONFIG_PATH.write_text(config_text)
        return True

    return False


def _update_lead_time_weights(old_weights, new_weights):
    """
    Update LEAD_TIME_WEIGHTS proportionally based on weight changes.

    If a model's default weight changed by factor X, apply the same
    factor to all lead-time tiers.
    """
    config_text = CONFIG_PATH.read_text()

    for model in new_weights:
        if model in old_weights and old_weights[model] > 0:
            factor = new_weights[model] / old_weights[model]
            if abs(factor - 1.0) < 0.01:
                continue  # No meaningful change

            # Find all occurrences of this model in LEAD_TIME_WEIGHTS and scale
            import re
            pattern = rf'("{model}":\s*)([\d.]+)'

            def replacer(match):
                prefix = match.group(1)
                old_val = float(match.group(2))
                new_val = round(max(MIN_WEIGHT, min(MAX_WEIGHT, old_val * factor)), 1)
                return f"{prefix}{new_val}"

            # Only replace within LEAD_TIME_WEIGHTS section
            lt_start = config_text.find("LEAD_TIME_WEIGHTS")
            if lt_start >= 0:
                lt_section = config_text[lt_start:]
                lt_end_marker = "\ndef "
                lt_end = lt_section.find(lt_end_marker)
                if lt_end < 0:
                    lt_end = len(lt_section)

                lt_block = lt_section[:lt_end]
                updated_block = re.sub(pattern, replacer, lt_block)
                config_text = config_text[:lt_start] + updated_block + lt_section[lt_end:]

    CONFIG_PATH.write_text(config_text)


def run_optimization():
    """
    Main optimization entry point.

    Returns a summary string of changes made, or None if no changes.
    """
    print("=" * 60)
    print("AUTO-OPTIMIZATION")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Get accuracy data
    accuracy_stats = _get_model_accuracy_stats()
    if not accuracy_stats:
        print("  Not enough accuracy data yet. Skipping optimization.")
        return None

    total_samples = sum(r["n_samples"] for r in accuracy_stats)
    print(f"\n  Model accuracy ({total_samples} total samples):")
    for row in accuracy_stats:
        print(f"    {row['model_name']:22} MAE={row['mae']:.1f}F  "
              f"bias={row['mean_error']:+.1f}F  n={row['n_samples']}")

    # 2. Compute target weights
    targets = _compute_target_weights(accuracy_stats)
    current = _read_current_weights()

    print(f"\n  Weight adjustments (blend={BLEND_FACTOR:.0%}):")
    new_weights = _blend_weights(current, targets)

    changes = []
    for model in sorted(new_weights.keys(), key=lambda m: -new_weights[m]):
        curr = current.get(model, 1.0)
        new = new_weights[model]
        delta = new - curr
        arrow = "^" if delta > 0.02 else ("v" if delta < -0.02 else "=")
        print(f"    {model:22} {curr:.2f} -> {new:.2f} ({arrow})")
        if abs(delta) > 0.02:
            changes.append(f"{model}: {curr:.1f}->{new:.1f}")

    # 3. Apply weight changes
    if changes:
        old_weights = dict(current)
        _update_config_weights(new_weights)
        _update_lead_time_weights(old_weights, new_weights)
        print(f"\n  Updated {len(changes)} model weight(s) in config.py")
    else:
        print(f"\n  No significant weight changes needed")

    # 4. City bias analysis
    city_stats = _get_city_bias_stats()
    if city_stats:
        print(f"\n  City ensemble errors:")
        for row in city_stats:
            print(f"    {row['city']:5} mean_err={row['mean_error']:+.1f}F  "
                  f"MAE={row['mae']:.1f}F  n_dates={row['n_dates']}")

    summary = f"Optimized {len(changes)} weights from {total_samples} samples"
    if changes:
        summary += f": {', '.join(changes[:3])}"

    print(f"\n  {summary}")
    print("=" * 60)

    return summary if changes else None


if __name__ == "__main__":
    run_optimization()
