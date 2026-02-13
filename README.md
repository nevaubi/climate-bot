# Kalshi Weather Trading Bot

Automated weather prediction market trading on Kalshi. Uses ensemble weather model forecasts (GFS, ECMWF, JMA) and real-time METAR observations to identify mispriced temperature brackets.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=C:\Users\YourName\.kalshi\private_key.txt
PAPER_MODE=true
```

### 3. Test connection

```bash
python kalshi_client.py
```

This verifies your API key, shows your balance, and lists available weather markets.

### 4. Test weather data

```bash
python weather.py
```

Pulls model forecasts and METAR observations for all cities.

## Usage

### Paper Trading (default)

```bash
# Single scan cycle (run once, see what it finds)
python scanner.py --once

# Continuous scanning (polls every 45 seconds)
python scanner.py
```

### Resolve trades (run next morning)

```bash
python resolve.py
```

Checks settled markets, calculates P&L, updates bankroll.

### Go Live

Once paper trading validates your edge (1-2 weeks recommended):

1. Edit `.env`: set `PAPER_MODE=false`
2. Verify bankroll: the bot starts with $50 by default (see `config.py`)
3. Run scanner as normal

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAPER_MODE` | `true` | Paper trade or live |
| `EDGE_THRESHOLD` | `0.08` | Minimum model-vs-market edge to trade |
| `KELLY_FRACTION` | `0.5` | Half-Kelly for position sizing |
| `MAX_POSITION_PER_CITY` | `0.25` | Max 25% of bankroll per city |
| `MAX_POSITION_PER_BRACKET` | `0.15` | Max 15% of bankroll per bracket |
| `SCAN_INTERVAL_SECONDS` | `45` | Polling frequency |

## Architecture

```
scanner.py (main loop)
  ├── kalshi_client.py    → fetch markets, place orders
  ├── weather.py          → model forecasts + METAR observations
  ├── config.py           → cities, thresholds, API keys
  ├── db.py               → SQLite trades, forecasts, observations
  └── resolve.py          → settle trades, compute P&L
```

## Cities

NYC, Chicago, Miami, Austin, Los Angeles, Philadelphia, Washington DC.
Each maps to a Kalshi ticker prefix (e.g., `KXHIGHNY`) and NWS METAR station.
