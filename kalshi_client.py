"""
Kalshi API client with RSA-PSS authentication and clock sync.

Handles:
  - Server clock synchronization (fixes ~103s skew)
  - RSA-PSS request signing
  - Market data fetching (events, markets, order books)
  - Order placement (when PAPER_MODE is off)
  - Portfolio and balance queries
"""

import base64
import time
from pathlib import Path
from email.utils import parsedate_to_datetime

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_BASE_URL


class KalshiClient:
    def __init__(self):
        self.base_url = KALSHI_BASE_URL
        self.key_id = KALSHI_API_KEY_ID
        self.private_key = self._load_private_key()
        self.clock_offset = 0.0
        self._sync_clock()

    def _load_private_key(self):
        key_path = Path(KALSHI_PRIVATE_KEY_PATH)
        if not key_path.exists():
            raise FileNotFoundError(
                f"Kalshi private key not found at: {key_path}\n"
                f"Update KALSHI_PRIVATE_KEY_PATH in your .env file."
            )
        key_data = key_path.read_bytes()
        return serialization.load_pem_private_key(key_data, password=None)

    def _sync_clock(self):
        """Compute offset between local clock and Kalshi server clock."""
        try:
            r = requests.get(f"{self.base_url}/exchange/status", timeout=10)
            r.raise_for_status()
            server_date = r.headers.get("Date")
            if server_date:
                server_ts = parsedate_to_datetime(server_date).timestamp()
                local_ts = time.time()
                self.clock_offset = local_ts - server_ts
                print(f"[KalshiClient] Clock offset: {self.clock_offset:.1f}s "
                      f"(local ahead of server)")
            else:
                print("[KalshiClient] WARNING: No Date header, using local clock")
        except Exception as e:
            print(f"[KalshiClient] WARNING: Clock sync failed ({e}), using local clock")

    def _get_timestamp_ms(self):
        """Get timestamp adjusted for server clock skew."""
        adjusted = time.time() - self.clock_offset
        return str(int(adjusted * 1000))

    def _sign_request(self, method: str, path: str, timestamp_ms: str) -> str:
        message = f"{timestamp_ms}{method}{path}"
        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        timestamp_ms = self._get_timestamp_ms()
        # Kalshi requires signing with the full path including /trade-api/v2 prefix
        full_path = f"/trade-api/v2{path}"
        sig = self._sign_request(method.upper(), full_path, timestamp_ms)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers("GET", path)
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers("POST", path)
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers("DELETE", path)
        resp = requests.delete(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # -- Account --------------------------------------------------------

    def get_balance(self) -> dict:
        """Get account balance. Returns dict with 'balance' in cents."""
        return self._get("/portfolio/balance")

    def get_exchange_status(self) -> dict:
        """Get exchange status (unauthenticated)."""
        r = requests.get(f"{self.base_url}/exchange/status", timeout=10)
        r.raise_for_status()
        return r.json()

    # -- Markets --------------------------------------------------------

    def get_events(self, cursor: str = None, limit: int = 100,
                   status: str = "open", series_ticker: str = None,
                   with_nested_markets: bool = True) -> dict:
        """Get events (groups of related markets)."""
        params = {
            "limit": limit,
            "status": status,
            "with_nested_markets": str(with_nested_markets).lower(),
        }
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._get("/events", params=params)

    def get_markets(self, cursor: str = None, limit: int = 100,
                    event_ticker: str = None, series_ticker: str = None,
                    tickers: str = None, status: str = "open") -> dict:
        """Get markets with optional filters."""
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if tickers:
            params["tickers"] = tickers
        return self._get("/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get a single market by ticker."""
        return self._get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Get order book for a market."""
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_trades(self, ticker: str = None, limit: int = 100,
                   cursor: str = None) -> dict:
        """Get recent trades, optionally filtered by ticker."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets/trades", params=params)

    # -- Weather-Specific Helpers ---------------------------------------

    def get_weather_events(self) -> list:
        """
        Fetch all open weather/climate events from Kalshi.
        Paginates through all results.
        """
        all_events = []
        cursor = None

        while True:
            data = self.get_events(cursor=cursor, limit=100, status="open")
            events = data.get("events", [])
            if not events:
                break

            for event in events:
                title = (event.get("title") or "").lower()
                category = (event.get("category") or "").lower()
                series = (event.get("series_ticker") or "").upper()

                is_weather = (
                    "temperature" in title
                    or "weather" in category
                    or "climate" in category
                    or series.startswith("KXHIGH")
                    or series.startswith("KXLOW")
                    or "highest temp" in title
                    or "lowest temp" in title
                )
                if is_weather:
                    all_events.append(event)

            cursor = data.get("cursor")
            if not cursor:
                break

        return all_events

    def get_weather_markets_for_city(self, ticker_prefix: str) -> list:
        """
        Get all open markets matching a city's ticker prefix.
        Returns list of market dicts with bracket info and prices.
        """
        all_markets = []
        cursor = None

        while True:
            data = self.get_markets(
                cursor=cursor,
                limit=100,
                series_ticker=ticker_prefix,
                status="open",
            )
            markets = data.get("markets", [])
            all_markets.extend(markets)

            cursor = data.get("cursor")
            if not cursor or not markets:
                break

        return all_markets

    # -- Orders ---------------------------------------------------------

    def place_order(self, ticker: str, side: str, type: str = "limit",
                    yes_price: int = None, no_price: int = None,
                    count: int = 1, action: str = "buy",
                    expiration_ts: int = None) -> dict:
        """
        Place an order on Kalshi.

        Args:
            ticker: market ticker
            side: 'yes' or 'no'
            type: 'limit' or 'market'
            yes_price: price in cents (1-99) for yes side
            no_price: price in cents (1-99) for no side
            count: number of contracts
            action: 'buy' or 'sell'
            expiration_ts: optional unix timestamp for order expiry
        """
        body = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "type": type,
            "count": count,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if expiration_ts:
            body["expiration_ts"] = expiration_ts

        return self._post("/portfolio/orders", body)

    def get_orders(self, ticker: str = None, status: str = None) -> dict:
        """Get your orders."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return self._get("/portfolio/orders", params=params)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        return self._delete(f"/portfolio/orders/{order_id}")

    def get_positions(self, ticker: str = None,
                      settlement_status: str = None) -> dict:
        """Get your positions."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if settlement_status:
            params["settlement_status"] = settlement_status
        return self._get("/portfolio/positions", params=params)


def test_connection():
    """Quick test to verify API connectivity and authentication."""
    print("=" * 60)
    print("KALSHI API CONNECTION TEST")
    print("=" * 60)

    try:
        client = KalshiClient()
        print("[OK] Private key loaded + clock synced")
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Error loading key: {e}")
        return False

    try:
        balance_data = client.get_balance()
        balance_cents = balance_data.get("balance", 0)
        balance_usd = balance_cents / 100.0
        print(f"[OK] Authenticated successfully")
        print(f"[OK] Account balance: ${balance_usd:.2f}")
    except requests.HTTPError as e:
        print(f"[FAIL] Auth failed (HTTP {e.response.status_code}): {e.response.text}")
        return False
    except Exception as e:
        print(f"[FAIL] Connection error: {e}")
        return False

    try:
        status = client.get_exchange_status()
        print(f"[OK] Exchange: {status}")
    except Exception as e:
        print(f"[WARN] Could not fetch exchange status: {e}")

    try:
        events = client.get_weather_events()
        print(f"[OK] Found {len(events)} weather events")
        if events:
            for event in events[:5]:
                title = event.get("title", "Unknown")
                markets = event.get("markets", [])
                print(f"     - {title} ({len(markets)} markets)")
    except Exception as e:
        print(f"[WARN] Could not fetch weather events: {e}")

    print("=" * 60)
    print("Connection test passed.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_connection()
