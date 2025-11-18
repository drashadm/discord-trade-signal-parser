import re
import pandas as pd
from datetime import datetime

# === Robust Patterns ===

# Matches tickers like SPY, TSLA, NVDA, META, SPX, QQQ, IWM, NDX, CVX, etc.
TICKER_RE = r'(?P<ticker>\b[A-Z]{1,5}\b)'

# Matches call/put/entry/exit variations
ALERT_TYPE_RE = r'(?P<alert_type>call|calls|put|puts|entry|exit|alert|buy|sell)'

# Matches prices - more flexible to capture both formats
PRICE_RE = r'(?P<price>\d+(?:\.\d+)?)'

# Precompiled combined pattern
PATTERN = re.compile(
    rf'''
    {ALERT_TYPE_RE}        # alert type
    .*?
    {TICKER_RE}?           # optional ticker
    .*?
    (?:\$|@|price|entry|fill)?\s*{PRICE_RE}?   # optional price
    ''',
    flags=re.IGNORECASE | re.VERBOSE
)

# Remove URLs, mentions, emoji, unicode noise
CLEANER = re.compile(
    r'(https?://\S+|<@\d+>|[#@]\w+)'
)


def parse_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse raw Discord messages into structured trade signal data.

    Output columns:
        alert_type, ticker, price, timestamp, date, time, channel
    """
    if df_raw.empty:
        return pd.DataFrame()

    parsed_rows = []

    for _, row in df_raw.iterrows():
        content = str(row.get("content", ""))
        # Don't clean emojis - they're important for signal patterns
        clean_text = CLEANER.sub("", content).strip()

        match = PATTERN.search(clean_text)
        if not match:
            continue

        data = match.groupdict()

        # --- Extract alert type ---
        at = (data.get("alert_type") or "").lower()
        if "call" in at:
            alert_type = "CALL"
        elif "put" in at:
            alert_type = "PUT"
        elif "exit" in at:
            alert_type = "EXIT"
        elif "entry" in at or "buy" in at:
            alert_type = "ENTRY"
        else:
            alert_type = "ALERT"

        # --- Extract ticker from content more robustly ---
        # First try regex match
        ticker = (data.get("ticker") or "").upper()
        
        # If no ticker from pattern, try to find any 2-5 letter uppercase word
        if not ticker or len(ticker) < 1:
            ticker_match = re.search(r'\b([A-Z]{2,5})\b', content)
            if ticker_match:
                ticker = ticker_match.group(1)
        
        if not ticker or len(ticker) < 1:
            continue

        # --- Extract price ---
        # Try multiple patterns for price
        price = None
        
        # Pattern 1: "Price: 24264.4009"
        price_match = re.search(r'[Pp]rice[:\s]+(\d+(?:\.\d+)?)', clean_text)
        if price_match:
            try:
                price = float(price_match.group(1))
            except (ValueError, TypeError):
                pass
        
        # Pattern 2: "at 24264.4009"
        if not price:
            price_match = re.search(r'\sat\s+(\d+(?:\.\d+)?)', clean_text)
            if price_match:
                try:
                    price = float(price_match.group(1))
                except (ValueError, TypeError):
                    pass
        
        # Pattern 3: fallback from regex
        if not price:
            price_str = data.get("price") or ""
            try:
                price = float(price_str) if price_str else None
            except (ValueError, TypeError):
                pass

        # Skip if no valid price
        if price is None or price <= 0:
            continue

        # --- Timestamps ---
        timestamp = row.get("created_at", datetime.now().isoformat())
        try:
            ts = pd.to_datetime(timestamp)
        except Exception:
            ts = pd.Timestamp.now()

        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M:%S")
        channel = row.get("channel", "unknown")

        parsed_rows.append({
            "alert_type": alert_type,
            "ticker": ticker,
            "price": price,
            "timestamp": ts.isoformat(),
            "date": date_str,
            "time": time_str,
            "channel": channel,
        })

    if not parsed_rows:
        return pd.DataFrame()

    df_parsed = pd.DataFrame(parsed_rows)
    return df_parsed
