import re
import pandas as pd
from datetime import datetime

# === Robust Patterns ===

# Matches tickers like SPY, TSLA, NVDA, META, SPX, QQQ, IWM
TICKER_RE = r'(?P<ticker>\b[A-Z]{1,5}\b)'

# Matches call/put/entry/exit variations
ALERT_TYPE_RE = r'(?P<alert_type>call|calls|put|puts|entry|exit|alert|buy|sell)'

# Matches prices like 1.23, $1.23, @1.23, 1, 1.2c, 1.2p
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
    r'(https?://\S+|<@\d+>|[#@]\w+|[\U00010000-\U0010ffff])'
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
        clean_text = CLEANER.sub("", content).strip()

        match = PATTERN.search(clean_text)
        if not match:
            continue

        data = match.groupdict()

        # --- Extract alert type ---
        at = (data.get("alert_type") or "").lower()
