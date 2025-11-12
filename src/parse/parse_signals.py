import re
import pandas as pd
from datetime import datetime

# === Enhanced Regex Patterns ===
PATTERN = re.compile(
    r'(?P<alert_type>call|put|exit|alert)\b'
    r'(?:.*?\bon\s*(?P<ticker>[A-Z]{1,6}))?'
    r'(?:.*?(?:price(?:\s*called|\s*entry|\s*at)?):?\s*\$?(?P<price>\d+(?:\.\d+)?))?',
    flags=re.IGNORECASE | re.UNICODE
)

# Emojis or noisy characters to strip
CLEANER = re.compile(r'(https?://\S+|<@\d+>|[#@]\w+|[\U00010000-\U0010ffff])')

def parse_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse raw Discord messages into structured trade signal data.

    Columns:
        ['alert_type', 'ticker', 'price', 'timestamp', 'date', 'time', 'channel']
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
        alert_type = (data.get("alert_type") or "").upper()
        ticker = (data.get("ticker") or "").upper()
        price = data.get("price")

        try:
            price = float(price) if price else None
        except ValueError:
            price = None

        ts_raw = row.get("created_at", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            ts = None

        parsed_rows.append({
            "alert_type": alert_type,
            "ticker": ticker,
            "price": price,
            "timestamp": ts.isoformat() if ts else ts_raw,
            "date": ts.date().isoformat() if ts else None,
            "time": ts.time().isoformat(timespec="seconds") if ts else None,
            "channel": row.get("channel", "unknown"),
        })

    df = pd.DataFrame(parsed_rows)
    if not df.empty:
        df.drop_duplicates(subset=["timestamp", "ticker", "alert_type"], inplace=True)
        df.sort_values(by="timestamp", inplace=True, ignore_index=True)

    return df
