import re
import pandas as pd
from datetime import datetime

# === Precompile Regex Pattern (emoji + lowercase support) ===
ALERT_PATTERN = re.compile(
    r'(?:[\u2600-\u26FF\u2700-\u27BF\uFE0F\u1F300-\u1F6FF\u1F900-\u1F9FF]*\s*)?'  # emoji prefix
    r'(?P<alert_type>call alert|put alert|exit alert)'                             # alert type (any case)
    r'(?: confirmed)?\s*on\s*(?P<ticker>[A-Z]+)'                                  # ticker
    r'(?:\s*(?P<comment>[A-Za-z ]+?))?'                                           # optional comment
    r'\s*at\s*(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'                # timestamp
    r'\s*\|\s*Price:\s*\$?(?P<price>\d+\.?\d*)',                                  # price
    re.IGNORECASE
)

def parse_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse trading alert messages (emoji + lowercase tolerant) into structured DataFrame.

    Args:
        df_raw (pd.DataFrame): Must contain 'content' and 'channel' columns.

    Returns:
        pd.DataFrame: Parsed alerts with columns:
            ['alert_type', 'ticker', 'comment', 'timestamp', 'date', 'time', 'price', 'channel']
    """
    parsed_rows = []

    for _, row in df_raw.iterrows():
        content = str(row.get("content", "")).strip()
        if not content:
            continue

        match = ALERT_PATTERN.search(content)
        if not match:
            continue

        data = match.groupdict()
        try:
            ts = datetime.strptime(data["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            continue  # Skip malformed timestamps

        parsed_rows.append({
            "alert_type": data["alert_type"].strip().lower().replace(" alert", ""),
            "ticker": data["ticker"].upper(),
            "comment": (data.get("comment") or "").strip(),
            "timestamp": data["timestamp"],
            "date": ts.date().isoformat(),
            "time": ts.time().isoformat(timespec="seconds"),
            "price": float(data["price"]),
            "channel": row.get("channel", "unknown"),
        })

    # Return empty DataFrame if nothing matched
    if not parsed_rows:
        return pd.DataFrame(columns=[
            "alert_type", "ticker", "comment", "timestamp",
            "date", "time", "price", "channel"
        ])

    df = pd.DataFrame(parsed_rows)
    df.drop_duplicates(subset=["timestamp", "ticker", "alert_type"], inplace=True)
    df.sort_values(by="timestamp", inplace=True, ignore_index=True)
    return df
