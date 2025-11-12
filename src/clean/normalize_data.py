import pandas as pd
import re

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and enrich alert DataFrame fields for consistency + analytics.

    Operations:
    - Lowercase alert types and remove trailing ' alert'
    - Uppercase tickers
    - Strip and fill comment fields
    - Derive helper columns:
        • signal_strength: maps emojis/keywords (🔥=A+, ♻️=pullback, 💀=exit)
        • is_call / is_put boolean flags
        • Cleans ticker of stray symbols

    Args:
        df (pd.DataFrame): DataFrame containing at least
            ['alert_type', 'ticker', 'comment', 'channel', 'content']

    Returns:
        pd.DataFrame: Normalized and enriched copy of the DataFrame
    """
    if df.empty:
        return df

    df = df.copy()  # Avoid mutating input

    # --- Normalize alert_type ---
    if "alert_type" in df.columns:
        df["alert_type"] = (
            df["alert_type"]
            .astype(str)
            .str.lower()
            .str.replace(" alert", "", regex=False)
            .str.strip()
        )

    # --- Normalize ticker ---
    if "ticker" in df.columns:
        df["ticker"] = (
            df["ticker"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.replace(r"[^A-Z]", "", regex=True)  # remove emoji/symbols
            .str.strip()
        )

    # --- Normalize comment ---
    if "comment" in df.columns:
        df["comment"] = df["comment"].fillna("").astype(str).str.strip()

    # --- Derive signal_strength (emoji or keywords) ---
    def classify_strength(text):
        text = str(text)
        if any(e in text for e in ["🔥", "A+", "strong", "momentum"]):
            return "A+"
        if any(e in text for e in ["♻️", "pullback", "retest"]):
            return "Pullback"
        if any(e in text for e in ["💀", "exit", "weaken"]):
            return "Exit"
        return "Neutral"

    if "content" in df.columns:
        df["signal_strength"] = df["content"].apply(classify_strength)
    else:
        df["signal_strength"] = df.get("comment", "").apply(classify_strength)

    # --- Boolean helpers ---
    df["is_call"] = df.get("alert_type", "").eq("call")
    df["is_put"] = df.get("alert_type", "").eq("put")

    # --- Chronological sort if timestamp exists ---
    if "timestamp" in df.columns:
        df.sort_values(by="timestamp", inplace=True, ignore_index=True)

    return df
