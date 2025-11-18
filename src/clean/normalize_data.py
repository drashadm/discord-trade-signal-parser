import pandas as pd
from datetime import datetime

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize trade signal data (NON-AI version).
    Produces all required chart/ML fields:
        - alert_type (CALL/PUT/EXIT/ENTRY/OTHER)
        - ticker (uppercase)
        - price (float)
        - timestamp (ISO8601)
        - is_call / is_put
        - signal_strength (Neutral default)
    """
    if df.empty:
        return df

    df = df.copy()

    # --- Normalize alert type ---
    if "alert_type" in df.columns:
        df["alert_type"] = (
            df["alert_type"]
            .astype(str)
            .str.lower()
            .str.strip()
        )
    else:
        df["alert_type"] = "other"

    # Standardize categories
    df["alert_type"] = df["alert_type"].replace({
        "call": "CALL",
        "c": "CALL",
        "buy call": "CALL",
        "put": "PUT",
        "p": "PUT",
        "sell put": "PUT",
        "exit": "EXIT",
        "entry": "ENTRY",
    })

    # Fallback
    df["alert_type"] = df["alert_type"].apply(
        lambda x: x if x in ["CALL", "PUT", "ENTRY", "EXIT"] else "OTHER"
    )

    # --- Normalize ticker ---
    if "ticker" in df.columns:
        df["ticker"] = (
            df["ticker"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.replace(r"[^A-Z\.]", "", regex=True)
            .str.strip()
        )
    else:
        df["ticker"] = ""

    # --- Normalize price ---
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = None

    # --- Normalize timestamp ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = datetime.utcnow()

    df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.utcnow())
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # --- Required row cleaning ---
    df = df[df["ticker"] != ""]
    df = df[df["price"].notna()]

    # --- Derived fields for charts ---
    df["is_call"] = (df["alert_type"] == "CALL").astype(int)
    df["is_put"] = (df["alert_type"] == "PUT").astype(int)

    # --- Basic signal strength default ---
    df["signal_strength"] = "Neutral"

    # OPTIONAL: smarter signal strength logic can go here

    return df
