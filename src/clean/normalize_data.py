import pandas as pd

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize trade signal data for consistent ML-ready formatting.
    """
    if df.empty:
        return df

    df = df.copy()

    # Normalize alert type
    if "alert_type" in df.columns:
        df["alert_type"] = (
            df["alert_type"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # Normalize ticker
    if "ticker" in df.columns:
        df["ticker"] = (
            df["ticker"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

    # Handle numeric prices
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Drop rows missing both ticker & price (to keep quality high)
    df.dropna(subset=["ticker", "price"], how="all", inplace=True)

    return df
