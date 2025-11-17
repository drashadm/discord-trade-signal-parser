"""
main.py
Discord Trade Alert ETL Pipeline (Non-AI Version)
Author: @drashadm
Version: 1.0.0 (Stable Non-AI Build)
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# === MODULE IMPORTS ===
from parse_signals import parse_dataframe
from normalize import normalize
from charts import plot_all_charts


# === DIRECTORIES ===
ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CLEAN_DIR = DATA_DIR / "clean"
CHART_DIR = DATA_DIR / "charts"

RAW_FILE = RAW_DIR / "discord_messages.csv"
PARSED_FILE = PARSED_DIR / "parsed_signals.csv"
CLEAN_FILE = CLEAN_DIR / "trade_signals.csv"

for p in [RAW_DIR, PARSED_DIR, CLEAN_DIR, CHART_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# === PIPELINE ===
def stage(title: str):
    print("\n" + "=" * 60)
    print(f"🟦 {title}")
    print("=" * 60)


def main():

    # LOAD RAW DISCORD DATA
    stage("1. Loading raw Discord messages")
    if not RAW_FILE.exists():
        print(f" ERROR: Raw file not found → {RAW_FILE}")
        print("Run fetch_messages.py first.")
        sys.exit(1)

    df_raw = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df_raw):,} raw messages.")

    # PARSE INTO STRUCTURED SIGNALS
    stage("2. Parsing raw messages → structured signals")
    df_parsed = parse_dataframe(df_raw)

    if df_parsed.empty:
        print(" No valid signals were parsed.")
    else:
        df_parsed.to_csv(PARSED_FILE, index=False)
        print(f"Parsed {len(df_parsed):,} signals → {PARSED_FILE}")

    # NORMALIZE FIELDS
    stage("3. Normalizing parsed signals")
    df_clean = normalize(df_parsed)

    if df_clean.empty:
        print(" Clean dataset is empty after normalization.")
    else:
        df_clean.to_csv(CLEAN_FILE, index=False)
        print(f"Clean dataset saved → {CLEAN_FILE}")
        print(df_clean.head())

    # GENERATE CHARTS
    stage("4. Generating charts")
    try:
        chart_paths = plot_all_charts(df_clean, output_dir=str(CHART_DIR))
        if chart_paths:
            print("\n Charts generated:")
            for p in chart_paths:
                print(f"   - {p}")
        else:
            print(" No charts generated (insufficient data).")
    except Exception as e:
        print(f"Chart error: {e}")

    # SUMMARY REPORT
    stage("5. Summary Overview")
    if not df_clean.empty:
        print(df_clean.describe(include="all"))
    else:
        print("No data available for summary.")


if __name__ == "__main__":
    main()
