import os
import sys
import time
import logging
import pandas as pd
from src.parse.parse_signals import parse_dataframe
from src.clean.normalize_data import normalize
from src.analyze.charts import plot_all_charts
from src.utils.db import write_to_sqlite

# === Logging configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# === Configuration ===
RAW = os.getenv("RAW_CSV_PATH", "data/raw/discord_messages.csv")
CSV_OUT = os.getenv("CLEAN_CSV_PATH", "data/clean/trade_signals.csv")
DB_PATH = os.getenv("DB_PATH", "data/clean/trade_signals.db")

os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/charts", exist_ok=True)
os.makedirs("data/raw/backups", exist_ok=True)


def main():
    start_time = time.time()
    log.info("🚀 Starting Discord Trade Signal ETL pipeline...")

    # --- Step 1: Load raw data ---
    if not os.path.exists(RAW):
        log.error(f"Missing input file: {RAW}")
        sys.exit(1)

    try:
        raw = pd.read_csv(RAW)
        log.info(f"Loaded {len(raw):,} raw messages from {RAW}")

        # Create timestamped backup of raw file
        backup_path = f"data/raw/backups/discord_messages_{int(time.time())}.csv"
        raw.to_csv(backup_path, index=False)
        log.info(f"Backup created → {backup_path}")

    except Exception as e:
        log.exception(f"Failed to load or backup CSV: {e}")
        sys.exit(1)

    # --- Step 2: Parse ---
    try:
        parsed = parse_dataframe(raw)
        if parsed.empty:
            log.warning("No valid signals parsed — pipeline stopped.")
            sys.exit(0)
        log.info(f"Parsed {len(parsed):,} valid alerts.")
    except Exception as e:
        log.exception(f"Parsing failed: {e}")
        sys.exit(1)

    # --- Step 3: Normalize ---
    try:
        clean = normalize(parsed)
        log.info(f"Normalized dataset → {len(clean):,} rows ready for output.")
    except Exception as e:
        log.exception(f"Normalization failed: {e}")
        sys.exit(1)

    # --- Step 4: Save outputs ---
    try:
        clean.to_csv(CSV_OUT, index=False)
        log.info(f"CSV exported → {CSV_OUT}")
    except Exception as e:
        log.exception(f"CSV export failed: {e}")

    try:
        write_to_sqlite(clean, db_path=DB_PATH)
    except Exception as e:
        log.exception(f"SQLite write failed: {e}")

    try:
        chart_paths = plot_all_charts(clean)
        for path in chart_paths:
            log.info(f"Chart generated → {path}")
    except Exception as e:
        log.exception(f"Chart plotting failed: {e}")

    # --- Step 5: Summary ---
    elapsed = time.time() - start_time
    log.info("\n=== Pipeline Summary ===")
    log.info(f"Raw messages:     {len(raw):,}")
    log.info(f"Parsed alerts:    {len(parsed):,}")
    log.info(f"Final clean rows: {len(clean):,}")
    log.info(f"Output CSV:       {CSV_OUT}")
    log.info(f"Database:         {DB_PATH}")
    log.info(f"Charts directory: data/charts/")
    log.info(f"Elapsed time:     {elapsed:.2f}s")
    log.info("✅ ETL pipeline completed successfully.")

    sys.exit(0)


if __name__ == "__main__":
    main()
