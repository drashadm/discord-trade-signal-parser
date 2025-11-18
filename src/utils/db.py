"""
SQLite Writer for Trade Signal Data
Author: @drashadm
Version: 1.1.0
"""

import os
import sqlite3
import pandas as pd
import time


def write_to_sqlite(
    df: pd.DataFrame,
    db_path: str = "data/clean/trade_signals.db",
    table_name: str = "signals",
    if_exists: str = "replace",
    chunk_size: int = 5000,
    max_retries: int = 3,
) -> dict:
    """
    Safely writes a DataFrame into SQLite with WAL mode, chunked writing,
    retry handling, and automatic indexing on (ticker, date).
    Returns a summary dict with write details.
    """

    if df.empty:
        print("[WARN] Empty DataFrame — no data written.")
        return {"db_path": os.path.abspath(db_path), "rows_written": 0}

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    retries = 0
    rows_written = 0

    while retries < max_retries:
        try:
            with sqlite3.connect(db_path, timeout=30) as conn:

                # --- Performance & Safety ---
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                conn.execute("PRAGMA synchronous=NORMAL;")

                # --- Chunked Write ---
                df.to_sql(
                    table_name,
                    conn,
                    if_exists=if_exists,
                    index=False,
                    chunksize=chunk_size,    # <-- Correct keyword
                )
                conn.commit()

                # Row count verification
                cur = conn.execute(f"SELECT COUNT(*) FROM {table_name};")
                rows_written = cur.fetchone()[0]

                print(f"[INFO] SQLite table '{table_name}' now has {rows_written:,} rows.")

                # --- Optional index creation ---
                if {"ticker", "date"}.issubset(df.columns):
                    try:
                        conn.execute(
                            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ticker_date "
                            f"ON {table_name}(ticker, date);"
                        )
                        conn.commit()
                    except sqlite3.Error as e:
                        print(f"[WARN] Index creation skipped: {e}")

                print(f"[INFO] Database successfully updated → {db_path}")

                return {
                    "db_path": os.path.abspath(db_path),
                    "rows_written": rows_written,
                    "table": table_name,
                }

        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                retries += 1
                print(f"[WARN] Database is locked. Retrying ({retries}/{max_retries})...")
                time.sleep(2 * retries)
                continue
            else:
                print(f"[ERROR] SQLite operational error: {e}")
                break

        except sqlite3.Error as e:
            print(f"[ERROR] SQLite write failed: {e}")
            break

    print("[ERROR] Max retries exceeded — write aborted.")
    return {
        "db_path": os.path.abspath(db_path),
        "rows_written": rows_written,
        "error": "max_retries_exceeded",
    }
