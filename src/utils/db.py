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
) -> str:
    """
    Write a DataFrame to an SQLite database safely and efficiently.

    Features:
    - WAL journal mode for concurrent reads
    - Chunked writes for large DataFrames
    - Auto index creation on (ticker, date)
    - Row-count confirmation
    - Retry handling for transient locks

    Args:
        df (pd.DataFrame): The DataFrame to write.
        db_path (str): Path to SQLite database file.
        table_name (str): Target table name.
        if_exists (str): 'replace', 'append', or 'fail' (default: 'replace').
        chunk_size (int): Rows per chunk for large DataFrames.
        max_retries (int): Retry attempts if database is temporarily locked.

    Returns:
        str: Absolute path to the database file.
    """
    if df.empty:
        print("[WARN] Empty DataFrame — no data written.")
        return os.path.abspath(db_path)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    retries = 0

    while retries < max_retries:
        try:
            with sqlite3.connect(db_path, timeout=30) as conn:
                # --- Performance PRAGMAs ---
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")

                # --- Write DataFrame ---
                df.to_sql(
                    table_name,
                    conn,
                    if_exists=if_exists,
                    index=False,
                    chunksize=chunk_size,
                )
                conn.commit()

                # --- Add helpful index for faster queries ---
                if {"ticker", "date"}.issubset(df.columns):
                    try:
                        conn.execute(
                            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ticker_date "
                            f"ON {table_name}(ticker, date);"
                        )
                    except sqlite3.Error as e:
                        print(f"[WARN] Index creation skipped: {e}")

                # --- Row count confirmation ---
                cur = conn.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cur.fetchone()[0]
                print(f"[INFO] SQLite table '{table_name}' now has {row_count:,} rows.")
                print(f"[INFO] Database successfully updated → {db_path}")

                return os.path.abspath(db_path)

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

    print("[ERROR] Max retries exceeded — database write aborted.")
    return os.path.abspath(db_path)
