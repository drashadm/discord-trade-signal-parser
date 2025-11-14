"""
main.py
Autonomous AI Data Intelligence Orchestrator
Author: @drashadm (DeAndrai Mullen)
Version: 1.1.2 (Production Hardened+)
"""

from __future__ import annotations
import os, sys, time, json, logging, traceback, signal
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# --- Module Imports ---
from src.ai_modules.semantic_parser import ai_parse_signal
from src.ai_modules.normalizer import normalize_dataset
from src.ai_modules.quality_scorer import score_dataset
from src.ai_modules.summarizer import generate_summary

# === Global Paths ===
ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = ROOT / "logs"
for d in (RAW_DIR, CLEAN_DIR, REPORT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_PATH = RAW_DIR / "discord_export.csv"
PARSED_PATH = RAW_DIR / "parsed_signals.csv"
CLEAN_PATH = CLEAN_DIR / "trade_signals.csv"

# === Logging ===
LOG_FILE = LOG_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")

# === Telemetry ===
TELEMETRY_LOG = DATA_DIR / "metrics" / "usage.json"
TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)
_lock = Lock()

def _rotate_telemetry_if_needed():
    try:
        if TELEMETRY_LOG.exists() and TELEMETRY_LOG.stat().st_size > 5_000_000:
            rotated = TELEMETRY_LOG.with_name(f"{TELEMETRY_LOG.stem}.{datetime.utcnow():%Y%m%d%H%M}.json")
            TELEMETRY_LOG.rename(rotated)
            logger.info(f"Telemetry rotated: {rotated}")
    except Exception as e:
        logger.warning(f"Telemetry rotation failed: {e}")

def log_telemetry(event: str, **kwargs):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    with _lock:
        _rotate_telemetry_if_needed()
        try:
            if TELEMETRY_LOG.exists():
                try:
                    data = json.loads(TELEMETRY_LOG.read_text(encoding="utf-8"))
                    if not isinstance(data, list): data = [data]
                except Exception:
                    data = []
            else:
                data = []
            data.append(entry)
            tmp = TELEMETRY_LOG.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(TELEMETRY_LOG)
        except Exception as e:
            logger.warning(f"Telemetry log failed: {e}")

# === Signal Handling ===
def handle_sigint(signum, frame):
    logger.warning(" Pipeline interrupted. Flushing telemetry...")
    log_telemetry("pipeline_interrupted")
    sys.exit(130)
signal.signal(signal.SIGINT, handle_sigint)

# === Stage Runner ===
def run_stage(stage_name: str, func, *args, retries: int = 1, **kwargs) -> bool:
    start = time.time()
    logger.info(f" Starting stage: {stage_name}")
    for attempt in range(retries + 1):
        try:
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)
            rec_count = len(result) if isinstance(result, pd.DataFrame) else None
            logger.info(f" Completed: {stage_name} in {duration}s ({rec_count or 'N/A'} records)")
            log_telemetry("stage_completed", stage=stage_name, duration_sec=duration, records=rec_count)
            return True
        except Exception as e:
            duration = round(time.time() - start, 2)
            logger.error(f" {stage_name} failed after {duration}s (attempt {attempt+1}): {e}")
            traceback.print_exc()
            log_telemetry("stage_failed", stage=stage_name, duration_sec=duration, error=str(e))
            if attempt < retries:
                logger.info(f" Retrying {stage_name} in 2s...")
                time.sleep(2)
            else:
                return False

# === Pre-flight Checks ===
def preflight_checks():
    logger.info(" Running pre-flight checks...")
    required_envs = ["OPENAI_API_KEY"]
    missing = [v for v in required_envs if not os.getenv(v)]
    if missing:
        logger.error(f" Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    if not RAW_PATH.exists():
        logger.error(f" Missing raw dataset at {RAW_PATH}. Please export Discord data first.")
        sys.exit(1)
    logger.info(" Environment and dataset verified.")

# === Semantic Parser Wrapper ===
def safe_parse(message: str) -> dict | None:
    try:
        return ai_parse_signal(str(message))
    except Exception as e:
        logger.warning(f"Parse failed: {e}")
        return None

# === Entry Point ===
def main():
    logger.info("=== AI DATA INTELLIGENCE PIPELINE START ===")
    start_time = time.time()
    preflight_checks()

    # Stage 1️⃣ Semantic Parsing
    logger.info(" Stage 1: Semantic Parsing (transforming raw Discord messages to structured data)...")
    df_raw = pd.read_csv(RAW_PATH)
    if "content" not in df_raw.columns:
        logger.error(" Missing 'content' column in raw dataset.")
        sys.exit(1)

    parsed_records = []
    max_workers = min(4, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(safe_parse, m): i for i, m in enumerate(df_raw["content"].astype(str))}
        for fut in as_completed(futures):
            res = fut.result()
            if res: parsed_records.append(res)

    if not parsed_records:
        logger.error(" No parsed records generated. Check raw dataset or API key.")
        log_telemetry("stage_failed", stage="Semantic Parsing", reason="no_output")
        sys.exit(1)

    parsed_df = pd.DataFrame(parsed_records)
    parsed_df.to_csv(PARSED_PATH, index=False, encoding="utf-8")
    logger.info(f" Parsed {len(parsed_df)} records → {PARSED_PATH}")
    log_telemetry("stage_completed", stage="Semantic Parsing", records=len(parsed_df))

    # Stage Normalization
    if not run_stage("Normalization", normalize_dataset, PARSED_PATH, CLEAN_PATH):
        log_telemetry("pipeline_failed", stage="Normalization")
        sys.exit(1)

    # Stage Quality Scoring
    if not run_stage("Quality Scoring", score_dataset, CLEAN_PATH):
        log_telemetry("pipeline_failed", stage="Quality Scoring")
        sys.exit(1)

    # Stage Summarization
    if not run_stage("Summarization", generate_summary, CLEAN_PATH, True):
        log_telemetry("pipeline_failed", stage="Summarization")
        sys.exit(1)

    total_runtime = round(time.time() - start_time, 2)
    logger.info(f" Pipeline completed successfully in {total_runtime}s")
    log_telemetry("pipeline_completed", duration_sec=total_runtime, total_records=len(parsed_df))
    logger.info("=== ALL STAGES COMPLETE ===")

# === CLI ===
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        log_telemetry("pipeline_interrupted")
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
        log_telemetry("pipeline_crashed", error=str(e))
        sys.exit(1)
