"""
normalizer.py
AI-Enhanced Dataset Normalizer for Discord Trade Signal Intelligence
Author: @drashadm (DeAndrai Mullen)
Version: 1.6 (Budget-Safe + OpenAI v1 Update + Production Hardened)
"""

from __future__ import annotations
import os, json, logging, re, time, threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import yaml
import pandas as pd
from jsonschema import validate, ValidationError
from dateutil import parser as date_parser
from openai import OpenAI

# =========================================================
# Paths & Config
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = ROOT / "config" / "prompt_templates.yaml"
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "discord_export.csv"
CLEAN_PATH = DATA_DIR / "clean" / "trade_signals.csv"
LOG_DIR = ROOT / "logs"
for d in (DATA_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

with open(CONF_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

NORM_CONF = CONFIG.get("normalizer", {})
GOV_CONF = CONFIG.get("governance", {})

TELEMETRY_LOG = Path(GOV_CONF.get("telemetry_log", "./metrics/usage.json")).resolve()
TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)

MODEL = NORM_CONF.get("model", "gpt-4.1-mini")
TEMPERATURE = float(NORM_CONF.get("temperature", 0.2))
MAX_TOKENS = int(NORM_CONF.get("max_tokens", 300))
PROMPT = NORM_CONF.get("prompt", "Normalize and clean the dataset fields for consistency.")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

USE_AI = bool(int(os.getenv("NORM_USE_AI", "1")))
AI_MAX_RECORDS = int(os.getenv("NORM_AI_MAX_RECORDS", "250"))

# === FinOps / Budget Control ===
CHARS_PER_TOKEN = 4
OPENAI_INPUT_PRICE = 0.40         # gpt-4.1-mini per 1M
OPENAI_OUTPUT_PRICE = 1.60        # gpt-4.1-mini per 1M
MAX_COST_PER_CALL_USD = 0.002     # 0.2 cents per record
MAX_TOTAL_AI_COST_USD = 2.00      # normalization stage budget cap

_lock = threading.Lock()

# =========================================================
# Logging
# =========================================================
logger = logging.getLogger("normalizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "normalizer.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

# =========================================================
# Telemetry
# =========================================================
def _rotate_telemetry_if_needed() -> None:
    try:
        if TELEMETRY_LOG.exists() and TELEMETRY_LOG.stat().st_size > 5_000_000:
            rotated = TELEMETRY_LOG.with_name(
                f"{TELEMETRY_LOG.stem}.{datetime.utcnow():%Y%m%d%H%M}.json"
            )
            TELEMETRY_LOG.rename(rotated)
            logger.info(f"Telemetry rotated: {rotated}")
    except Exception as e:
        logger.warning(f"Telemetry rotation failed: {e}")

def _log_telemetry(entry: Dict[str, Any]) -> None:
    entry.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "normalizer",
        "schema_version": str(CONFIG.get("meta", {}).get("version", "unknown")),
    })
    try:
        with _lock:
            _rotate_telemetry_if_needed()
            try:
                data = json.loads(TELEMETRY_LOG.read_text(encoding="utf-8")) if TELEMETRY_LOG.exists() else []
                if not isinstance(data, list):
                    data = [data]
            except Exception:
                data = []
            data.append(entry)
            tmp = TELEMETRY_LOG.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(TELEMETRY_LOG)
    except Exception as e:
        logger.warning(f"Telemetry write failed: {e}")

# =========================================================
# Cleaning Utilities
# =========================================================
ALIASES = {"TESLA": "TSLA", "SPY ETF": "SPY", "SPX500": "SPX"}

def _normalize_ticker(t: str) -> str:
    if not isinstance(t, str):
        return "N/A"
    raw = t.upper().strip()
    for alias, canonical in ALIASES.items():
        if alias in raw:
            return canonical
    sym = re.sub(r"[^A-Z\.-]", "", raw)
    return sym or "N/A"

def _normalize_price(p: Any) -> float:
    try:
        val = float(re.sub(r"[^0-9\.]", "", str(p)))
        return round(val, 2) if 0 < val <= 10000 else 0.0
    except Exception:
        return 0.0

def _normalize_timestamp(ts: Any) -> str:
    try:
        parsed = date_parser.parse(str(ts))
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()

def _normalize_alert_type(a: str) -> str:
    if not isinstance(a, str):
        return "UNKNOWN"
    a = a.strip().lower()
    if "call" in a: return "CALL"
    if "put" in a: return "PUT"
    if "entry" in a: return "ENTRY"
    if "exit" in a: return "EXIT"
    return "OTHER"

def _extract_json_block(text: str) -> Optional[str]:
    """Extract JSON even inside markdown triple backticks."""
    if "```" in text:
        parts = re.split(r"```(?:json)?", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[-1].split("```")[0].strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else None

# =========================================================
# AI Normalization (NEW OpenAI Responses API)
# =========================================================
def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * OPENAI_INPUT_PRICE +
        (output_tokens / 1_000_000) * OPENAI_OUTPUT_PRICE
    )

def _ai_normalize_record(record: Dict[str, Any], retry: int = 1) -> Optional[Dict[str, Any]]:
    if client is None or not USE_AI:
        return None

    for attempt in range(retry + 1):
        try:
            prompt = (
                f"{PROMPT}\n\n"
                "Return ONLY valid JSON. No markdown, no explanations.\n\n"
                f"Record:\n{json.dumps(record, indent=2)}"
            )

            start = time.time()

            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": "Return strictly valid JSON. No explanations."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            latency = round(time.time() - start, 3)
            output = resp.output_text

            usage = resp.usage or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost_est = _estimate_cost(input_tokens, output_tokens)

            if cost_est > MAX_COST_PER_CALL_USD:
                logger.warning(
                    f"Normalize call cost ({cost_est:.6f}) exceeded per-call budget "
                    f"({MAX_COST_PER_CALL_USD:.6f})"
                )

            json_str = _extract_json_block(output)
            if not json_str:
                raise ValueError("No JSON block in AI response.")

            data = json.loads(json_str)

            _log_telemetry({
                "event": "ai_normalize_record",
                "model": MODEL,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "latency_sec": latency,
                "est_cost_usd": round(cost_est, 6),
            })

            return data

        except Exception as e:
            if attempt < retry:
                time.sleep(1.5 * (attempt + 1))
                continue
            _log_telemetry({
                "event": "ai_normalize_failed",
                "error": str(e),
                "model": MODEL,
            })
            return None

# =========================================================
# Main Normalization Pipeline
# =========================================================
def normalize_dataset(raw_path: str | Path = RAW_PATH, out_path: str | Path = CLEAN_PATH) -> pd.DataFrame:
    raw_path, out_path = Path(raw_path), Path(out_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing dataset: {raw_path}")

    df = pd.read_csv(raw_path)
    if df.empty:
        logger.warning("Empty raw dataset.")
        _log_telemetry({"event": "normalize_skipped", "reason": "no_records"})
        pd.DataFrame(columns=["alert_type","ticker","price","timestamp","sentiment","source"]).to_csv(out_path, index=False)
        return pd.DataFrame()

    schema = NORM_CONF.get("schema")
    normalized_records: List[Dict[str, Any]] = []

    ai_calls = 0
    ai_total_cost = 0.0

    for rec in df.to_dict(orient="records"):
        norm = {
            "alert_type": _normalize_alert_type(rec.get("alert_type", "")),
            "ticker": _normalize_ticker(rec.get("ticker", "")),
            "price": _normalize_price(rec.get("price", 0)),
            "timestamp": _normalize_timestamp(rec.get("timestamp", datetime.utcnow())),
            "sentiment": str(rec.get("sentiment", "neutral")).lower(),
            "source": rec.get("source", "discord"),
        }

        needs_ai = (norm["ticker"] == "N/A" or norm["price"] == 0.0)

        if USE_AI and needs_ai and ai_calls < AI_MAX_RECORDS and ai_total_cost < MAX_TOTAL_AI_COST_USD:
            ai_result = _ai_normalize_record(rec)
            ai_calls += 1

            if ai_result and isinstance(ai_result, dict):
                for k, v in ai_result.items():
                    if k in norm and v not in [None, "", "null"]:
                        if k == "ticker":
                            norm[k] = _normalize_ticker(v)
                        elif k == "price":
                            norm[k] = _normalize_price(v)
                        elif k == "timestamp":
                            norm[k] = _normalize_timestamp(v)
                        elif k == "alert_type":
                            norm[k] = _normalize_alert_type(v)
                        else:
                            norm[k] = v

            # simple estimate if usage missing
            ai_total_cost += MAX_COST_PER_CALL_USD

            if ai_total_cost >= MAX_TOTAL_AI_COST_USD:
                logger.warning("AI normalization budget reached. Disabling further AI calls.")
                USE_AI = False

        if schema:
            try:
                validate(instance=norm, schema=schema)
            except ValidationError as e:
                logger.warning(f"Schema validation failed: {e.message}")

        normalized_records.append(norm)

    clean_df = pd.DataFrame(normalized_records)
    clean_df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f" Normalized dataset saved → {out_path} ({len(clean_df)} records)")

    _log_telemetry({
        "event": "normalize_completed",
        "records": len(clean_df),
        "ai_calls": ai_calls,
        "total_ai_cost_usd": round(ai_total_cost, 4),
        "ai_used": USE_AI,
    })

    return clean_df
