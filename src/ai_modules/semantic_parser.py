"""
semantic_parser.py
AI-Enhanced Semantic Parser for Discord Trade Signal Intelligence
Author: @drashadm (DeAndrai Mullen)
Version: 1.3 (Production Release)
"""

import os
import json
import yaml
import hashlib
import time
import logging
import threading
from datetime import datetime
from jsonschema import validate, ValidationError
from openai import OpenAI
from dateutil import parser as date_parser  # tolerant ISO parser

# === Load Configuration ===
with open("config/prompt_templates.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

AI_CONF = CONFIG["semantic_parser"]
FALLBACKS = CONFIG["model_fallbacks"]
TELEMETRY_LOG = CONFIG["governance"].get("telemetry_log", "metrics/usage.json")

# === API Clients ===
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

# === Paths ===
CACHE_DIR = "data/ai_parsed"
LOG_DIR = "logs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TELEMETRY_LOG), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "parser.log")),
        logging.StreamHandler(),
    ],
)

_lock = threading.Lock()  # thread safety for telemetry writes

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def cache_path(raw_text: str) -> str:
    return os.path.join(CACHE_DIR, f"{hash_key(raw_text)}.json")

def cache_lookup(raw_text: str) -> dict | None:
    path = cache_path(raw_text)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("_schema_version") == CONFIG["meta"]["version"]:
                return data
        except Exception as e:
            logging.warning(f"Cache read failed: {e}")
    return None

def cache_save(raw_text: str, data: dict) -> None:
    tmp_path = cache_path(raw_text) + ".tmp"
    data["_schema_version"] = CONFIG["meta"]["version"]
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, tmp_path[:-4])
    except Exception as e:
        logging.error(f"Cache save failed: {e}")

def rotate_telemetry_if_needed():
    """Rotate telemetry log if file grows too large (>5MB)."""
    if os.path.exists(TELEMETRY_LOG) and os.path.getsize(TELEMETRY_LOG) > 5_000_000:
        rotated = f"{TELEMETRY_LOG}.{datetime.utcnow():%Y%m%d%H%M}"
        os.rename(TELEMETRY_LOG, rotated)
        logging.info(f"Telemetry rotated: {rotated}")

def log_telemetry(entry: dict) -> None:
    """Thread-safe append telemetry data to metrics/usage.json"""
    rotate_telemetry_if_needed()
    entry.update({
        "timestamp": datetime.utcnow().isoformat(),
        "schema_version": CONFIG["meta"]["version"],
    })
    try:
        with _lock:
            if os.path.exists(TELEMETRY_LOG):
                with open(TELEMETRY_LOG, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
            else:
                with open(TELEMETRY_LOG, "w", encoding="utf-8") as f:
                    json.dump([entry], f, indent=2)
    except Exception as e:
        logging.warning(f"Telemetry log failed: {e}")

# ---------------------------------------------------------
# Core Model Call
# ---------------------------------------------------------
def call_model(prompt: str, retries: int = 2) -> tuple[str | None, dict]:
    """Call OpenAI model with retries and usage tracking."""
    if not client:
        raise RuntimeError("Missing OPENAI_API_KEY; cannot call model.")
    for attempt in range(retries + 1):
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=AI_CONF["model"],
                temperature=AI_CONF.get("temperature", 0.2),
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                timeout=25,
                max_tokens=AI_CONF.get("max_tokens", 512),
                stream=False,
            )
            latency = round(time.time() - start, 3)
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                "total_tokens": getattr(resp.usage, "total_tokens", 0),
                "latency_sec": latency,
            }
            return resp.choices[0].message.content, usage
        except Exception as e:
            logging.warning(f"Model call attempt {attempt + 1} failed: {e}")
            time.sleep(1.5 ** attempt)  # exponential backoff
    logging.error("All model retries failed.")
    return None, {}

# ---------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------
def score_quality(record: dict) -> dict:
    """Simple rule-based hybrid quality scorer."""
    score, rationale = 1.0, []
    if not record.get("price") or record["price"] <= 0:
        score -= 0.2; rationale.append("Missing/malformed price")
    if record.get("ticker") in ["N/A", None, ""]:
        score -= 0.1; rationale.append("Unrecognized ticker")
    try:
        ts = date_parser.isoparse(record["timestamp"])
        if (datetime.utcnow() - ts).days > 30:
            score -= 0.1; rationale.append("Timestamp older than 30d")
    except Exception:
        score -= 0.1; rationale.append("Invalid timestamp")
    record["quality_score"] = max(0.0, min(1.0, round(score, 2)))
    record["quality_rationale"] = ", ".join(rationale) or "Good quality"
    return record

# ---------------------------------------------------------
# Main Parser
# ---------------------------------------------------------
def ai_parse_signal(message: str) -> dict | None:
    """Parse a Discord trading message, validate, score, and log telemetry."""
    cached = cache_lookup(message)
    if cached:
        logging.info("Cache hit.")
        return cached

    prompt = f"{AI_CONF['prompt']}\n\nMessage:\n{message.strip()}"
    raw_json, usage = call_model(prompt)

    if not raw_json:
        logging.warning("Fallback to placeholder result.")
        raw_json = json.dumps({
            "alert_type": "UNKNOWN",
            "ticker": "N/A",
            "price": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": "neutral",
            "confidence": 0.0,
        })
        usage = {"total_tokens": 0, "latency_sec": 0}

    try:
        raw_json = raw_json.strip().split("```")[-1]  # handle markdown wrapping
        data = json.loads(raw_json)
        validate(instance=data, schema=AI_CONF["schema"])

        # Sanity corrections
        if not (0 < data["price"] < 2000):
            raise ValueError("Unrealistic price value")
        data["ticker"] = data["ticker"].upper().strip()

        data = score_quality(data)
        cache_save(message, data)

        log_telemetry({
            "event": "ai_parse_signal",
            "model": AI_CONF["model"],
            **usage,
            "quality_score": data["quality_score"],
        })

        logging.info(
            f"Parsed {data['ticker']} ({data['alert_type']}) | "
            f"Price: {data['price']} | Quality: {data['quality_score']}"
        )
        return data

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logging.error(f"Validation failed: {e}")
        log_telemetry({
            "event": "parse_error",
            "error": str(e),
            "model": AI_CONF["model"],
        })
        return None
