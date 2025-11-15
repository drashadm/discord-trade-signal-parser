"""
semantic_parser.py
AI-Enhanced Semantic Parser for Discord Trade Signal Intelligence
Author: @drashadm (DeAndrai Mullen)
Version: 1.4 (Budget-Safe + OpenAI v1 Upgrade + Production Hardened)
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
from dateutil import parser as date_parser
from openai import OpenAI

# =========================================================
# Load Configuration
# =========================================================
with open("config/prompt_templates.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

AI_CONF = CONFIG["semantic_parser"]
FALLBACKS = CONFIG["model_fallbacks"]
TELEMETRY_LOG = CONFIG["governance"].get("telemetry_log", "metrics/usage.json")

# =========================================================
# API Clients
# =========================================================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
if not client:
    raise RuntimeError("OPENAI_API_KEY missing. Cannot run parser.")

# Pricing reference for telemetry & budget governance
OPENAI_INPUT_PRICE = 0.40      # per 1M input tokens (gpt-4.1-mini)
OPENAI_OUTPUT_PRICE = 1.60     # per 1M output tokens

# Budget protection
MAX_TOKENS_PER_CALL = AI_CONF.get("max_tokens", 512)
MAX_PARSE_COST_USD = 0.002     # ≈ 2/10ths of a cent per parse
CHARS_PER_TOKEN = 4

# =========================================================
# Paths + Logging
# =========================================================
CACHE_DIR = "data/ai_parsed"
LOG_DIR = "logs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TELEMETRY_LOG), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "parser.log")),
        logging.StreamHandler(),
    ],
)

_lock = threading.Lock()

# =========================================================
# Utility Functions
# =========================================================
def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def cache_path(raw_text: str) -> str:
    return os.path.join(CACHE_DIR, f"{hash_key(raw_text)}.json")

def cache_lookup(raw_text: str) -> dict | None:
    """Returns cached JSON if schema version matches."""
    path = cache_path(raw_text)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("_schema_version") == CONFIG["meta"]["version"]:
            return data
    except Exception as e:
        logging.warning(f"Cache read failed: {e}")
    return None

def cache_save(raw_text: str, data: dict) -> None:
    """Safely write cache JSON with version tag."""
    tmp = cache_path(raw_text) + ".tmp"
    data["_schema_version"] = CONFIG["meta"]["version"]
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, tmp[:-4])
    except Exception as e:
        logging.error(f"Cache save failed: {e}")

def rotate_telemetry_if_needed():
    if os.path.exists(TELEMETRY_LOG) and os.path.getsize(TELEMETRY_LOG) > 5_000_000:
        rotated = f"{TELEMETRY_LOG}.{datetime.utcnow():%Y%m%d%H%M}"
        os.rename(TELEMETRY_LOG, rotated)
        logging.info(f"Telemetry rotated: {rotated}")

def log_telemetry(entry: dict) -> None:
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

# =========================================================
# OpenAI - Correct v1 API Call (responses.create)
# =========================================================
def call_model(prompt: str, retries: int = 2):
    """
    Calls OpenAI (gpt-4.1-mini by default) safely with:
    - strict JSON return
    - token cap
    - retry logic
    - real usage tracking
    """
    for attempt in range(retries + 1):
        try:
            start = time.time()

            resp = client.responses.create(
                model=AI_CONF["model"],        # e.g., gpt-4.1-mini
                input=[
                    {"role": "system", "content": "Return ONLY valid JSON. No explanations."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=MAX_TOKENS_PER_CALL,
                temperature=AI_CONF.get("temperature", 0.2),
            )

            latency = round(time.time() - start, 3)

            usage = resp.usage or {}
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # === Cost Estimation (for logging only) ===
            est_cost = (
                (prompt_tokens / 1_000_000) * OPENAI_INPUT_PRICE +
                (completion_tokens / 1_000_000) * OPENAI_OUTPUT_PRICE
            )

            if est_cost > MAX_PARSE_COST_USD:
                logging.warning(
                    f"Single call cost {est_cost:.6f} exceeds safe per-parse threshold "
                    f"{MAX_PARSE_COST_USD:.6f}"
                )

            return resp.output_text, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_sec": latency,
                "est_cost_usd": round(est_cost, 6),
            }

        except Exception as e:
            logging.warning(f"Model call failed (attempt {attempt+1}): {e}")
            time.sleep(1.2 ** attempt)

    logging.error("All retries for call_model() failed.")
    return None, {}

# =========================================================
# Quality Scoring
# =========================================================
def score_quality(record: dict) -> dict:
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

# =========================================================
# Main Semantic Parser
# =========================================================
def ai_parse_signal(message: str) -> dict | None:
    """Parses Discord trading messages using OpenAI, with caching + validation."""
    
    # 1. Cache-first execution
    cached = cache_lookup(message)
    if cached:
        logging.info("Cache hit.")
        return cached

    # 2. Build safe prompt
    prompt = (
        f"{AI_CONF['prompt']}\n\n"
        "RULE: Return strictly valid JSON. No comments, no markdown.\n"
        "RULE: If uncertain, leave fields empty but valid.\n\n"
        f"Message:\n{message.strip()}"
    )

    # 3. Call model
    raw_json, usage = call_model(prompt)

    if not raw_json:
        logging.warning("Model returned no output — using safe fallback.")
        raw_json = json.dumps({
            "alert_type": "UNKNOWN",
            "ticker": "N/A",
            "price": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": "neutral",
            "confidence": 0.0,
        })
        usage = {"total_tokens": 0, "latency_sec": 0, "est_cost_usd": 0}

    # 4. Clean and decode model output
    try:
        raw_json = raw_json.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw_json)

        # Validate using JSON Schema from config
        validate(instance=data, schema=AI_CONF["schema"])

        # Sanity correction
        if not (0 < data["price"] < 2000):
            raise ValueError("Unrealistic price value")

        data["ticker"] = data["ticker"].upper().strip()

        # Apply quality scoring
        data = score_quality(data)

        # Save to cache
        cache_save(message, data)

        # Telemetry
        log_telemetry({
            "event": "ai_parse_signal",
            "model": AI_CONF["model"],
            **usage,
            "quality_score": data["quality_score"],
        })

        logging.info(
            f"Parsed {data['ticker']} ({data['alert_type']}) | "
            f"Price: {data['price']} | Q={data['quality_score']}"
        )
        return data

    except Exception as e:
        logging.error(f"Validation/decoding error: {e}")
        log_telemetry({
            "event": "parse_error",
            "error": str(e),
            "model": AI_CONF["model"],
        })
        return None
