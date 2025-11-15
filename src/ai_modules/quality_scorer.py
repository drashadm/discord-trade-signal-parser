"""
quality_scorer.py
AI-Assisted Quality Evaluation for Discord Trade Signal Intelligence
Author: @drashadm (DeAndrai Mullen)
Version: 2.0 (Budget-Safe + OpenAI v1 Upgrade + Production Hardened)
"""

from __future__ import annotations
import os, json, logging, hashlib, threading, re, gzip, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any, Iterable, List

import yaml
import pandas as pd
from jsonschema import validate, ValidationError
from dateutil import parser as date_parser
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# OpenAI Client (NEW Responses API)
# =========================================================
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

# =========================================================
# Paths & Config
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = ROOT / "config" / "prompt_templates.yaml"
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "data" / "reports"

LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONF_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

AI_CONF: Dict[str, Any] = CONFIG.get("quality_scorer", {})
GOV_CONF: Dict[str, Any] = CONFIG.get("governance", {})

TELEMETRY_LOG = Path(GOV_CONF.get("telemetry_log", "./metrics/usage.json")).resolve()
TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# Model & Behavior Settings
# =========================================================
MODEL = AI_CONF.get("model", "gpt-4.1-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

AI_GATE_THRESHOLD = float(os.getenv("QS_AI_GATE_THRESHOLD", AI_CONF.get("ai_gate_threshold", 0.70)))
AI_MAX_EVALS = int(os.getenv("QS_AI_MAX_EVALS", AI_CONF.get("ai_max_evals", 250)))
AI_MAX_WORKERS = int(os.getenv("QS_AI_MAX_WORKERS", AI_CONF.get("ai_max_workers", 4)))
AI_RPM_LIMIT = int(os.getenv("QS_AI_RPM_LIMIT", AI_CONF.get("ai_rpm_limit", 180)))
AI_TIMEOUT = float(os.getenv("QS_AI_TIMEOUT", AI_CONF.get("ai_timeout", 20.0)))
MAX_PRICE = float(os.getenv("QS_MAX_PRICE", AI_CONF.get("max_price", 5000)))
MIN_PRICE = float(os.getenv("QS_MIN_PRICE", AI_CONF.get("min_price", 0.0)))
CHUNK_SIZE = int(os.getenv("QS_CHUNK_SIZE", 0))
STRICT_SCHEMA = bool(int(os.getenv("QS_STRICT_SCHEMA", AI_CONF.get("strict_schema", 0))))
DRY_RUN = bool(int(os.getenv("QS_DRY_RUN", 0)))

# =========================================================
# Cost Controls (FinOps Governance)
# =========================================================
CHARS_PER_TOKEN = 4
INPUT_TOKEN_PRICE = 0.40       # gpt-4.1-mini
OUTPUT_TOKEN_PRICE = 1.60
MAX_COST_PER_CALL = 0.002      # USD
MAX_TOTAL_AI_COST = 2.00       # USD cap for entire stage

# =========================================================
# Logging
# =========================================================
logger = logging.getLogger("quality_scorer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "quality_scorer.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_min = [0, time.time()]  # [count, window_start_ts]

# =========================================================
# OpenAI Client Init (Unified)
# =========================================================
client = OpenAI(api_key=OPENAI_KEY) if (OPENAI_AVAILABLE and OPENAI_KEY) else None


# =========================================================
# Telemetry
# =========================================================
def _rotate_and_compress(path: Path):
    try:
        if path.exists() and path.stat().st_size > 5_000_000:
            rotated = path.with_suffix(
                path.suffix + f".{datetime.utcnow():%Y%m%d%H%M%S}"
            )
            path.replace(rotated)
            with rotated.open("rb") as src, gzip.open(str(rotated) + ".gz", "wb") as dst:
                dst.writelines(src)
            rotated.unlink(missing_ok=True)
            logger.info(f"Telemetry rotated & gzipped: {rotated}.gz")
    except Exception as e:
        logger.warning(f"Telemetry rotation failed: {e}")

def log_telemetry(entry: Dict[str, Any]):
    entry.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "quality_scorer",
        "schema_version": CONFIG.get("meta", {}).get("version", "unknown"),
    })
    _rotate_and_compress(TELEMETRY_LOG)
    try:
        with _lock:
            try:
                data = json.loads(TELEMETRY_LOG.read_text(encoding="utf-8")) if TELEMETRY_LOG.exists() else []
                if not isinstance(data, list):
                    data = [data]
            except Exception:
                backup = TELEMETRY_LOG.with_suffix(TELEMETRY_LOG.suffix + ".bak")
                TELEMETRY_LOG.replace(backup)
                logger.warning(f"Telemetry corrupted → moved to {backup}")
                data = []
            data.append(entry)
            tmp = TELEMETRY_LOG.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(TELEMETRY_LOG)
    except Exception as e:
        logger.warning(f"Telemetry write failed: {e}")


# =========================================================
# Rule-Based Scoring
# =========================================================
def compute_quality_score(record: Dict[str, Any]) -> Tuple[float, str]:
    score = 1.0
    rationale = []

    # Price
    try:
        price_val = float(record.get("price", 0))
    except Exception:
        price_val = None

    if price_val is None or price_val <= MIN_PRICE:
        score -= 0.25; rationale.append("Missing/malformed price")
    elif price_val > MAX_PRICE:
        score -= 0.15; rationale.append("Price unusually high")

    # Ticker
    ticker = record.get("ticker")
    if not ticker or not re.fullmatch(r"[A-Z]{1,6}([\.-][A-Z]{1,3})?", str(ticker).upper()):
        score -= 0.15; rationale.append("Unrecognized ticker")

    # Timestamp
    ts_str = record.get("timestamp")
    try:
        ts = date_parser.isoparse(str(ts_str))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).days
        if age_days > 30:
            score -= 0.10; rationale.append("Timestamp older than 30d")
    except Exception:
        score -= 0.15; rationale.append("Invalid timestamp")

    # Full schema check
    if all(record.get(f) not in [None, "", "N/A"] for f in ["alert_type", "ticker", "price", "timestamp"]):
        score += 0.25; rationale.append("Full schema present")

    # Sentiment normalization
    if str(record.get("sentiment", "")).lower() in {"bullish", "bearish", "neutral"}:
        score += 0.05; rationale.append("Sentiment normalized")

    return max(0.0, min(1.0, round(score, 2))), (", ".join(rationale) or "Good quality")


# =========================================================
# OpenAI Responses API (NEW)
# =========================================================
def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * INPUT_TOKEN_PRICE +
        (output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE
    )


def _ai_eval_once(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if client is None or DRY_RUN:
        return None

    prompt = (
        f"{AI_CONF.get('prompt','You are a data quality auditor.')}\n\n"
        "Return ONLY valid JSON with fields:\n"
        "{quality_score: float (0-1), rationale: string}\n\n"
        f"Record:\n{json.dumps(record, indent=2)}"
    )

    try:
        # Rate limit
        if AI_RPM_LIMIT > 0:
            with _rate_lock:
                count, start = _last_min
                now = time.time()
                if now - start >= 60:
                    _last_min[:] = [1, now]
                else:
                    if count >= AI_RPM_LIMIT:
                        time.sleep(60 - (now - start))
                        _last_min[:] = [1, time.time()]
                    else:
                        _last_min[0] += 1

        start = time.time()
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=AI_CONF.get("max_tokens", 300),
            temperature=float(AI_CONF.get("temperature", 0.0)),
        )
        latency = round(time.time() - start, 3)

        # Extract JSON cleanly
        text = resp.output_text.strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None

        parsed = json.loads(m.group(0))

        # Usage + cost
        usage = resp.usage or {}
        inp_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cost = _estimate_cost(inp_tok, out_tok)

        log_telemetry({
            "event": "ai_quality_eval",
            "model": MODEL,
            "latency_sec": latency,
            "input_tokens": inp_tok,
            "output_tokens": out_tok,
            "est_cost_usd": round(cost, 6),
        })

        parsed["quality_score"] = float(parsed.get("quality_score", 0))
        parsed["rationale"] = parsed.get("rationale", "AI evaluation")

        return parsed

    except Exception as e:
        log_telemetry({"event": "ai_eval_failed", "error": str(e)})
        return None


# =========================================================
# Dataset Scoring Pipeline
# =========================================================
def score_dataset(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    in_path = Path(csv_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing dataset: {in_path}")

    schema = AI_CONF.get("schema", {})
    all_frames = []
    total_cost = 0.0

    def iter_chunks():
        if CHUNK_SIZE > 0:
            for df in pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False):
                yield df
        else:
            yield pd.read_csv(in_path, low_memory=False)

    for df in iter_chunks():
        if df.empty:
            continue

        # Ensure expected columns exist
        for col in ("alert_type", "ticker", "price", "timestamp", "sentiment"):
            if col not in df.columns:
                df[col] = None

        # Rule-based scoring
        records = df.to_dict(orient="records")
        scored_records = []

        for rec in records:
            try:
                validate(rec, schema)
            except ValidationError as e:
                if STRICT_SCHEMA:
                    continue
                logger.warning(f"Schema mismatch: {e.message}")

            score, rationale = compute_quality_score(rec)
            rec["quality_score"] = score
            rec["quality_rationale"] = rationale
            scored_records.append(rec)

        df_scored = pd.DataFrame(scored_records)
        if df_scored.empty:
            all_frames.append(df_scored)
            continue

        # AI gating
        gated_idx = (
            df_scored.sort_values("quality_score").index[:AI_MAX_EVALS]
        )

        # AI evaluations (only if needed & within budget)
        ai_inputs = [df_scored.loc[i].to_dict() for i in gated_idx]
        ai_results = [None] * len(ai_inputs)

        if client and not DRY_RUN:
            with ThreadPoolExecutor(max_workers=AI_MAX_WORKERS) as ex:
                futures = {
                    ex.submit(_ai_eval_once, rec): idx
                    for idx, rec in enumerate(ai_inputs)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        out = fut.result()
                        if out:
                            ai_results[idx] = out
                            total_cost += out.get("est_cost_usd", 0)
                            if total_cost >= MAX_TOTAL_AI_COST:
                                logger.warning("⚠ AI scoring budget reached. Stopping further LLM calls.")
                                break
                    except Exception as e:
                        log_telemetry({"event": "ai_eval_exception", "error": str(e)})

        # Merge AI results
        df_scored["ai_quality_score"] = df_scored["quality_score"]
        df_scored["ai_rationale"] = df_scored["quality_rationale"]

        for idx_local, out in enumerate(ai_results):
            global_idx = gated_idx[idx_local]
            if out:
                df_scored.at[global_idx, "ai_quality_score"] = out["quality_score"]
                df_scored.at[global_idx, "ai_rationale"] = out["rationale"]

        df_scored["final_quality_score"] = df_scored[
            ["quality_score", "ai_quality_score"]
        ].max(axis=1)

        all_frames.append(df_scored)

    # Combine results
    result = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        output_path = REPORT_DIR / f"quality_report_{ts}.csv"

    if not result.empty:
        result.to_csv(output_path, index=False)
        logger.info(f" Quality report → {output_path}")

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(result),
            "mean_rule_score": float(result["quality_score"].mean()),
            "mean_ai_score": float(result["ai_quality_score"].mean()),
            "mean_final_score": float(result["final_quality_score"].mean()),
            "ai_evals_used": int((result["ai_quality_score"] != result["quality_score"]).sum()),
            "total_ai_cost_usd": round(total_cost, 6),
        }

        summary_path = REPORT_DIR / "quality_summary.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(summary_path)

        logger.info(f" Summary → {summary_path}")
        logger.info(
            f"Avg Quality — Rule: {summary['mean_rule_score']:.2f}, "
            f"AI: {summary['mean_ai_score']:.2f}, "
            f"Final: {summary['mean_final_score']:.2f}"
        )
    else:
        logger.warning("No rows scored.")

    return result


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    try:
        score_dataset(os.getenv("CLEAN_CSV_PATH", str(ROOT / "data" / "clean" / "trade_signals.csv")))
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
