"""
quality_scorer.py
AI-Assisted Quality Evaluation for Discord Trade Signal Intelligence
Author: @drashadm (DeAndrai Mullen)
Version: 1.4 (Production Hardened+)
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

# OpenAI client (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from concurrent.futures import ThreadPoolExecutor, as_completed

# === Paths & Config ===
ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = ROOT / "config" / "prompt_templates.yaml"
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "data" / "reports"

with open(CONF_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

AI_CONF: Dict[str, Any] = CONFIG.get("quality_scorer", {})
GOV_CONF: Dict[str, Any] = CONFIG.get("governance", {})

TELEMETRY_LOG = Path(GOV_CONF.get("telemetry_log", "./metrics/usage.json")).resolve()
TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = AI_CONF.get("model", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# === Tuning (env overrides config) ===
AI_GATE_THRESHOLD = float(os.getenv("QS_AI_GATE_THRESHOLD", AI_CONF.get("ai_gate_threshold", 0.70)))
AI_MAX_EVALS = int(os.getenv("QS_AI_MAX_EVALS", AI_CONF.get("ai_max_evals", 250)))
AI_MAX_WORKERS = int(os.getenv("QS_AI_MAX_WORKERS", AI_CONF.get("ai_max_workers", 4)))
AI_RPM_LIMIT = int(os.getenv("QS_AI_RPM_LIMIT", AI_CONF.get("ai_rpm_limit", 180)))  # safety throttle
AI_TIMEOUT = float(os.getenv("QS_AI_TIMEOUT", AI_CONF.get("ai_timeout", 20.0)))
MAX_PRICE = float(os.getenv("QS_MAX_PRICE", AI_CONF.get("max_price", 5000)))
MIN_PRICE = float(os.getenv("QS_MIN_PRICE", AI_CONF.get("min_price", 0.0)))
CHUNK_SIZE = int(os.getenv("QS_CHUNK_SIZE", 0))  # 0 = read whole file
STRICT_SCHEMA = bool(int(os.getenv("QS_STRICT_SCHEMA", AI_CONF.get("strict_schema", 0))))
DRY_RUN = bool(int(os.getenv("QS_DRY_RUN", 0)))  # if set, skip LLM entirely

# === Logging ===
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
_last_minute = [0, time.time()]  # [count, window_start_ts]

# === Optional OpenAI Client ===
client = None
if OPENAI_KEY and OpenAI:
    try:
        client = OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        logger.warning(f"OpenAI client init failed: {e}. Continuing without AI.")

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def _is_missing(x: Any) -> bool:
    try:
        import math
        return x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip() == "")
    except Exception:
        return x is None

def hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def _rotate_and_compress(path: Path) -> None:
    """Rotate and gzip telemetry if >5MB."""
    try:
        if path.exists() and path.stat().st_size > 5_000_000:
            rotated = path.with_suffix(path.suffix + f".{datetime.now(timezone.utc):%Y%m%d%H%M%S}")
            path.replace(rotated)
            with rotated.open("rb") as src, gzip.open(str(rotated) + ".gz", "wb") as dst:
                dst.writelines(src)
            rotated.unlink(missing_ok=True)
            logger.info(f"Telemetry rotated & gzipped: {rotated}.gz")
    except Exception as e:
        logger.warning(f"Telemetry rotation failed: {e}")

def log_telemetry(entry: Dict[str, Any]) -> None:
    entry.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "quality_scorer",
        "schema_version": str(CONFIG.get("meta", {}).get("version", "unknown")),
    })
    _rotate_and_compress(TELEMETRY_LOG)
    try:
        with _lock:
            data: List[Dict[str, Any]] = []
            if TELEMETRY_LOG.exists():
                try:
                    data = json.loads(TELEMETRY_LOG.read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        data = [data]
                except Exception:
                    # recover from partial/corrupt file
                    backup = TELEMETRY_LOG.with_suffix(TELEMETRY_LOG.suffix + ".bak")
                    TELEMETRY_LOG.replace(backup)
                    logger.warning(f"Telemetry corrupted; moved to {backup}")
                    data = []
            data.append(entry)
            tmp = TELEMETRY_LOG.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(TELEMETRY_LOG)
    except Exception as e:
        logger.warning(f"Telemetry log failed: {e}")

def _ticker_ok(t: Any) -> bool:
    if _is_missing(t): return False
    s = str(t).upper().strip()
    # Allow typical US tickers + dots/dashes (BRK.B, RDS-A), length 1..6
    return bool(re.fullmatch(r"[A-Z]{1,6}([\.-][A-Z]{1,3})?", s))

def _throttle_rpm():
    """Simple per-process RPM limiter."""
    if AI_RPM_LIMIT <= 0: return
    with _rate_lock:
        count, start = _last_minute
        now = time.time()
        if now - start >= 60:
            _last_minute[0], _last_minute[1] = 0, now
            return
        if count >= AI_RPM_LIMIT:
            sleep_for = 60 - (now - start)
            if sleep_for > 0:
                time.sleep(min(sleep_for, 5))  # micro-sleep in small slices
        _last_minute[0] += 1

# ---------------------------------------------------------
# Rule-Based Scorer
# ---------------------------------------------------------
def compute_quality_score(record: Dict[str, Any]) -> Tuple[float, str]:
    score = 1.0
    rationale: List[str] = []

    # price
    price_raw = record.get("price")
    price_val: Optional[float] = None
    try:
        if not _is_missing(price_raw):
            price_val = float(price_raw)
    except Exception:
        price_val = None

    if price_val is None or price_val <= MIN_PRICE:
        score -= 0.25; rationale.append("Missing/malformed price")
    elif price_val > MAX_PRICE:
        score -= 0.15; rationale.append("Price unusually high")

    # ticker
    ticker = record.get("ticker")
    if not _ticker_ok(ticker):
        score -= 0.15; rationale.append("Unrecognized ticker")

    # timestamp
    ts_str = record.get("timestamp")
    if _is_missing(ts_str):
        score -= 0.15; rationale.append("Missing timestamp")
    else:
        try:
            ts = date_parser.isoparse(str(ts_str))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts).days
            if age_days > 30:
                score -= 0.10; rationale.append("Timestamp older than 30d")
        except Exception:
            score -= 0.15; rationale.append("Invalid timestamp format")

    # schema completeness
    required_fields = ("alert_type", "ticker", "price", "timestamp")
    if all(not _is_missing(record.get(f)) for f in required_fields):
        score += 0.25; rationale.append("Full schema present")

    # sentiment
    if str(record.get("sentiment", "")).lower() in {"bullish", "bearish", "neutral"}:
        score += 0.05; rationale.append("Sentiment normalized")

    score = max(0.0, min(1.0, round(score, 2)))
    return score, (", ".join(rationale) if rationale else "Good quality")

# ---------------------------------------------------------
# Optional LLM-Assisted Evaluation (with retry + usage)
# ---------------------------------------------------------
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract first JSON object from model output."""
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL)
    s = m.group(1) if m else text
    m2 = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if not m2: return None
    try:
        return json.loads(m2.group(1))
    except json.JSONDecodeError:
        return None

def _ai_eval_once(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if client is None or DRY_RUN:
        return None
    prompt = f"{AI_CONF.get('prompt','You are a data quality auditor.')}\n\nRecord:\n{json.dumps(record, indent=2)}"
    try:
        _throttle_rpm()
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=float(AI_CONF.get("temperature", 0.0)),
            messages=[
                {"role": "system", "content": "Return ONLY a JSON object with fields: quality_score (0-1) and rationale (string)."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=AI_CONF.get("max_tokens", 300),
            timeout=AI_TIMEOUT,
        )
        content = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json_from_text(content)
        if not parsed:
            return None
        qs = parsed.get("quality_score")
        try:
            qs = float(qs)
        except Exception:
            return None
        parsed["quality_score"] = max(0.0, min(1.0, qs))
        parsed["rationale"] = str(parsed.get("rationale", "")).strip() or "AI evaluation"
        # Telemetry with token usage if available
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", None) if usage else None
        log_telemetry({
            "event": "ai_eval",
            "model": MODEL,
            "tokens": tokens if tokens is not None else -1,
        })
        return parsed
    except Exception as e:
        # simple exponential backoff retry once
        time.sleep(0.8)
        try:
            _throttle_rpm()
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=float(AI_CONF.get("temperature", 0.0)),
                messages=[
                    {"role": "system", "content": "Return ONLY a JSON object with fields: quality_score (0-1) and rationale (string)."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=AI_CONF.get("max_tokens", 300),
                timeout=AI_TIMEOUT,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _extract_json_from_text(content)
            if not parsed:
                return None
            qs = float(parsed.get("quality_score", 0))
            parsed["quality_score"] = max(0.0, min(1.0, qs))
            parsed["rationale"] = str(parsed.get("rationale", "")).strip() or "AI evaluation"
            usage = getattr(resp, "usage", None)
            tokens = getattr(usage, "total_tokens", None) if usage else None
            log_telemetry({
                "event": "ai_eval_retry",
                "model": MODEL,
                "tokens": tokens if tokens is not None else -1,
                "error": str(e),
            })
            return parsed
        except Exception as e2:
            log_telemetry({"event": "ai_eval_failed", "error": str(e2), "model": MODEL})
            return None

# ---------------------------------------------------------
# Main Scoring Pipeline
# ---------------------------------------------------------
def _iter_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    if chunk_size and chunk_size > 0:
        for df in pd.read_csv(path,
                              dtype={"ticker": "string", "alert_type": "string", "timestamp": "string"},
                              low_memory=False, chunksize=chunk_size):
            yield df
    else:
        yield pd.read_csv(path,
                          dtype={"ticker": "string", "alert_type": "string", "timestamp": "string"},
                          low_memory=False)

def _apply_schema(record: Dict[str, Any], schema: Dict[str, Any]) -> None:
    if not schema:
        return
    try:
        validate(instance=record, schema=schema)
    except ValidationError as e:
        if STRICT_SCHEMA:
            raise
        else:
            logger.warning(f"Schema validation failed: {e.message}")

def _gate_ai_indices(df: pd.DataFrame) -> List[int]:
    """Return subset indices that warrant AI evaluation under budget limits."""
    # lower scores are prioritized for AI
    sorted_idx = list(df.sort_values("quality_score").index)
    gated = [i for i in sorted_idx if df.loc[i, "quality_score"] < AI_GATE_THRESHOLD]
    return gated[:AI_MAX_EVALS]

def _evaluate_ai_concurrent(records: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
    if client is None or DRY_RUN or not records:
        return [None] * len(records)
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=AI_MAX_WORKERS) as ex:
        futures = {ex.submit(_ai_eval_once, rec): idx for idx, rec in enumerate(records)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                log_telemetry({"event": "ai_eval_exception", "error": str(e)})
                results[idx] = None
    return results

def score_dataset(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    in_path = Path(csv_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing dataset: {in_path}")

    schema = AI_CONF.get("schema", {})
    all_frames: List[pd.DataFrame] = []

    for df in _iter_chunks(in_path, CHUNK_SIZE):
        if df.empty:
            continue

        # Normalize columns if missing
        for col in ("alert_type", "ticker", "price", "timestamp", "sentiment"):
            if col not in df.columns:
                df[col] = pd.Series([None] * len(df))

        # Rule scores
        records = df.to_dict(orient="records")
        scored: List[Dict[str, Any]] = []
        for rec in records:
            try:
                _apply_schema(rec, schema)
            except ValidationError as e:
                # If strict, this raises; otherwise it was already warned
                continue

            rule_score, rationale = compute_quality_score(rec)
            rec["quality_score"] = rule_score
            rec["quality_rationale"] = rationale
            scored.append(rec)

            log_telemetry({
                "event": "quality_scored_rule",
                "ticker": rec.get("ticker"),
                "rule_score": rule_score,
            })

        df_scored = pd.DataFrame(scored)
        if df_scored.empty:
            all_frames.append(df_scored)
            continue

        # AI gating + concurrent eval
        idx_for_ai = _gate_ai_indices(df_scored)
        ai_inputs = [df_scored.loc[i].to_dict() for i in idx_for_ai]
        ai_outputs = _evaluate_ai_concurrent(ai_inputs)

        # Merge AI results
        df_scored["ai_quality_score"] = df_scored["quality_score"]
        df_scored["ai_rationale"] = df_scored["quality_rationale"]

        for i, out in zip(idx_for_ai, ai_outputs):
            if out:
                df_scored.at[i, "ai_quality_score"] = out.get("quality_score", df_scored.at[i, "quality_score"])
                df_scored.at[i, "ai_rationale"] = out.get("rationale", df_scored.at[i, "quality_rationale"])

        df_scored["final_quality_score"] = df_scored[["quality_score", "ai_quality_score"]].max(axis=1)
        all_frames.append(df_scored)

    scored_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    # Outputs
    if output_path is None:
        ts_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        output_path = str(REPORT_DIR / f"quality_report_{ts_suffix}.csv")

    if not scored_df.empty:
        scored_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f" Quality report saved → {output_path}")

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": int(len(scored_df)),
            "mean_rule_score": round(float(scored_df["quality_score"].mean()), 3),
            "mean_ai_score": round(float(scored_df["ai_quality_score"].mean()), 3),
            "mean_final_score": round(float(scored_df["final_quality_score"].mean()), 3),
            "ai_evals_used": int((scored_df["ai_quality_score"] != scored_df["quality_score"]).sum()),
            "ai_gate_threshold": AI_GATE_THRESHOLD,
            "ai_max_evals": AI_MAX_EVALS,
        }
        summary_path = REPORT_DIR / "quality_summary.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(summary_path)
        logger.info(f"📊 Summary report saved → {summary_path}")

        logger.info("Avg Quality → Rule: %.2f, AI: %.2f, Final: %.2f",
                    summary["mean_rule_score"], summary["mean_ai_score"], summary["mean_final_score"])
    else:
        logger.warning("No rows scored; empty dataset or all rows failed strict schema.")

    return scored_df

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        INPUT_CSV = os.getenv("CLEAN_CSV_PATH", str(ROOT / "data" / "clean" / "trade_signals.csv"))
        score_dataset(INPUT_CSV)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
