"""
summarizer.py
Executive Summary Generator for Discord Trade Signal Intelligence
Author: @drashadm (DeAndrai Mullen)
Version: 2.0 (Budget-Safe + OpenAI v1 Upgrade + Production Hardened)
"""

from __future__ import annotations
import os, json, logging, re, time, threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import yaml
import pandas as pd
import requests

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
DATA_DIR = ROOT / "data"
CLEAN_CSV = DATA_DIR / "clean" / "trade_signals.csv"
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = ROOT / "logs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open(CONF_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SUMM_CONF = CONFIG.get("summarizer", {})
GOV_CONF = CONFIG.get("governance", {})
STYLE_GUIDE = CONFIG.get("style_guide", {})

TELEMETRY_LOG = Path(GOV_CONF.get("telemetry_log", "./metrics/usage.json")).resolve()
TELEMETRY_LOG.parent.mkdir(parents=True, exist_ok=True)

MODEL = SUMM_CONF.get("model", "gpt-4.1-mini")
TEMPERATURE = float(SUMM_CONF.get("temperature", 0.4))
MAX_TOKENS = int(SUMM_CONF.get("max_tokens", 700))
PROMPT = SUMM_CONF.get("prompt", "Summarize this dataset.")
TONE = STYLE_GUIDE.get("tone", "analytical, concise, trustworthy")

DISCORD_WEBHOOK = os.getenv("DISCORD_ALERT_WEBHOOK", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUMMARY_GATE_MIN = int(os.getenv("SUMMARY_GATE_MIN_ALERTS", "20"))

_lock = threading.Lock()

# =========================================================
# FinOps / Cost Controls
# =========================================================
CHARS_PER_TOKEN = 4
INPUT_COST = 0.40      # 4.1-mini input per 1M
OUTPUT_COST = 1.60     # 4.1-mini output per 1M
MAX_COST_PER_CALL = 0.003     # 0.3 cents
MAX_TOTAL_SUMMARY_COST = 0.50 # 50 cents hard cap for summarizer

# =========================================================
# Logging
# =========================================================
logger = logging.getLogger("summarizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "summarizer.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

# =========================================================
# OpenAI Client (Unified)
# =========================================================
client = OpenAI(api_key=OPENAI_KEY) if (OPENAI_AVAILABLE and OPENAI_KEY) else None


# =========================================================
# Telemetry
# =========================================================
def _rotate_telemetry_if_needed() -> None:
    try:
        if TELEMETRY_LOG.exists() and TELEMETRY_LOG.stat().st_size > 5_000_000:
            rotated = TELEMETRY_LOG.with_name(f"{TELEMETRY_LOG.stem}.{datetime.utcnow():%Y%m%d%H%M}.json")
            TELEMETRY_LOG.rename(rotated)
            logger.info(f"Telemetry rotated: {rotated}")
    except Exception as e:
        logger.warning(f"Telemetry rotation failed: {e}")

def _log_telemetry(entry: Dict[str, Any]):
    entry.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "summarizer",
        "schema_version": CONFIG.get("meta", {}).get("version", "unknown"),
    })
    try:
        _rotate_telemetry_if_needed()
        with _lock:
            if TELEMETRY_LOG.exists():
                try:
                    data = json.loads(TELEMETRY_LOG.read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        data = [data]
                except Exception:
                    data = []
            else:
                data = []
            data.append(entry)
            tmp = TELEMETRY_LOG.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(TELEMETRY_LOG)
    except Exception as e:
        logger.warning(f"Telemetry write failed: {e}")


# =========================================================
# Aggregation
# =========================================================
def _aggregate(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total_alerts": 0,
            "alert_types": {},
            "top_tickers": [],
            "avg_price": None,
            "date_min": None,
            "date_max": None
        }
    df = df.copy()
    df["ticker"] = df.get("ticker", "").astype(str).str.upper().str.strip()
    df["alert_type"] = df.get("alert_type", "").astype(str).str.upper().str.strip()

    out = {
        "total_alerts": int(len(df)),
        "alert_types": df["alert_type"].value_counts().head(10).to_dict(),
        "avg_price": float(df["price"].astype(float).mean()),
    }

    top = df["ticker"].value_counts().head(10)
    out["top_tickers"] = [(k, int(v)) for k, v in top.items()]

    for col in ("date", "timestamp"):
        if col in df.columns:
            try:
                dmin, dmax = pd.to_datetime(df[col]).min(), pd.to_datetime(df[col]).max()
                out["date_min"] = dmin.strftime("%Y-%m-%d") if pd.notnull(dmin) else None
                out["date_max"] = dmax.strftime("%Y-%m-%d") if pd.notnull(dmax) else None
                break
            except Exception:
                pass
    return out


# =========================================================
# Utility: Extract Markdown Section Safely
# =========================================================
def _extract_fenced_md(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = re.split(r"```(?:md|markdown)?", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].split("```")[0].strip()
    return text.strip()


# =========================================================
# Cost Estimation
# =========================================================
def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * INPUT_COST +
        (output_tokens / 1_000_000) * OUTPUT_COST
    )


# =========================================================
# LLM Summarizer (OpenAI Responses API)
# =========================================================
def _summarize_with_llm(agg: Dict[str, Any], qual: Optional[Dict[str, Any]]) -> Optional[str]:
    if client is None:
        return None
    if agg.get("total_alerts", 0) < SUMMARY_GATE_MIN:
        return None

    payload = {
        "aggregates": agg,
        "quality_summary": qual or {},
        "style_tone": TONE,
    }

    prompt = (
        f"{PROMPT}\n\n"
        "Return well-formatted Markdown. Be analytical and concise.\n\n"
        f"JSON INPUT:\n{json.dumps(payload, indent=2)}"
    )

    start = time.time()
    try:
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": "You are a senior data summarizer. Output Markdown only."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        latency = round(time.time() - start, 3)

        text = resp.output_text.strip()

        # Cost Estimate
        usage = resp.usage or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = _estimate_cost(input_tokens, output_tokens)

        if cost > MAX_COST_PER_CALL:
            logger.warning(
                f"⚠ Summary LLM call cost {cost:.6f} exceeded per-call cap {MAX_COST_PER_CALL:.6f}"
            )

        _log_telemetry({
            "event": "llm_summary",
            "model": MODEL,
            "latency_sec": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "est_cost_usd": round(cost, 6),
        })

        return _extract_fenced_md(text)

    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        _log_telemetry({"event": "llm_summary_failed", "error": str(e)})
        return None


# =========================================================
# Deterministic 0-Cost Summary (fallback)
# =========================================================
def _deterministic_summary_md(agg: Dict[str, Any], qual: Optional[Dict[str, Any]]) -> str:
    lines = [
        "# Trade Signal Summary",
        "",
        f"- **Total alerts**: {agg.get('total_alerts', 0)}",
    ]
    if agg.get("date_min") or agg.get("date_max"):
        lines.append(f"- **Date range**: {agg.get('date_min','N/A')} → {agg.get('date_max','N/A')}")
    if agg.get("avg_price") is not None:
        lines.append(f"- **Average price**: ${agg['avg_price']:,.2f}")

    lines += ["", "## Type Breakdown"]
    for k, v in (agg.get("alert_types") or {}).items():
        lines.append(f"- **{k}**: {v}")

    lines += ["", "## Top Tickers"]
    for t, c in agg.get("top_tickers", []):
        lines.append(f"- **{t}**: {c}")

    if qual:
        lines += ["", "## Data Quality Snapshot"]
        for k in ("total_records", "mean_rule_score", "mean_ai_score", "mean_final_score"):
            lines.append(f"- **{k.replace('_',' ').title()}**: {qual.get(k,'N/A')}")

    lines.append("")
    lines.append("_Auto-generated report — analytical, concise, trustworthy._")
    return "\n".join(lines)


# =========================================================
# Save Reports & Optional Discord Notification
# =========================================================
def _save_reports(md_text: str, plain_text: str, agg_json: Dict[str, Any]) -> Dict[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    paths = {
        "md": REPORT_DIR / f"summary_{ts}.md",
        "txt": REPORT_DIR / f"summary_{ts}.txt",
        "json": REPORT_DIR / f"summary_{ts}.json",
    }
    paths["md"].write_text(md_text, encoding="utf-8")
    paths["txt"].write_text(plain_text, encoding="utf-8")
    paths["json"].write_text(json.dumps(agg_json, indent=2), encoding="utf-8")

    logger.info(f"Summary saved → {', '.join(p.name for p in paths.values())}")
    _log_telemetry({"event": "summary_saved", "files": [p.name for p in paths.values()]})

    return {k: str(v) for k, v in paths.items()}


def _maybe_post_discord(message: str, retries: int = 2):
    if not DISCORD_WEBHOOK:
        return
    for attempt in range(retries + 1):
        try:
            resp = requests.post(DISCORD_WEBHOOK, json={"content": message[:1950]}, timeout=8)
            if resp.status_code < 300:
                logger.info("Summary posted to Discord.")
                _log_telemetry({"event": "discord_posted"})
                return
            logger.warning(f"Discord webhook failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Discord post failed: {e}")
        time.sleep(1.5 * attempt)


# =========================================================
# Public API
# =========================================================
def generate_summary(clean_csv_path: str | Path = CLEAN_CSV, use_llm: bool = True) -> Dict[str, str]:
    clean_csv_path = Path(clean_csv_path)
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {clean_csv_path}")

    df = pd.read_csv(clean_csv_path)
    agg = _aggregate(df)

    # Load quality summary if exists
    qual_path = REPORT_DIR / "quality_summary.json"
    qual = None
    if qual_path.exists():
        try:
            qual = json.loads(qual_path.read_text())
        except Exception:
            qual = None

    # LLM summary with cost-safety
    md_text = None
    if use_llm:
        md_text = _summarize_with_llm(agg, qual)

    # Fallback summary (0-cost)
    if not md_text:
        md_text = _deterministic_summary_md(agg, qual)

    # Save files
    plain = re.sub(r"^#+\s*", "", md_text, flags=re.MULTILINE).replace("**", "")
    paths = _save_reports(md_text, plain, {"aggregates": agg, "quality": qual or {}})

    # Post small text preview to Discord
    preview = (
        f"**Trade Summary**\n"
        f"Alerts: {agg.get('total_alerts',0)} | "
        f"Top: {', '.join([t for t, _ in agg.get('top_tickers', [])][:5])}"
    )
    _maybe_post_discord(preview)

    return paths


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    try:
        USE_LLM = bool(int(os.getenv("SUMMARY_USE_LLM", "1")))
        generate_summary(CLEAN_CSV, use_llm=USE_LLM)
    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        raise
