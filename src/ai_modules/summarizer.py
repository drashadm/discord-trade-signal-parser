"""
summarizer.py
Executive Summary Generator for Discord Trade Signal Intelligence
Author: @drashadm (DeAndrai Mullen)
Version: 1.4.1 (Production-Ready Enhancements)
"""

from __future__ import annotations
import os, json, logging, re, time, threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import yaml
import pandas as pd
import requests  # ensure listed in requirements

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ------------------ Paths & Config ------------------
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

MODEL = SUMM_CONF.get("model", "gpt-4o-mini")
TEMPERATURE = float(SUMM_CONF.get("temperature", 0.4))
MAX_TOKENS = int(SUMM_CONF.get("max_tokens", 800))
PROMPT = SUMM_CONF.get("prompt", "Summarize this dataset.")
TONE = STYLE_GUIDE.get("tone", "analytical, concise, trustworthy")

DISCORD_WEBHOOK = os.getenv("DISCORD_ALERT_WEBHOOK", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUMMARY_GATE_MIN = int(os.getenv("SUMMARY_GATE_MIN_ALERTS", "20"))  # skip LLM for tiny sets

_lock = threading.Lock()

client = None
if OPENAI_KEY and OpenAI:
    try:
        client = OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        logging.warning(f"OpenAI client init failed: {e}")

# ------------------ Logging ------------------
logger = logging.getLogger("summarizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "summarizer.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

# ------------------ Helpers ------------------
def _rotate_telemetry_if_needed() -> None:
    try:
        if TELEMETRY_LOG.exists() and TELEMETRY_LOG.stat().st_size > 5_000_000:
            rotated = TELEMETRY_LOG.with_name(f"{TELEMETRY_LOG.stem}.{datetime.utcnow():%Y%m%d%H%M}.json")
            TELEMETRY_LOG.rename(rotated)
            logger.info(f"Telemetry rotated: {rotated}")
    except Exception as e:
        logger.warning(f"Telemetry rotation failed: {e}")

def _log_telemetry(entry: Dict[str, Any]) -> None:
    entry.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "summarizer",
        "schema_version": str(CONFIG.get("meta", {}).get("version", "unknown")),
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

def _aggregate(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return dict(total_alerts=0, alert_types={}, top_tickers=[], avg_price=None, date_min=None, date_max=None)
    df = df.copy()
    df["ticker"] = df.get("ticker", pd.Series([""]*len(df))).astype(str).str.upper().str.strip()
    df["alert_type"] = df.get("alert_type", pd.Series([""]*len(df))).astype(str).str.upper().str.strip()
    out = {
        "total_alerts": int(len(df)),
        "alert_types": df["alert_type"].value_counts(dropna=False).head(10).to_dict(),
        "avg_price": float(df["price"].astype(float).mean()) if "price" in df.columns else None,
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

def _extract_fenced_md(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = re.split(r"```(?:md|markdown)?", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].split("```")[0].strip()
    return text.strip()

# ------------------ LLM Summarization ------------------
def _summarize_with_llm(agg: Dict[str, Any], qual: Optional[Dict[str, Any]]) -> Optional[str]:
    if client is None or agg.get("total_alerts", 0) < SUMMARY_GATE_MIN:
        return None
    try:
        payload = {"aggregates": agg, "quality_summary": qual or {}, "style_tone": TONE}
        start = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": "You are an analytical, trustworthy data summarizer. Output Markdown."},
                {"role": "user", "content": f"{PROMPT}\n\nJSON INPUT:\n{json.dumps(payload, indent=2)}"},
            ],
            timeout=30,
        )
        latency = round(time.time() - start, 3)
        text = _extract_fenced_md((resp.choices[0].message.content or "").strip())
        usage = getattr(resp, "usage", None)
        _log_telemetry({
            "event": "llm_summary",
            "model": MODEL,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "latency_sec": latency,
        })
        return text
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        _log_telemetry({"event": "llm_summary_failed", "error": str(e), "model": MODEL})
        return None

# ------------------ Deterministic Summary ------------------
def _deterministic_summary_md(agg: Dict[str, Any], qual: Optional[Dict[str, Any]]) -> str:
    lines = [
        "#  Trade Signal Summary",
        "",
        f"- **Total alerts**: {agg.get('total_alerts', 0)}",
    ]
    if agg.get("date_min") or agg.get("date_max"):
        lines.append(f"- **Date range**: {agg.get('date_min','N/A')} → {agg.get('date_max','N/A')}")
    if agg.get("avg_price") is not None:
        lines.append(f"- **Average price**: ${agg['avg_price']:,.2f}")
    lines += ["", "## Type Breakdown"]
    for k, v in (agg.get("alert_types") or {}).items():
        lines.append(f"- **{k or 'UNKNOWN'}**: {v}")
    lines += ["", "## Top Tickers"]
    for t, c in (agg.get("top_tickers") or []):
        lines.append(f"- **{t}**: {c}")
    if qual:
        lines += ["", "## Data Quality Snapshot"]
        for k in ("total_records", "mean_rule_score", "mean_ai_score", "mean_final_score"):
            lines.append(f"- **{k.replace('_',' ').title()}**: {qual.get(k,'N/A')}")
    lines += ["", "_Auto-generated report — style: analytical, concise, trustworthy._"]
    return "\n".join(lines)

# ------------------ I/O & Notifications ------------------
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
    logger.info(f" Summary saved → {', '.join(p.name for p in paths.values())}")
    _log_telemetry({"event": "summary_saved", "files": [p.name for p in paths.values()]})
    return {k: str(v) for k, v in paths.items()}

def _maybe_post_discord(message: str, retries: int = 2) -> None:
    if not DISCORD_WEBHOOK:
        return
    for attempt in range(retries + 1):
        try:
            resp = requests.post(DISCORD_WEBHOOK, json={"content": message[:1950]}, timeout=8)
            if resp.status_code < 300:
                logger.info(" Summary posted to Discord.")
                _log_telemetry({"event": "discord_posted"})
                return
            logger.warning(f"Discord webhook failed: {resp.status_code} {resp.text}")
        except requests.Timeout:
            logger.warning("Discord post timed out.")
        except Exception as e:
            logger.warning(f"Discord post failed: {e}")
        time.sleep(1.5 * attempt)

# ------------------ Public API ------------------
def generate_summary(clean_csv_path: str | Path = CLEAN_CSV, use_llm: bool = True) -> Dict[str, str]:
    clean_csv_path = Path(clean_csv_path)
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {clean_csv_path}")

    df = pd.read_csv(clean_csv_path)
    agg = _aggregate(df)
    qual = None
    qpath = REPORT_DIR / "quality_summary.json"
    if qpath.exists():
        try:
            qual = json.loads(qpath.read_text(encoding="utf-8"))
        except Exception:
            qual = None

    md_text = _summarize_with_llm(agg, qual) if use_llm else None
    if not md_text:
        md_text = _deterministic_summary_md(agg, qual)

    plain = re.sub(r"^#+\s*", "", md_text, flags=re.MULTILINE).replace("**", "")
    paths = _save_reports(md_text, plain, {"aggregates": agg, "quality": qual or {}})

    preview = f"**Trade Signal Summary**\nTotal alerts: {agg.get('total_alerts',0)}\nTop tickers: " \
              + ", ".join([t for t, _ in agg.get("top_tickers", [])][:5])
    _maybe_post_discord(preview)
    return paths

# ------------------ CLI ------------------
if __name__ == "__main__":
    try:
        USE_LLM = bool(int(os.getenv("SUMMARY_USE_LLM", "1")))
        generate_summary(CLEAN_CSV, use_llm=USE_LLM)
    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        raise
