import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless safe (no display needed)
import matplotlib.pyplot as plt


def plot_all_charts(df: pd.DataFrame, output_dir: str = "data/charts") -> list:
    """
    Generate and save multiple analytics charts for trade signal data.

    Charts generated:
        1. ticker_frequency.png        – count of alerts by ticker
        2. signal_strength_dist.png    – distribution of signal strengths (A+, Pullback, Exit, Neutral)
        3. call_put_ratio.png          – comparison of CALL vs PUT counts

    Args:
        df (pd.DataFrame): Cleaned and normalized DataFrame.
        output_dir (str): Directory where charts are saved.

    Returns:
        list[str]: List of saved chart file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # --- Ticker Frequency ---
    if not df.empty and "ticker" in df.columns:
        counts = df["ticker"].value_counts().sort_values(ascending=False)
        if not counts.empty:
            file_path = os.path.join(output_dir, "ticker_frequency.png")
            plt.figure(figsize=(10, 6))
            counts.plot(kind="bar", color="steelblue")
            plt.title("Alert Frequency by Ticker")
            plt.xlabel("Ticker")
            plt.ylabel("Count")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(file_path, dpi=150)
            plt.close()
            print(f"[INFO] Chart saved → {file_path}")
            saved_paths.append(file_path)
        else:
            print("[WARN] No valid ticker entries found.")

    # --- Signal Strength Distribution ---
    if "signal_strength" in df.columns:
        strength_counts = df["signal_strength"].value_counts()
        if not strength_counts.empty:
            file_path = os.path.join(output_dir, "signal_strength_dist.png")
            plt.figure(figsize=(8, 6))
            colors = {
                "A+": "#FF4500",
                "Pullback": "#32CD32",
                "Exit": "#8B0000",
                "Neutral": "#708090"
            }
            strength_counts.plot(kind="bar", color=[colors.get(s, "#4682B4") for s in strength_counts.index])
            plt.title("Signal Strength Distribution")
            plt.xlabel("Signal Strength")
            plt.ylabel("Count")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(file_path, dpi=150)
            plt.close()
            print(f"[INFO] Chart saved → {file_path}")
            saved_paths.append(file_path)

    # --- CALL vs PUT Ratio ---
    if {"is_call", "is_put"}.issubset(df.columns):
        call_count = df["is_call"].sum()
        put_count = df["is_put"].sum()
        if call_count + put_count > 0:
            file_path = os.path.join(output_dir, "call_put_ratio.png")
            plt.figure(figsize=(6, 6))
            plt.pie(
                [call_count, put_count],
                labels=["CALL Alerts", "PUT Alerts"],
                autopct="%1.1f%%",
                startangle=140,
                colors=["#1E90FF", "#FFD700"],
                wedgeprops={"edgecolor": "white", "linewidth": 1.5}
            )
            plt.title("CALL vs PUT Alert Ratio")
            plt.tight_layout()
            plt.savefig(file_path, dpi=150)
            plt.close()
            print(f"[INFO] Chart saved → {file_path}")
            saved_paths.append(file_path)

    if not saved_paths:
        print("[WARN] No charts were generated (insufficient data).")

    return saved_paths


def plot_ticker_counts(df: pd.DataFrame, output_dir: str = "data/charts") -> str:
    """
    Legacy single-chart helper retained for backward compatibility.
    """
    paths = plot_all_charts(df, output_dir)
    return paths[0] if paths else None
