"""
fetch_messages.py
Stable v1.1.0 — Safe Discord Message Archiver
Author: @drashadm
"""

import os
import asyncio
import discord
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
import time

# === Load Environment ===
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_IDS = [int(x.strip()) for x in os.getenv("CHANNEL_IDS", "").split(",") if x.strip()]

# === Discord Intents ===
intents = discord.Intents.default()
intents.message_content = True  # Required for reading messages
intents.guilds = True

client = discord.Client(intents=intents)
messages = []


# === Safe CSV Writer ===
def save_partial(data, suffix="partial"):
    os.makedirs("data/raw", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"data/raw/discord_messages_{suffix}_{ts}.csv"
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved partial {len(data)} messages → {path}")
    return path


async def save_final(data):
    os.makedirs("data/raw", exist_ok=True)
    path = "data/raw/discord_messages.csv"
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved final {len(data)} messages → {path}")
    return path


# === Message Fetch Logic (rate-limit safe) ===
async def fetch_channel_messages(channel):
    print(f"[INFO] Fetching messages from #{channel.name}")

    batch_size = 100
    last_message = None
    downloaded = 0

    while True:

                messages.append({
                    "channel": channel.name,
                    "author": msg.author.name,
                    "is_bot": msg.author.bot,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                })

        # Save once all messages are collected
        await save_to_csv(messages)

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        await client.close()


# === Async CSV writer ===
async def save_to_csv(data):
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(data)
    path = "data/raw/discord_messages.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(data)} messages → {path}")


# === Entry point ===
async def main():
    if not TOKEN:
        raise ValueError("Missing DISCORD_BOT_TOKEN in environment.")
    if not CHANNEL_IDS:
        raise ValueError("Missing CHANNEL_IDS in environment.")
    await client.start(TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
