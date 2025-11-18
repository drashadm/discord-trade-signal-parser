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
async def save_messages(data, filename="discord_messages.csv"):
    """Save messages to CSV file"""
    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/{filename}"
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(data)} messages → {path}")
    return path


# === Message Fetch Logic (rate-limit safe) ===
async def fetch_channel_messages(channel):
    """Fetch all messages from a channel"""
    print(f"[INFO] Fetching messages from #{channel.name}...")
    
    batch_messages = []
    async for msg in channel.history(limit=None, oldest_first=True):
        # Skip system messages with no content
        if not msg.content:
            continue

        batch_messages.append({
            "channel": channel.name,
            "author": msg.author.name,
            "is_bot": msg.author.bot,
            "content": msg.content,
            "created_at": msg.created_at.isoformat(),
        })
        
        # Save in batches to avoid memory issues
        if len(batch_messages) % 500 == 0:
            print(f"[INFO] Fetched {len(batch_messages)} messages from #{channel.name}...")

    print(f"[INFO] Finished fetching {len(batch_messages)} messages from #{channel.name}")
    return batch_messages


# === Event: Ready ===
@client.event
async def on_ready():
    print(f"[INFO] Logged in as {client.user}")
    
    all_messages = []
    
    try:
        for cid in CHANNEL_IDS:
            try:
                channel = await client.fetch_channel(cid)
                channel_messages = await fetch_channel_messages(channel)
                all_messages.extend(channel_messages)
            except discord.NotFound:
                print(f"[ERROR] Channel {cid} not found")
            except discord.Forbidden:
                print(f"[ERROR] No permission to access channel {cid}")
            except Exception as e:
                print(f"[ERROR] Failed to fetch channel {cid}: {e}")

        # Save all messages to CSV
        if all_messages:
            await save_messages(all_messages)
            print(f"[SUCCESS] Fetched total {len(all_messages)} messages from all channels")
        else:
            print("[WARN] No messages found")

    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")

    finally:
        await client.close()


# === Entry point ===
async def main():
    if not TOKEN:
        raise ValueError("Missing DISCORD_BOT_TOKEN in environment.")
    if not CHANNEL_IDS:
        raise ValueError("Missing CHANNEL_IDS in environment. Format: ID1,ID2,ID3")
    
    print(f"[INFO] Fetching from {len(CHANNEL_IDS)} channels...")
    await client.start(TOKEN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")

