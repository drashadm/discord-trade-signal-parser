import os
import asyncio
import discord
import pandas as pd
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_IDS = [int(x.strip()) for x in os.getenv("CHANNEL_IDS", "").split(",") if x.strip()]

# === Configure Discord client ===
intents = discord.Intents.none()
intents.guilds = True
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

messages = []  # async-safe in memory

# === Event: Ready ===
@client.event
async def on_ready():
    print(f"[INFO] Logged in as {client.user}")

    try:
        for cid in CHANNEL_IDS:
            channel = await client.fetch_channel(cid)
            print(f"[INFO] Fetching messages from #{channel.name}")

            # Fetch complete history (Discord rate-limit safe)
            async for msg in channel.history(limit=None, oldest_first=True):
                # ✅ include bot messages (important for TradingView alerts)
                # Skip only system messages with no content
                if not msg.content:
                    continue

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
