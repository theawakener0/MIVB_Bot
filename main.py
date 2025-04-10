import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import asyncio
import logging

# Load environment variables (for API keys)
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN_NEW")
MOD_CHANNEL_ID = int(os.getenv("MOD_REPORT_CHANNEL_ID")) if os.getenv("MOD_REPORT_CHANNEL_ID") else None

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True # Required for potential future text commands
intents.voice_states = True    # Required for voice channel operations

bot = commands.Bot(command_prefix="!", intents=intents, name="أبو إسماعيل الجدا جدا جامد")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Cog Loading ---
async def load_extensions():
    """Loads all cogs."""
    initial_extensions = ['recorder_cog']  # Changed to point to the new file
    for extension in initial_extensions:
        try:
            await bot.load_extension(extension)
            logger.info(f"Successfully loaded extension: {extension}")
        except Exception as e:
            logger.error(f"Failed to load extension {extension}.", exc_info=True)

@bot.event
async def on_ready():
    """Event triggered when the bot is ready."""
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info('------')
    
    # Load extensions first
    await load_extensions()
    
    # Then sync commands
    try:
        # Sync commands globally
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s) globally")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
    
    # Set bot activity
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="Voice Channels"))

# --- Main Execution ---
async def main():
    if not DISCORD_TOKEN:
        logger.error("Discord bot token not found. Please set DISCORD_BOT_TOKEN in your .env file.")
        return
    if not MOD_CHANNEL_ID:
        logger.warning("MOD_CHANNEL_ID not set. Recordings won't be sent to a specific channel.")

    async with bot:
        await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    # Ensure the 'recordings' directory exists
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    asyncio.run(main())