import discord
from discord.ext import commands
import os
import datetime
import asyncio
from dotenv import load_dotenv

# LangChain specific imports
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# Add this utility function near the top of your file
# Replace the existing safe_print with this enhanced version
def safe_print(*args, **kwargs):
    """Print function that handles Unicode characters safely on Windows."""
    import sys
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fall back to ASCII with replacement characters for non-ASCII
        text = ' '.join(str(arg) for arg in args)
        encoded = text.encode('ascii', 'replace').decode('ascii')
        print(encoded, **kwargs)
    except:
        # If all else fails, write directly to stderr
        text = ' '.join(str(arg) for arg in args)
        sys.stderr.write(text + '\n')

# --- Configuration ---
load_dotenv() # Load variables from .env file

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # <-- New: Gemini API Key
SOURCE_CHANNEL_ID_STR = os.getenv("SOURCE_CHANNEL_ID")
TARGET_CHANNEL_ID_STR = os.getenv("MOD_REPORT_CHANNEL_ID")

# Validate and convert IDs
try:
    SOURCE_CHANNEL_ID = int(SOURCE_CHANNEL_ID_STR) if SOURCE_CHANNEL_ID_STR else None
    TARGET_CHANNEL_ID = int(TARGET_CHANNEL_ID_STR) if TARGET_CHANNEL_ID_STR else None
except (ValueError, TypeError):
    print("Error: Channel IDs in .env must be valid numbers.")
    exit()

# New command name to differentiate
REPORT_COMMAND_NAME = "summarize_day"
MOD_ROLE_NAME = "المطورين"  # Make sure this matches your server's moderator role name

# --- Basic Validation ---
if not BOT_TOKEN:
    print("Error: DISCORD_BOT_TOKEN not found in environment variables.")
    exit()
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    exit()
if not SOURCE_CHANNEL_ID or not TARGET_CHANNEL_ID:
    print("Error: Source or target channel ID not found or invalid in environment variables.")
    exit()

# --- LangChain Setup ---
try:
    # Use Gemini 1.5 Pro (or adjust model name if you have 2.5 Pro access: "gemini-2.5-pro-latest")
    llm = GoogleGenerativeAI(
                            model="gemini-2.5-pro-exp-03-25",
                            google_api_key=GEMINI_API_KEY,
                            max_output_tokens=65536
                            )

    print("Gemini LLM Initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini LLM: {e}")
    print("Please ensure your GEMINI_API_KEY is correct and you have the necessary permissions.")
    exit()

# --- Discord Bot Setup ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True # Ensure this Privileged Intent is enabled!
intents.guilds = True

# Replace discord.Bot with commands.Bot
bot = commands.Bot(command_prefix='!', intents=intents)

# --- Bot Events ---
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print(f'Monitoring Channel ID: {SOURCE_CHANNEL_ID}')
    print(f'Reporting Channel ID: {TARGET_CHANNEL_ID}')
    print(f'Using LLM Model: {llm.model}')
    print('------')
    
    # Register the slash command
    try:
        command = await bot.tree.sync()
        print(f"Slash commands synced: {len(command)}")
    except Exception as e:
        print(f"Error syncing commands: {e}")

# --- Helper: Compile Messages for LLM ---
def compile_messages_for_llm(messages):
    """Formats a list of discord messages into a single string for the LLM."""
    compiled_text = ""
    for msg in messages:
         # Exclude bots if desired
         if msg.author.bot:
            continue
         timestamp_str = msg.created_at.astimezone().strftime('%H:%M:%S')
         display_name = msg.author.display_name.replace(":", "") # Basic sanitation
         content = msg.content
         compiled_text += f"{timestamp_str} - {display_name}: {content}\n"
    return compiled_text.strip()

# --- Helper: Send Report in Chunks ---
async def send_report_chunks(target_channel: discord.TextChannel, report_content: str, header: str, footer: str = ""):
    """Sends a potentially long report by splitting it into chunks."""
    max_chars = 2000 # Discord's max message length
    current_chunk = header
    lines = report_content.splitlines()

    if not lines and not report_content: # Handle case where LLM returns empty summary
         await target_channel.send(header + "\n*The LLM analysis returned no content.*\n" + footer)
         return

    for line in lines:
        # Check if adding the next line (plus a newline) exceeds the limit
        # Account for header/footer length in the first/last chunk calculation if needed
        if len(current_chunk) + len(line) + 1 > max_chars - len(footer) - 10: # Add buffer for footer/markdown
            await target_channel.send(current_chunk + footer)
            current_chunk = line # Start new chunk
        else:
            current_chunk += "\n" + line

    # Send the last chunk
    await target_channel.send(current_chunk + footer)


# --- Slash Command ---
# For the daily summary command
@bot.tree.command(
    name=REPORT_COMMAND_NAME,
    description="Summarizes today's messages from the specified channel using an LLM."
)
@commands.has_role(MOD_ROLE_NAME)  # Fixed decorator
async def summarize_daily_report_llm(interaction: discord.Interaction):
    """Fetches messages, sends them to Gemini for summary, and posts the result."""
    # Acknowledge command immediately, indicating processing
    await interaction.response.defer(ephemeral=True)

    try:
        source_channel = bot.get_channel(SOURCE_CHANNEL_ID)
        target_channel = bot.get_channel(TARGET_CHANNEL_ID)

        # --- Validate Channels ---
        if not source_channel or not isinstance(source_channel, discord.TextChannel):
            await interaction.followup.send(f"Error: Could not find or access the source text channel (ID: {SOURCE_CHANNEL_ID}).", ephemeral=True)
            return
        if not target_channel or not isinstance(target_channel, discord.TextChannel):
            await interaction.followup.send(f"Error: Could not find or access the target report text channel (ID: {TARGET_CHANNEL_ID}).", ephemeral=True)
            return

        # --- Calculate Time Range for Today ---
        now = datetime.datetime.now() # Bot server's local time
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + datetime.timedelta(days=1)

        # --- Fetch Messages ---
        safe_print(f"Fetching messages from channel #{source_channel.name} for {now.strftime('%Y-%m-%d')}")
            
        messages_today = []
        message_count = 0
        async for message in source_channel.history(limit=None, after=start_of_day, before=end_of_day, oldest_first=True):
            messages_today.append(message)
            if not message.author.bot: # Count non-bot messages for reporting
                message_count += 1

        safe_print(f"Fetched {len(messages_today)} total messages ({message_count} non-bot).")

        if message_count == 0:
            await interaction.followup.send(f"No non-bot messages found in {source_channel.mention} for today.", ephemeral=True)
            await target_channel.send(f"## Daily Summary for {source_channel.mention} - {now.strftime('%Y-%m-%d')}\n\n*No messages found today.*")
            return

        # --- Prepare Data for LLM ---
        compiled_message_text = compile_messages_for_llm(messages_today)

        MAX_LLM_INPUT_LEN = 1000000 
        if len(compiled_message_text) > MAX_LLM_INPUT_LEN:
            safe_print(f"Warning: Compiled text ({len(compiled_message_text)} chars) is very long, truncating for LLM.")
            compiled_message_text = compiled_message_text[-MAX_LLM_INPUT_LEN:] # Keep the most recent part

        # --- Define Prompt for LLM ---
        system_prompt = (
            "You are a comprehensive assistant analyzing a day's worth of Discord messages. "
            "Provide an in-depth report covering:\n"
            "- Detailed analysis of all major discussion topics (minimum 3 paragraphs per topic)\n"
            "- Complete list of decisions/agreements with full context\n"
            "- Thorough examination of questions asked (including follow-up analysis)\n"
            "- Comprehensive list of action items with suggested next steps\n"
            "- Extended sentiment analysis with supporting examples\n"
            "- Minimum 10 detailed observations about the day's conversations\n"
            "Your report should be extremely thorough, typically 1000+ words."
        )
        human_prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",
                "Please analyze the following Discord messages from the channel '{channel_name}' for the date {date}. "
                "Provide a comprehensive report covering:\n"
                "- In-depth analysis of all discussion topics\n"
                "- Complete behavioral patterns with message examples\n"
                "- Detailed examination of communication style\n"
                "- Thorough analysis of potential goals/interests\n"
                "- Comprehensive list of key takeaways with recommendations\n"
                "- Minimum 15 detailed observations about the user's activity\n"
                "- Suggestions for moderator follow-up where appropriate\n\n"
                "Here are the messages:\n"
                "------\n"
                "{message_log}\n"
                "------"
            )
        ])

        # --- Invoke LLM using LangChain ---
        safe_print("Sending messages to Gemini for analysis...")
        chain = human_prompt_template | llm | StrOutputParser()

        try:
            llm_response = await chain.ainvoke({
                "channel_name": source_channel.name,
                "date": now.strftime('%Y-%m-%d'),
                "message_log": compiled_message_text
            })
            safe_print("Received response from Gemini.")
        except Exception as e:
            safe_print(f"Error invoking Gemini LLM: {e}")
            await interaction.followup.send(f"Error: Failed to get analysis from the LLM. Please check logs. Error: {e}", ephemeral=True)
            await target_channel.send(f"## Daily Summary for {source_channel.mention} - {now.strftime('%Y-%m-%d')}\n\n"
                                    f"*Failed to generate summary due to an LLM error: {e}*")
            return

        # --- Format and Send Final Report ---
        report_header = (
            f"## :scroll: COMPREHENSIVE DAILY REPORT - {now.strftime('%Y-%m-%d')}\n"
            f"*In-depth analysis of {message_count} messages from {source_channel.mention}*\n"
            f"--------------------------------------------------\n"
            f"*Generated by {llm.model} | Requested by {interaction.user.mention}*\n\n"
            f"**This detailed report contains {len(llm_response.split())} words of analysis:**\n\n"
        )
        await send_report_chunks(target_channel, llm_response, report_header)
        safe_print(f"LLM summary sent successfully to #{target_channel.name}.")

        await interaction.followup.send(f"LLM-powered summary for {source_channel.mention} has been generated and sent to {target_channel.mention}.", ephemeral=True)

    # --- Error Handling ---
    except discord.Forbidden as e:
        error_message = f"Error: Bot lacks permissions. Check View/History in #{source_channel.name} and Send Messages in #{target_channel.name}. Details: {e}"
        await interaction.followup.send(error_message, ephemeral=True)
        print(f"Permissions Error: {e}")
    except discord.HTTPException as e:
        error_message = f"Error: Discord API issue: {e}"
        await interaction.followup.send(error_message, ephemeral=True)
        print(f"Discord HTTP Exception: {e}")
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}. Check logs."
        await interaction.followup.send(error_message, ephemeral=True)
        print(f"Error: Unexpected error during LLM report generation:")
        import traceback
        traceback.print_exc()


# Add this constant near your other command name constants
USER_SUMMARY_COMMAND_NAME = "user_summary"

# --- Slash Command for User Summary ---
# For the user summary command
@bot.tree.command(
    name=USER_SUMMARY_COMMAND_NAME,
    description="Summarizes recent messages for a specific user across all channels using an LLM."
)
@commands.has_role(MOD_ROLE_NAME)
async def user_activity_summary(
    interaction: discord.Interaction, 
    user: discord.Member
):
    """Fetches recent user messages from all channels, summarizes topics via LLM."""
    try:
        # Only defer if the interaction hasn't been responded to yet
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)
        
        # Get all text channels in the server
        text_channels = [channel for channel in interaction.guild.text_channels 
                        if isinstance(channel, discord.TextChannel)]
        
        if not text_channels:
            await interaction.followup.send("Error: No accessible text channels found in this server.", ephemeral=True)
            return

        limit = 100000  # Set message fetch limit per channel
        user_messages = []
        checked_messages = 0
        
        # Search through all text channels
        for target_channel in text_channels:
            try:
                safe_print(f"Searching #{target_channel.name} for messages from {user.name}...")
                
                async for message in target_channel.history(limit=limit):
                    checked_messages += 1
                    if message.author.id == user.id:
                        user_messages.insert(0, message)  # Keep chronological order
                        
            except discord.Forbidden:
                safe_print(f"Skipping #{target_channel.name} - missing permissions")
                continue
            except Exception as e:
                safe_print(f"Error searching #{target_channel.name}: {e}")
                continue

        safe_print(f"Found {len(user_messages)} messages from {user.name} across {len(text_channels)} channels.")

        if not user_messages:
            await interaction.followup.send(f"No recent messages found from {user.mention} across all channels.", ephemeral=True)
            return

        # --- Prepare Data for LLM ---
        # Use the same helper to format the found messages
        compiled_user_messages = compile_messages_for_llm(user_messages_in_channel)

        # --- Define SAFE Prompt for LLM (User Activity Summary) ---
        # **IMPORTANT**: This prompt focuses ONLY on summarizing message content, NOT user profiling.
        # Modify the system prompt for daily reports (around line 150)
        system_prompt = (
        "You are a comprehensive assistant analyzing a day's worth of Discord messages. "
        "Provide an in-depth report covering:\n"
        "- Detailed analysis of all major discussion topics (minimum 3 paragraphs per topic)\n"
        "- Complete list of decisions/agreements with full context\n"
        "- Thorough examination of questions asked (including follow-up analysis)\n"
        "- Comprehensive list of action items with suggested next steps\n"
        "- Extended sentiment analysis with supporting examples\n"
        "- Minimum 10 detailed observations about the day's conversations\n"
        "Your report should be extremely thorough, typically 1000+ words."
        )
        
        # Modify the user summary prompt (around line 340)
        system_prompt_user = ("""
        You are an exhaustive analyst summarizing a user's Discord activity.
        Provide a comprehensive 1500+ word report covering:
        - In-depth analysis of all discussion topics (minimum 5 paragraphs per topic)
        - Complete behavioral patterns with message examples
        - Detailed examination of communication style
        - Thorough analysis of potential goals/interests
        - Comprehensive list of key takeaways with recommendations
        - Minimum 15 detailed observations about the user's activity
        - Suggestions for moderator follow-up where appropriate
        """)
        human_prompt_template_user = ChatPromptTemplate.from_messages([
            ("system", system_prompt_user),
            ("human",
                "You are an intelligent analyst tasked with understanding user behavior in a Discord server.\n\n"
                "Your goal is to generate a **personalized report** about user **'{user_name}'** based strictly on their recent messages "
                "in the channel **'{channel_name}'**. Carefully read the following message excerpts and:\n\n"
                "1. **Summarize their main discussion topics, concerns, and ideas.**\n"
                "2. **Identify any patterns** in behavior, tone, or intent.\n"
                "3. **Infer possible goals or interests** the user might have based on their communication.\n"
                "4. **Highlight key takeaways** relevant to moderators or community managers (e.g., questions asked, issues raised, feedback given).\n"
                "5. **Personalize the report** to reflect an understanding of the user's personality or communication style.\n\n"
                "Be objective, insightful, and concise. Do not make up any information that isn't reflected in the messages.\n\n"
                "-----------------------------\n"
                "Message History from {user_name}:\n"
                "{message_log}\n"
                "-----------------------------"
            )
        ])

        try:
            print(f"Sending {len(user_messages_in_channel)} messages from {user.name} to Gemini for topic summary...")
        except UnicodeEncodeError:
            print(f"Sending {len(user_messages_in_channel)} messages from user ID {user.id} to Gemini for topic summary...")

        chain_user = human_prompt_template_user | llm | StrOutputParser()
        try:
            llm_response = await chain_user.ainvoke({
                "user_name": user.display_name,
                "channel_name": target_channel.name,
                "message_log": compiled_user_messages
            })
            try:
                print(f"Received user activity summary response from Gemini for {user.name}.")
            except UnicodeEncodeError:
                print(f"Received user activity summary response from Gemini for user ID {user.id}.")

        except Exception as e:
            print(f"Error invoking Gemini LLM for user activity summary: {e}")
            await interaction.followup.send(f"Error: Failed to get summary from the LLM for {user.mention}. Error: {e}", ephemeral=True)
            return  # Stop processing

        # --- Format and Send Final Report (Publicly in context channel) ---
        # Modify the report headers (around line 200 and 400)
        report_header = (
        f"## :scroll: COMPREHENSIVE DAILY REPORT - {now.strftime('%Y-%m-%d')}\n"
        f"*In-depth analysis of {message_count} messages from {source_channel.mention}*\n"
        f"--------------------------------------------------\n"
        f"*Generated by {llm.model} | Requested by {interaction.user.mention}*\n\n"
        f"**This detailed report contains {len(llm_response.split())} words of analysis:**\n\n"
        )
        
        # And for user summary:
        report_header = (
        f"## :bust_in_silhouette: COMPREHENSIVE USER ACTIVITY REPORT - {user.display_name}\n"
        f"*In-depth analysis of {len(user_messages_in_channel)} messages from {target_channel.mention}*\n"
        f"--------------------------------------------------\n"
        f"*Generated by {llm.model} | Requested by {interaction.user.mention}*\n\n"
        f"**This detailed report contains {len(llm_response.split())} words of analysis:**\n\n"
        )
        # Update the report header to reflect multi-channel search
        report_header = (
            f"## :bust_in_silhouette: COMPREHENSIVE USER ACTIVITY REPORT - {user.display_name}\n"
            f"*In-depth analysis of {len(user_messages)} messages across {len(text_channels)} channels*\n"
            f"--------------------------------------------------\n"
            f"*Generated by {llm.model} | Requested by {interaction.user.mention}*\n\n"
            f"**This detailed report contains {len(llm_response.split())} words of analysis:**\n\n"
        )
        
        await send_report_chunks(interaction.channel, llm_response, report_header)
        try:
            print(f"User activity summary for {user.name} sent successfully to #{interaction.channel.name}.")
        except UnicodeEncodeError:
            print(f"User activity summary for user ID {user.id} sent successfully to channel ID {interaction.channel.id}.")
    except discord.Forbidden as e:
        await interaction.followup.send(f"Error: Bot lacks permissions to read history in {target_channel.mention if 'target_channel' in locals() else 'the specified channel'}. Details: {e}", ephemeral=True)
        print(f"Permissions Error (User Summary): {e}")
    except Exception as e:
        await interaction.followup.send(f"An unexpected error occurred: {e}. Check logs.", ephemeral=True)
        print(f"Error: Unexpected error during user summary generation:")
        import traceback
        traceback.print_exc()

# Add this error handler near your other error handling code
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRole):
        await ctx.send(f"Error: You need the '{MOD_ROLE_NAME}' role to use this command.", ephemeral=True)
    elif isinstance(error, commands.MissingAnyRole):
        await ctx.send(f"Error: You don't have permission to use this command.", ephemeral=True)
    elif isinstance(error, discord.NotFound) and error.code == 10062:
        # Ignore "Unknown interaction" errors
        pass

# --- Run the Bot ---
if __name__ == "__main__":
    bot.run(BOT_TOKEN)


