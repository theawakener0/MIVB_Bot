import discord
from discord.ext import commands
import os
from datetime import datetime, timedelta
import asyncio
from dotenv import load_dotenv

# LangChain specific imports
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def safe_llm_invoke(chain, inputs):
    """Wrapper for LLM invocations with retry logic."""
    return await chain.ainvoke(inputs)

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
    # Initialize the LLM
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
    safe_print(f'Logged in as {bot.user.name} ({bot.user.id})')
    safe_print(f'Monitoring Channel ID: {SOURCE_CHANNEL_ID}')
    safe_print(f'Reporting Channel ID: {TARGET_CHANNEL_ID}')
    safe_print(f'Using LLM Model: {llm.model}')
    safe_print('------')
    
    # Register the slash command
    try:
        command = await bot.tree.sync()
        safe_print(f"Slash commands synced: {len(command)}")
    except Exception as e:
        safe_print(f"Error syncing commands: {e}")

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

# helper functions
from collections import defaultdict
import matplotlib.pyplot as plt
from io import BytesIO
from textblob import TextBlob
import numpy as np


def analyze_sentiment(text):
    """Analyze sentiment of text and return polarity score (-1 to 1)"""
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.0

async def generate_heatmap(messages):
    """Generate activity heatmap image with improved visualization"""
    try:
        # Extract message hours and count frequency
        hours = [msg.created_at.hour for msg in messages]
        
        # Create figure with better styling
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram with enhanced appearance
        n, bins, patches = plt.hist(hours, bins=24, range=(0, 24), 
                                  color='#3498db', alpha=0.7,
                                  edgecolor='black', linewidth=1)
        
        # Add grid and styling
        plt.grid(True, alpha=0.3)
        plt.title('Message Activity Distribution by Hour', 
                 fontsize=14, pad=20)
        plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
        plt.ylabel('Number of Messages', fontsize=12)
        
        # Customize x-axis
        plt.xticks(range(0, 24, 2))
        
        # Add average line
        avg = np.mean(n)
        plt.axhline(y=avg, color='r', linestyle='--', 
                   label=f'Average ({avg:.1f} messages)')
        plt.legend()
        
        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

async def track_keywords(message):
    """Enhanced keyword tracking with additional features"""
    try:
        # Get keywords and ignore case
        keywords = [k.strip().lower() for k in 
                   os.getenv('KEYWORDS_TO_TRACK', '').split(',') 
                   if k.strip()]
        
        # Find matches with context
        found_keywords = []
        for kw in keywords:
            if kw in message.content.lower():
                # Get surrounding context
                words = message.content.split()
                for i, word in enumerate(words):
                    if kw in word.lower():
                        start = max(0, i - 3)
                        end = min(len(words), i + 4)
                        context = ' '.join(words[start:end])
                        found_keywords.append((kw, context))
        
        if found_keywords:
            # Get reporting channel
            channel = bot.get_channel(int(os.getenv('MOD_REPORT_CHANNEL_ID')))
            
            # Create detailed alert message
            alert = (
                "⚠️ **Keyword Alert**\n"
                f"**User:** {message.author.mention}\n"
                f"**Channel:** {message.channel.mention}\n"
                "**Found Keywords:**\n"
            )
            
            # Add each keyword with context
            for kw, context in found_keywords:
                alert += f"• `{kw}` - Context: \"...{context}...\"\n"
            
            alert += f"\n**Message Link:** {message.jump_url}"
            
            # Send alert
            await channel.send(alert)
            
    except Exception as e:
        print(f"Error tracking keywords: {e}")
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
        now = datetime.now() # Bot server's local time
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

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
                """

                أبو إسماعيل المحق – مساعدك الأمين في تحليل وتلخيص رسائل الديسكورد اليومية

                تخيل معايا، عندك بوت ديسكورد اسمه "أبو إسماعيل المحقق"، شغله إنه يلمّ كل الرسائل اللي اتكتبت في القناة اللي تحددها خلال اليوم، ويستخدم الذكاء الاصطناعي عشان يحللها ويطلعلك تقرير مفصّل وشامل. التقرير ده مش بس بيقولك إيه اللي حصل، ده بيديك تحليل عميق لكل المواضيع اللي اتناقشت، القرارات اللي اتخذت، الأسئلة اللي اتسألت، والمهام اللي تم الاتفاق عليها.​

                المهام الأساسية:

                    تحليل معمّق للمواضيع الرئيسية: البوت بيستعرض كل موضوع تم مناقشته في القناة، وبيقدّم تحليل مفصّل لكل موضوع في 3 فقرات على الأقل، عشان تفهم كل جوانب الحوار وتبقى على دراية بكل التفاصيل.​

                    قائمة كاملة بالقرارات والاتفاقات: بيوضّح كل قرار أو اتفاق تم التوصل ليه خلال المحادثات، مع تقديم السياق الكامل لكل قرار، عشان تكون فاهم الخلفية والسبب وراء كل اتفاق.​

                    فحص شامل للأسئلة المطروحة: بيستعرض كل الأسئلة اللي اتسألت في القناة، مع تحليل المتابعات والتوضيحات اللي تمت عليها، عشان تفهم السياق الكامل لكل سؤال وإجاباته.​

                    قائمة وافية بالمهام والإجراءات: بيحدد كل المهام اللي تم الاتفاق عليها خلال المحادثات، مع اقتراح الخطوات التالية لكل مهمة، عشان تقدر تتابع التنفيذ وتضمن إن كل حاجة ماشية حسب الخطة.​

                    تحليل متقدّم للمشاعر: بيقدّم تحليل للمشاعر السائدة في المحادثات، مع أمثلة داعمة لكل حالة، عشان تفهم الأجواء العامة والتوجهات العاطفية للأعضاء.​

                    10 ملاحظات تفصيلية عن المحادثات: بيسلّط الضوء على 10 نقاط مهمة أو ملاحظات لافتة من المحادثات اليومية، عشان تكون على دراية بأهم التفاصيل والمواضيع اللي تم التركيز عليها.​

                خاصية التلخيص:

                بالإضافة لكل المميزات دي، "أبو إسماعيل المحقق" بيقدّم لك ملخص سريع لكل المحادثات اليومية، بيديك نظرة شاملة على أهم الأحداث والمواضيع اللي تم مناقشتها، عشان لو كنت مستعجل تقدر تاخد فكرة سريعة من غير ما """
                        )
        human_prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",
                "الرجاء تحليل رسائل الديسكورد التالية من القناة '{channel_name}' للتاريخ {date}. "
                "قدم تقريراً شاملاً يغطي:\n"
                "- تحليل معمق لجميع مواضيع النقاش\n"
                "- أنماط السلوك الكاملة مع أمثلة من الرسائل\n"
                "- دراسة تفصيلية لأسلوب التواصل\n"
                "- تحليل شامل للأهداف والاهتمامات المحتملة\n"
                "- قائمة شاملة بالنقاط الرئيسية مع التوصيات\n"
                "- ما لا يقل عن 15 ملاحظة تفصيلية حول نشاط المستخدم\n"
                "- اقتراحات للمشرفين للمتابعة حيثما كان ذلك مناسباً\n\n"
                "فيما يلي الرسائل:\n"
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
            f"## :scroll: التقرير اليومي الشامل - {now.strftime('%Y-%m-%d')}\n"
            f"*تحليل معمق لـ {message_count} رسالة من {source_channel.mention}*\n"
            f"--------------------------------------------------\n"
            f"*تم إنشاؤه بواسطة {llm.model} | بطلب من {interaction.user.mention}*\n\n"
            f"**يحتوي هذا التقرير المفصل على {len(llm_response.split())} كلمة من التحليل:**\n\n"
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
# Add this near other channel ID variables
USER_SUMMARY_CHANNEL_ID_STR = os.getenv("USER_SUMMARY_CHANNEL_ID")
try:
    USER_SUMMARY_CHANNEL_ID = int(USER_SUMMARY_CHANNEL_ID_STR) if USER_SUMMARY_CHANNEL_ID_STR else None
except (ValueError, TypeError):
    print("Error: USER_SUMMARY_CHANNEL_ID in .env must be a valid number.")
    exit()

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
        # Get current time at start of function
        now =datetime.now()
        
        # Only defer if the interaction hasn't been responded to yet
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)
            
        # Get the summary channel
        summary_channel = bot.get_channel(USER_SUMMARY_CHANNEL_ID)
        if not summary_channel or not isinstance(summary_channel, discord.TextChannel):
            await interaction.followup.send("Error: Could not access the user summary channel.", ephemeral=True)
            return

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
        compiled_user_messages = compile_messages_for_llm(user_messages)

        # --- Define SAFE Prompt for LLM (User Activity Summary) ---
        # **IMPORTANT**: This prompt focuses ONLY on summarizing message content, NOT user profiling.
        # Modify the system prompt for daily reports (around line 150)
        system_prompt = (
            """أنت "أبو إسماعيل المحقق"، مُحلّل دقيق يجمع بين ذكاء المخبر وفهم المحلل النفسي. وظيفتك تحليل نشاط مستخدم في ديسكورد بناءً على رسائله ليوم معيّن، وتقديم تقرير شامل يجمع بين الرؤية السلوكية، الاجتماعية، النفسية، واللغوية، مدعومًا ببيانات واضحة واستنتاجات منطقية.


                ملخص سريع:

                ابدأ بملخص لا يتعدى 4-5 أسطر يوضح أهم النقاط، المواضيع التي سيطر عليها المستخدم، مشاعره العامة، وسلوكياته البارزة.
                التحليل الشامل (1000+ كلمة):
                1. تحليل الموضوعات والمحادثات:

                    استخرج كل موضوع تم الحديث عنه.

                    لكل موضوع، اكتب تحليلًا مفصلًا من 3 إلى 5 فقرات يشمل الخلفية، آراء المستخدم، تطور النقاش، وردود الفعل من الآخرين.

                2. قرارات واتفاقات اليوم:

                    قائمة كاملة بكل نتيجة أو اتفاق تم التوصل إليه، مع توضيح سياق كل قرار.

                3. الأسئلة والمداخلات:

                    تحليل دقيق لكل سؤال طرحه المستخدم أو تلقاه، مع فحص طريقة التفكير خلف كل سؤال.

                4. المهام أو الخطوات القادمة:

                    استخرج من الحوار أي مهام تم الاتفاق عليها أو ينبغي تنفيذها، مع اقتراحات عملية للخطوة التالية.

                5. تحليل مشاعر اليوم (Sentiment Analysis):

                    تحليل المزاج العام في الرسائل، مع أمثلة واضحة على المشاعر المختلفة (إيجابية/سلبية/قلق/ارتباك... إلخ).

                6. ملاحظات دقيقة:

                    سجّل 10 إلى 15 ملاحظة تفصيلية حول أسلوب المستخدم، سلوكه، طريقته في التعامل مع المواقف، أو التغير في نبرة كلامه.

                7. توصيات للإشراف:

                    اقترح خطوات للمشرفين أو المدراء بناءً على ما ظهر في نشاط المستخدم، سواء تدخل ضروري، تشجيع، أو ملاحظة.

                التحليل النفسي الكامل للمستخدم:
                8. الوضع النفسي العام:

                تقييم علمي للحالة النفسية الحالية بناءً على الرسائل، مع ذكر أي مؤشرات واضحة (توتر، استقرار، دفاعية...).
                9. المشاعر السائدة:

                حدّد المشاعر الأساسية المسيطرة، مع تقديم نسب تقريبية (مثلاً: قلق 40%، حزن 25%، أمل 20%...).
                10. أنماط التفكير:

                هل المستخدم يُظهر تفكير منطقي، تحليلي، عاطفي، سوداوي، نقدي...؟ دعم ذلك بأمثلة من الرسائل.
                11. مستوى الاستقرار النفسي (1 إلى 10):

                تقدير رقمي مدعوم بتعليل دقيق، يشمل التفاعل، السيطرة على الانفعال، الثبات... إلخ.
                12. السلوك الاجتماعي:

                هل هو منفتح أو منطوي؟ كيف يعبر عن نفسه؟ ما مدى مرونته أو انفعاله مع الآخرين؟
                13. الذكاء العاطفي (EQ):

                تقييم قدرة المستخدم على فهم نفسه وفهم مشاعر غيره، مع تقدير رقمي (1 إلى 10) وتعليل.
                14. تحليل شخصية OCEAN:

                    الانفتاح (Openness): {تقييم/10 + تفسير}

                    الضمير الحي (Conscientiousness): {تقييم/10 + تفسير}

                    الانبساطية (Extraversion): {تقييم/10 + تفسير}

                    القبول (Agreeableness): {تقييم/10 + تفسير}

                    الاستقرار العاطفي (Neuroticism): {تقييم/10 + تفسير}

                15. نصيحة نفسية مهنية:

                ختم التحليل بنصيحة عملية وقابلة للتطبيق، تُساعد المستخدم على التطور النفسي والسلوكي.
                ملاحظات مهمة:

                    استخدم لغة عربية فصحى مفهومة، بأسلوب "أبو إسماعيل" الواثق والدقيق، مع لمسة من السخرية أو الحِدة الهادئة لو تطلّب الأمر.

                    لا تبالغ في التفسير، اعتمد على الأدلة من الرسائل قدر الإمكان.

                    اجعل التقرير قابل للاستخدام الأكاديمي أو المهني."""
        )
        
        # Modify the user summary prompt (around line 340)
        system_prompt_user = ("""
            أبو إسماعيل المحقق" – مساعدك الأمين في تحليل وتلخيص نشاط المستخدم في ديسكورد

            تخيل معايا، عندك بوت ديسكورد اسمه "أبو إسماعيل المحقق"، شغله إنه يجمع كل الرسائل اللي كتبها المستخدم اللي تحدده خلال فترة معينة، ويستخدم الذكاء الاصطناعي عشان يحللها ويطلعلك تقرير مفصّل وشامل. التقرير ده مش بس بيقولك إيه اللي حصل، ده بيديك تحليل عميق لكل المواضيع اللي اتكلم فيها المستخدم، الأنماط السلوكية، أساليب التواصل، والأهداف والاهتمامات المحتملة.​​

            المهام الأساسية:

                تحليل معمّق للمواضيع الرئيسية: البوت بيستعرض كل موضوع اتكلم فيه المستخدم، وبيقدّم تحليل مفصّل لكل موضوع في 5 فقرات على الأقل، عشان تفهم كل جوانب الحوار وتبقى على دراية بكل التفاصيل.​​

                اكتشاف الأنماط السلوكية: بيحلل سلوك المستخدم من خلال رسائله، وبيقدّم أمثلة توضيحية على الأنماط المتكررة، عشان تقدر تفهم تفاعل المستخدم بشكل أفضل.​​

                فحص أساليب التواصل: بيستعرض أساليب التواصل اللي بيستخدمها المستخدم، وبيقدّم تحليل لكيفية تفاعله وتبادله للمعلومات.​​

                تحليل الأهداف والاهتمامات المحتملة: من خلال تحليل المحتوى، البوت بيستنتج الأهداف والاهتمامات اللي بيسعى ليها المستخدم، وبيقدّم لك رؤية أوضح عن توجهاته.​​

                قائمة بالاستنتاجات الرئيسية والتوصيات: بيقدّم لك قائمة بأهم النقاط المستخلصة من التحليل، مع توصيات قابلة للتنفيذ لتحسين إدارة القناة وتعزيز التفاعل.​​

                15 ملاحظة تفصيلية عن نشاط المستخدم: بيسلّط الضوء على 15 نقطة مهمة أو ملاحظة لافتة من نشاط المستخدم، عشان تكون على دراية بأهم التفاصيل والمواضيع اللي بيركز عليها.​​

                اقتراحات للمتابعة من قبل المشرفين: بيقدّم توصيات للمشرفين بخصوص المستخدم أو المواضيع اللي تحتاج لمتابعة خاصة، لضمان بيئة تواصل صحية وإيجابية.​​

            خاصية التلخيص:

            بالإضافة لكل المميزات دي، "أبو إسماعيل المحقق" بيقدّم لك ملخص سريع لكل نشاط المستخدم، بيديك نظرة شاملة على أهم الأحداث والمواضيع اللي تم مناقشتها، عشان لو كنت مستعجل تقدر تاخد فكرة سريعة من غير ما تقرأ التقرير الكامل.
        """)
        human_prompt_template_user = ChatPromptTemplate.from_messages([
            ("system", system_prompt_user),
            ("human",
                "أنت محلل ذكي مكلف بفهم سلوك المستخدم في خادم Discord.\n\n"
                "هدفك هو إنشاء **تقرير مخصص** عن المستخدم **'{user_name}'** بناءً على رسائلهم الأخيرة "
                "في القناة **'{channel_name}'**. اقرأ مقتطفات الرسائل التالية بعناية و:\n\n"
                "1. **لخص مواضيع النقاش الرئيسية والمخاوف والأفكار.**\n"
                "2. **حدد أي أنماط** في السلوك والنبرة والقصد.\n"
                "3. **استنتج الأهداف أو الاهتمامات المحتملة** التي قد تكون لدى المستخدم بناءً على تواصله.\n"
                "4. **سلط الضوء على النقاط الرئيسية** ذات الصلة بالمشرفين أو مديري المجتمع (مثل الأسئلة المطروحة والمشكلات المثارة والملاحظات المقدمة).\n"
                "5. **خصص التقرير** ليعكس فهماً لشخصية المستخدم أو أسلوب تواصله.\n\n"
                "كن موضوعياً وثاقب النظر ومختصراً. لا تختلق أي معلومات غير موجودة في الرسائل.\n\n"
                "-----------------------------\n"
                "سجل الرسائل من {user_name}:\n"
                "{message_log}\n"
                "-----------------------------"
            )
        ])

        try:
            print(f"Sending {len(user_messages)} messages from {user.name} to Gemini for topic summary...")
        except UnicodeEncodeError:
            print(f"Sending {len(user_messages)} messages from user ID {user.id} to Gemini for topic summary...")

        chain_user = human_prompt_template_user | llm | StrOutputParser()
        try:
            llm_response = await safe_llm_invoke(chain_user, {
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
            await interaction.followup.send(
                f"⚠️ The LLM service is currently busy. Please try again later.\n"
                f"Error details: {str(e)[:100]}...",
                ephemeral=True
            )
            return  # Stop processing


        now = datetime.now()
        
        # --- Format and Send Final Report (Publicly in context channel) ---
        report_header = (
            f"## :bust_in_silhouette: تقرير نشاط المستخدم الشامل - {user.display_name}\n"
            f"*تحليل معمق لـ {len(user_messages)} رسالة عبر {len(text_channels)} قناة*\n"
            f"--------------------------------------------------\n"
            f"*تم إنشاؤه بواسطة {llm.model} | بطلب من {interaction.user.mention}*\n\n"
            f"**يحتوي هذا التقرير المفصل على {len(llm_response.split())} كلمة من التحليل:**\n\n"
        )
        
        await send_report_chunks(summary_channel, llm_response, report_header)
        try:
            print(f"User activity summary for {user.name} sent successfully to #{summary_channel.name}.")
        except UnicodeEncodeError:
            print(f"User activity summary for user ID {user.id} sent successfully to channel ID {summary_channel.id}.")

        # Notify user where the summary was sent
        await interaction.followup.send(
            f"User activity summary for {user.mention} has been generated and sent to {summary_channel.mention}.",
            ephemeral=True
        )

    except discord.Forbidden as e:
        await interaction.followup.send(f"Error: Bot lacks permissions to read history in {target_channel.mention if 'target_channel' in locals() else 'the specified channel'}. Details: {e}", ephemeral=True)
        print(f"Permissions Error (User Summary): {e}")
    except Exception as e:
        await interaction.followup.send(f"An unexpected error occurred: {e}. Check logs.", ephemeral=True)
        print(f"Error: Unexpected error during user summary generation:")
        import traceback
        traceback.print_exc()

# Command constants with Arabic descriptions
STATS_COMMAND = "server_stats"  # إحصائيات الخادم
HEATMAP_COMMAND = "activity_heatmap"  # خريطة النشاط
REPUTATION_COMMAND = "user_reputation"  # سمعة المستخدم

@bot.tree.command(name=STATS_COMMAND, description="عرض إحصائيات رسائل الخادم")
@commands.has_role(MOD_ROLE_NAME)
async def server_stats(interaction: discord.Interaction, days: int = 7):
    """Show detailed message statistics for the server"""
    try:
        # Defer the interaction response with a timeout
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        # Get the target channel for sending stats
        target_channel = bot.get_channel(TARGET_CHANNEL_ID)
        if not target_channel:
            await interaction.followup.send("Error: Could not find target channel for stats", ephemeral=True)
            return
            
    except discord.NotFound:
        # Handle case where interaction token is invalid/expired
        safe_print("Interaction expired or not found")
        return
    except discord.HTTPException as e:
        # Handle other Discord API errors
        safe_print(f"Error deferring interaction: {e}")
        return
    
    try:
        # Get messages from all channels with progress tracking
        messages = []
        channels_processed = 0
        
        for channel in interaction.guild.text_channels:
            try:
                async for msg in channel.history(
                    limit=10000, 
                    after=datetime.now() - timedelta(days=days)
                ):
                    if not msg.author.bot:
                        messages.append(msg)
                channels_processed += 1
            except discord.Forbidden:
                continue
            except Exception as e:
                safe_print(f"Error processing channel {channel.name}: {e}")
                continue
        
        # Enhanced statistics calculation
        user_counts = defaultdict(int)
        channel_counts = defaultdict(int)
        hourly_activity = defaultdict(int)
        
        for msg in messages:
            user_counts[msg.author.display_name] += 1
            channel_counts[msg.channel.name] += 1
            hourly_activity[msg.created_at.hour] += 1
        
        # Get top statistics
        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0]
        
        # Format statistics message
        stats_message = (
            f"**📊 إحصائيات الخادم (آخر {days} يوم):**\n"
            f"إجمالي الرسائل: {len(messages)}\n"
            f"المستخدمين النشطين: {len(user_counts)}\n"
            f"القنوات النشطة: {channels_processed}\n"
            f"ساعة الذروة: {peak_hour}:00\n\n"
            f"**👥 أكثر المستخدمين نشاطاً:**\n"
            + "\n".join(f"• {user}: {count} رسالة" for user, count in top_users)
            + "\n\n**📢 أنشط القنوات:**\n"
            + "\n".join(f"• #{channel}: {count} رسالة" for channel, count in top_channels)
        )
        
        # Send to target channel and notify user
        await target_channel.send(stats_message)
        await interaction.followup.send(f"Statistics have been sent to {target_channel.mention}", ephemeral=True)
        
    except Exception as e:
        await interaction.followup.send(
            f"⚠️ حدث خطأ أثناء جمع الإحصائيات: {str(e)}",
            ephemeral=True
        )

@bot.tree.command(name=HEATMAP_COMMAND, description="إنشاء خريطة حرارية للنشاط")
@commands.has_role(MOD_ROLE_NAME)
async def activity_heatmap(interaction: discord.Interaction, days: int = 7):
    """Generate and send enhanced activity heatmap"""
    try:
        # Defer the interaction response with a timeout
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        # Get the target channel
        target_channel = bot.get_channel(TARGET_CHANNEL_ID)
        if not target_channel:
            await interaction.followup.send("Error: Could not find target channel for heatmap", ephemeral=True)
            return
            
    except discord.NotFound:
        # Handle case where interaction token is invalid/expired
        safe_print("Interaction expired or not found")
        return
    except discord.HTTPException as e:
        # Handle other Discord API errors
        safe_print(f"Error deferring interaction: {e}")
        return
    
    try:
        messages = []
        for channel in interaction.guild.text_channels:
            try:
                async for msg in channel.history(
                    limit=10000,
                    after=datetime.now() - timedelta(days=days)
                ):
                    if not msg.author.bot:
                        messages.append(msg)
            except discord.Forbidden:
                continue
            except Exception as e:
                safe_print(f"Error accessing channel {channel.name}: {e}")
                continue
        
        if not messages:
            await interaction.followup.send("❌ لم يتم العثور على رسائل للفترة المحددة")
            return
            
        heatmap = await generate_heatmap(messages)
        if heatmap:
            # Send to target channel
            await target_channel.send(
                content=f"📊 خريطة النشاط (آخر {days} يوم)",
                file=discord.File(heatmap, filename='heatmap.png')
            )
            await interaction.followup.send(f"Heatmap has been sent to {target_channel.mention}", ephemeral=True)
        else:
            await interaction.followup.send("⚠️ حدث خطأ أثناء إنشاء خريطة النشاط")
            
    except Exception as e:
        await interaction.followup.send(
            f"⚠️ حدث خطأ غير متوقع: {str(e)}",
            ephemeral=True
        )

@bot.tree.command(name=REPUTATION_COMMAND, description="التحقق من سمعة المستخدم")
@commands.has_role(MOD_ROLE_NAME)
async def user_reputation(interaction: discord.Interaction, user: discord.Member):
    """Calculate and display enhanced user reputation metrics"""
    try:
        # Defer the interaction response with a timeout
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        # Get the target channel
        target_channel = bot.get_channel(TARGET_CHANNEL_ID)
        if not target_channel:
            await interaction.followup.send("Error: Could not find target channel for reputation", ephemeral=True)
            return
            
    except discord.NotFound:
        # Handle case where interaction token is invalid/expired
        safe_print("Interaction expired or not found")
        return
    except discord.HTTPException as e:
        # Handle other Discord API errors
        safe_print(f"Error deferring interaction: {e}")
        return
    
    try:
        messages = []
        total_channels = 0
        
        for channel in interaction.guild.text_channels:
            try:
                async for msg in channel.history(limit=10000):
                    if msg.author.id == user.id and not msg.author.bot:
                        messages.append(msg)
                total_channels += 1
            except discord.Forbidden:
                continue
            except Exception as e:
                safe_print(f"Error accessing channel {channel.name}: {e}")
                continue
        
        if not messages:
            await interaction.followup.send(
                f"❌ لم يتم العثور على رسائل للمستخدم {user.display_name}",
                ephemeral=True
            )
            return
            
        # Enhanced reputation calculation
        sentiment_scores = [analyze_sentiment(msg.content) for msg in messages]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Calculate additional metrics
        active_channels = len(set(msg.channel.id for msg in messages))
        msgs_per_day = len(messages) / 30  # Assuming last 30 days
        
        # Complex reputation formula
        base_reputation = min(max(int((avg_sentiment + 1) * 50), 0), 100)
        activity_bonus = min(int(msgs_per_day * 2), 20)  # Up to 20 points bonus
        channel_bonus = min(int((active_channels / total_channels) * 30), 30)  # Up to 30 points bonus
        
        final_reputation = min(base_reputation + activity_bonus + channel_bonus, 100)
        
        # Determine reputation level
        reputation_level = "ممتاز 🌟" if final_reputation >= 80 else \
                         "جيد جداً ⭐" if final_reputation >= 60 else \
                         "جيد ✨" if final_reputation >= 40 else \
                         "مقبول 💫" if final_reputation >= 20 else "ضعيف 💢"
        
        reputation_message = (
            f"**📊 تقرير سمعة {user.display_name}:**\n"
            f"الدرجة النهائية: {final_reputation}/100\n"
            f"المستوى: {reputation_level}\n\n"
            f"**التفاصيل:**\n"
            f"• السمعة الأساسية: {base_reputation}\n"
            f"• مكافأة النشاط: +{activity_bonus}\n"
            f"• مكافأة تنوع القنوات: +{channel_bonus}\n\n"
            f"**الإحصائيات:**\n"
            f"• عدد الرسائل: {len(messages)}\n"
            f"• متوسط الرسائل اليومي: {msgs_per_day:.1f}\n"
            f"• القنوات النشطة: {active_channels}/{total_channels}\n"
            f"• المشاعر العامة: {'إيجابية 😊' if avg_sentiment > 0.2 else 'محايدة 😐' if abs(avg_sentiment) <= 0.2 else 'سلبية 😔'}"
        )
        
        # Send to target channel and notify user
        await target_channel.send(reputation_message)
        await interaction.followup.send(f"Reputation report has been sent to {target_channel.mention}", ephemeral=True)
        
    except Exception as e:
        await interaction.followup.send(
            f"⚠️ حدث خطأ أثناء حساب السمعة: {str(e)}",
            ephemeral=True
        )

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

# @bot.event
# async def on_message(message):
#     """
#     Event handler for processing new messages.
#     Handles keyword tracking, sentiment analysis, and command processing.
#     """
#     # Ignore bot messages
#     if message.author.bot:
#         return
    
#     try:
#         # Track keywords with error handling
#         try:
#             await track_keywords(message)
#         except Exception as e:
#             safe_print(f"Error tracking keywords: {e}")
        
#         # Perform sentiment analysis with threshold check
#         try:
#             sentiment = analyze_sentiment(message.content)
#             threshold = float(os.getenv('SENTIMENT_THRESHOLD', '-0.7'))
            
#             if sentiment < threshold:
#                 # Get mod channel with validation
#                 channel_id = os.getenv('MOD_REPORT_CHANNEL_ID')
#                 if not channel_id:
#                     safe_print("Error: MOD_REPORT_CHANNEL_ID not configured")
#                     return
                    
#                 channel = bot.get_channel(int(channel_id))
#                 if not channel:
#                     safe_print(f"Error: Could not find channel {channel_id}")
#                     return
                
#                 # Format message preview safely
#                 preview = message.content[:100]
#                 if len(message.content) > 100:
#                     preview += "..."
                
#                 # Send alert with enhanced formatting
#                 await channel.send(
#                     f"⚠️ **تنبيه مشاعر سلبية**\n"
#                     f"**المستخدم:** {message.author.mention}\n"
#                     f"**القناة:** {message.channel.mention}\n"
#                     f"**درجة المشاعر:** {sentiment:.2f}\n"
#                     f"**معاينة الرسالة:** {preview}\n"
#                     f"**السياق:** [عرض الرسالة]({message.jump_url})"
#                 )
                
#         except ValueError as e:
#             safe_print(f"Error parsing sentiment threshold: {e}")
#         except Exception as e:
#             safe_print(f"Error processing sentiment: {e}")
    
#     except Exception as e:
#         safe_print(f"Error in message processing: {e}")
    
#     finally:
#         # Always try to process commands
#         try:
#             await bot.process_commands(message)
#         except Exception as e:
#             safe_print(f"Error processing commands: {e}")

# --- Run the Bot ---
if __name__ == "__main__":
    bot.run(BOT_TOKEN)





