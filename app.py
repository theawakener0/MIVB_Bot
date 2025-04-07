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
                """

                "أبو إسماعيل المحقق" – مساعدك الأمين في تحليل وتلخيص رسائل الديسكورد اليومية

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
        now = datetime.datetime.now()
        
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
            """
                أبو إسماعيل المحقق" – مساعدك الأمين في تحليل وتلخيص رسائل الديسكورد اليومية

    تخيل معايا، عندك بوت ديسكورد اسمه "أبو إسماعيل المحقق"، شغله إنه يلمّ كل الرسائل اللي اتكتبت في القناة اللي تحددها خلال اليوم، ويستخدم الذكاء الاصطناعي عشان يحللها ويطلعلك تقرير مفصّل وشامل. التقرير ده مش بس بيقولك إيه اللي حصل، ده بيديك تحليل عميق لكل المواضيع اللي اتناقشت، القرارات اللي اتخذت، الأسئلة اللي اتسألت، والمهام اللي تم الاتفاق عليها.​​

    المهام الأساسية:

        تحليل معمّق للمواضيع الرئيسية: البوت بيستعرض كل موضوع تم مناقشته في القناة، وبيقدّم تحليل مفصّل لكل موضوع في 3 فقرات على الأقل، عشان تفهم كل جوانب الحوار وتبقى على دراية بكل التفاصيل.​​

        قائمة كاملة بالقرارات والاتفاقات: بيوضّح كل قرار أو اتفاق تم التوصل ليه خلال المحادثات، مع تقديم السياق الكامل لكل قرار، عشان تكون فاهم الخلفية والسبب وراء كل اتفاق.​​

        فحص شامل للأسئلة المطروحة: بيستعرض كل الأسئلة اللي اتسألت في القناة، مع تحليل المتابعات والتوضيحات اللي تمت عليها، عشان تفهم السياق الكامل لكل سؤال وإجاباته.​​

        قائمة وافية بالمهام والإجراءات: بيحدد كل المهام اللي تم الاتفاق عليها خلال المحادثات، مع اقتراح الخطوات التالية لكل مهمة، عشان تقدر تتابع التنفيذ وتضمن إن كل حاجة ماشية حسب الخطة.​​

        تحليل متقدّم للمشاعر: بيقدّم تحليل للمشاعر السائدة في المحادثات، مع أمثلة داعمة لكل حالة، عشان تفهم الأجواء العامة والتوجهات العاطفية للأعضاء.​​

        10 ملاحظات تفصيلية عن المحادثات: بيسلّط الضوء على 10 نقاط مهمة أو ملاحظات لافتة من المحادثات اليومية، عشان تكون على دراية بأهم التفاصيل والمواضيع اللي تم التركيز عليها.​​

    خاصية التلخيص:

    بالإضافة لكل المميزات دي، "أبو إسماعيل المحقق" بيقدّم لك ملخص سريع لكل المحادثات اليومية، بيديك نظرة شاملة على أهم الأحداث والمواضيع اللي تم مناقشتها، عشان لو كنت مستعجل تقدر تاخد فكرة سريعة من غير ما تقرأ التقرير الكامل.
            """
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


        now = datetime.datetime.now()
        
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





