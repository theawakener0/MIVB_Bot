# main.py - Main bot file
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
    initial_extensions = ['record']  # Changed from 'cogs.recorder' to 'record'
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
    await load_extensions()
    # Sync slash commands globally
    try:
        # Just sync commands globally without guild-specific sync
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

# --- Cog Structure ---
# cogs/recorder.py - Cog for recording functionality
import discord
from discord.ext import commands
from discord import app_commands # For slash commands
import asyncio
import os
import time
import logging
from datetime import datetime
import speech_recognition as sr
from pydub import AudioSegment # For potential audio conversion
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Choose the appropriate LangChain LLM wrapper for Gemini
# Option 1: Using google-generativeai directly (simpler for basic use)
# Option 2: Using LangChain's community Google GenerativeAI integration
from langchain_google_genai import ChatGoogleGenerativeAI # Use this if available and preferred

# --- Constants ---
RECORDINGS_DIR = "recordings"
MAX_RECORDING_MINUTES = 60 # Set a limit for recording duration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MOD_CHANNEL_ID = int(os.getenv("MOD_REPORT_CHANNEL_ID"))

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.warning("GEMINI_API_KEY not found. Summarization feature will be disabled.")

logger = logging.getLogger(__name__)

# --- Helper Functions ---

async def transcribe_audio(audio_path: str) -> str | None:
    """
    Transcribes the given audio file using SpeechRecognition.
    NOTE: This uses the default Sphinx engine if offline=True, which has limited accuracy.
          For better results, configure it to use an online service like Google Cloud Speech
          or Wit.ai by installing necessary libraries and providing API keys.
          Example (using Google Web Speech API - requires internet, no key needed but rate limited):
          r = sr.Recognizer()
          with sr.AudioFile(audio_path) as source:
              audio_data = r.record(source)
              try:
                  text = r.recognize_google(audio_data)
                  return text
              except sr.UnknownValueError:
                  logger.error("Google Web Speech API could not understand audio")
                  return None
              except sr.RequestError as e:
                  logger.error(f"Could not request results from Google Web Speech API service; {e}")
                  return None
    """
    logger.info(f"Attempting to transcribe: {audio_path}")
    r = sr.Recognizer()
    try:
        # Ensure the file is in a compatible format (like WAV)
        # If you save as MP3, you might need to convert first using pydub
        # Example conversion:
        # sound = AudioSegment.from_mp3(audio_path)
        # wav_path = audio_path.replace(".mp3", ".wav")
        # sound.export(wav_path, format="wav")
        # audio_path = wav_path # Use the WAV path for transcription

        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise (optional but recommended)
            # r.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = r.record(source) # Read the entire audio file

        # Use Sphinx for offline transcription (lower accuracy)
        # text = r.recognize_sphinx(audio_data)

        # Use Google Web Speech API (requires internet, better accuracy, rate limits)
        text = r.recognize_google(audio_data, language="ar-EG")
        if text:
            logger.info(f"Transcription successful. First 50 chars: {text[:50]}...")
        else:
            logger.warning("Google Speech Recognition returned empty result")

        logger.info("Transcription successful.")
        return text
    except sr.UnknownValueError:
        logger.error("Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logger.error(f"Could not request results from Speech Recognition service; {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Audio file not found for transcription: {audio_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during transcription: {e}", exc_info=True)
        return None


async def summarize_text(text: str) -> str | None:
    """Summarizes the text using Gemini via LangChain."""
    if not GEMINI_API_KEY or not text:
        logger.warning("Gemini API key or text is missing. Skipping summarization.")
        return None

    logger.info("Starting summarization...")
    try:
        # --- LangChain Setup ---
        # Select the Gemini model
        # Check available models: for m in genai.list_models(): print(m.name)
        # Use a model suitable for summarization, e.g., 'gemini-1.5-flash' or 'gemini-pro'
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GEMINI_API_KEY,
                                     temperature=0.3, convert_system_message_to_human=True) # convert_system_message_to_human might be needed depending on LangChain version

        # Define the prompt template
        prompt_template = PromptTemplate(
            input_variables=["transcript"],
            template="""يرجى تقديم ملخص موجز باللغة العربية لمحادثة الصوت التالية. تأكد من أن الملخص:
1. يكون بالكامل باللغة العربية
2. يلتقط النقاط الرئيسية
3. يكون مناسبًا ثقافيًا
4. يستخدم اللغة العربية الفصحى الواضحة

النص:
"{transcript}"

الملخص (باللغة العربية، من 3 إلى 5 جمل كحد أقصى):"""
        )

        # Create the LLMChain
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Run the chain
        summary = await chain.arun(transcript=text) # Use arun for async

        logger.info("Summarization successful.")
        return summary.strip()

    except Exception as e:
        logger.error(f"An error occurred during summarization: {e}", exc_info=True)
        return None

# --- Audio Sink ---
# Remove the entire old WavFileSink class (lines 210-245) and keep only this version:
class WavFileSink:
    def __init__(self, destination_dir: str):
        self.destination_dir = destination_dir
        self.start_time = time.time()
        self.file_path = None
        self.file = None

    def write(self, data):
        if self.file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_path = os.path.join(self.destination_dir, f"recording_{timestamp}.wav")
            self.file = open(self.file_path, 'wb')
        self.file.write(data)

    def cleanup(self):
        if self.file and not self.file.closed:
            self.file.close()
            duration = time.time() - self.start_time
            # Rename file to include duration
            new_path = self.file_path.replace('.wav', f'_{int(duration)}s.wav')
            os.rename(self.file_path, new_path)
            self.file_path = new_path
            logger.info(f"Recording saved successfully to {self.file_path}")

    def get_filepath(self):
        return self.file_path


# --- Recording View (Buttons) ---
class RecordingView(discord.ui.View):
    def __init__(self, recorder_cog, interaction: discord.Interaction):
        super().__init__(timeout=None) # Persistent view or set timeout
        self.recorder_cog = recorder_cog
        self.initial_interaction = interaction # Store the interaction that spawned the view
        self.voice_client = interaction.guild.voice_client
        self.sink = None # Will hold the WavFileSink instance
        self.recording_task = None # Will hold the asyncio task for timeout

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # Optional: Check if the user interacting is the one who started recording or has permissions
        # For simplicity, allowing anyone in the server to stop for now
        if not interaction.guild.voice_client:
             await interaction.response.send_message("The bot is not currently in a voice channel.", ephemeral=True)
             return False
        return True

    # Replace the start_recording method
    async def start_recording(self, interaction: discord.Interaction):
        """Starts the actual recording process."""
        if not self.voice_client or not self.voice_client.is_connected():
            await interaction.response.send_message("I'm not connected to a voice channel!", ephemeral=True)
            return

        if hasattr(self.voice_client, '_recording') and self.voice_client._recording:
            await interaction.response.send_message("I'm already recording!", ephemeral=True)
            return

        # Create recorder and start recording
        self.recorder = VoiceRecorder(self.voice_client, RECORDINGS_DIR)
        success = self.recorder.start()
        
        if not success:
            await interaction.response.send_message("Failed to start recording!", ephemeral=True)
            return
            
        # Mark voice client as recording
        self.voice_client._recording = True
        
        logger.info(f"Starting recording in channel: {self.voice_client.channel.name}")
        
        # Start timeout task
        self.recording_task = asyncio.create_task(self.recording_timeout())

        # Update button states
        self.start_button.disabled = True
        self.stop_button.disabled = False
        await interaction.message.edit(view=self) # Update the original message with new button states

    async def recording_timeout(self):
        """Automatically stops recording after MAX_RECORDING_MINUTES."""
        await asyncio.sleep(MAX_RECORDING_MINUTES * 60)
        logger.warning(f"Recording automatically stopped after {MAX_RECORDING_MINUTES} minutes.")
        if self.voice_client and self.voice_client.is_recording():
            # Need the original interaction or a way to get the context to send message
            channel = self.initial_interaction.channel
            if channel:
                await channel.send(f"Automatically stopping recording due to time limit ({MAX_RECORDING_MINUTES} mins).")
            await self.stop_recording_logic(self.initial_interaction) # Pass the initial interaction


    async def recording_finished_callback(self, sink: WavFileSink, *args):
        """Called by discord.py when recording stops."""
        # Ensure the timeout task is cancelled if we stopped manually
        if self.recording_task and not self.recording_task.done():
            self.recording_task.cancel()

        # The recording is stopped, now we wait for the sink cleanup (file writing)
        # Schedule the post-processing in the event loop
        await self.recorder_cog.bot.loop.create_task(self.post_process_recording(sink))

    async def post_process_recording(self, sink: WavFileSink):
        """Handles transcription and summarization after recording finishes."""
        # Wait a brief moment to ensure file writing/renaming is complete
        await asyncio.sleep(1)

        audio_filepath = sink.get_filepath()
        if not audio_filepath or not os.path.exists(audio_filepath):
            logger.error("Recording file path not found after recording finished.")
            # Try sending a message back to the original channel if possible
            try:
                 await self.initial_interaction.followup.send("Error: Could not find the saved recording file.", ephemeral=True)
            except Exception:
                 logger.error("Could not send error followup message.") # Avoid breaking if interaction expired
            return

        logger.info(f"Recording finished. File: {audio_filepath}")
        mod_channel = self.recorder_cog.bot.get_channel(MOD_CHANNEL_ID)

        if not mod_channel:
            logger.warning(f"Mod channel ({MOD_CHANNEL_ID}) not found. Cannot send recording/summary.")
            # Optionally send to the interaction channel as fallback
            mod_channel = self.initial_interaction.channel
            if not mod_channel: return # Give up if no channel found

        # Send the recording file first
        try:
            file = discord.File(audio_filepath)
            await mod_channel.send(f"Voice chat recording finished:", file=file)
        except discord.errors.HTTPException as e:
             if e.status == 413: # Payload too large
                 await mod_channel.send(f"Voice chat recording finished, but the file is too large ({os.path.getsize(audio_filepath)/(1024*1024):.2f} MB) to upload to Discord.")
                 logger.warning(f"Recording file {audio_filepath} too large for Discord.")
             else:
                 await mod_channel.send(f"Voice chat recording finished, but an error occurred sending the file: {e.status}")
                 logger.error(f"Error sending recording file: {e}", exc_info=True)
        except Exception as e:
            await mod_channel.send(f"Voice chat recording finished, but an unexpected error occurred sending the file.")
            logger.error(f"Unexpected error sending recording file: {e}", exc_info=True)


        # --- Summarization ---
        await mod_channel.send("Attempting to transcribe and summarize the recording...")
        transcript = await transcribe_audio(audio_filepath)

        if transcript:
            summary = await summarize_text(transcript)
            if summary:
                # Split summary if too long for one message
                for i in range(0, len(summary), 2000):
                     await mod_channel.send(f"**Summary:**\n{summary[i:i+2000]}")
            else:
                await mod_channel.send("Could not generate a summary for the recording.")
                # Optionally send the transcript if summarization failed but transcription worked
                # Be mindful of transcript length
                # transcript_preview = transcript[:1900] + "..." if len(transcript) > 1900 else transcript
                # await mod_channel.send(f"**Transcript (Partial):**\n```\n{transcript_preview}\n```")
        else:
            await mod_channel.send("Could not transcribe the audio. No summary available.")

        # Optional: Clean up the local audio file after processing
        # try:
        #     os.remove(audio_filepath)
        #     logger.info(f"Removed local recording file: {audio_filepath}")
        # except OSError as e:
        #     logger.error(f"Error removing local recording file {audio_filepath}: {e}")

    # Replace the stop_recording_logic method
    async def stop_recording_logic(self, interaction: discord.Interaction):
        """Contains the logic to stop recording, callable by button or timeout."""
        if not self.voice_client or not hasattr(self.voice_client, '_recording') or not self.voice_client._recording:
            # Check if interaction is deferred or responded to avoid errors
            if not interaction.response.is_done():
                await interaction.response.send_message("I'm not currently recording!", ephemeral=True)
            else:
                await interaction.followup.send("I'm not currently recording!", ephemeral=True)
            return

        logger.info("Stopping recording...")
        
        # Stop the recorder
        file_path = self.recorder.stop()
        self.voice_client._recording = False
        
        # Update buttons immediately for responsiveness
        self.start_button.disabled = False
        self.stop_button.disabled = True
        
        # Check if the original message still exists before editing
        try:
            original_message = await interaction.channel.fetch_message(interaction.message.id)
            if original_message:
                await original_message.edit(view=self)
        except discord.NotFound:
            logger.warning("Original message for recording view not found, cannot update buttons.")
        except discord.Forbidden:
            logger.error("Missing permissions to edit the original message.")
        except Exception as e:
            logger.error(f"Error editing message: {e}", exc_info=True)

        # Acknowledge the interaction
        if not interaction.response.is_done():
            await interaction.response.send_message("Recording stopped. Processing audio...", ephemeral=True)
        else:
            # If already responded (e.g., by timeout message), use followup
            try:
                await interaction.followup.send("Recording stopped. Processing audio...", ephemeral=True)
            except discord.NotFound:  # Interaction might have expired
                logger.warning("Interaction expired before followup could be sent for stop command.")
                # Optionally send to the channel directly if interaction expired
                await interaction.channel.send("Recording stopped. Processing audio...")
        
        # Create a sink-like object for compatibility with existing code
        class SinkAdapter:
            def __init__(self, file_path):
                self.file_path = file_path
                
            def get_filepath(self):
                return self.file_path
                
            def cleanup(self):
                pass
        
        # Process the recording
        if file_path:
            sink_adapter = SinkAdapter(file_path)
            await self.recording_finished_callback(sink_adapter)


    @discord.ui.button(label="Start Recording", style=discord.ButtonStyle.green, custom_id="start_rec_button")
    async def start_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'{interaction.user}' clicked Start Recording.")
        # Defer the response to avoid "Interaction failed" if processing takes time
        await interaction.response.defer(ephemeral=True)
        await self.start_recording(interaction)
        # Send followup after starting
        await interaction.followup.send("Recording started!", ephemeral=True)


    @discord.ui.button(label="Stop Recording", style=discord.ButtonStyle.red, custom_id="stop_rec_button", disabled=True)
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'{interaction.user}' clicked Stop Recording.")
        # Deferring here might cause issues if stop_recording_logic also tries to respond.
        # Let stop_recording_logic handle the response/followup.
        await self.stop_recording_logic(interaction)
        # No followup needed here as stop_recording_logic handles it


# --- Cog Class ---
class RecorderCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_views = {} # Store active views per guild {guild_id: view}
        
    @app_commands.command(
        name="record_voice",  # Changed from "record" to "record_voice"
        description="[AR] يبدأ تسجيل قناة الصوت الحالية ويوفر عناصر تحكم تفاعلية"
    )
    async def record_command(self, interaction: discord.Interaction):
        """Slash command to start recording with interactive controls."""
        voice_state = interaction.user.voice
        if not voice_state or not voice_state.channel:
            await interaction.response.send_message("يجب أن تكون في قناة صوتية لاستخدام هذا الأمر!", ephemeral=True)
            return

        if interaction.guild.voice_client and interaction.guild.voice_client.is_connected():
            if interaction.guild.voice_client.channel != voice_state.channel:
                await interaction.response.send_message("أنا بالفعل في قناة صوتية أخرى!", ephemeral=True)
                return

        # Create and store the view
        view = RecordingView(self, interaction)
        self.active_views[interaction.guild.id] = view

        # Join voice channel if not already connected
        if not interaction.guild.voice_client:
            try:
                await voice_state.channel.connect()
            except discord.ClientException as e:
                await interaction.response.send_message(f"فشل الانضمام إلى القناة الصوتية: {e}", ephemeral=True)
                return

        await interaction.response.send_message(
            "سيتم بدء التسجيل قريبًا. استخدم الأزرار أدناه للتحكم:",
            view=view,
            ephemeral=True
        )

    @app_commands.command(name="join", description="[AR] ينضم البوت إلى قناتك الصوتية الحالية")
    async def join_voice(self, interaction: discord.Interaction):
        """Slash command to join the user's voice channel."""
        voice_state = interaction.user.voice
        if not voice_state or not voice_state.channel:
            await interaction.response.send_message("يجب أن تكون في قناة صوتية لاستخدام هذا الأمر!", ephemeral=True)
            return
            
        if interaction.guild.voice_client and interaction.guild.voice_client.is_connected():
            if interaction.guild.voice_client.channel == voice_state.channel:
                await interaction.response.send_message("أنا بالفعل في قناتك الصوتية!", ephemeral=True)
                return
            
        # Defer the response since connection might take time
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Increase timeout to give more time for connection
            await voice_state.channel.connect(timeout=20.0)
            await interaction.followup.send("تم الانضمام إلى القناة الصوتية بنجاح!", ephemeral=True)
            logger.info(f"Successfully connected to voice channel {voice_state.channel.name} in {interaction.guild.name}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout while connecting to voice channel {voice_state.channel.name}")
            await interaction.followup.send("فشل الانضمام إلى القناة الصوتية: انتهت مهلة الاتصال. يرجى المحاولة مرة أخرى لاحقًا.", ephemeral=True)
        except discord.ClientException as e:
            logger.error(f"Client exception while connecting to voice: {e}")
            await interaction.followup.send(f"فشل الانضمام إلى القناة الصوتية: {e}", ephemeral=True)
        except Exception as e:
            logger.error(f"Unexpected error while connecting to voice: {e}", exc_info=True)
            await interaction.followup.send("حدث خطأ غير متوقع أثناء محاولة الانضمام إلى القناة الصوتية.", ephemeral=True)

    @app_commands.command(name="leave", description="Leaves the current voice channel.")
    async def leave_voice(self, interaction: discord.Interaction):
        """Slash command to leave the current voice channel."""
        voice_client = interaction.guild.voice_client
        if not voice_client:
            await interaction.response.send_message("I'm not in a voice channel!", ephemeral=True)
            return

        try:
            await voice_client.disconnect()
            await interaction.response.send_message("Left the voice channel.")
        except Exception as e:
            logger.error(f"Error leaving voice channel: {e}", exc_info=True)
            await interaction.response.send_message("Couldn't leave the voice channel.", ephemeral=True)

    @app_commands.command(name="record_controls", description="Shows controls to record the current voice channel.")  # Changed from "record" to "record_controls"
    @app_commands.checks.has_permissions(administrator=True) # Example permission check
    async def record_slash(self, interaction: discord.Interaction):
        """Slash command to initiate recording controls."""
        voice_state = interaction.user.voice
        if not voice_state or not voice_state.channel:
            await interaction.response.send_message("You need to be in a voice channel first!", ephemeral=True)
            return

        if not interaction.guild.voice_client:
            await interaction.response.send_message("I need to be in a voice channel first! Use /join first.", ephemeral=True)
            return

        # Check if a view is already active for this guild
        if interaction.guild_id in self.active_views:
            # Optionally try to fetch the old message and reuse/delete it
            await interaction.response.send_message("Recording controls are already active in this server.", ephemeral=True)
            return

        # Create and send the view
        view = RecordingView(self, interaction)
        self.active_views[interaction.guild_id] = view # Store the view

        # Initial state: only Start is enabled
        view.start_button.disabled = False
        view.stop_button.disabled = True

        await interaction.response.send_message("Recording Controls:", view=view)
        logger.info(f"Recording controls sent to channel {interaction.channel.name} in guild {interaction.guild.name}")

        # Store the message ID for potential future reference/cleanup
        message = await interaction.original_response()
        view.message_id = message.id

    @app_commands.command(name="stop", description="Forces the bot to stop recording and leave the voice channel.")
    @app_commands.checks.has_permissions(administrator=True)
    async def stop_slash(self, interaction: discord.Interaction):
        """Slash command to forcefully stop recording and disconnect."""
        vc = interaction.guild.voice_client
        if not vc or not vc.is_connected():
            await interaction.response.send_message("I'm not currently in a voice channel.", ephemeral=True)
            return

        # Check if recording using the older approach
        if hasattr(vc, '_recording') and vc._recording:
            logger.info("Force stop command received. Stopping recording...")
            await interaction.response.defer(ephemeral=True)  # Defer while processing

            # Find the active view to call its stop logic if possible
            active_view = self.active_views.get(interaction.guild_id)
            if active_view:
                # Use the view's logic to handle cleanup and processing
                await active_view.stop_recording_logic(interaction)
                # Remove the view reference after stopping
                if interaction.guild_id in self.active_views:
                    del self.active_views[interaction.guild_id]
            else:
                # Fallback if view not found (shouldn't normally happen)
                logger.warning("Could not find active view during force stop. Stopping recording directly.")
                # For the new approach, we can't stop directly without the recorder object
                await interaction.followup.send("Recording stopped forcefully (view not found). Processing may be incomplete.", ephemeral=True)

            # Disconnect after stopping recording and potential processing starts
            await vc.disconnect(force=True)
            logger.info(f"Disconnected from {vc.channel.mention} due to force stop command.")
        else:
            # If connected but not recording, just disconnect
            logger.info("Force stop command received. Disconnecting...")
            await vc.disconnect(force=True)
            await interaction.response.send_message("Disconnected from the voice channel.", ephemeral=True)
            logger.info(f"Disconnected from {vc.channel.mention} due to force stop command (was not recording).")

        # Clean up view reference if it exists
        if interaction.guild_id in self.active_views:
            del self.active_views[interaction.guild_id]


    @record_slash.error
    @stop_slash.error
    async def slash_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        """Handles errors for the slash commands."""
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("You don't have the required permissions to use this command.", ephemeral=True)
        elif isinstance(error, app_commands.CheckFailure):
             await interaction.response.send_message("You don't meet the requirements to use this command (e.g., not in VC).", ephemeral=True)
        else:
            logger.error(f"An error occurred in a slash command: {error}", exc_info=True)
            if not interaction.response.is_done():
                 await interaction.response.send_message("An unexpected error occurred. Please check the bot logs.", ephemeral=True)
            else:
                 await interaction.followup.send("An unexpected error occurred. Please check the bot logs.", ephemeral=True)


# --- Cog Setup Function ---
async def setup(bot):
    """Required for extension loading."""
    await bot.add_cog(RecorderCog(bot))  # Changed from bot.add_cog

# Add these imports at the top of your file
import pyaudio
import wave
import threading
import struct

# Add this class after the WavFileSink class
class VoiceRecorder:
    def __init__(self, voice_client, destination_dir):
        self.voice_client = voice_client
        self.destination_dir = destination_dir
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.start_time = None
        self.file_path = None
        self.recording_thread = None
        
    def start(self):
        if self.recording:
            return False
            
        self.recording = True
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.destination_dir, f"recording_{timestamp}.wav")
        self.frames = []
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        return True
        
    def _record(self):
        # Configure audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=48000,
            input=True,
            frames_per_buffer=4096
        )
        
        # Record audio
        while self.recording:
            data = self.stream.read(4096)
            self.frames.append(data)
            
        # Close and cleanup
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        
        # Save the recording
        self._save_recording()
        
    def _save_recording(self):
        if not self.frames:
            return None
            
        wf = wave.open(self.file_path, 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(48000)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        # Rename file to include duration
        duration = time.time() - self.start_time
        new_path = self.file_path.replace('.wav', f'_{int(duration)}s.wav')
        os.rename(self.file_path, new_path)
        self.file_path = new_path
        logger.info(f"Recording saved successfully to {self.file_path}")
        
        return self.file_path
        
    def stop(self):
        if not self.recording:
            return False
            
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
            self.recording_thread = None
            
        return self.file_path

