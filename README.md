# MIVB Discord Bot

A comprehensive Discord bot for analyzing and summarizing channel messages using Gemini LLM.

## Features
- Daily channel message summaries
- User activity analysis across channels
- Detailed behavioral pattern detection
- Sentiment analysis and topic tracking

## Setup
1. Install Python 3.10+
2. Clone this repository
3. Install dependencies: `pip install -r requirements.txt`
4. Configure `.env` file with:
   - DISCORD_BOT_TOKEN
   - GEMINI_API_KEY
   - SOURCE_CHANNEL_ID
   - MOD_REPORT_CHANNEL_ID

## Usage
- `/daily_report`: Generates a comprehensive summary of today's messages
- `/user_summary @username`: Creates a detailed activity report for a specific user

## Requirements
- discord.py
- google-generativeai
- python-dotenv
- langchain