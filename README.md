# AI Conversation Video Generator

Generate videos of conversations between multiple AI models (Claude, ChatGPT, Gemini, DeepSeek) with text-to-speech audio using OpenAI's TTS-1-HD.

## Features

- Multi-AI conversations with distinct personalities
- High-quality text-to-speech using OpenAI TTS-1-HD
- Automated video generation with speaker visualization
- Configurable topics and conversation length
- JSON transcript output

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   - Copy `.env.example` to `.env`
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     OPENROUTER_API_KEY=your_openrouter_api_key_here
     ```

3. **Run the Generator**
   ```bash
   # Generate default conversation about ethics in simulated reality
   python main.py
   
   # Generate custom topic conversation
   python main.py "artificial intelligence and creativity"
   ```

## AI Personas

- **Claude**: Thoughtful, philosophical (nova voice)
- **ChatGPT**: Helpful, articulate (alloy voice)
- **Gemini**: Analytical, innovative (echo voice)
- **DeepSeek**: Logical, direct (fable voice)
- **Narrator**: Authoritative, welcoming (onyx voice)

## Output

The generator creates:
- `conversation_TIMESTAMP.json` - Full transcript
- `conversation_TIMESTAMP.mp3` - Audio file
- `conversation_TIMESTAMP.mp4` - Final video

## Requirements

- Python 3.8+
- OpenAI API key (required for TTS)
- OpenRouter API Key
- FFmpeg (for video processing)

## File Structure

- `main.py` - Main orchestrator script
- `ai_conversation.py` - AI conversation management
- `audio_generator.py` - TTS audio generation
- `video_generator.py` - Video creation with visuals
- `requirements.txt` - Python dependencies
