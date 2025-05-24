from openai import OpenAI
import os
from typing import List, Dict
from pydub import AudioSegment
import io

class AudioGenerator:
    def __init__(self):
        # Get API key directly from shell environment (no dotenv)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Debug: Print what we're getting (masked for security)
        if self.openai_api_key:
            print(f"OpenAI API key loaded from environment: {self.openai_api_key[:10]}...{self.openai_api_key[-4:] if len(self.openai_api_key) > 14 else '[too short]'}")
        else:
            print("Warning: No OpenAI API key found in shell environment variables")
            print("Make sure OPENAI_API_KEY is set in your shell")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.temp_audio_dir = "temp_audio"  # Temporary directory for individual files
        self.voice_mapping = {
            "calm and measured": "nova",
            "clear and professional": "alloy", 
            "enthusiastic and articulate": "echo",
            "confident and direct": "fable",
            "authoritative and welcoming": "onyx"
        }
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_audio_dir, exist_ok=True)
    
    def text_to_speech(self, text: str, voice_style: str = "alloy") -> AudioSegment:
        """Convert text to speech using OpenAI TTS-1-HD"""
        voice = self.voice_mapping.get(voice_style, "alloy")
        
        try:
            response = self.client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=text,
                speed=1.0
            )
            
            # Convert to AudioSegment
            audio_data = io.BytesIO(response.content)
            audio = AudioSegment.from_mp3(audio_data)
            return audio
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            # Return silence as fallback
            return AudioSegment.silent(duration=1000)
    
    def generate_conversation_audio(self, conversation: List[Dict], pause_duration: int = 1000) -> AudioSegment:
        """Generate complete audio for the conversation with pauses between speakers"""
        combined_audio = AudioSegment.empty()
        
        for entry in conversation:
            speaker = entry["speaker"]
            content = entry["content"]
            voice_style = entry.get("voice_style", "alloy")
            
            print(f"Generating audio for {speaker}...")
            
            # Add speaker identification
            speaker_intro = f"{speaker}:"
            intro_audio = self.text_to_speech(speaker_intro, voice_style)
            
            # Generate main content audio
            content_audio = self.text_to_speech(content, voice_style)
            
            # Combine intro and content with a short pause
            speaker_segment = intro_audio + AudioSegment.silent(duration=300) + content_audio
            
            # Add to combined audio with pause
            combined_audio += speaker_segment + AudioSegment.silent(duration=pause_duration)
        
        return combined_audio
    
    def generate_speech(self, text: str, voice_id: str = "alloy") -> AudioSegment:
        """Generate speech from text using OpenAI TTS"""
        try:
            if not self.openai_api_key or self.openai_api_key.startswith('your_'):
                print(f"Warning: Invalid or missing OpenAI API key. Skipping audio generation.")
                return None
                
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice_id,
                input=text
            )
            
            # Convert to AudioSegment
            audio_data = io.BytesIO(response.content)
            audio_segment = AudioSegment.from_mp3(audio_data)
            return audio_segment
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None

    def generate_audio_for_speaker(self, content: str, speaker: str, voice_style: str) -> AudioSegment:
        """Generate audio for a specific speaker with their voice style"""
        try:
            if not self.openai_api_key or self.openai_api_key.startswith('your_'):
                print(f"Warning: Invalid or missing OpenAI API key. Skipping audio generation for {speaker}.")
                return None
            
            # Map voice style to OpenAI voice
            voice = self.voice_mapping.get(voice_style, "alloy")
            
            # Generate speech
            response = self.client.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=content,
                speed=1.0
            )
            
            # Convert to AudioSegment
            audio_data = io.BytesIO(response.content)
            audio_segment = AudioSegment.from_mp3(audio_data)
            return audio_segment
            
        except Exception as e:
            print(f"Error generating speech for {speaker}: {e}")
            return None

    def save_audio(self, audio: AudioSegment, filename: str, is_final: bool = False):
        """Save audio to file with error handling"""
        if audio is None:
            print(f"Warning: No audio to save for {filename}")
            return None
        
        # Use temp directory for individual files, main directory for final files
        if not is_final:
            filename = os.path.join(self.temp_audio_dir, os.path.basename(filename))
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Export with specific settings for better compatibility
            audio.export(
                filename, 
                format="mp3",
                bitrate="192k",
                parameters=["-ar", "44100", "-ac", "2"]  # Sample rate and stereo
            )
            print(f"Audio saved: {filename}")
            return filename  # Return the actual filename used
        except FileNotFoundError as e:
            if 'ffmpeg' in str(e):
                print(f"Error: ffmpeg not found. Please install ffmpeg to generate audio files.")
                print(f"On macOS: brew install ffmpeg")
                print(f"On Ubuntu: sudo apt update && sudo apt install ffmpeg")
                print(f"On Windows: Download from https://ffmpeg.org/download.html")
            else:
                print(f"Error saving audio: {e}")
        except Exception as e:
            print(f"Error saving audio: {e}")
        return None
    
    def cleanup_temp_files(self):
        """Delete all temporary audio files"""
        try:
            import shutil
            if os.path.exists(self.temp_audio_dir):
                shutil.rmtree(self.temp_audio_dir)
                print(f"Cleaned up temporary audio files from {self.temp_audio_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")