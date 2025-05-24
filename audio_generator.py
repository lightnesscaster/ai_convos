# Import audioop-lts to provide missing audioop functionality for pydub
try:
    import audioop
except ImportError:
    import audioop_lts as audioop
    import sys
    sys.modules['audioop'] = audioop

from openai import OpenAI
import os
from typing import List, Dict
from pydub import AudioSegment
import io

class AudioGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.voice_mapping = {
            "calm and measured": "nova",
            "clear and professional": "alloy", 
            "enthusiastic and articulate": "echo",
            "confident and direct": "fable",
            "authoritative and welcoming": "onyx"
        }
    
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
    
    def save_audio(self, audio: AudioSegment, filename: str):
        """Save audio to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        audio.export(filename, format="mp3")
        print(f"Audio saved to {filename}")