#!/usr/bin/env python3
"""
AI Conversation Video Generator
Creates videos of conversations between different AI models
"""

import os
import sys
from datetime import datetime
from ai_conversation import AIConversationManager
from audio_generator import AudioGenerator
from video_generator import VideoGenerator
import json

def main():
    """Main function to generate AI conversation video"""
    
    # Ensure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Initialize managers
    print("Initializing AI Conversation Manager...")
    conversation_manager = AIConversationManager()
    
    if len(conversation_manager.personas) < 2:
        print("Error: Need at least 2 AI personas configured with API keys!")
        print("Please add your API keys to a .env file (see .env.example)")
        return
    
    print(f"Found {len(conversation_manager.personas)} AI personas:")
    for persona in conversation_manager.personas:
        print(f"  - {persona.name} ({persona.voice_style})")
    
    print("\nInitializing Audio Generator...")
    audio_generator = AudioGenerator()
    
    print("Initializing Video Generator...")
    video_generator = VideoGenerator()
    
    # Generate conversation about ethics in simulated reality
    topic = "ethics in a simulated reality"
    print(f"\nGenerating conversation about: {topic}")
    
    conversation = conversation_manager.generate_conversation(topic, num_exchanges=6)
    
    # Save conversation transcript
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_file = f"output/conversation_{timestamp}.json"
    
    with open(transcript_file, 'w') as f:
        json.dump(conversation, f, indent=2)
    print(f"Conversation transcript saved to {transcript_file}")
    
    # Generate audio
    print("\nGenerating audio...")
    audio = audio_generator.generate_conversation_audio(conversation)
    audio_file = f"output/conversation_{timestamp}.mp3"
    audio_generator.save_audio(audio, audio_file)
    
    # Generate video
    print("\nGenerating video...")
    video_file = f"output/conversation_{timestamp}.mp4"
    video_generator.create_video_from_conversation(conversation, audio_file, video_file)
    
    print(f"\n✅ Complete! Files generated:")
    print(f"📝 Transcript: {transcript_file}")
    print(f"🎵 Audio: {audio_file}")
    print(f"🎬 Video: {video_file}")

def generate_custom_conversation(topic: str, num_exchanges: int = 6):
    """Generate a conversation on a custom topic"""
    conversation_manager = AIConversationManager()
    audio_generator = AudioGenerator()
    video_generator = VideoGenerator()
    
    # Ensure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    print(f"Generating conversation about: {topic}")
    conversation = conversation_manager.generate_conversation(topic, num_exchanges)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transcript
    transcript_file = f"output/conversation_{timestamp}.json"
    with open(transcript_file, 'w') as f:
        json.dump(conversation, f, indent=2)
    
    # Generate audio and video
    audio = audio_generator.generate_conversation_audio(conversation)
    audio_file = f"output/conversation_{timestamp}.mp3"
    audio_generator.save_audio(audio, audio_file)
    
    video_file = f"output/conversation_{timestamp}.mp4"
    video_generator.create_video_from_conversation(conversation, audio_file, video_file)
    
    return transcript_file, audio_file, video_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom topic provided
        topic = " ".join(sys.argv[1:])
        num_exchanges = 6
        generate_custom_conversation(topic, num_exchanges)
    else:
        # Default conversation
        main()