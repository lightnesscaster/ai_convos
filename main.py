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
import argparse

def main():
    """Main function to generate AI conversation video"""
    
    parser = argparse.ArgumentParser(
        description="Generate an AI Agora conversation"
    )
    parser.add_argument("topic", help="Topic to discuss")
    parser.add_argument(
        "--num-exchanges",
        type=int,
        default=12,
        help="Number of exchanges between AIs (default: 12)"
    )
    parser.add_argument(
        "--debate",
        action="store_true",
        help="Enable debate mode (assign PRO/CON stances)"
    )
    parser.add_argument(
        "--next-topic",
        type=str,
        default=None,
        help="Topic for next week's episode, to be mentioned in the conclusion"
    )
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)
    
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
    topic = args.topic
    num_exchanges = args.num_exchanges
    debate_mode = args.debate
    print(f"\nGenerating conversation about: {topic}")
    
    conversation = conversation_manager.generate_conversation(
        topic, 
        num_exchanges=num_exchanges, 
        debate_mode=debate_mode, 
        next_topic=args.next_topic
    )
    
    # Save conversation transcript
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_file = f"output/conversation_{timestamp}.json"
    
    with open(transcript_file, 'w') as f:
        json.dump(conversation, f, indent=2)
    print(f"Conversation transcript saved to {transcript_file}")
    
    # Generate audio
    print("\nGenerating audio...")
    audio_segments = []
    
    for i, turn in enumerate(conversation):
        print(f"Generating audio for {turn['speaker']}...")
        
        audio_file = f"temp_audio/audio_{timestamp}_{i:02d}_{turn['speaker'].lower()}.mp3"
        
        audio = audio_generator.generate_audio_for_speaker(
            turn['content'],
            turn['speaker'],
            turn['voice_style']
        )
        
        if audio is not None:
            audio_segments.append(audio)
            audio_generator.save_audio(audio, audio_file)
        else:
            print(f"Skipping audio for {turn['speaker']} due to generation error")
    
    # Only proceed with video generation if we have audio
    if audio_segments:
        # Combine all audio segments
        combined_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            combined_audio += segment
        
        # Save combined audio to output directory (not temp_audio)
        combined_audio_file = f"output/full_conversation_{timestamp}.mp3"
        audio_generator.save_audio(combined_audio, combined_audio_file, is_final=True)
        
        # Generate video with precise timing from individual segments
        if combined_audio_file and os.path.exists(combined_audio_file):
            print("\nGenerating video...")
            video_file = f"output/conversation_video_{timestamp}.mp4"
            video_generator.create_conversation_video(
                conversation, 
                combined_audio_file, 
                video_file,
                individual_audio_segments=audio_segments  # Pass individual segments for timing
            )
            print(f"Video generation complete! Saved to: {video_file}")
            
            # Clean up temporary audio files after video is created
            print("\nCleaning up temporary files...")
            audio_generator.cleanup_temp_files()
    else:
        print("\nSkipping video generation - no audio was successfully generated")
        print("To generate audio and video:")
        print("1. Set up a valid OpenAI API key in your .env file")
        print("2. Install ffmpeg (brew install ffmpeg on macOS)")

if __name__ == "__main__":
    main()