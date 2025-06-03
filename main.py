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
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic to discuss (required unless --mp3-file is provided)"
    )
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
    parser.add_argument(
        "--mp3-file",
        type=str,
        help="Path to an existing MP3 file to generate a video from"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        help="Path to a JSON conversation file to generate audio from"
    )
    args = parser.parse_args()

    # Ensure either topic, MP3 file, or JSON file is provided
    if not args.topic and not args.mp3_file and not args.json_file:
        print("Error: You must provide either a topic, an MP3 file, or a JSON file.")
        parser.print_help()
        return

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
    
    # Check if MP3 file is provided
    if args.mp3_file:
        if not os.path.exists(args.mp3_file):
            print(f"Error: Specified MP3 file does not exist: {args.mp3_file}")
            return

        print("Initializing Video Generator...")
        video_generator = VideoGenerator()

        # Generate video from the provided MP3 file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = f"output/conversation_video_{timestamp}.mp4"
        video_generator.create_custom_video_flow_ffmpeg(args.mp3_file, video_file)
        print(f"Video successfully created: {video_file}")
        return
    
    # Check if JSON file is provided
    if args.json_file:
        if not os.path.exists(args.json_file):
            print(f"Error: Specified JSON file does not exist: {args.json_file}")
            return

        print("Loading conversation from JSON file...")
        try:
            with open(args.json_file, 'r') as f:
                conversation = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading JSON file: {e}")
            return

        print("Initializing Audio Generator...")
        audio_generator = AudioGenerator()

        # Generate audio from JSON conversation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                combined_audio += segment
            
            combined_audio_file = f"output/full_conversation_{timestamp}.mp3"
            audio_generator.save_audio(combined_audio, combined_audio_file, is_final=True)
            print(f"Audio file generated: {combined_audio_file}")
            
            # Clean up temporary audio files
            print("\nCleaning up temporary files...")
            audio_generator.cleanup_temp_files()
        else:
            print("\nNo audio was successfully generated")
        
        return
    
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
    
    # Combine all audio segments
    if audio_segments:
        combined_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            combined_audio += segment
        
        combined_audio_file = f"output/full_conversation_{timestamp}.mp3"
        audio_generator.save_audio(combined_audio, combined_audio_file, is_final=True)
        
        # Use custom video flow
        print("\nGenerating custom video flow...")
        video_file = f"output/conversation_video_{timestamp}.mp4"
        video_generator.create_custom_video_flow_ffmpeg(combined_audio_file, video_file)
        print(f"Custom video flow complete! Saved to: {video_file}")
        
        # Clean up temporary audio files
        print("\nCleaning up temporary files...")
        audio_generator.cleanup_temp_files()
    else:
        print("\nSkipping video generation - no audio was successfully generated")
        print("To generate audio and video:")
        print("1. Set up a valid OpenAI API key in your .env file")
        print("2. Install ffmpeg (brew install ffmpeg on macOS)")

if __name__ == "__main__":
    main()