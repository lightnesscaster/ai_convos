try:
    from moviepy import AudioFileClip, ImageClip, CompositeVideoClip
except ImportError:
    print("Warning: MoviePy not properly installed. Video generation may not work.")
    # Fallback classes to prevent import errors
    class AudioFileClip:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        @property
        def duration(self): return 10.0
    
    class ImageClip:
        def __init__(self, *args, **kwargs): pass
        def set_start(self, time): return self
        def set_duration(self, duration): return self
    
    class CompositeVideoClip:
        def __init__(self, *args, **kwargs): pass
        def set_duration(self, duration): return self
        def set_audio(self, audio): return self
        def write_videofile(self, *args, **kwargs): pass
        def close(self): pass

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from typing import List, Dict

class VideoGenerator:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.fps = 24
        
    def create_speaker_visual(self, speaker_name: str, is_speaking: bool = False) -> np.ndarray:
        """Create a visual representation for each speaker"""
        # Create a simple colored background with speaker name
        img = Image.new('RGB', (self.width, self.height), color='black')
        draw = ImageDraw.Draw(img)
        
        # Color scheme for different AIs
        colors = {
            'Claude': '#FF6B6B' if is_speaking else '#FF6B6B80',
            'ChatGPT': '#4ECDC4' if is_speaking else '#4ECDC480',
            'Gemini': '#45B7D1' if is_speaking else '#45B7D180',
            'DeepSeek': '#96CEB4' if is_speaking else '#96CEB480',
            'Narrator': '#FFEAA7' if is_speaking else '#FFEAA780'
        }
        
        color = colors.get(speaker_name, '#FFFFFF' if is_speaking else '#FFFFFF80')
        
        # Draw background gradient effect
        for y in range(self.height):
            alpha = int(255 * (1 - y / self.height) * 0.3)
            gradient_color = color + format(alpha, '02x') if len(color) == 7 else color
            draw.line([(0, y), (self.width, y)], fill=gradient_color)
        
        # Add speaker name
        try:
            # Try to use a nice font
            font_size = 120 if is_speaking else 80
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Center the text
        bbox = draw.textbbox((0, 0), speaker_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2
        
        text_color = 'white' if is_speaking else '#CCCCCC'
        draw.text((x, y), speaker_name, fill=text_color, font=font)
        
        # Add speaking indicator
        if is_speaking:
            # Draw animated-style border
            border_width = 10
            draw.rectangle([border_width//2, border_width//2, 
                          self.width-border_width//2, self.height-border_width//2], 
                         outline=color, width=border_width)
        
        return np.array(img)
    
    def create_video_from_conversation(self, conversation: List[Dict], audio_file: str, output_file: str):
        """Create video with synchronized audio and visuals"""
        try:
            # Load the audio to get duration information
            audio_clip = AudioFileClip(audio_file)
            
            # Calculate timing for each speaker segment
            clips = []
            current_time = 0
            
            # Estimate duration per segment (you might want to make this more precise)
            total_segments = len(conversation)
            segment_duration = audio_clip.duration / total_segments
            
            for entry in conversation:
                speaker = entry["speaker"]
                
                # Create visual for this speaker
                img_array = self.create_speaker_visual(speaker, is_speaking=True)
                
                # Create video clip for this segment
                img_clip = ImageClip(img_array, duration=segment_duration)
                img_clip = img_clip.set_start(current_time)
                
                clips.append(img_clip)
                current_time += segment_duration
            
            # Combine all video clips
            video = CompositeVideoClip(clips, size=(self.width, self.height))
            video = video.set_duration(audio_clip.duration)
            
            # Add audio
            final_video = video.set_audio(audio_clip)
            
            # Export video
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            final_video.write_videofile(
                output_file,
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            print(f"Video saved to {output_file}")
            
            # Clean up
            audio_clip.close()
            video.close()
            final_video.close()
            
        except Exception as e:
            print(f"Error creating video: {e}")
            print("This might be due to missing FFmpeg or MoviePy installation issues.")
    
    def create_waveform_video(self, conversation: List[Dict], audio_file: str, output_file: str):
        """Alternative: Create video with waveform visualization"""
        # This would create a more dynamic video with audio waveform
        # Implementation would involve analyzing audio levels and creating animated waveforms
        print("Waveform video generation not yet implemented")
        print(f"Would create waveform video for {len(conversation)} segments from {audio_file} to {output_file}")