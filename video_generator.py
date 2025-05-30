from moviepy import AudioFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from typing import List, Dict

class VideoGenerator:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 4):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._visual_cache = {}  # Cache for generated speaker visuals
        self._clip_cache = {}    # Cache for ImageClip objects per speaker

    def create_speaker_visual(self, speaker_name: str, is_speaking: bool = False) -> np.ndarray:
        """Load and display the corresponding PNG image for each speaker."""
        # Check cache first
        cache_key = (speaker_name, is_speaking)
        if cache_key in self._visual_cache:
            return self._visual_cache[cache_key]
        
        # Map speaker names to image files
        image_path = f"images/{speaker_name.lower()}.png"
        
        # Check if the image file exists
        if os.path.exists(image_path):
            try:
                # Load the speaker's image
                img = Image.open(image_path).convert('RGBA')
                
                # Resize image to fit the video dimensions while maintaining aspect ratio
                img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
                
                # Create a background canvas
                canvas = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 255))
                
                # Center the image on the canvas
                x = (self.width - img.width) // 2
                y = (self.height - img.height) // 2
                canvas.paste(img, (x, y), img)
                
                result = np.array(canvas.convert('RGB'))
                self._visual_cache[cache_key] = result
                return result
                
            except Exception as e:
                print(f"Error loading image for {speaker_name}: {e}")
                # Fall back to creating a simple colored background with text
                result = self._create_fallback_visual(speaker_name, is_speaking)
                self._visual_cache[cache_key] = result
                return result
        else:
            print(f"Image not found: {image_path}, using fallback visual")
            result = self._create_fallback_visual(speaker_name, is_speaking)
            self._visual_cache[cache_key] = result
            return result

    def _create_fallback_visual(self, speaker_name: str, is_speaking: bool = False) -> np.ndarray:
        """Create a simple fallback visual when image is not available."""
        # Check if we already have this cached (in case called directly)
        cache_key = (speaker_name, is_speaking)
        if cache_key in self._visual_cache:
            return self._visual_cache[cache_key]
            
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        # Color schemes
        base_colors = {
            'Claude':   (0xFF, 0x6B, 0x6B),
            'ChatGPT':  (0x4E, 0xCC, 0xDC),
            'Gemini':   (0x45, 0xB7, 0xD1),
            'DeepSeek': (0x96, 0xCE, 0xB4),
            'Narrator': (0xFF, 0xEA, 0xA7),
        }
        r, g, b = base_colors.get(speaker_name, (0xCC, 0xCC, 0xCC))
        alpha = 255 if is_speaking else 128
        fill  = (r, g, b, alpha)

        # Draw background
        draw.rectangle([(0, 0), (self.width, self.height)], fill=fill)

        # Speaker name text
        font_size = 120 if is_speaking else 80
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        # Get text width/height using textbbox (Pillow ≥ 8.0)
        bbox = draw.textbbox((0, 0), speaker_name, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = (self.width  - text_w)  // 2
        y = (self.height - text_h) // 2
        draw.text((x, y), speaker_name, font=font, fill=(255,255,255,255))

        result = np.array(img.convert('RGB'))
        self._visual_cache[cache_key] = result
        return result

    def _get_speaker_clip(self, speaker_name: str) -> ImageClip:
        """Get or create a cached ImageClip for the speaker."""
        if speaker_name not in self._clip_cache:
            frame = self.create_speaker_visual(speaker_name, is_speaking=True)
            self._clip_cache[speaker_name] = ImageClip(frame)
        return self._clip_cache[speaker_name]

    def create_conversation_video(
        self,
        conversation: List[Dict[str, str]],
        audio_file: str,
        output_file: str,
        individual_audio_segments: List = None
    ):
        """Alias wrapper for create_video_from_conversation."""
        return self.create_video_from_conversation(conversation, audio_file, output_file, individual_audio_segments)

    def create_video_from_conversation(
        self,
        conversation: List[Dict[str, str]],
        audio_file: str,
        output_file: str,
        individual_audio_segments: List = None
    ):
        """Render a conversation to a video with synchronized audio."""
        # Verify audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        # Load the combined audio for the video track
        audio = AudioFileClip(audio_file)
        total_duration = audio.duration

        clips = []
        t_start = 0.0

        # Use individual audio segments for precise timing if provided
        if individual_audio_segments and len(individual_audio_segments) == len(conversation):
            print("Using individual audio segments for precise speaker timing...")
            for i, entry in enumerate(conversation):
                speaker = entry['speaker']
                # Get duration from the actual individual audio segment
                segment_duration = len(individual_audio_segments[i]) / 1000.0  # Convert ms to seconds
                
                base_clip = self._get_speaker_clip(speaker)
                img_clip = base_clip.with_duration(segment_duration).with_start(t_start)
                clips.append(img_clip)
                t_start += segment_duration
        else:
            # Fallback to equal duration segments
            print("Using equal duration segments for speaker timing...")
            num_segments = len(conversation)
            seg_dur = total_duration / num_segments
            
            for entry in conversation:
                speaker = entry['speaker']
                base_clip = self._get_speaker_clip(speaker)
                img_clip = base_clip.with_duration(seg_dur).with_start(t_start)
                clips.append(img_clip)
                t_start += seg_dur

        # Composite video and add the combined audio
        video = CompositeVideoClip(clips, size=(self.width, self.height))
        final = video.with_duration(total_duration).with_audio(audio)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the file
        print(f"Writing video with audio from: {audio_file}")
        
        # Check if audio file exists and has content
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found: {audio_file}")
        else:
            print(f"Audio file found: {audio_file}, duration: {audio.duration:.2f} seconds")
        
        try:
            final.write_videofile(
                output_file,
                fps=self.fps,
                audio_codec='aac',
                codec='libx264',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                threads = 6
            )
        except Exception as e:
            print(f"Error during video creation: {e}")
            # Try fallback without custom audio codec
            print("Attempting fallback video creation...")
            try:
                final.write_videofile(
                    output_file,
                    fps=self.fps,
                    codec='libx264',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
            except Exception as e2:
                print(f"Fallback video creation also failed: {e2}")
                raise

        # Cleanup
        audio.close()
        video.close()
        final.close()
        # Clean up cached clips
        for clip in self._clip_cache.values():
            clip.close()
        self._clip_cache.clear()
        print(f"Video with audio successfully created: {output_file}")

    def create_waveform_video(
        self,
        conversation: List[Dict[str, str]],
        audio_file: str,
        output_file: str
    ):
        """Alternative: stub for waveform‐based visuals."""
        print("Waveform video generation not implemented.")
        print(f"Would create waveform video for {len(conversation)} segments, audio={audio_file}, output={output_file}")

# Example usage
if __name__ == "__main__":
    convo = [
        {"speaker": "Claude",   "text": "Hello, I'm Claude."},
        {"speaker": "ChatGPT",  "text": "And I'm ChatGPT."},
        {"speaker": "DeepSeek", "text": "DeepSeek checking in."},
    ]
    vg = VideoGenerator()
    vg.create_conversation_video(convo, "dialogue.mp3", "out/video.mp4")
