from moviepy import AudioFileClip, ImageClip, CompositeVideoClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
from typing import List, Dict

class VideoGenerator:
    def __init__(self, width: int = 1080, height: int = 720, fps: int = 30):
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
                preset='ultrafast',  # Use ultrafast for faster processing
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

    def create_custom_video_flow(self, podcast_audio_file: str, output_file: str):
        """Create a custom video flow combining intro, outro, and podcast visuals."""
        # Verify podcast audio file exists
        if not os.path.exists(podcast_audio_file):
            raise FileNotFoundError(f"Podcast audio file not found: {podcast_audio_file}")

        # Load podcast audio
        podcast_audio = AudioFileClip(podcast_audio_file)
        duration = podcast_audio.duration  # Total duration of the audio
        segment_length = 8  # Length of each segment in seconds

        # Define intro and outro clips
        intro_video_path = "output/clips/intro_and_outro/Video_Ready_AI_Emergence.mp4"
        outro_video_path = "output/clips/intro_and_outro/A_setting_sun_202505311911.mp4"
        aerial_video_path = "output/clips/intro_and_outro/A_majestic_aerial_202505311907.mp4"

        if not os.path.exists(intro_video_path) or not os.path.exists(outro_video_path) or not os.path.exists(aerial_video_path):
            raise FileNotFoundError("One or more intro/outro video files are missing.")

        intro_clip = VideoFileClip(intro_video_path)  # Keep audio
        aerial_clip = VideoFileClip(aerial_video_path).with_audio(None)  # Remove audio
        outro_clip = VideoFileClip(outro_video_path)  # Keep audio

        # Define podcast visuals and cache 8-second subclips
        podcast_visuals_dir = "output/clips/random"
        if not os.path.exists(podcast_visuals_dir):
            raise FileNotFoundError(f"Podcast visuals directory not found: {podcast_visuals_dir}")

        podcast_visuals = {}  # Cache for 8-second subclips
        for file_name in os.listdir(podcast_visuals_dir):
            if file_name.endswith(".mp4"):
                full_path = os.path.join(podcast_visuals_dir, file_name)
                base_clip = VideoFileClip(full_path).without_audio()
                subclip = base_clip.with_duration(segment_length).resized((self.width, self.height))
                podcast_visuals[file_name] = subclip

        # Ensure there are enough clips for randomness
        if len(podcast_visuals) < 3:
            raise ValueError("Not enough podcast visual clips for randomness.")

        # Combine podcast visuals with audio
        podcast_clips = []
        t = 0.0
        last_name = None
        second_last = None

        print("Preparing podcast clips...")
        while t < duration:
            # Determine end of this segment (clamp at total duration)
            t_end = min(t + segment_length, duration)

            # Select a visual clip avoiding repetition of last and second_last filenames
            candidates = [fn for fn in podcast_visuals.keys() if fn not in (last_name, second_last)]
            chosen_name = random.choice(candidates) if candidates else random.choice(list(podcast_visuals))
            chosen_clip = podcast_visuals[chosen_name]

            # Handle shorter final segment
            actual_dur = t_end - t
            # Adjusting for MoviePy v2.0 changes
            segment = chosen_clip.with_duration(segment_length).resized((self.width, self.height))
            if actual_dur < segment_length:
                segment = segment.with_duration(actual_dur).with_start(t)
            else:
                segment = segment.with_start(t)

            podcast_clips.append(segment)

            # Update tracking of last filenames
            second_last = last_name
            last_name = chosen_name
            t = t_end
            print(f"Prepared clip for segment {t:.2f}/{duration:.2f} seconds.")

        print("Podcast clips preparation complete. Compositing video...")

        # Composite podcast clips without reattaching full audio
        podcast_video = CompositeVideoClip(podcast_clips, size=(self.width, self.height))

        # Combine all clips into final video using CompositeVideoClip
        final_video = CompositeVideoClip([
            intro_clip.with_start(0),
            aerial_clip.with_start(intro_clip.duration),
            podcast_video.with_start(intro_clip.duration + aerial_clip.duration),
            outro_clip.with_start(intro_clip.duration + aerial_clip.duration + podcast_video.duration)
        ]).with_audio(podcast_audio)  # Use only the MP3 audio for the main content

        print("Final video composition complete. Starting rendering...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the final video file with built-in logger for progress tracking
        print(f"Writing custom video flow to: {output_file}")
        try:
            final_video.write_videofile(
                output_file,
                fps=self.fps,
                audio_codec="aac",
                codec="libx264",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                preset="ultrafast",
                threads=62,
            )
        except Exception as e:
            print(f"Error during video rendering: {e}")
            raise

        # Cleanup
        final_video.close()
        podcast_audio.close()
        intro_clip.close()
        aerial_clip.close()
        outro_clip.close()
        for clip in podcast_visuals.values():
            clip.close()

        print(f"Custom video flow successfully created: {output_file}")

# Example usage
if __name__ == "__main__":
    convo = [
        {"speaker": "Claude",   "text": "Hello, I'm Claude."},
        {"speaker": "ChatGPT",  "text": "And I'm ChatGPT."},
        {"speaker": "DeepSeek", "text": "DeepSeek checking in."},
    ]
    vg = VideoGenerator()
    vg.create_conversation_video(convo, "dialogue.mp3", "out/video.mp4")
