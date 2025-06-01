from moviepy import AudioFileClip, ImageClip, CompositeVideoClip, VideoFileClip, CompositeAudioClip
from moviepy import vfx
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
from typing import List, Dict

class VideoGenerator:
    def __init__(self, width: int = 256, height: int = 144, fps: int = 6):
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
        """Create a custom video flow combining intro, aerial, podcast visuals, and outro with synchronized audio."""
        # Verify podcast audio file exists
        if not os.path.exists(podcast_audio_file):
            raise FileNotFoundError(f"Podcast audio file not found: {podcast_audio_file}")

        # Define paths for intro, aerial, and outro clips
        intro_video_path = "output/clips/intro_and_outro/Video_Ready_AI_Emergence.mp4"
        outro_video_path = "output/clips/intro_and_outro/A_setting_sun_202505311911.mp4"
        aerial_video_path = "output/clips/intro_and_outro/A_majestic_aerial_202505311907.mp4"

        for path in [intro_video_path, outro_video_path, aerial_video_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

        intro_clip   = VideoFileClip(intro_video_path).resized((self.width, self.height))
        aerial_clip  = VideoFileClip(aerial_video_path).resized((self.width, self.height))
        outro_clip   = VideoFileClip(outro_video_path).resized((self.width, self.height))


        # Load podcast audio
        podcast_audio = AudioFileClip(podcast_audio_file)
        podcast_duration = podcast_audio.duration

        # Generate random B-roll visuals for podcast
        podcast_visuals_dir = "output/clips/random"
        if not os.path.exists(podcast_visuals_dir):
            raise FileNotFoundError(f"Podcast visuals directory not found: {podcast_visuals_dir}")

        podcast_visuals = {}
        for file_name in os.listdir(podcast_visuals_dir):
            if file_name.endswith(".mp4"):
                full_path = os.path.join(podcast_visuals_dir, file_name)
                base_clip = VideoFileClip(full_path).without_audio()
                podcast_visuals[file_name] = base_clip.resized((self.width, self.height))

        if len(podcast_visuals) < 3:
            raise ValueError("Not enough podcast visual clips for randomness.")

        # Adjust podcast clips for true crossfade
        fade_dur = 0.5
        seg_len = 8.0
        podcast_clips = []

        t = 0.0
        last_name = second_last = None
        prev_segment = None
        prev_start = None
        prev_end = None

        while t < podcast_duration:
            t_end = min(t + seg_len - fade_dur, podcast_duration)

            # Pick a new B-roll clip that isn’t one of the last two
            candidates = [name for name in podcast_visuals if name not in (last_name, second_last)]
            chosen_name = random.choice(candidates) if candidates else random.choice(list(podcast_visuals))
            base = podcast_visuals[chosen_name]

            # Take exactly seg_len from the front of this base clip
            cur_segment = base.subclipped(0, seg_len)

            if prev_segment is None:
                # First segment: no fade-in, just add it at t=0
                first = cur_segment.with_effects([vfx.FadeOut(fade_dur)]).with_start(0)
                podcast_clips.append(first)
                prev_segment = cur_segment
                prev_start = 0.0
                prev_end = seg_len  # Correctly set prev_end to seg_len
            else:
                overlap_start = prev_end - fade_dur
                B_faded = cur_segment.with_effects([vfx.FadeIn(fade_dur)], [vfx.FadeOut(fade_dur)]).with_start(overlap_start)

                podcast_clips.append(B_faded)

                prev_segment = cur_segment
                prev_start = overlap_start
                prev_end = prev_start + seg_len  # Fix timing update here

            second_last = last_name
            last_name = chosen_name
            t = t_end

        # Fade out the last segment if no next segment exists
        if prev_segment is not None:
            tail = prev_segment.with_effects([vfx.FadeOut(fade_dur)]).with_start(prev_start)
            podcast_clips.append(tail)

        podcast_video = CompositeVideoClip(podcast_clips, size=(self.width, self.height))

        # Calculate durations
        intro_dur = intro_clip.duration
        aerial_dur = aerial_clip.duration
        podcast_dur = podcast_video.duration
        outro_dur = outro_clip.duration

        # Apply crossfade between intro, aerial, podcast, and outro
        intro_faded = intro_clip.with_effects([vfx.FadeOut(fade_dur)]).with_start(0)
        aerial_faded = aerial_clip.with_start(intro_dur - fade_dur)
        podcast_start = intro_dur + aerial_dur - fade_dur
        podcast_faded = podcast_video.with_effects([vfx.FadeOut(fade_dur)]).with_start(podcast_start)
        outro_start = intro_dur + aerial_dur + podcast_dur - fade_dur
        outro_faded = outro_clip.with_effects([vfx.FadeIn(fade_dur)]).with_start(outro_start)

        final_video = CompositeVideoClip(
            [intro_faded, aerial_faded, podcast_faded, outro_faded],
            size=(self.width, self.height)
        )

        # Set audio tracks with proper timing
        audio_tracks = [
            intro_clip.audio.with_start(0),
            aerial_clip.audio.with_start(intro_dur - fade_dur),
            podcast_audio.with_start(podcast_start),
            outro_clip.audio.with_start(outro_start)
        ]
        final_audio = CompositeAudioClip(audio_tracks)
        final_video = final_video.with_audio(final_audio)

        # Render the final video
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_video.write_videofile(
            output_file,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            preset="ultrafast",
            threads=62
        )

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
