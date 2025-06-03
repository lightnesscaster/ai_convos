from moviepy import AudioFileClip, ImageClip, CompositeVideoClip, VideoFileClip, CompositeAudioClip
from moviepy import vfx
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
import subprocess
import tempfile
import json
import sys
import threading
from typing import List, Dict

class VideoGenerator:
    def __init__(self, width: int = 1080, height: int = 720, fps: int = 24):
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
                codec='h264_videotoolbox',  # Use hardware-accelerated codec
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
                    codec='h264_videotoolbox',  # Use hardware-accelerated codec
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
        last_three_names = [] # Stores the names of the last three clips used
        prev_segment = None
        prev_start = None
        prev_end = None

        while t < podcast_duration:
            t_end = min(t + seg_len - fade_dur, podcast_duration)
            remaining_time = podcast_duration - t

            # Pick a new B-roll clip that isn't one of the last three
            candidates = [name for name in podcast_visuals if name not in last_three_names]
            if not candidates: # If all clips have been used in the last three turns, pick any
                candidates = list(podcast_visuals.keys())
            
            chosen_name = random.choice(candidates)
            base = podcast_visuals[chosen_name]

            # For the last clip, use only the remaining time
            clip_duration = min(seg_len, remaining_time)
            cur_segment = base.subclipped(0, clip_duration)

            if prev_segment is None:
                # First segment: no fade-in, just add it at t=0
                first = cur_segment.with_effects([vfx.CrossFadeOut(fade_dur)]).with_start(0)
                podcast_clips.append(first)
                prev_segment = cur_segment
                prev_start = 0.0
                prev_end = clip_duration
            else:
                overlap_start = prev_end - fade_dur
                B_faded = cur_segment.with_effects([vfx.CrossFadeIn(fade_dur), vfx.CrossFadeOut(fade_dur)]).with_start(overlap_start)

                podcast_clips.append(B_faded)

                prev_segment = cur_segment
                prev_start = overlap_start
                prev_end = prev_start + clip_duration

            # Update the list of last three used clips
            last_three_names.append(chosen_name)
            if len(last_three_names) > 3:
                last_three_names.pop(0)
            
            t = t_end


        podcast_video = CompositeVideoClip(podcast_clips, size=(self.width, self.height))

        # Calculate durations
        intro_dur = intro_clip.duration
        aerial_dur = aerial_clip.duration
        podcast_dur = podcast_video.duration
        outro_dur = outro_clip.duration

        # Apply crossfade between intro, aerial, podcast, and outro
        intro_faded = intro_clip.with_effects([vfx.CrossFadeOut(fade_dur)]).with_start(0)
        aerial_faded = aerial_clip.with_start(intro_dur - fade_dur)
        podcast_start = intro_dur + aerial_dur - fade_dur
        podcast_faded = podcast_video.with_effects([vfx.CrossFadeOut(fade_dur)]).with_start(podcast_start)
        outro_start = intro_dur + aerial_dur + podcast_dur - 2*fade_dur
        outro_faded = outro_clip.with_effects([vfx.CrossFadeIn(fade_dur)]).with_start(outro_start)

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
            codec="h264_videotoolbox",  # Use hardware-accelerated codec
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            preset="ultrafast",
            threads=8
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

    def create_custom_video_flow_ffmpeg(self, podcast_audio_file: str, output_file: str, skip_normalize: bool = True):
        """Pure FFmpeg-based custom video flow with proper crossfades using correct offset calculations."""
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

        # Get durations using ffprobe
        def get_duration(file_path):
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())

        def get_video_duration(file_path):
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "stream=duration",
                "-select_streams", "v:0", "-of", "csv=p=0", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())

        # Get all durations - using consistent variable names
        P = get_duration(podcast_audio_file)  # podcast_duration
        I = get_video_duration(intro_video_path)  # intro_dur
        A = get_video_duration(aerial_video_path)  # aerial_dur
        O = get_video_duration(outro_video_path)  # outro_dur
        F = 0.5  # fade_dur

        print(f"Durations: Intro={I:.2f}s, Aerial={A:.2f}s, Podcast={P:.2f}s, Outro={O:.2f}s, Fade={F}s")

        # Generate random B-roll visuals for podcast
        podcast_visuals_dir = "output/clips/random"
        if not os.path.exists(podcast_visuals_dir):
            raise FileNotFoundError(f"Podcast visuals directory not found: {podcast_visuals_dir}")

        podcast_visual_files = [
            os.path.join(podcast_visuals_dir, f) for f in os.listdir(podcast_visuals_dir)
            if f.endswith(".mp4")
        ]

        if len(podcast_visual_files) < 3:
            raise ValueError("Not enough podcast visual clips for randomness.")

        seg_len = 8.0

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            if not skip_normalize:
                print("Step 1: Resize and prepare all input videos...")
                # Step 1: Resize all inputs to consistent H.264 encoding (video-only)
                intro_resized = os.path.join(tmpdir, "intro_resized.mp4")
                aerial_resized = os.path.join(tmpdir, "aerial_resized.mp4")
                outro_resized = os.path.join(tmpdir, "outro_resized.mp4")

                # Resize intro, aerial, outro (strip audio to avoid confusion)
                for src, dst, duration in [(intro_video_path, intro_resized, I), (aerial_video_path, aerial_resized, A), (outro_video_path, outro_resized, O)]:
                    cmd = [
                        "ffmpeg", "-i", src, 
                        "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,fps={self.fps}",
                        "-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
                        "-r", str(self.fps),  # Force 24fps
                        "-an",  # Remove audio track entirely
                        "-preset", "ultrafast", 
                        "-progress", "pipe:1", "-nostats",  # Enable progress tracking
                        "-y", dst
                    ]
                    ret = self._run_ffmpeg_with_progress(cmd, duration, f"resize {os.path.basename(src)}")
                    if ret != 0:
                        raise RuntimeError(f"FFmpeg resize failed for {src}")
            else:
                print("Skipping resize step (clips already consistent)...")
                intro_resized = intro_video_path
                aerial_resized = aerial_video_path
                outro_resized = outro_video_path

            print("Step 2: Create B-roll sequence...")
            # Step 2: Create B-roll sequence - collect all segments first
            broll_segments = []
            segment_durations = []
            t = 0.0
            last_three_files = []
            segment_idx = 0

            # Calculate all segments and their files
            while t < P:
                remaining_time = P - t
                clip_duration = min(seg_len, remaining_time)
                
                # Pick a random B-roll clip that wasn't used in last 3 segments
                candidates = [f for f in podcast_visual_files if f not in last_three_files]
                if not candidates:
                    candidates = podcast_visual_files

                chosen_file = random.choice(candidates)
                
                # Update last three used files
                last_three_files.append(chosen_file)
                if len(last_three_files) > 3:
                    last_three_files.pop(0)

                broll_segments.append(chosen_file)
                segment_durations.append(clip_duration)
                
                # Fix: Always advance by at least the remaining time to avoid infinite loop
                if remaining_time <= seg_len:
                    # This is the final segment, advance by remaining time to exit loop
                    t = P
                else:
                    # Normal advancement with fade overlap
                    t += seg_len - F
                segment_idx += 1

            print(f"Generated {len(broll_segments)} B-roll segments")

            if not skip_normalize:
                # Normalize all B-roll clips first to ensure consistent frame rates
                normalized_broll = []
                for i, segment_file in enumerate(broll_segments):
                    normalized_file = os.path.join(tmpdir, f"broll_normalized_{i}.mp4")
                    duration = segment_durations[i]
                    
                    cmd = [
                        "ffmpeg", "-i", segment_file,
                        "-t", str(duration),
                        "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,fps={self.fps}",
                        "-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
                        "-r", str(self.fps),  # Force consistent frame rate
                        "-g", str(self.fps),  # Set keyframe interval for better compatibility
                        "-an", "-preset", "ultrafast",
                        "-avoid_negative_ts", "make_zero",  # Fix timestamp issues
                        "-progress", "pipe:1", "-nostats",
                        "-y", normalized_file
                    ]
                    ret = self._run_ffmpeg_with_progress(cmd, duration, f"normalize broll {i+1}/{len(broll_segments)}")
                    if ret != 0:
                        raise RuntimeError(f"FFmpeg B-roll normalization failed for segment {i}")
                    normalized_broll.append(normalized_file)
            else:
                print("Skipping B-roll normalization (clips already consistent)...")
                normalized_broll = broll_segments

            # Create B-roll video using concat instead of xfade for simpler approach
            if len(normalized_broll) == 1:
                # Single segment case - just copy the normalized file
                podcast_video = os.path.join(tmpdir, "podcast_video.mp4")
                cmd = [
                    "ffmpeg", "-i", normalized_broll[0],
                    "-c", "copy",
                    "-y", podcast_video
                ]
                ret = self._run_ffmpeg_with_progress(cmd, segment_durations[0], "b-roll single")
                if ret != 0:
                    raise RuntimeError("FFmpeg B-roll creation failed")
            else:
                # Use concat with crossfade for more reliable results
                # First create a concat list file
                concat_file = os.path.join(tmpdir, "broll_list.txt")
                with open(concat_file, 'w') as f:
                    for normalized_file in normalized_broll:
                        # Ensure we use absolute paths for concat
                        abs_path = os.path.abspath(normalized_file)
                        f.write(f"file '{abs_path}'\n")
                
                # Simple concatenation without xfade for now (more reliable)
                podcast_video = os.path.join(tmpdir, "podcast_video.mp4")
                cmd = [
                    "ffmpeg",
                    "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-c", "copy" if skip_normalize else "h264_videotoolbox",
                    "-progress", "pipe:1", "-nostats",
                    "-y", podcast_video
                ]
                if not skip_normalize:
                    # Add encoding parameters only when normalizing
                    cmd.extend(["-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0", "-r", str(self.fps), "-preset", "ultrafast"])
                
                ret = self._run_ffmpeg_with_progress(cmd, P, "b-roll concat")
                if ret != 0:
                    raise RuntimeError("FFmpeg B-roll concat failed")

            print("Step 3: Create video-only final sequence with correct xfade offsets...")
            # Step 3: Create final video sequence with correct xfade offsets
            # Calculate correct offsets using the fixed formula:
            # First xfade offset = I - F
            # Second xfade offset = I + A - 2*F  
            # Third xfade offset = I + A + P - 3*F
            
            offset1 = I - F
            offset2 = I + A - 2*F
            offset3 = I + A + P - 3*F
            
            print(f"Crossfade offsets: {offset1:.2f}s, {offset2:.2f}s, {offset3:.2f}s")

            silent_video = os.path.join(tmpdir, "silent_video.mp4")
            
            video_filter = f"""
[0:v]trim=0:{I},setpts=PTS-STARTPTS,fps={self.fps}[intro];
[1:v]trim=0:{A},setpts=PTS-STARTPTS,fps={self.fps}[aerial];
[2:v]trim=0:{P},setpts=PTS-STARTPTS,fps={self.fps}[podcast];
[3:v]trim=0:{O},setpts=PTS-STARTPTS,fps={self.fps}[outro];
[intro][aerial]xfade=transition=fade:duration={F}:offset={offset1}[xa];
[xa][podcast]xfade=transition=fade:duration={F}:offset={offset2}[xab];
[xab][outro]xfade=transition=fade:duration={F}:offset={offset3}[finalv]
"""

            cmd = [
                "ffmpeg",
                "-i", intro_resized,
                "-i", aerial_resized,
                "-i", podcast_video,
                "-i", outro_resized,
                "-filter_complex", video_filter,
                "-map", "[finalv]",
                "-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
                "-preset", "ultrafast", "-threads", "8", 
                "-progress", "pipe:1", "-nostats",  # Enable progress tracking
                "-y", silent_video
            ]
            ret = self._run_ffmpeg_with_progress(cmd, I + A + P + O - 3*F, "final video")
            if ret != 0:
                raise RuntimeError("FFmpeg final video creation failed")

            print("Step 4: Create audio track with proper timing...")
            # Step 4: Create audio track with correct delays to match video crossfades
            # Audio delays must match exactly where video segments start
            intro_delay = 0
            aerial_delay = int((I - F) * 1000)  # Aerial starts when intro crossfade begins
            podcast_delay = int((I + A - 2*F) * 1000)  # Podcast starts when aerial-podcast crossfade begins  
            outro_delay = int((I + A + P - 3*F) * 1000)  # Outro starts when podcast-outro crossfade begins
            
            print(f"Audio delays: intro={intro_delay}ms, aerial={aerial_delay}ms, podcast={podcast_delay}ms, outro={outro_delay}ms")
            
            final_audio = os.path.join(tmpdir, "final_audio.aac")
            
            # Create silent audio for intro/aerial/outro and use actual podcast audio
            audio_filter = f"""
[0:a]afade=out:st={I-F}:d={F}[intro_a];
[1:a]afade=in:st=0:d={F},afade=out:st={A-F}:d={F}[aerial_a];
[2:a]afade=in:st=0:d={F},afade=out:st={P-F}:d={F}[podcast_a];
[3:a]afade=in:st=0:d={F}[outro_a];
[intro_a]adelay={intro_delay}|{intro_delay}[intro_delayed];
[aerial_a]adelay={aerial_delay}|{aerial_delay}[aerial_delayed];
[podcast_a]adelay={podcast_delay}|{podcast_delay}[podcast_delayed];
[outro_a]adelay={outro_delay}|{outro_delay}[outro_delayed];
[intro_delayed][aerial_delayed][podcast_delayed][outro_delayed]amix=inputs=4:duration=longest[finala]
"""

            cmd = [
                "ffmpeg",
                "-i", intro_video_path,  # Use original files with audio
                "-i", aerial_video_path,
                "-i", podcast_audio_file,  # Direct podcast audio (not embedded in video)
                "-i", outro_video_path,
                "-filter_complex", audio_filter,
                "-map", "[finala]",
                "-c:a", "aac", 
                "-progress", "pipe:1", "-nostats",  # Enable progress tracking
                "-y", final_audio
            ]
            ret = self._run_ffmpeg_with_progress(cmd, I + A + P + O - 3*F, "audio mix")
            if ret != 0:
                raise RuntimeError("FFmpeg audio mixing failed")

            print("Step 5: Mux video and audio...")
            # Step 5: Final mux - combine silent video with timed audio
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            cmd = [
                "ffmpeg",
                "-i", silent_video,
                "-i", final_audio,
                "-c:v", "copy", "-c:a", "copy", "-shortest",
                "-progress", "pipe:1", "-nostats",  # Enable progress tracking
                "-y", output_file
            ]
            ret = self._run_ffmpeg_with_progress(cmd, I + A + P + O - 3*F, "final mux")
            if ret != 0:
                raise RuntimeError("FFmpeg final mux failed")
        print(f"Custom video flow (Pure FFmpeg with correct crossfades) successfully created: {output_file}")
        print(f"Total expected duration: {I + A + P + O - 3*F:.2f} seconds")

    def _run_ffmpeg_with_progress(
        self,
        cmd: List[str],
        total_duration: float = None,
        description: str = "ffmpeg"
    ) -> int:
        """
        Run the given ffmpeg command with real-time progress tracking.
        
        Args:
            cmd: FFmpeg command as list of arguments (must include -progress pipe:1 -nostats)
            total_duration: Expected output duration in seconds for progress calculation
            description: Label to show in progress output
            
        Returns:
            Return code from ffmpeg process
        """
        # Launch ffmpeg process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # line-buffered
        )

        # If no duration provided, just run without progress
        if total_duration is None:
            print(f"[{description}] Running without progress tracking...")
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"\n[{description}] Error: {stderr}")
            return proc.returncode

        # Parse progress info from ffmpeg -progress pipe:1 output
        percent_done = 0.0
        last_print = 0.0
        print(f"[{description}] Starting...")

        def reader():
            nonlocal percent_done, last_print
            try:
                for raw in proc.stdout:
                    line = raw.strip()
                    if line.startswith("out_time_ms="):
                        try:
                            ms = int(line.split("=")[1])
                            sec = ms / 1000.0
                            p = min(sec / total_duration * 100.0, 100.0)
                            # Only print if progress increased by ≥2% to avoid spam
                            if p - last_print >= 2.0 or p >= 99.0:
                                last_print = p
                                percent_done = p
                                sys.stdout.write(f"\r[{description}] {p:6.2f}% ")
                                sys.stdout.flush()
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("out_time_us="):
                        try:
                            us = int(line.split("=")[1])
                            sec = us / 1000000.0
                            p = min(sec / total_duration * 100.0, 100.0)
                            # Only print if progress increased by ≥2% to avoid spam
                            if p - last_print >= 2.0 or p >= 99.0:
                                last_print = p
                                percent_done = p
                                sys.stdout.write(f"\r[{description}] {p:6.2f}% ")
                                sys.stdout.flush()
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("progress="):
                        if line.endswith("end"):
                            percent_done = 100.0
                            sys.stdout.write(f"\r[{description}] {100.00:6.2f}% ")
                            sys.stdout.flush()
                            print()  # New line after completion
            except Exception as e:
                print(f"\n[{description}] Progress reader error: {e}")

        # Run reader in separate thread
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        
        # Wait for process to complete and get stderr
        _, stderr = proc.communicate()
        thread.join(timeout=1)  # Give thread a moment to finish
        
        # Ensure we always print 100% at the end
        if proc.returncode == 0:
            sys.stdout.write(f"\r[{description}] 100.00% ")
            sys.stdout.flush()
            print()  # New line
        
        # Print any errors from stderr if process failed
        if proc.returncode != 0:
            if stderr:
                print(f"\n[{description}] Error: {stderr}")
        
        return proc.returncode
# Example usage

if __name__ == "__main__":
    convo = [
        {"speaker": "Claude",   "text": "Hello, I'm Claude."},
        {"speaker": "ChatGPT",  "text": "And I'm ChatGPT."},
        {"speaker": "DeepSeek", "text": "DeepSeek checking in."},
    ]
    vg = VideoGenerator()
    vg.create_conversation_video(convo, "dialogue.mp3", "out/video.mp4")
