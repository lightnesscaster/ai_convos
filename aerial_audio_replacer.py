#!/usr/bin/env python3
"""
Aerial Video Audio Replacer
Extracts audio from aerial video, replaces "The AI Agora" phrase with TTS-1-HD voice,
while preserving background music and other audio elements.
"""

import os
import sys
import argparse
from datetime import datetime
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.effects import normalize, low_pass_filter, high_pass_filter
import speech_recognition as sr
import tempfile
import numpy as np
from audio_generator import AudioGenerator

class AerialAudioReplacer:
    def __init__(self):
        self.audio_generator = AudioGenerator()
        self.recognizer = sr.Recognizer()
        
    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> bool:
        """Extract audio from video file and save as WAV for processing"""
        try:
            print(f"Extracting audio from video: {video_path}")
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Export as WAV for better speech recognition
            audio.write_audiofile(output_audio_path, codec='pcm_s16le', logger=None)
            
            video.close()
            audio.close()
            
            print(f"Audio extracted to: {output_audio_path}")
            return True
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False
    
    def separate_vocals_and_music(self, audio_segment: AudioSegment) -> tuple:
        """
        Attempt to separate vocals from music using simple audio processing.
        Returns (vocals, music) tuple of AudioSegments.
        'vocals' is L-R (music estimate), 'music' is (L+R)/2 (vocal estimate).
        This is a basic implementation - for better results, consider using spleeter or similar.
        """
        try:
            print("Attempting to separate vocals from background music...")
            
            # Convert to stereo if not already
            if audio_segment.channels == 1:
                audio_segment = audio_segment.set_channels(2)
            
            # Extract left and right channels
            left = audio_segment.split_to_mono()[0]
            right = audio_segment.split_to_mono()[1]
            
            # Convert to numpy arrays for processing
            left_samples = np.array(left.get_array_of_samples(), dtype=np.float32)
            right_samples = np.array(right.get_array_of_samples(), dtype=np.float32)
            
            # Vocal isolation using center channel extraction
            # Vocals are typically in the center, so subtracting L-R can isolate them
            vocals_samples = left_samples - right_samples # This is the L-R music estimate
            
            # Music isolation (what remains when vocals are removed)
            # Average of left and right for background music
            music_samples = (left_samples + right_samples) / 2.0 # This is the (L+R)/2 vocal estimate
            
            # Convert back to AudioSegment
            vocals = AudioSegment(
                vocals_samples.astype(np.int16).tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=audio_segment.sample_width,
                channels=1
            )
            
            music = AudioSegment(
                music_samples.astype(np.int16).tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=audio_segment.sample_width,
                channels=1
            )
            
            # Apply some filtering to clean up the separation
            # For 'vocals' (L-R music estimate), only normalize. Filters removed to avoid "weird noises".
            # For 'music' ((L+R)/2 vocal estimate), normalize.
            
            # Normalize the separated tracks
            vocals = normalize(vocals) # L-R music estimate
            music = normalize(music)   # (L+R)/2 vocal estimate
            
            print("Audio separation completed (basic method)")
            return vocals, music
            
        except Exception as e:
            print(f"Error in audio separation: {e}")
            # Fallback: return silent as music component, original audio as vocal component
            return AudioSegment.silent(duration=len(audio_segment)), audio_segment
    
    def find_spoken_segments(self, audio_path: str) -> list:
        """
        Find segments that contain speech using audio analysis.
        Returns list of (start_time, end_time) tuples in seconds where speech is detected.
        """
        try:
            print("Analyzing audio for speech segments...")
            
            # Load audio with pydub
            audio = AudioSegment.from_wav(audio_path)
            duration = len(audio) / 1000.0
            print(f"Audio duration: {duration:.2f} seconds")
            
            # Try multiple approaches to find speech
            speech_segments = []
            
            # Approach 1: Analyze original audio directly
            print("\nApproach 1: Analyzing original audio for speech...")
            original_segments = self._detect_speech_in_audio(audio, "original")
            speech_segments.extend(original_segments)
            
            # Approach 2: Try vocal separation and analyze vocals
            print("\nApproach 2: Separating vocals and analyzing...")
            try:
                vocals, music = self.separate_vocals_and_music(audio)
                vocal_segments = self._detect_speech_in_audio(vocals, "vocals")
                speech_segments.extend(vocal_segments)
            except Exception as e:
                print(f"Vocal separation failed: {e}")
            
            # Approach 3: Use different silence detection settings
            print("\nApproach 3: Using more sensitive speech detection...")
            sensitive_segments = self._detect_speech_sensitive(audio)
            speech_segments.extend(sensitive_segments)
            
            # Remove duplicates and merge overlapping segments
            speech_segments = self._merge_overlapping_segments(speech_segments, duration)
            
            if speech_segments:
                print(f"\nFinal speech segments found:")
                for start_sec, end_sec in speech_segments:
                    print(f"  {start_sec:.2f}s - {end_sec:.2f}s (duration: {end_sec - start_sec:.2f}s)")
            else:
                print("\nNo speech segments detected with any method.")
            
            return speech_segments
            
        except Exception as e:
            print(f"Error analyzing speech segments: {e}")
            return []
    
    def _detect_speech_in_audio(self, audio_segment: AudioSegment, label: str) -> list:
        """Detect speech in an audio segment using silence detection"""
        try:
            # Use different thresholds for different audio types
            if label == "vocals":
                min_silence_len = 300  # 300ms
                silence_thresh = audio_segment.dBFS - 20  # 20dB below average
            else:
                min_silence_len = 200  # 200ms  
                silence_thresh = audio_segment.dBFS - 16  # 16dB below average
            
            print(f"  {label} audio dBFS: {audio_segment.dBFS:.1f}, silence threshold: {silence_thresh:.1f}")
            
            non_silent_ranges = detect_nonsilent(
                audio_segment,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            speech_segments = []
            for start_ms, end_ms in non_silent_ranges:
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                segment_duration = end_sec - start_sec
                
                # Be more lenient with segment duration
                if 0.3 <= segment_duration <= 8.0:  # Allow up to 8 seconds for full phrase
                    speech_segments.append((start_sec, end_sec))
                    print(f"  {label} speech segment: {start_sec:.2f}s - {end_sec:.2f}s (duration: {segment_duration:.2f}s)")
            
            return speech_segments
            
        except Exception as e:
            print(f"Error detecting speech in {label}: {e}")
            return []
    
    def _detect_speech_sensitive(self, audio_segment: AudioSegment) -> list:
        """More sensitive speech detection for quiet speech"""
        try:
            # Very sensitive settings to catch quiet speech
            min_silence_len = 100  # 100ms
            silence_thresh = audio_segment.dBFS - 25  # 25dB below average (very sensitive)
            
            print(f"  Sensitive detection - dBFS: {audio_segment.dBFS:.1f}, threshold: {silence_thresh:.1f}")
            
            non_silent_ranges = detect_nonsilent(
                audio_segment,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            speech_segments = []
            for start_ms, end_ms in non_silent_ranges:
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                segment_duration = end_sec - start_sec
                
                # Very lenient duration requirements
                if 0.2 <= segment_duration <= 8.0:
                    speech_segments.append((start_sec, end_sec))
                    print(f"  Sensitive speech segment: {start_sec:.2f}s - {end_sec:.2f}s (duration: {segment_duration:.2f}s)")
            
            return speech_segments
            
        except Exception as e:
            print(f"Error in sensitive speech detection: {e}")
            return []
    
    def _merge_overlapping_segments(self, segments: list, max_duration: float) -> list:
        """Merge overlapping segments and remove duplicates"""
        if not segments:
            return []
        
        # Remove duplicates and sort
        unique_segments = list(set(segments))
        unique_segments.sort(key=lambda x: x[0])
        
        merged = []
        for start, end in unique_segments:
            # Ensure we don't exceed audio duration
            end = min(end, max_duration)
            
            if not merged:
                merged.append((start, end))
            else:
                last_start, last_end = merged[-1]
                # If segments overlap or are very close (within 0.5s), merge them
                if start <= last_end + 0.5:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))
        
        return merged

    def analyze_speech_segment_for_phrase(self, audio_path: str, start_time: float, end_time: float, 
                                        target_phrase: str = "The AI Agora") -> bool:
        """
        Analyze a specific speech segment to see if it contains the target phrase.
        """
        try:
            # Load and extract the segment
            audio = AudioSegment.from_wav(audio_path)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment = audio[start_ms:end_ms]
            
            # Export segment for speech recognition
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                segment.export(temp_file.name, format="wav")
                
                try:
                    with sr.AudioFile(temp_file.name) as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                        audio_data = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio_data).lower()
                        
                        print(f"Segment {start_time:.2f}s-{end_time:.2f}s: '{text}'")
                        
                        # Check for target phrase or partial matches
                        target_words = set(target_phrase.lower().split())
                        recognized_words = set(text.split())
                        
                        # Calculate overlap score
                        overlap = len(target_words.intersection(recognized_words))
                        overlap_ratio = overlap / len(target_words) if target_words else 0
                        
                        print(f"  Overlap score: {overlap}/{len(target_words)} ({overlap_ratio:.2f})")
                        
                        # Consider it a match if we have significant overlap
                        return overlap_ratio >= 0.6  # 60% of words must match
                        
                except sr.UnknownValueError:
                    print(f"Segment {start_time:.2f}s-{end_time:.2f}s: [unintelligible]")
                    return False
                except sr.RequestError as e:
                    print(f"Error with speech recognition: {e}")
                    return False
                finally:
                    os.unlink(temp_file.name)
                    
        except Exception as e:
            print(f"Error analyzing speech segment: {e}")
            return False
    
    def find_target_phrase_segments(self, audio_path: str, target_phrase: str = "The AI Agora") -> list:
        """
        Find speech segments using audio waveform analysis instead of speech recognition.
        Returns list of (start_time, end_time) tuples.
        """
        try:
            print(f"Analyzing audio waveform for speech patterns...")
            
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            duration = len(audio) / 1000.0
            print(f"Audio duration: {duration:.2f} seconds")
            
            # Convert to numpy array for analysis
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)  # Convert to mono
            
            # Normalize samples
            samples = samples.astype(np.float32)
            max_abs_sample = np.max(np.abs(samples))
            if max_abs_sample > 1e-5: # Use a small epsilon to avoid division by zero/small number
                samples = samples / max_abs_sample
            else:
                print("Audio appears to be silent or near-silent, no speech detected.")
                return [] # Audio is silent or too quiet
            
            # Calculate the sample rate
            sample_rate = audio.frame_rate
            
            # Find speech using multiple audio features
            speech_segments = self._detect_speech_from_waveform(samples, sample_rate, duration)
            
            if not speech_segments:
                print("No speech detected in audio waveform")
                return []

            # Merge potentially fragmented speech segments
            merged_speech_segments = self._merge_overlapping_segments(speech_segments, duration)

            if not merged_speech_segments:
                print("No speech segments remained after merging.")
                return []
            
            # For short clips like this aerial video, likely only one speech segment
            # containing "The AI Agora" - return the longest/most prominent one
            # from the merged segments.
            if merged_speech_segments:
                # Sort by duration and take the longest
                merged_speech_segments.sort(key=lambda x: x[1] - x[0], reverse=True)
                main_speech = merged_speech_segments[0]
                print(f"Main speech segment identified (after merging): {main_speech[0]:.2f}s - {main_speech[1]:.2f}s")
                return [main_speech]
            
            return []
            
        except Exception as e:
            print(f"Error in waveform analysis: {e}")
            return []
    
    def _detect_speech_from_waveform(self, samples: np.ndarray, sample_rate: int, duration: float) -> list:
        """
        Detect speech segments using waveform analysis.
        """
        try:
            # Calculate frame size for analysis (25ms frames, 10ms hop)
            frame_size = int(0.025 * sample_rate)  # 25ms
            hop_size = int(0.010 * sample_rate)    # 10ms

            if len(samples) < frame_size:
                print("Audio too short for frame analysis.")
                return []
            
            # Calculate energy and spectral features for each frame
            frames = []
            for i in range(0, len(samples) - frame_size, hop_size):
                frame = samples[i:i + frame_size]
                
                # Calculate energy
                energy = np.sum(frame ** 2)
                
                # Calculate zero crossing rate (speech has moderate ZCR)
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                
                # Calculate spectral centroid (speech has characteristic spectral shape)
                fft = np.fft.fft(frame)
                magnitude = np.abs(fft[:len(fft)//2])
                freqs = np.fft.fftfreq(len(fft), 1/sample_rate)[:len(fft)//2]
                if np.sum(magnitude) > 0:
                    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                else:
                    spectral_centroid = 0
                
                frames.append({
                    'time': i / sample_rate,
                    'energy': energy,
                    'zcr': zcr,
                    'spectral_centroid': spectral_centroid
                })

            if not frames:
                print("No frames generated for analysis (audio might be too short).")
                return []
            
            # Normalize features
            energies = [f['energy'] for f in frames]
            zcrs = [f['zcr'] for f in frames]
            centroids = [f['spectral_centroid'] for f in frames]
            
            if not energies: # Should be caught by 'if not frames' but as a safeguard
                print("No energy features extracted from frames.")
                return []

            energy_mean = np.mean(energies)
            energy_std = np.std(energies)
            # Handle case of constant energy or very low variance
            if energy_std < 1e-6: 
                energy_std = energy_mean * 0.1 if energy_mean > 1e-6 else 0.001 

            zcr_mean = np.mean(zcrs)
            centroid_mean = np.mean(centroids)
            
            print(f"Audio features - Energy mean: {energy_mean:.6f}, Energy std: {energy_std:.6f}, ZCR mean: {zcr_mean:.4f}, Spectral centroid mean: {centroid_mean:.1f} Hz")
            
            # Identify speech frames using thresholds
            speech_frames = []
            for frame in frames:
                # Speech detection criteria:
                # 1. Higher than average energy
                # 2. Moderate zero crossing rate (not too high like noise, not too low like music)
                # 3. Spectral centroid in speech range (roughly 250-5500 Hz)
                
                # Adaptive primary energy threshold
                factor_std_primary = 0.3
                if energy_std > 1.5 * energy_mean: # If highly dynamic audio
                    factor_std_primary = 0.05 # More sensitive for high std
                energy_threshold = energy_mean + factor_std_primary * energy_std
                
                zcr_threshold_low = zcr_mean * 0.4 # More permissive
                zcr_threshold_high = zcr_mean * 2.5 # More permissive
                centroid_threshold_low = 250  # Hz, More permissive
                centroid_threshold_high = 5500  # Hz, More permissive
                
                is_speech = (
                    frame['energy'] > energy_threshold and
                    zcr_threshold_low < frame['zcr'] < zcr_threshold_high and
                    centroid_threshold_low < frame['spectral_centroid'] < centroid_threshold_high
                )
                
                if is_speech:
                    speech_frames.append(frame['time'])
            
            if not speech_frames:
                # Fallback: use energy-based detection only
                print("No speech detected with full criteria, falling back to energy-based detection")
                
                # Adaptive fallback relative energy threshold
                factor_std_fallback = 0.1
                if energy_std > 1.5 * energy_mean: # If highly dynamic audio
                    factor_std_fallback = 0.02 # More sensitive for high std
                energy_threshold_relative = energy_mean + factor_std_fallback * energy_std
                
                # Absolute minimum threshold: 0.2% of max possible frame energy
                min_abs_energy_threshold = 0.002 * frame_size 
                
                final_energy_threshold = max(energy_threshold_relative, min_abs_energy_threshold)
                print(f"  Fallback energy threshold: {final_energy_threshold:.6f} (relative: {energy_threshold_relative:.6f}, min_abs: {min_abs_energy_threshold:.6f})")

                speech_frames = [f['time'] for f in frames if f['energy'] > final_energy_threshold]
            
            if not speech_frames:
                print("No speech detected with any waveform analysis method")
                return []
            
            # Group consecutive speech frames into segments
            speech_segments = []
            if speech_frames:
                segment_start = speech_frames[0]
                last_time = speech_frames[0]
                
                for time in speech_frames[1:]:
                    if time - last_time > 0.25:  # 250ms gap indicates new segment (increased from 0.1s)
                        # End current segment
                        speech_segments.append((segment_start, last_time + 0.025))  # Add frame duration
                        segment_start = time
                    last_time = time
                
                # Add final segment
                speech_segments.append((segment_start, last_time + 0.025))
            
            # Filter segments by duration (remove very short ones)
            filtered_segments = []
            for start, end in speech_segments:
                if end - start >= 0.15:  # At least 150ms (reduced from 300ms)
                    filtered_segments.append((start, end))
                    print(f"Speech segment detected: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            
            return filtered_segments
            
        except Exception as e:
            print(f"Error in waveform analysis: {e}")
            return []
    
    def generate_replacement_audio(self, text: str, voice_style: str = "authoritative and welcoming") -> AudioSegment:
        """Generate TTS-1-HD audio for replacement text"""
        try:
            print(f"Generating TTS replacement for: '{text}'")
            audio_segment = self.audio_generator.text_to_speech(text, voice_style)
            return audio_segment
            
        except Exception as e:
            print(f"Error generating replacement audio: {e}")
            return None
    
    def replace_vocal_segments_preserve_music(self, original_audio_path: str, replacements: list, output_path: str) -> bool:
        """
        Replace vocal segments with TTS-generated audio while preserving background music
        using a "duck and overlay" method.
        replacements: list of (start_time, end_time, replacement_audio_segment) tuples
        """
        try:
            print("Replacing vocal segments using duck and overlay method...")
            
            # Load original audio
            original_audio = AudioSegment.from_wav(original_audio_path)
            
            # Start with a copy of the original audio
            modified_audio = original_audio
            
            # Sort replacements by start time (in reverse to avoid index shifting)
            # This is important if segments could overlap, though less critical here
            # as we are modifying in place on a copy.
            replacements.sort(key=lambda x: x[0], reverse=True)
            
            # Apply replacements
            for start_time, end_time, replacement_audio in replacements:
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                print(f"Replacing segment {start_time:.2f}s - {end_time:.2f}s")
                
                # Extract the original segment that will be ducked
                original_segment_to_duck = modified_audio[start_ms:end_ms]
                
                # Ensure replacement audio matches the duration
                segment_duration_ms = end_ms - start_ms
                
                if len(replacement_audio) != segment_duration_ms:
                    if len(replacement_audio) > segment_duration_ms:
                        replacement_audio = replacement_audio[:segment_duration_ms]
                    else:
                        padding_duration_ms = segment_duration_ms - len(replacement_audio)
                        padding = AudioSegment.silent(duration=padding_duration_ms)
                        replacement_audio = replacement_audio + padding
                
                # Duck the original segment (e.g., reduce volume by 20dB for a more noticeable effect)
                ducking_amount_db = -20  # Increased ducking
                ducked_original_segment = original_segment_to_duck + ducking_amount_db
                
                # No boost for TTS for now, to isolate ducking effect
                tts_boost_db = 0 # Removed TTS boost for diagnosis
                boosted_replacement_audio = replacement_audio + tts_boost_db
                
                # Overlay the TTS onto the ducked original segment
                mixed_segment = ducked_original_segment.overlay(boosted_replacement_audio)
                
                # Replace the segment in the main audio
                before = modified_audio[:start_ms]
                after = modified_audio[end_ms:]
                modified_audio = before + mixed_segment + after
            
            # Export modified audio
            modified_audio.export(output_path, format="wav")
            print(f"Modified audio with ducking saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error replacing vocal segments with ducking: {e}")
            # Fallback to simple replacement (no ducking, just overwrite)
            print("Falling back to simple audio replacement (no ducking)...")
            return self.replace_audio_segments_simple(original_audio_path, replacements, output_path)
    
    def replace_audio_segments_simple(self, original_audio_path: str, replacements: list, output_path: str) -> bool:
        """
        Simple replacement method (fallback) - replaces entire audio segment.
        """
        try:
            print("Using simple audio replacement method...")
            
            # Load original audio
            original_audio = AudioSegment.from_wav(original_audio_path)
            
            # Sort replacements by start time (in reverse to avoid index shifting)
            replacements.sort(key=lambda x: x[0], reverse=True)
            
            # Apply replacements
            modified_audio = original_audio
            for start_time, end_time, replacement_audio in replacements:
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)
                
                print(f"Replacing segment {start_time:.2f}s - {end_time:.2f}s")
                
                # Split audio and insert replacement
                before = modified_audio[:start_ms]
                after = modified_audio[end_ms:]
                
                # Combine with replacement audio
                modified_audio = before + replacement_audio + after
            
            # Export modified audio
            modified_audio.export(output_path, format="wav")
            print(f"Modified audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error replacing audio segments: {e}")
            return False
    
    def create_new_video(self, original_video_path: str, new_audio_path: str, output_video_path: str) -> bool:
        """Create new video with replaced audio"""
        try:
            print("Creating new video with replaced audio...")
            
            # Load original video and new audio
            video = VideoFileClip(original_video_path)
            new_audio = AudioFileClip(new_audio_path)
            
            # Ensure audio duration matches video duration
            if new_audio.duration > video.duration:
                new_audio = new_audio.subclipped(0, video.duration)
            elif new_audio.duration < video.duration:
                # Pad with silence if needed
                silence_duration = video.duration - new_audio.duration
                print(f"Padding audio with {silence_duration:.2f}s of silence")
            
            # Set new audio to video
            final_video = video.with_audio(new_audio)
            
            # Write output video
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            final_video.write_videofile(
                output_video_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None
            )
            
            # Cleanup
            video.close()
            new_audio.close()
            final_video.close()
            
            print(f"New video created: {output_video_path}")
            return True
            
        except Exception as e:
            print(f"Error creating new video: {e}")
            return False
    
    def process_aerial_video(self, video_path: str, target_phrase: str = "The AI Agora", 
                           replacement_text: str = None, output_dir: str = "output") -> str:
        """
        Main processing function that handles the complete workflow.
        Returns path to the new video file or None if failed.
        """
        if replacement_text is None:
            replacement_text = target_phrase  # Use same text with TTS voice
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create temporary and output paths
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_audio_path = os.path.join(temp_dir, f"{base_name}_extracted_{timestamp}.wav")
        modified_audio_path = os.path.join(temp_dir, f"{base_name}_modified_{timestamp}.wav")
        output_video_path = os.path.join(output_dir, f"{base_name}_tts_replaced_{timestamp}.mp4")
        
        try:
            # Step 1: Extract audio from video
            if not self.extract_audio_from_video(video_path, extracted_audio_path):
                return None
            
            # Step 2: Find target phrase in speech segments
            phrase_locations = self.find_target_phrase_segments(extracted_audio_path, target_phrase)
            
            if not phrase_locations:
                print(f"No occurrences of '{target_phrase}' found in audio.")
                print("You may need to manually specify the time ranges or check the audio quality.")
                return None
            
            # Step 3: Generate replacement audio for each occurrence
            replacements = []
            for start_time, end_time in phrase_locations:
                replacement_audio = self.generate_replacement_audio(replacement_text)
                if replacement_audio:
                    replacements.append((start_time, end_time, replacement_audio))
            
            if not replacements:
                print("Failed to generate replacement audio.")
                return None
            
            # Step 4: Replace vocal segments while preserving music
            if not self.replace_vocal_segments_preserve_music(extracted_audio_path, replacements, modified_audio_path):
                return None
            
            # Step 5: Create new video with modified audio
            if not self.create_new_video(video_path, modified_audio_path, output_video_path):
                return None
            
            print(f"\nSuccess! New video with TTS-replaced vocals and preserved music: {output_video_path}")
            return output_video_path
            
        except Exception as e:
            print(f"Error in processing workflow: {e}")
            return None
        
        finally:
            # Cleanup temporary files
            for temp_file in [extracted_audio_path, modified_audio_path]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass

    def process_aerial_video_manual(self, video_path: str, start_time: float, end_time: float, 
                                   replacement_text: str = "The AI Agora", output_dir: str = "output") -> str:
        """
        Processing function that uses manual timing for the target phrase.
        Returns path to the new video file or None if failed.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create temporary and output paths
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_audio_path = os.path.join(temp_dir, f"{base_name}_extracted_{timestamp}.wav")
        modified_audio_path = os.path.join(temp_dir, f"{base_name}_modified_{timestamp}.wav")
        output_video_path = os.path.join(output_dir, f"{base_name}_tts_replaced_{timestamp}.mp4")
        
        try:
            # Step 1: Extract audio from video
            if not self.extract_audio_from_video(video_path, extracted_audio_path):
                return None
            
            # Step 2: Use manual timing for phrase locations
            phrase_locations = [(start_time, end_time)]
            print(f"Using manual phrase location: {start_time}s - {end_time}s")
            
            # Step 3: Generate replacement audio for each occurrence
            replacements = []
            for start_time, end_time in phrase_locations:
                replacement_audio = self.generate_replacement_audio(replacement_text)
                if replacement_audio:
                    replacements.append((start_time, end_time, replacement_audio))
            
            if not replacements:
                print("Failed to generate replacement audio.")
                return None
            
            # Step 4: Replace audio segments
            if not self.replace_audio_segments_simple(extracted_audio_path, replacements, modified_audio_path):
                return None
            
            # Step 5: Create new video with modified audio
            if not self.create_new_video(video_path, modified_audio_path, output_video_path):
                return None
            
            print(f"\nSuccess! New video with TTS-replaced audio: {output_video_path}")
            return output_video_path
            
        except Exception as e:
            print(f"Error in processing workflow: {e}")
            return None
        
        finally:
            # Cleanup temporary files
            for temp_file in [extracted_audio_path, modified_audio_path]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass

    def _remove_overlapping_detections(self, detections: list, min_gap: float = 0.5) -> list:
        """Remove overlapping or very close detections"""
        if not detections:
            return detections
        
        # Sort by start time
        detections.sort(key=lambda x: x[0])
        
        filtered = [detections[0]]
        for current in detections[1:]:
            last = filtered[-1]
            # If current detection starts before last one ends + min_gap, skip it
            if current[0] < last[1] + min_gap:
                continue
            filtered.append(current)
        
        return filtered

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Replace 'The AI Agora' phrase in aerial video with TTS-1-HD voice"
    )
    parser.add_argument(
        "video_path",
        help="Path to the aerial video file"
    )
    parser.add_argument(
        "--phrase",
        default="The AI Agora",
        help="Target phrase to replace (default: 'The AI Agora')"
    )
    parser.add_argument(
        "--replacement",
        default=None,
        help="Replacement text (default: same as target phrase)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for the new video (default: 'output')"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Initialize processor
    processor = AerialAudioReplacer()
    
    # Process the video with automatic detection
    result = processor.process_aerial_video(
        args.video_path,
        args.phrase,
        args.replacement,
        args.output_dir
    )
    
    if result:
        print(f"\nProcessing complete! Output: {result}")
        return 0
    else:
        print("\nProcessing failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())