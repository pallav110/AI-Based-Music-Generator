import torch
import numpy as np
import scipy.io.wavfile
import time
import warnings
import re
import os
import random
import traceback
from audiocraft.data.audio import audio_write
# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# This would need to be installed first: pip install audiocraft

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class EnhancedMusicComposer:
    def __init__(self):
        """Fully automated CPU-only music composition system with enhanced musicality"""
        print("🎵 Initializing Advanced Music Composer with melodic focus...")
        
        # Force CPU usage
        self.device = "cpu"
        torch.set_num_threads(4)  # Optimize CPU core usage
        torch.set_grad_enabled(False)
        
        # Load model from AudioCraft - Using medium for better quality
        self.model_name = "medium"  # Options: small, medium, melody, large
        print(f"🔄 Loading MusicGen-{self.model_name} for CPU...")
        
        try:
            # Use the AudioCraft implementation directly
            self.model = MusicGen.get_pretrained(self.model_name, device=self.device)
            
            # Set generation parameters for better musical quality
            self.model.set_generation_params(
                duration=10,  # Default duration in seconds
                temperature=0.88,  # Slightly higher for more creativity
                top_p=0.93,
                top_k=250,
                cfg_coef=3.0  # Guidance scale for better prompt adherence
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            raise RuntimeError("Failed to initialize model. See error above.")
        
        # Enhanced genre settings with focus on melodic coherence
        self.genre_settings = {
            'pop': {
                'bpm': (90, 120), 
                'key': ['C major', 'G major', 'A minor', 'F major'],
                'instruments': 'acoustic guitar, light drums, piano, bass guitar, synth pad',
                'style': 'melodic modern pop with clear vocal line, catchy chorus and flowing arrangement',
                'structure': 'intro-verse-chorus-verse-chorus-bridge-chorus-outro',
                'melodic_qualities': 'memorable hooks, consistent motifs that repeat and evolve'
            },
            'electronic': {
                'bpm': (110, 140), 
                'key': ['F minor', 'A minor', 'G minor', 'C minor'],
                'instruments': 'synth pads, electronic beats, bass, atmospheric textures, arpeggiated leads',
                'style': 'layered electronic music with evolving textures and cohesive sound design',
                'structure': 'intro-buildup-drop-breakdown-buildup-drop-outro',
                'melodic_qualities': 'pulsing motifs, gradual timbral evolution, interweaving layers'
            },
            'ballad': {
                'bpm': (60, 85), 
                'key': ['D major', 'G major', 'E minor', 'B minor'],
                'instruments': 'piano, strings, light percussion, acoustic guitar, subtle pads',
                'style': 'emotional ballad with expressive lead melody and supportive harmonies',
                'structure': 'intro-verse-chorus-verse-chorus-bridge-chorus-outro',
                'melodic_qualities': 'lyrical phrases, emotional contours, dynamic swells'
            },
            'rock': {
                'bpm': (100, 130), 
                'key': ['E minor', 'A minor', 'D major', 'G major'],
                'instruments': 'electric guitar, drums, bass, keyboard, occasional strings',
                'style': 'energetic rock with integrated guitar and rhythm section, balanced mix',
                'structure': 'intro-verse-chorus-verse-chorus-solo-chorus-outro',
                'melodic_qualities': 'guitar riffs that complement vocals, cohesive band sound'
            },
            'folk': {
                'bpm': (80, 110), 
                'key': ['G major', 'D major', 'C major', 'A minor'],
                'instruments': 'acoustic guitar, violin, mandolin, light percussion, acoustic bass',
                'style': 'organic folk with cohesive acoustic ensemble and natural blend',
                'structure': 'intro-verse-chorus-verse-chorus-bridge-chorus-outro',
                'melodic_qualities': 'interweaving acoustic instruments, natural harmonies'
            }
        }
        
        # Enhanced emotion settings with musical focus
        self.emotion_settings = {
            'happy': {
                'key_preference': ['major', 'lydian'],
                'modifiers': 'uplifting harmonies, bright timbres, flowing progressions',
                'melodic_style': 'rising melodic contours with cohesive instrumental support'
            },
            'sad': {
                'key_preference': ['minor', 'dorian'],
                'modifiers': 'rich harmonic texture, emotional depth, subtle dynamics',
                'melodic_style': 'expressive melodies with supportive harmonic movement'
            },
            'emotional': {
                'key_preference': ['minor', 'major'],
                'modifiers': 'dynamic range, expressive phrasing, cohesive arrangement',
                'melodic_style': 'melodic phrases that build and resolve with full arrangement'
            },
            'energetic': {
                'key_preference': ['major', 'mixolydian'],
                'modifiers': 'driving cohesive rhythm, dynamic band interplay, clear mix',
                'melodic_style': 'rhythmic motifs with full ensemble support and momentum'
            },
            'romantic': {
                'key_preference': ['major', 'minor'],
                'modifiers': 'rich harmonies, warm timbres, flowing melodic movement',
                'melodic_style': 'expressive melodic lines supported by harmonic texture'
            },
            'relaxed': {
                'key_preference': ['major', 'mixolydian'],
                'modifiers': 'transparent textures, gentle rhythmic integration, breathing space',
                'melodic_style': 'flowing melodic patterns with cohesive ambient support'
            }
        }
        
        print("✅ Enhanced Melodic Music Engine Ready")

    def _analyze_lyrics(self, lyrics):
        """Enhanced lyrical analysis for better genre and emotion matching"""
        lyrics_lower = lyrics.lower()
        
        # Extract structure elements
        sections = re.findall(r'\[(.*?)\]', lyrics)
        
        # Genre detection with improved keyword mapping
        genre_keywords = {
            'pop': ['love', 'dance', 'baby', 'heart', 'tonight', 'feel', 'dream'],
            'electronic': ['beat', 'rhythm', 'move', 'light', 'night', 'energy', 'pulse', 'glow'],
            'ballad': ['heart', 'tears', 'lonely', 'soul', 'cry', 'forever', 'memory'],
            'rock': ['fire', 'burn', 'fight', 'power', 'strong', 'alive', 'freedom'],
            'folk': ['river', 'mountain', 'home', 'land', 'wind', 'story', 'road']
        }
        
        genre_scores = {genre: 0 for genre in self.genre_settings.keys()}
        for genre, keywords in genre_keywords.items():
            score = sum(lyrics_lower.count(word) for word in keywords)
            # Check for section names that hint at genre
            if 'solo' in sections or 'guitar solo' in sections:
                genre_scores['rock'] += 3
            if 'drop' in sections or 'build' in sections:
                genre_scores['electronic'] += 3
            genre_scores[genre] += score
            
        # Determine most likely genre
        genre = max(genre_scores.items(), key=lambda x: x[1])[0]
        if genre_scores[genre] == 0:
            genre = 'pop'  # Default if no strong genre detected
        
        # Enhanced emotion detection
        emotion_keywords = {
            'happy': ['love', 'joy', 'smile', 'bright', 'light', 'dance', 'happy', 'sun'],
            'sad': ['sad', 'pain', 'tears', 'cry', 'lost', 'gone', 'alone', 'broken'],
            'emotional': ['heart', 'soul', 'deep', 'feel', 'emotion', 'truth', 'real'],
            'energetic': ['fire', 'run', 'jump', 'alive', 'energy', 'fast', 'power'],
            'romantic': ['kiss', 'touch', 'hold', 'close', 'passion', 'desire', 'embrace'],
            'relaxed': ['peace', 'calm', 'slow', 'gentle', 'flow', 'easy', 'breeze']
        }
        
        emotion_scores = {emotion: 0 for emotion in self.emotion_settings.keys()}
        for emotion, keywords in emotion_keywords.items():
            score = sum(lyrics_lower.count(word) for word in keywords)
            emotion_scores[emotion] += score
            
        # Determine dominant emotion
        emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        if emotion_scores[emotion] == 0:
            emotion = 'emotional'  # Default if no strong emotion detected
            
        return genre, emotion, sections

    def _calculate_duration(self, lyrics):
        """Determine song length based on sections with improved timing"""
        section_count = lyrics.count('[')  # Count all section markers
        word_count = len(re.findall(r'\b\w+\b', lyrics))  # Count words
        
        # More realistic timing calculation
        # Average vocal delivery is about 2-3 words per second in a song
        words_per_second = 2.5
        estimated_lyrics_duration = word_count / words_per_second
        
        # Each section typically has musical interludes
        section_duration = section_count * 5  # 5 seconds per section for music parts
        
        # Calculate total with minimum and maximum constraints
        total_duration = estimated_lyrics_duration + section_duration
        return max(5, min(total_duration, 30))  # 5-30 seconds (MusicGen limit)

    def _create_enhanced_prompt(self, lyrics, genre, emotion, sections, is_continuation=False, prev_segment_data=None):
        """Generate music-theory informed prompt for better composition with musical continuity"""
        settings = self.genre_settings[genre]
        emotion_config = self.emotion_settings[emotion]
        
        # Select musically appropriate settings
        bpm = random.randint(*settings['bpm'])
        key = random.choice(settings['key'])
        
        # Ensure key consistency across segments
        if is_continuation and prev_segment_data and 'key' in prev_segment_data:
            key = prev_segment_data['key']
            bpm = prev_segment_data['bpm']
        
        # Check if key aligns with emotion
        if not is_continuation:
            if 'major' in emotion_config['key_preference'] and 'minor' in key:
                # Try to find a major key alternative
                major_keys = [k for k in settings['key'] if 'major' in k]
                if major_keys:
                    key = random.choice(major_keys)
            elif 'minor' in emotion_config['key_preference'] and 'major' in key:
                # Try to find a minor key alternative
                minor_keys = [k for k in settings['key'] if 'minor' in k]
                if minor_keys:
                    key = random.choice(minor_keys)
        
        # Create specific section guidance
        section_guidance = ""
        if sections:
            unique_sections = list(dict.fromkeys(sections))  # Remove duplicates
            section_guidance = "Song sections: "
            for section in unique_sections:
                if section.lower() == 'verse':
                    section_guidance += f"{section} (establish melody, cohesive instrumental backing), "
                elif section.lower() == 'chorus':
                    section_guidance += f"{section} (unified full arrangement, memorable melodic hook), "
                elif section.lower() == 'bridge':
                    section_guidance += f"{section} (harmonic contrast with cohesive band texture), "
                elif section.lower() == 'intro':
                    section_guidance += f"{section} (establish theme with full instrumental blend), "
                elif section.lower() == 'outro':
                    section_guidance += f"{section} (resolve with cohesive arrangement), "
                else:
                    section_guidance += f"{section}, "
            section_guidance = section_guidance.rstrip(", ") + ". "
        
        # Create enhanced composition prompt with focus on melodic cohesion
        continuity_text = ""
        if is_continuation:
            continuity_text = (
                "Maintain exact musical continuity from previous segment. "
                "Continue with the same instruments, melodic themes, and harmonic progressions. "
                "Use seamless transitions between sections. "
            )
            
        # Core prompt with enhanced instrument integration focus
        base_prompt = (
            f"Compose a {emotion} {genre} song in {key} at {bpm} BPM with cohesive instrument blend. "
            f"Instruments: {settings['instruments']} working as an integrated ensemble. "
            f"Style: {settings['style']} with balanced instrumental mix. "
            f"Structure: {settings['structure']} with seamless transitions. "
            f"{section_guidance}"
            f"Musical characteristics: {emotion_config['modifiers']}. "
            f"Melody should have {emotion_config['melodic_style']} with full instrumental support. "
            f"{settings['melodic_qualities']}. "
            f"Incorporate these lyrics: {lyrics}. "
            f"{continuity_text}"
            "Create clear melodic themes with cohesive instrumental arrangement. "
            "Ensure all instruments blend naturally like a professional recording. "
            "Add musical cohesion between all instrument parts. "
            "Include natural dynamic changes with full band response. "
            "Maintain consistent harmonic framework throughout."
        )
        
        # Return prompt and settings for continuity
        return base_prompt, {'key': key, 'bpm': bpm, 'genre': genre, 'emotion': emotion}

    def compose_song(self, lyrics, output_path="enhanced_melodic_song.wav"):
        """Create a full-length song by generating and stitching segments with smooth transitions"""
        print("\n🔍 Analyzing full song structure for melodic composition...")
        
        # Extract sections from lyrics
        section_pattern = r'\[(.*?)\](.*?)(?=\[|$)'
        sections = re.findall(section_pattern, lyrics, re.DOTALL)
        
        if not sections:
            print("No section markers found. Falling back to single generation.")
            return self._generate_single_segment(lyrics, output_path)
        
        # Group sections for generation (keeping under 30s each)
        segment_groups = []
        current_group = []
        word_count = 0
        
        for section_name, section_text in sections:
            section_words = len(re.findall(r'\b\w+\b', section_text))
            
            if word_count + section_words > 60:  # ~30 seconds worth
                if current_group:
                    segment_groups.append(current_group)
                    current_group = []
                    word_count = 0
                    
            current_group.append((section_name, section_text))
            word_count += section_words
        
        if current_group:
            segment_groups.append(current_group)
        
        print(f"Creating {len(segment_groups)} coherent musical segments with unified sound")
        
        # Generate each segment with continuity information
        audio_segments = []
        prev_segment_data = None
        
        for i, segment in enumerate(segment_groups):
            # Create segment lyrics with headers
            segment_lyrics = ""
            for section_name, section_text in segment:
                segment_lyrics += f"[{section_name}]{section_text}"
            
            print(f"\nProcessing segment {i+1}/{len(segment_groups)} with melodic continuity...")
            # Generate audio with continuity information
            is_continuation = i > 0  # All segments after first are continuations
            audio_data, segment_data = self._generate_audio_segment(
                segment_lyrics, 
                is_continuation=is_continuation,
                prev_segment_data=prev_segment_data
            )
            
            if audio_data is not None:
                audio_segments.append(audio_data)
                prev_segment_data = segment_data  # Store for continuity
                print(f"✅ Successfully generated melodic segment {i+1}")
            else:
                print(f"⚠️ Failed to generate segment {i+1}")
        
        # Combine segments with crossfading for smooth transitions
        if audio_segments:
            try:
                print("\nCombining all segments with crossfading for seamless transitions...")
                full_audio = self._combine_segments_with_crossfade(audio_segments)
                
                # Save the final audio
                if self._write_audio(full_audio, output_path):
                    print(f"\n✨ Complete cohesive song saved to: {output_path}")
                    print(f"Total duration: {len(full_audio)/self.model.sample_rate:.1f}s")
                    return output_path
                else:
                    print("❌ Failed to save complete song")
                    return None
                    
            except Exception as e:
                print(f"❌ Error combining segments: {e}")
                traceback.print_exc()
                return None
        else:
            print("❌ No audio segments were successfully generated")
            return None
    
    def _combine_segments_with_crossfade(self, audio_segments):
        """Combine audio segments with crossfading for smooth transitions"""
        if not audio_segments:
            return np.array([])
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Define crossfade duration in samples (0.5 seconds)
        crossfade_duration = int(self.model.sample_rate * 0.5)
        
        # Initialize with first segment
        result = audio_segments[0]
        
        # Add each subsequent segment with crossfade
        for i in range(1, len(audio_segments)):
            # Ensure segments are at least as long as crossfade
            if len(result) < crossfade_duration or len(audio_segments[i]) < crossfade_duration:
                # If too short, just concatenate
                result = np.concatenate([result, audio_segments[i]])
                continue
            
            # Create linear crossfade weights
            fade_out = np.linspace(1.0, 0.0, crossfade_duration)
            fade_in = np.linspace(0.0, 1.0, crossfade_duration)
            
            # Apply crossfade
            result_end = result[-crossfade_duration:]
            next_start = audio_segments[i][:crossfade_duration]
            
            # Crossfaded section
            crossfaded = (result_end * fade_out) + (next_start * fade_in)
            
            # Combine everything
            result = np.concatenate([result[:-crossfade_duration], crossfaded, audio_segments[i][crossfade_duration:]])
        
        return result

    def _generate_audio_segment(self, lyrics, is_continuation=False, prev_segment_data=None):
        """Generate audio with musical continuity information"""
        try:
            # Analyze segment characteristics
            genre, emotion, sections = self._analyze_lyrics(lyrics)
            
            # Override with previous segment data if continuing
            if is_continuation and prev_segment_data:
                genre = prev_segment_data.get('genre', genre)
                emotion = prev_segment_data.get('emotion', emotion)
            
            # Create the segment prompt with continuity information
            prompt, segment_data = self._create_enhanced_prompt(
                lyrics, genre, emotion, sections, 
                is_continuation=is_continuation,
                prev_segment_data=prev_segment_data
            )
            
            # Adjust generation parameters based on whether continuing or starting
            if is_continuation:
                # Lower temperature for more consistent continuation
                self.model.set_generation_params(
                    duration=min(30, self._calculate_duration(lyrics)),
                    temperature=0.7,  # Lower for consistency
                    top_p=0.85,
                    top_k=150,
                    cfg_coef=3.5  # Higher guidance for better prompt adherence
                )
            else:
                # More creative for first segment
                self.model.set_generation_params(
                    duration=min(30, self._calculate_duration(lyrics)),
                    temperature=0.85,
                    top_p=0.92,
                    top_k=250,
                    cfg_coef=3.0
                )
            
            # Generate audio
            print(f"Generating {'continuation' if is_continuation else 'initial'} audio...")
            wav = self.model.generate([prompt])
            
            # Get numpy array from tensor - ENSURE SHAPE IS CORRECT
            audio = wav[0, 0].cpu().numpy()  # Extract correctly - accessing only the waveform
            
            print(f"Audio shape: {audio.shape}, Sample rate: {self.model.sample_rate}")
            print(f"Audio range: {np.min(audio):.6f} to {np.max(audio):.6f}")
            
            # Properly normalize the audio
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak) * 0.95  # 5% headroom to avoid clipping
            
            # Return the audio data and segment info for continuity
            return audio, segment_data
                
        except Exception as e:
            print(f"⚠️ Error generating audio: {e}")
            traceback.print_exc()
            return None, None

    def _generate_single_segment(self, lyrics, output_path):
        """Generate a complete song as a single segment with enhanced musicality"""
        try:
            # Get audio data - no continuity needed for single segment
            audio, _ = self._generate_audio_segment(lyrics)
            
            if audio is not None:
                # Save using our fixed audio write function
                success = self._write_audio(audio, output_path)
                if success:
                    return output_path
            
            return None
                
        except Exception as e:
            print(f"⚠️ Error in single segment generation: {e}")
            traceback.print_exc()
            return None

    def _write_audio(self, audio_data, file_path):
        """Write audio using a robust method that properly handles dimensions"""
        try:
            # Ensure audio_data is a numpy array with correct dimensions
            if isinstance(audio_data, torch.Tensor):
                # Make sure it's a 1D array
                audio_data = audio_data.squeeze().cpu().numpy()
            
            # Ensure it's a 1D numpy array
            audio_data = audio_data.squeeze()
            
            # If audio is silent or invalid, reject it
            if np.max(np.abs(audio_data)) < 0.01 or np.isnan(audio_data).any():
                print("⚠️ Generated audio is silent or contains invalid values")
                return False
                
            # Normalize again just to be safe
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                audio_data = (audio_data / peak) * 0.95
                
            try:
                # Try scipy first - most reliable for simple WAV writing
                scipy.io.wavfile.write(
                    file_path, 
                    self.model.sample_rate, 
                    (audio_data * 32767).astype(np.int16)
                )
                print(f"✅ Successfully saved audio to {file_path}")
                return True
            except Exception as scipy_error:
                print(f"⚠️ scipy.io.wavfile failed: {scipy_error}")
                
                # Fallback method using wave module
                try:
                    print("Trying fallback save method...")
                    import wave
                    import struct
                    
                    # Ensure audio is in -1 to 1 range
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    
                    # Convert to 16-bit PCM
                    audio_int16 = (audio_data * 32767.0).astype(np.int16)
                    
                    with wave.open(file_path, 'wb') as wf:
                        wf.setnchannels(1)  # Mono audio
                        wf.setsampwidth(2)  # 2 bytes for int16
                        wf.setframerate(self.model.sample_rate)
                        wf.writeframes(audio_int16.tobytes())
                    
                    print(f"✅ Successfully saved audio using fallback method to {file_path}")
                    return True
                    
                except Exception as e2:
                    print(f"⚠️ Fallback save method also failed: {e2}")
                    return False
                
        except Exception as e:
            print(f"⚠️ Error saving audio: {e}")
            traceback.print_exc()
            return False


# Example Usage
if __name__ == "__main__":
    # Test song with clear structure
    test_lyrics = """
    [Intro]
    
    [VERSE]
    pour up, terra no la e te de un monton
    or all of me sweep it
    around sum burn wats konvict.
    bear ah... come back up fire! waist bring en meri mas, en calle de sorat es noche
    vent dans un louco hai ha........ like he hit the volver of desires ilusin.

    [VERSE]
    for her contempt sillanp! trains and hurray the fire and the sergeant lies is consumption is burn by
    the land of desires hereditary lies tengas lo a celebration of discontent
    the blind lead us with the blind to arms, fear of death
    is screaming self obsessed burn the of lives for consumption feeding the away. self
    unholy wig the of with or in the land and my knees

    [CHORUS]
    and rest against the ground. to hate
    the price of the sky of a bottle or and the air
    the trees of god. hand of their rule and the winner. their power
    is the aesthetic of the ground. your and caution against their world

    [VERSE]
    into the passing of my brain, for all your hands he cripping
    the sheet blows the literature i'll run for a human please
    though the sun goes around you call me feel the next night my soul is like heaven were always resting
    for all the whiskey of the engine in a basement every feet will be leavin'
    you are sure that's the way they feel as we know the world and the hunted that i

    [CHORUS]
    feel all right this is the best way and live in the shadows of this world is
    that all there is and [music] is just the only way that i am in my soul [music] [applause] [music] it's
    the flash in the soul [music]
    [music] me oh yeah in the beginning there is a

    [BRIDGE]
    light in the [music] soul there is the
    [music] past [music] oh let us know the lord i set on you from a crowded
    into the city [music] [applause] [music] [applause]

    [CHORUS]
    oh [music] a hosana hosana in the
    highest hosana and worship you lord of the highest [music] hosana [music] we are here in the
    [music] world a [music]
    flash up hear my soul away and

    [CHORUS]
    sing our knees we're in my soul is fire my god hear us from [music] the flash the highest light and worship
    you need to make your name light sing your praise sing our praise to sing the
    praise of our glory cover sing our praise hosana bow bow [music] hosana bow up brea on the highest
    hosana and praise my soul the world of my heart lord lord lord lord i
    
    [Outro]
    """
    
    # Run enhanced composer
    composer = EnhancedMusicComposer()
    result = composer.compose_song(
        lyrics=test_lyrics,
        output_path="melodic_integrated_song.wav"
    )
    
    if result:
        print("\n✨ Enhanced melodic song created successfully!")
    else:
        print("\nFailed to generate enhanced song")