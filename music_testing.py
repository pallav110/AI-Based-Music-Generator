# import os
# import subprocess
# import numpy as np
# import soundfile as sf
# import mido
# from mido import Message, MidiFile, MidiTrack
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# import re
# import os
# from midi2audio import FluidSynth  # Easier alternative to direct fluidsynth calls
# nltk.download('vader_lexicon')

# class AI_LyricsToMusic:
#     def __init__(self, bpm=120, sample_rate=44100):
#         self.bpm = bpm
#         self.sample_rate = sample_rate
#         self.sia = SentimentIntensityAnalyzer()
#         self.notes = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}
        
#     def analyze_lyrics(self, lyrics):
#         """
#         Detects the mood, genre, and BPM based on lyrics.
#         """
#         sentiment = self.sia.polarity_scores(lyrics)
        
#         if sentiment['compound'] >= 0.5:
#             mood = 'happy'
#             genre = 'pop'
#             self.bpm = 120 + len(lyrics.split()) // 2  # Faster tempo for more words
#         elif sentiment['compound'] <= -0.3:
#             mood = 'sad'
#             genre = 'ballad'
#             self.bpm = 80
#         else:
#             mood = 'calm'
#             genre = 'lofi'
#             self.bpm = 90

#         print(f"🎶 Detected Mood: {mood}, Genre: {genre}, BPM: {self.bpm}")
#         return mood, genre, self.bpm

#     def generate_midi(self, lyrics, filename="generated_song.mid"):
#         """
#         Convert lyrics into a MIDI melody.
#         """
#         midi = MidiFile()
#         track = MidiTrack()
#         midi.tracks.append(track)

#         words = lyrics.split()
#         note_sequence = list(self.notes.values())[:len(words)]  # Assign notes to words
        
#         for note in note_sequence:
#             track.append(Message('note_on', note=note, velocity=64, time=0))
#             track.append(Message('note_off', note=note, velocity=64, time=int(mido.bpm2tempo(self.bpm) / 2)))

#         midi.save(filename)
#         print(f"🎵 MIDI file saved as {filename}")
#         return filename



#     import subprocess

#     import os
#     import subprocess

#     import os
#     import subprocess

#     def convert_midi_to_wav(self, midi_file, soundfont="soundfont.sf2", output_file="final_song.wav"):
#         """
#         Convert MIDI to WAV using FluidSynth and log any errors.
#         """
#         fluidsynth_path = "C:\\Users\\pallav\\Downloads\\fluidsynth-2.4.4-win10-x64\\bin\\fluidsynth.exe"
#         soundfont_path = "C:\\Users\\pallav\\Desktop\\Python\\music_generation_folder\\soundfont.sf2"

#         # Ensure paths exist
#         if not os.path.exists(fluidsynth_path):
#             raise FileNotFoundError(f"❌ Fluidsynth not found at {fluidsynth_path}!")
#         if not os.path.exists(soundfont_path):
#             raise FileNotFoundError(f"❌ SoundFont file not found at {soundfont_path}!")

#         print(f"🎼 Converting {midi_file} to {output_file} using {fluidsynth_path} ...")

#         # Run Fluidsynth and capture output for debugging
#         result = subprocess.run([
#             fluidsynth_path, "-ni", soundfont_path, midi_file,
#             "-F", output_file, "-r", "44100"
#         ], capture_output=True, text=True)

#         # Print Fluidsynth logs
#         print("🔹 Fluidsynth Output:")
#         print(result.stdout)
#         print(result.stderr)

#         # Check if the file was actually created
#         if not os.path.exists(output_file):
#             raise RuntimeError(f"❌ Conversion failed! {output_file} was not created.")

#         print(f"✅ Conversion Complete: {output_file}")
#         return output_file


#     def align_lyrics_to_beat(self, lyrics, bpm):
#         """
#         Align lyrics to the generated music beat.
#         """
#         words = lyrics.split()
#         beats_per_word = 60 / bpm  # Assign beats based on tempo
#         timestamps = np.arange(0, len(words) * beats_per_word, beats_per_word)
#         return list(zip(words, timestamps))

#     def generate_full_song(self, lyrics):
#         """
#         End-to-end song generation pipeline.
#         """
#         mood, genre, bpm = self.analyze_lyrics(lyrics)
#         midi_file = self.generate_midi(lyrics)
#         final_audio = self.convert_midi_to_wav(midi_file)
#         print("🎶 Song Generation Complete!")

# # Example Usage
# if __name__ == "__main__":
#     lyrics = """I'm sure I've seen you
    # I see the fire in your eyes 
    # It makes me feel, it's taking my dreams
    # I'm feeling like ain't the first time we ever meet
    # You make the time fly slow, slow
    # Just like slow rivers flow, slow
    # Dance with me like it's my party
    # We go wild, we're in safari
#     """

#     generator = AI_LyricsToMusic()
#     generator.generate_full_song(lyrics)






#TESTING 2
# import streamlit as st
# import torch
# import torchaudio
# import os
# import numpy as np
# import base64
# import pandas as pd
# from audiocraft.models import MusicGen

# # Create output directory if it doesn't exist
# os.makedirs("audio_output", exist_ok=True)

# @st.cache_resource
# def load_model():
#     """Load the MusicGen model and cache it to avoid reloading."""
#     model = MusicGen.get_pretrained('facebook/musicgen-small')
#     return model

# def load_lyrics_dataset(csv_path):
#     """Load lyrics dataset from a CSV file."""
#     try:
#         df = pd.read_csv(csv_path)
#         return df
#     except Exception as e:
#         st.error(f"Error loading dataset: {str(e)}")
#         return None

# def generate_music_from_lyrics(lyrics, artist=None, title=None, duration=10):
#     """Generate music based on lyrics and optional metadata."""
#     model = load_model()
    
#     # Create a prompt that includes all available information
#     if artist and title:
#         prompt = f"Create music for the song '{title}' by {artist} with lyrics: {lyrics[:300]}"
#     elif title:
#         prompt = f"Create music for the song '{title}' with lyrics: {lyrics[:300]}"
#     elif artist:
#         prompt = f"Create music in the style of {artist} with lyrics: {lyrics[:300]}"
#     else:
#         prompt = f"Create music for these lyrics: {lyrics[:300]}"
    
#     # Set generation parameters
#     model.set_generation_params(
#         use_sampling=True,
#         top_k=250,
#         duration=duration
#     )
    
#     # Generate the music
#     output = model.generate(
#         descriptions=[prompt],
#         progress=True,
#         return_tokens=False
#     )
    
#     return output

# def save_audio(samples, filename="generated_music.wav"):
#     """Save the generated audio to a file."""
#     samples = samples.detach().cpu()
#     sample_rate = 32000
#     filepath = os.path.join("audio_output", filename)
    
#     if samples.dim() == 3:
#         samples = samples[0]
    
#     torchaudio.save(filepath, samples, sample_rate)
#     return filepath

# def get_download_link(file_path, link_text="Download Audio"):
#     """Create a download link for the generated audio."""
#     with open(file_path, 'rb') as f:
#         audio_data = f.read()
    
#     b64_audio = base64.b64encode(audio_data).decode()
#     href = f'<a href="data:audio/wav;base64,{b64_audio}" download="{os.path.basename(file_path)}">{link_text}</a>'
    
#     return href

# def main():
#     st.set_page_config(
#         page_title="Lyrics to Music Generator 🎵",
#         page_icon="🎵"
#     )
    
#     st.title("Lyrics to Music Generator 🎵")
    
#     with st.expander("About this app"):
#         st.markdown("""
#         This app generates music based on song lyrics using Meta's Audiocraft MusicGen model.
        
#         You can:
#         - Enter lyrics manually
#         - Select a song from your dataset (if loaded)
#         - Adjust music duration and generation parameters
        
#         The app will create music that fits the mood and content of the lyrics.
#         """)
    
#     # Sidebar for dataset loading
#     with st.sidebar:
#         st.header("Dataset Options")
        
#         # File uploader for CSV
#         uploaded_file = st.file_uploader("Upload lyrics dataset (CSV)", type="csv")
        
#         if uploaded_file is not None:
#             df = load_lyrics_dataset(uploaded_file)
#             if df is not None:
#                 st.success(f"Dataset loaded with {len(df)} songs!")
                
#                 # Show column info
#                 st.write("Columns found:", df.columns.tolist())
#         else:
#             st.info("Upload a CSV file with lyrics or enter lyrics manually below.")
#             df = None
    
#     # Main content area
#     tab1, tab2 = st.tabs(["Manual Entry", "Use Dataset"])
    
#     # Tab 1: Manual lyrics entry
#     with tab1:
#         st.header("Enter Lyrics Manually")
        
#         lyrics = st.text_area("Enter song lyrics:", height=200,
#                             placeholder="Enter the lyrics of your song here...")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             artist_name = st.text_input("Artist name (optional):")
#         with col2:
#             song_title = st.text_input("Song title (optional):")
            
#         duration = st.slider("Music duration (seconds):", min_value=5, max_value=30, value=15)
        
#         if st.button("Generate Music from Lyrics", key="manual_gen"):
#             if not lyrics:
#                 st.error("Please enter lyrics first!")
#             else:
#                 with st.spinner("Generating music from lyrics... This may take a minute."):
#                     try:
#                         music_tensor = generate_music_from_lyrics(lyrics, artist_name, song_title, duration)
#                         safe_title = ''.join(c if c.isalnum() else '_' for c in (song_title or "untitled"))
#                         file_path = save_audio(music_tensor, f"{safe_title}_{hash(lyrics)%10000}.wav")
                        
#                         st.success("Music generated successfully!")
                        
#                         # Display player and download
#                         st.subheader("Your generated track:")
#                         st.audio(file_path)
#                         st.markdown(get_download_link(file_path), unsafe_allow_html=True)
#                     except Exception as e:
#                         st.error(f"Error generating music: {str(e)}")
    
#     # Tab 2: Dataset-based generation
#     with tab2:
#         st.header("Generate from Dataset")
        
#         if df is not None:
#             # Check for required columns
#             has_lyrics = 'lyrics' in df.columns
#             has_artist = 'artist_name' in df.columns
#             has_title = 'title' in df.columns
            
#             if has_lyrics:
#                 # Create filtered view if needed
#                 if has_artist:
#                     artists = ["All Artists"] + sorted(df['artist_name'].unique().tolist())
#                     selected_artist = st.selectbox("Filter by artist:", artists)
                    
#                     if selected_artist != "All Artists":
#                         filtered_df = df[df['artist_name'] == selected_artist]
#                     else:
#                         filtered_df = df
#                 else:
#                     filtered_df = df
                
#                 # Show song selection
#                 if has_title:
#                     song_options = filtered_df['title'].tolist()
#                     selected_song = st.selectbox("Select a song:", song_options)
                    
#                     # Get the selected song data
#                     song_data = filtered_df[filtered_df['title'] == selected_song].iloc[0]
                    
#                     # Display song info
#                     st.subheader("Song Information")
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         if has_artist:
#                             st.info(f"Artist: {song_data['artist_name']}")
#                         st.info(f"Title: {song_data['title']}")
                    
#                     # Display lyrics
#                     st.subheader("Lyrics")
#                     st.text_area("", value=song_data['lyrics'], height=200, disabled=True)
                    
#                     # Generation options
#                     duration = st.slider("Music duration (seconds):", min_value=5, max_value=30, value=15, 
#                                         key="dataset_duration")
                    
#                     # Generate button
#                     if st.button("Generate Music for This Song", key="dataset_gen"):
#                         with st.spinner("Generating music... This may take a minute."):
#                             try:
#                                 artist_val = song_data['artist_name'] if has_artist else None
#                                 lyrics_val = song_data['lyrics']
#                                 title_val = song_data['title']
                                
#                                 music_tensor = generate_music_from_lyrics(
#                                     lyrics_val, artist_val, title_val, duration
#                                 )
                                
#                                 safe_title = ''.join(c if c.isalnum() else '_' for c in title_val)
#                                 file_path = save_audio(music_tensor, f"{safe_title}.wav")
                                
#                                 st.success(f"Music generated for '{title_val}'!")
                                
#                                 # Display player and download
#                                 st.subheader("Your generated track:")
#                                 st.audio(file_path)
#                                 st.markdown(get_download_link(file_path), unsafe_allow_html=True)
#                             except Exception as e:
#                                 st.error(f"Error generating music: {str(e)}")
#                 else:
#                     st.error("Dataset must contain a 'title' column to select songs.")
#             else:
#                 st.error("Dataset must contain a 'lyrics' column.")
#         else:
#             st.info("Please upload a dataset in the sidebar to use this feature.")

# if __name__ == "__main__":
#     main()














# import torch
# import torchaudio
# import numpy as np
# from transformers import pipeline
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write
# import nltk
# from nltk.tokenize import sent_tokenize
# import soundfile as sf
# import librosa
# import pandas as pd
# from scipy.io import wavfile
# import os
# import re
# import time
# from tqdm import tqdm
# import scipy.io.wavfile
# # Download necessary NLTK data
# nltk.download('punkt')

# class LyricsToMusicGenerator:
#     def __init__(self, model_size="large", device="cuda" if torch.cuda.is_available() else "cpu"):
#         """
#         Initialize the Lyrics-to-Music Generator
        
#         Args:
#             model_size (str): Size of MusicGen model ('medium', 'large', or 'melody')
#             device (str): Device to run the model on ('cuda' or 'cpu')
#         """
#         self.device = device
#         print(f"Using device: {self.device}")
        
#         # Load MusicGen model
#         print("Loading MusicGen model...")
#         self.music_gen = MusicGen.get_pretrained(model_size, device=self.device)
#         self.music_gen.set_generation_params(
#             duration=15,  # Generate 15 seconds of music by default
#             temperature=0.95,  # Higher temperature for more creative outputs
#             top_k=250,
#             top_p=0.95,
#             cfg_coef=4.0,  # Strength of the text conditioning
#         )
        
#         # Load text-to-speech model for vocal synthesis
#         print("Loading TTS model...")
#         self.tts_model = pipeline("text-to-speech", "suno/bark-small", device=0 if self.device == "cuda" else -1)
        
#         # Sentiment analysis for emotional context
#         print("Loading sentiment analysis model...")
#         self.sentiment_model = pipeline("sentiment-analysis", device=0 if self.device == "cuda" else -1)
        
#         print("All models loaded successfully!")
    
#     def _analyze_lyrics(self, lyrics):
#         """
#         Analyze lyrics to extract musical style, emotion, and structure
        
#         Args:
#             lyrics (str): Song lyrics
            
#         Returns:
#             dict: Analysis results including style, emotion, and sections
#         """
#         # Split lyrics into sections (verses, chorus, etc.)
#         sections = []
#         lines = lyrics.strip().split("\n")
#         current_section = []
#         current_section_name = "verse"
        
#         for line in lines:
#             # Check if line indicates a section
#             section_indicators = {
#                 "chorus": ["chorus", "[chorus]", "refrain"],
#                 "verse": ["verse", "[verse]"],
#                 "bridge": ["bridge", "[bridge]"],
#                 "intro": ["intro", "[intro]"],
#                 "outro": ["outro", "[outro]"]
#             }
            
#             line_lower = line.lower()
#             found_section = False
            
#             for section_name, indicators in section_indicators.items():
#                 if any(indicator in line_lower for indicator in indicators):
#                     # Save previous section
#                     if current_section:
#                         sections.append({
#                             "type": current_section_name,
#                             "content": "\n".join(current_section)
#                         })
#                     # Start new section
#                     current_section = []
#                     current_section_name = section_name
#                     found_section = True
#                     break
            
#             if not found_section and line.strip():
#                 current_section.append(line)
        
#         # Add the last section
#         if current_section:
#             sections.append({
#                 "type": current_section_name,
#                 "content": "\n".join(current_section)
#             })
        
#         # Analyze sentiment of whole lyrics
#         sentiment_result = self.sentiment_model(lyrics)
        
#         # Determine tempo and style based on content and sentiment
#         sentiment_score = sentiment_result[0]["score"]
#         label = sentiment_result[0]["label"]
        
#         if label == "POSITIVE":
#             if any(word in lyrics.lower() for word in ["dance", "party", "beat", "groove"]):
#                 style = "upbeat pop"
#                 tempo = "fast"
#             else:
#                 style = "cheerful acoustic"
#                 tempo = "medium"
#         else:  # NEGATIVE
#             if any(word in lyrics.lower() for word in ["love", "heart", "pain", "cry"]):
#                 style = "emotional ballad"
#                 tempo = "slow"
#             else:
#                 style = "alternative rock"
#                 tempo = "medium"
        
#         return {
#             "sentiment": label,
#             "sentiment_score": sentiment_score,
#             "style": style,
#             "tempo": tempo,
#             "sections": sections
#         }
    
#     def _generate_prompt(self, analysis):
#         """
#         Generate a descriptive prompt for MusicGen based on lyrics analysis
        
#         Args:
#             analysis (dict): Analysis results from _analyze_lyrics
            
#         Returns:
#             str: Prompt for MusicGen
#         """
#         style = analysis["style"]
#         tempo = analysis["tempo"]
#         sentiment = analysis["sentiment"].lower()
        
#         # Map sentiment to musical elements
#         if sentiment == "positive":
#             key = "major key"
#             mood = "uplifting"
#         else:
#             key = "minor key"
#             mood = "melancholic"
        
#         # Create base prompt
#         prompt = f"A {mood} {style} song in {key} with {tempo} tempo. "
        
#         # Add instrumentation based on style
#         if "pop" in style:
#             prompt += "Features drums, bass, synths, and piano. "
#         elif "acoustic" in style:
#             prompt += "Features acoustic guitar, piano, and light percussion. "
#         elif "ballad" in style:
#             prompt += "Features piano, strings, and minimal percussion. "
#         elif "rock" in style:
#             prompt += "Features electric guitars, bass, and drums. "
        
#         # Add vocal style
#         prompt += f"Clear vocal melody with {sentiment} emotional delivery. "
        
#         # Add structure hints
#         if len(analysis["sections"]) > 1:
#             prompt += "Song has dynamic structure with verse and chorus. "
        
#         return prompt
    
#     def _generate_vocals(self, lyrics, output_path="vocals.wav", gender="female", emotion="neutral"):
#         """
#         Generate vocals from lyrics using TTS
        
#         Args:
#             lyrics (str): Song lyrics
#             output_path (str): Path to save the vocal audio
#             gender (str): Gender of the vocalist ('male' or 'female')
#             emotion (str): Emotional tone of the vocals
            
#         Returns:
#             str: Path to the generated vocal audio
#         """
#         print("Generating vocals...")
        
#         # Clean lyrics for TTS
#         cleaned_lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove section markers
#         sentences = sent_tokenize(cleaned_lyrics)
        
#         # Select appropriate voice preset
#         voice_presets = {
#             "female": {
#                 "neutral": "v2/en_speaker_6",
#                 "happy": "v2/en_speaker_9",
#                 "sad": "v2/en_speaker_5"
#             },
#             "male": {
#                 "neutral": "v2/en_speaker_0",
#                 "happy": "v2/en_speaker_3",
#                 "sad": "v2/en_speaker_2"
#             }
#         }
        
#         voice = voice_presets.get(gender, {}).get(emotion, "v2/en_speaker_6")
        
#         # Generate audio for each sentence
#         audio_chunks = []
        
#         # Process in smaller chunks to avoid memory issues
#         for sentence in tqdm(sentences, desc="Generating vocals"):
#             if not sentence.strip():
#                 continue
            
#             # Generate speech
#             speech = self.tts_model(
#                 sentence, 
#                 forward_params={"do_sample": True}
#             )
            
#             # Apply singing-like modifications
#             audio = speech["audio"]
#             sr = speech["sampling_rate"]
            
#             audio_chunks.append((audio, sr))
            
#             # Avoid overloading GPU memory
#             torch.cuda.empty_cache() if self.device == "cuda" else None
#             time.sleep(0.05)  # Small delay to prevent potential issues
        
#         # Combine all chunks
#         if audio_chunks:
#             # Resample all chunks to the same sample rate
#             target_sr = audio_chunks[0][1]
            
#             # Debug print to check sample rate
#             print(f"DEBUG: Sample rate is {target_sr}")
            
#             # If sample rate is too high for WAV format, downsample to a standard value
#             if target_sr > 48000:  # Most WAV files use 44100 or 48000 Hz
#                 print(f"Warning: Sample rate {target_sr} is too high, downsampling to 48000 Hz")
#                 target_sr = 48000
            
#             resampled_chunks = []
            
#             for audio, sr in audio_chunks:
#                 if sr != target_sr:
#                     audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
#                 resampled_chunks.append(audio)
            
#             # Concatenate chunks
#             combined_audio = np.concatenate(resampled_chunks)
            
#             # Scale to int16 range
#             combined_audio = (combined_audio * 32767).astype(np.int16)
            
#             # Use soundfile instead of scipy.io.wavfile which has better handling of sample rates
#             try:
#                 sf.write(output_path, combined_audio, target_sr)
#                 print(f"Successfully wrote audio to {output_path}")
#                 return output_path
#             except Exception as e:
#                 print(f"Error writing audio with sf.write: {e}")
#                 # Fall back to librosa
#                 try:
#                     librosa.output.write_wav(output_path, combined_audio, target_sr)
#                     print(f"Successfully wrote audio using librosa fallback")
#                     return output_path
#                 except Exception as e2:
#                     print(f"Error writing audio with librosa: {e2}")
#                     # Last resort, try scipy with very conservative settings
#                     try:
#                         # Ensure sample rate is within bounds for scipy.io.wavfile
#                         safe_sr = min(target_sr, 48000)  # Cap at 48kHz
#                         # Ensure we have the right number of channels (mono)
#                         if len(combined_audio.shape) > 1 and combined_audio.shape[1] > 1:
#                             combined_audio = np.mean(combined_audio, axis=1)
#                         scipy.io.wavfile.write(output_path, safe_sr, combined_audio)
#                         print(f"Successfully wrote audio using scipy fallback")
#                         return output_path
#                     except Exception as e3:
#                         print(f"All audio writing methods failed: {e3}")
#                         return None
#         else:
#             print("Warning: No audio was generated for the vocals")
#             return None
#     def _generate_instrumental(self, prompt, duration=15, output_path="instrumental.wav"):
#         """
#         Generate instrumental music using MusicGen
        
#         Args:
#             prompt (str): Descriptive prompt for MusicGen
#             duration (int): Duration in seconds
#             output_path (str): Path to save the instrumental audio
            
#         Returns:
#             str: Path to the generated instrumental audio
#         """
#         print(f"Generating instrumental with prompt: {prompt}")
        
#         # Set duration for generation
#         self.music_gen.set_generation_params(
#             duration=duration,
#             temperature=0.95,
#             top_k=250,
#             top_p=0.95,
#             cfg_coef=4.0,
#         )
        
#         # Generate music
#         wav = self.music_gen.generate([prompt])
        
#         # Convert to numpy and save
#         wav = wav.detach().cpu().numpy()
#         sampling_rate = self.music_gen.sample_rate
#         audio_write(output_path.replace(".wav", ""), wav[0], sampling_rate, strategy="loudness", loudness_compressor=True)
        
#         return output_path.replace(".wav", ".wav")
    
#     def _mix_audio(self, vocal_path, instrumental_path, output_path="final_song.wav"):
#         """
#         Mix vocals and instrumental together
        
#         Args:
#             vocal_path (str): Path to vocal audio
#             instrumental_path (str): Path to instrumental audio
#             output_path (str): Path to save the final mixed audio
            
#         Returns:
#             str: Path to the final mixed audio
#         """
#         print("Mixing vocals and instrumental...")
        
#         # Load audio files
#         vocal, vocal_sr = librosa.load(vocal_path, sr=None)
#         instrumental, instrumental_sr = librosa.load(instrumental_path, sr=None)
        
#         # Resample if needed
#         if vocal_sr != instrumental_sr:
#             vocal = librosa.resample(vocal, orig_sr=vocal_sr, target_sr=instrumental_sr)
        
#         # Adjust volumes
#         vocal_gain = 1.0  # Adjust as needed
#         instrumental_gain = 0.5  # Lower instrumental volume to make vocals clearer
        
#         vocal = vocal * vocal_gain
#         instrumental = instrumental * instrumental_gain
        
#         # Pad or trim to match lengths
#         max_length = max(len(vocal), len(instrumental))
        
#         if len(vocal) < max_length:
#             vocal = np.pad(vocal, (0, max_length - len(vocal)))
#         else:
#             vocal = vocal[:max_length]
            
#         if len(instrumental) < max_length:
#             instrumental = np.pad(instrumental, (0, max_length - len(instrumental)))
#         else:
#             instrumental = instrumental[:max_length]
        
#         # Mix audio
#         mixed_audio = vocal + instrumental
        
#         # Normalize
#         mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
#         # Save mixed audio
#         sf.write(output_path, mixed_audio, instrumental_sr)
        
#         return output_path
    
#     def generate_song(self, lyrics, output_dir="output", duration=30, gender="female"):
#         """
#         Generate a complete song with vocals from lyrics
        
#         Args:
#             lyrics (str): Song lyrics
#             output_dir (str): Directory to save output files
#             duration (int): Duration in seconds for the instrumental
#             gender (str): Gender of the vocalist ('male' or 'female')
            
#         Returns:
#             str: Path to the final song
#         """
#         # Create output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Analyze lyrics
#         print("Analyzing lyrics...")
#         analysis = self._analyze_lyrics(lyrics)
        
#         # Generate prompt for MusicGen
#         prompt = self._generate_prompt(analysis)
        
#         # Generate vocals
#         emotion = "happy" if analysis["sentiment"] == "POSITIVE" else "sad"
#         vocal_path = self._generate_vocals(
#             lyrics, 
#             output_path=os.path.join(output_dir, "vocals.wav"),
#             gender=gender,
#             emotion=emotion
#         )
        
#         # Generate instrumental
#         instrumental_path = self._generate_instrumental(
#             prompt, 
#             duration=duration,
#             output_path=os.path.join(output_dir, "instrumental.wav")
#         )
        
#         # Mix vocals and instrumental
#         final_path = self._mix_audio(
#             vocal_path,
#             instrumental_path,
#             output_path=os.path.join(output_dir, "final_song.wav")
#         )
        
#         print(f"Song generation complete! Final song saved to: {final_path}")
#         return final_path

# # Example usage
# def main():
#     lyrics = """
#     [Verse]
#     Walking through the empty streets
#     Memories of you are all I keep
#     The city lights reflect my tears
#     I've been missing you for years
    
#     [Chorus]
#     But I'm letting go tonight
#     Spreading my wings to fly
#     No more looking back
#     I'm finding my own track
    
#     [Verse]
#     New horizons call my name
#     I'm not the person I became
#     When you left me all alone
#     Now I'm stronger on my own
    
#     [Chorus]
#     But I'm letting go tonight
#     Spreading my wings to fly
#     No more looking back
#     I'm finding my own track
    
#     [Bridge]
#     The pain is fading away
#     A new life begins today
    
#     [Chorus]
#     I'm letting go tonight
#     Spreading my wings to fly
#     No more looking back
#     I'm finding my own track
#     """
    
#     # Initialize generator
#     generator = LyricsToMusicGenerator(model_size="large")
    
#     # Generate song
#     output_path = generator.generate_song(lyrics, duration=60, gender="female")
    
#     print(f"Song generated successfully! Output: {output_path}")





# if __name__ == "__main__":
#     main()























































# import os
# import re
# import time
# import torch
# import numpy as np
# import scipy.io.wavfile
# import librosa
# import soundfile as sf
# from tqdm import tqdm
# from transformers import pipeline, AutoProcessor, AutoModel
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write
# from nltk.tokenize import sent_tokenize

# class OptimizedLyricsToMusicGenerator:
#     def __init__(self, model_size="small", device="cpu"):
#         """
#         Initialize the Lyrics-to-Music Generator with optimizations for CPU
        
#         Args:
#             model_size (str): Size of MusicGen model ('small', 'medium', or 'large')
#             device (str): Device to run the model on (forced to 'cpu')
#         """
#         # Force CPU usage and optimize threading
#         self.device = "cpu"
#         torch.set_num_threads(os.cpu_count())  # Use all available CPU cores
#         print(f"Using device: {self.device} with {os.cpu_count()} threads")
        
#         # Load MusicGen model - using smaller model for speed
#         print("Loading MusicGen model...")
#         self.music_gen = MusicGen.get_pretrained(model_size, device=self.device)
#         self.music_gen.set_generation_params(
#             duration=8,  # Shorter duration for faster generation
#             temperature=0.9,
#             top_k=250,
#             top_p=0.95,
#             cfg_coef=4.0,
#         )
        
#         # Use Microsoft's Speecht5 model which is compatible with transformers
#         print("Loading TTS model...")
#         try:
#             self.tts_processor = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
#             self.tts_model = AutoModel.from_pretrained("microsoft/speecht5_tts")
#             self.tts_vocoder = AutoModel.from_pretrained("microsoft/speecht5_hifigan")
#             print("TTS model loaded successfully!")
#         except Exception as e:
#             print(f"Error loading TTS model: {e}")
#             print("Falling back to simple text-only mode (no vocals)")
#             self.tts_processor = None
#             self.tts_model = None
#             self.tts_vocoder = None
        
#         # Simple rule-based sentiment analysis instead of model for speed
#         print("Using rule-based sentiment analysis for speed...")
        
#         print("All components initialized successfully!")
    
#     def _analyze_lyrics_rule_based(self, lyrics):
#         """
#         Analyze lyrics using rule-based approach instead of ML model for faster processing
#         """
#         # Simple word-based sentiment analysis
#         positive_words = ["happy", "joy", "love", "light", "hope", "bright", "smile", 
#                           "good", "beautiful", "dream", "fly", "free", "peace"]
#         negative_words = ["sad", "pain", "hurt", "cry", "alone", "dark", "fear", 
#                           "hate", "lost", "broken", "tear", "die", "sorry"]
        
#         lyrics_lower = lyrics.lower()
#         pos_count = sum(lyrics_lower.count(word) for word in positive_words)
#         neg_count = sum(lyrics_lower.count(word) for word in negative_words)
        
#         sentiment = "POSITIVE" if pos_count >= neg_count else "NEGATIVE"
#         sentiment_score = 0.5 + (0.5 * (pos_count - neg_count) / max(1, pos_count + neg_count))
        
#         # Simple section analysis
#         sections = []
#         lines = lyrics.strip().split("\n")
#         current_section = []
#         current_section_name = "verse"
        
#         for line in lines:
#             section_indicators = {
#                 "chorus": ["chorus", "[chorus]", "refrain"],
#                 "verse": ["verse", "[verse]"],
#                 "bridge": ["bridge", "[bridge]"],
#                 "intro": ["intro", "[intro]"],
#                 "outro": ["outro", "[outro]"]
#             }
            
#             line_lower = line.lower()
#             found_section = False
            
#             for section_name, indicators in section_indicators.items():
#                 if any(indicator in line_lower for indicator in indicators):
#                     if current_section:
#                         sections.append({
#                             "type": current_section_name,
#                             "content": "\n".join(current_section)
#                         })
#                     current_section = []
#                     current_section_name = section_name
#                     found_section = True
#                     break
            
#             if not found_section and line.strip():
#                 current_section.append(line)
        
#         if current_section:
#             sections.append({
#                 "type": current_section_name,
#                 "content": "\n".join(current_section)
#             })
        
#         # Determine style based on content and sentiment
#         if sentiment == "POSITIVE":
#             if any(word in lyrics_lower for word in ["dance", "party", "beat", "groove"]):
#                 style = "upbeat pop"
#                 tempo = "fast"
#             else:
#                 style = "cheerful acoustic"
#                 tempo = "medium"
#         else:  # NEGATIVE
#             if any(word in lyrics_lower for word in ["love", "heart", "pain", "cry"]):
#                 style = "emotional ballad"
#                 tempo = "slow"
#             else:
#                 style = "alternative rock"
#                 tempo = "medium"
        
#         return {
#             "sentiment": sentiment,
#             "sentiment_score": sentiment_score,
#             "style": style,
#             "tempo": tempo,
#             "sections": sections
#         }
    
#     def _generate_prompt(self, analysis):
#         """Generate a descriptive prompt for MusicGen based on lyrics analysis"""
#         style = analysis["style"]
#         tempo = analysis["tempo"]
#         sentiment = analysis["sentiment"].lower()
        
#         if sentiment == "positive":
#             key = "major key"
#             mood = "uplifting"
#         else:
#             key = "minor key"
#             mood = "melancholic"
        
#         prompt = f"A {mood} {style} song in {key} with {tempo} tempo. "
        
#         if "pop" in style:
#             prompt += "Features drums, bass, synths, and piano. "
#         elif "acoustic" in style:
#             prompt += "Features acoustic guitar, piano, and light percussion. "
#         elif "ballad" in style:
#             prompt += "Features piano, strings, and minimal percussion. "
#         elif "rock" in style:
#             prompt += "Features electric guitars, bass, and drums. "
        
#         prompt += f"Clear vocal melody with {sentiment} emotional delivery. "
        
#         if len(analysis["sections"]) > 1:
#             prompt += "Song has dynamic structure with verse and chorus. "
        
#         return prompt
    
#     def _generate_simplified_vocals(self, lyrics, output_path="vocals.wav"):
#         """Simplified vocal generation for faster processing"""
#         print("Generating simplified vocals...")
        
#         # If no TTS model is available, skip vocal generation
#         if self.tts_model is None or self.tts_processor is None:
#             print("No TTS model available, skipping vocal generation")
#             return None
            
#         # Clean lyrics and take only first few lines for speed
#         cleaned_lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove section markers
#         sentences = sent_tokenize(cleaned_lyrics)
        
#         # Limit to first 2 sentences for quick testing
#         if len(sentences) > 2:
#             print(f"Limiting vocal generation to first 2 sentences (of {len(sentences)}) for speed")
#             sentences = sentences[:2]
        
#         # Process each sentence with progress tracking
#         all_audio = []
#         sr = 16000  # Standard sample rate for SpeechT5
        
#         # Process text in smaller batches
#         for i, sentence in enumerate(sentences):
#             if not sentence.strip():
#                 continue
                
#             print(f"Generating vocals for segment {i+1}/{len(sentences)}: '{sentence[:30]}...'")
#             start_time = time.time()
            
#             try:
#                 # Using Microsoft SpeechT5 TTS
#                 inputs = self.tts_processor(text=sentence, return_tensors="pt")
                
#                 # Generate speech with progress tracking
#                 print("  Processing text...")
#                 speech = self.tts_model.generate_speech(
#                     inputs["input_ids"], 
#                     self.tts_vocoder,
#                     vocoder_speaker_embedding=torch.zeros((1, 512))  # Default speaker embedding
#                 )
                
#                 # Convert to numpy array
#                 audio = speech.numpy()
                
#                 all_audio.append(audio)
                
#                 end_time = time.time()
#                 print(f"✓ Segment {i+1} complete in {end_time - start_time:.1f} seconds")
                
#             except Exception as e:
#                 print(f"Error processing segment {i+1}: {e}")
#                 print("Continuing with other segments...")
        
#         if all_audio:
#             # Concatenate audio
#             combined_audio = np.concatenate(all_audio)
            
#             # Scale to int16 range for WAV
#             combined_audio = (combined_audio * 32767).astype(np.int16)
            
#             # Write to file
#             try:
#                 scipy.io.wavfile.write(output_path, sr, combined_audio)
#                 print(f"Successfully wrote vocals to {output_path}")
#                 return output_path
#             except Exception as e:
#                 print(f"Error writing vocal audio: {e}")
#                 return None
#         else:
#             print("No audio was generated for vocals")
#             return None
    
#     def _generate_instrumental(self, prompt, duration=8, output_path="instrumental.wav"):
#         """Generate instrumental music with progress reporting"""
#         print(f"Generating instrumental with prompt: {prompt}")
#         print(f"Estimated time: ~{duration * 5} seconds on CPU")
        
#         # Set shorter duration for faster generation
#         self.music_gen.set_generation_params(
#             duration=duration,
#             temperature=0.9,
#             top_k=250,
#             top_p=0.95,
#             cfg_coef=4.0,
#         )
        
#         # Generate music with timing info
#         start_time = time.time()
#         print("Starting generation...")
        
#         # Generate the music
#         try:
#             # Add this patch to fix the numpy.dtype issue
#             original_is_floating_point = torch.is_floating_point
#             torch.is_floating_point = lambda x: x.is_floating_point() if hasattr(x, 'is_floating_point') else x.dtype.is_floating_point
            
#             wav = self.music_gen.generate([prompt])
            
#             # Restore original function
#             torch.is_floating_point = original_is_floating_point
            
#             end_time = time.time()
#             print(f"Instrumental generation completed in {end_time - start_time:.1f} seconds")
            
#             # Convert to numpy and save
#             wav = wav.detach().cpu().numpy()
#             sampling_rate = self.music_gen.sample_rate
            
#             # Save the audio
#             audio_write(output_path.replace(".wav", ""), wav[0], sampling_rate, 
#                         strategy="loudness", loudness_compressor=True)
            
#             print(f"Instrumental saved to {output_path.replace('.wav', '.wav')}")
#             return output_path.replace(".wav", ".wav")
        
#         except Exception as e:
#             print(f"Error generating instrumental: {e}")
#             return None
    
#     def _quick_mix_audio(self, vocal_path, instrumental_path, output_path="final_song.wav"):
#         """Mix vocals and instrumental with simplified approach for speed"""
#         print("Quick-mixing vocals and instrumental...")
        
#         # Check if both files exist
#         if vocal_path is None and instrumental_path is None:
#             print("No audio files to mix")
#             return None
        
#         # If only instrumental exists, just return that
#         if vocal_path is None:
#             print("No vocals available, returning instrumental only")
#             return instrumental_path
            
#         # If only vocals exist, just return that
#         if instrumental_path is None:
#             print("No instrumental available, returning vocals only")
#             return vocal_path
        
#         try:
#             # Load audio files
#             vocal, vocal_sr = librosa.load(vocal_path, sr=None)
#             instrumental, instrumental_sr = librosa.load(instrumental_path, sr=None)
            
#             # Resample if needed
#             if vocal_sr != instrumental_sr:
#                 print(f"Resampling vocals from {vocal_sr}Hz to {instrumental_sr}Hz")
#                 vocal = librosa.resample(vocal, orig_sr=vocal_sr, target_sr=instrumental_sr)
            
#             # Adjust volumes
#             vocal = vocal * 1.5  # Boost vocals
#             instrumental = instrumental * 0.5  # Lower instrumental
            
#             # Ensure lengths match
#             if len(vocal) > len(instrumental):
#                 print("Trimming vocals to match instrumental length")
#                 vocal = vocal[:len(instrumental)]
#             else:
#                 print("Padding vocals to match instrumental length")
#                 vocal = np.pad(vocal, (0, len(instrumental) - len(vocal)))
            
#             # Simple mix
#             mixed_audio = vocal + instrumental
            
#             # Normalize
#             mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
            
#             # Save
#             sf.write(output_path, mixed_audio, instrumental_sr)
#             print(f"Mixed song saved to {output_path}")
            
#             return output_path
            
#         except Exception as e:
#             print(f"Error mixing audio: {e}")
#             if instrumental_path:
#                 print("Returning instrumental only due to mixing error")
#                 return instrumental_path
#             return None
    
#     def generate_song_quick(self, lyrics, output_dir="output", duration=8):
#         """Generate a song quickly with simplified approach"""
#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)
        
#         print("\n=== QUICK SONG GENERATION STARTED ===")
#         start_time = time.time()
        
#         # Step 1: Analyze lyrics (rule-based for speed)
#         print("\n[1/4] Analyzing lyrics...")
#         analysis = self._analyze_lyrics_rule_based(lyrics)
#         print(f"Analysis complete - Detected {analysis['sentiment']} sentiment, {analysis['style']} style")
        
#         # Step 2: Generate prompt
#         prompt = self._generate_prompt(analysis)
#         print(f"Generated prompt: {prompt}")
        
#         # Step 3: Generate vocals (optional - may be skipped if no TTS)
#         print("\n[2/4] Generating vocals...")
#         vocal_path = self._generate_simplified_vocals(
#             lyrics, 
#             output_path=os.path.join(output_dir, "vocals.wav")
#         )
        
#         # Step 4: Generate instrumental
#         print("\n[3/4] Generating instrumental...")
#         instrumental_path = self._generate_instrumental(
#             prompt, 
#             duration=duration,
#             output_path=os.path.join(output_dir, "instrumental.wav")
#         )
        
#         # Step 5: Mix (if both parts are available)
#         print("\n[4/4] Mixing audio...")
#         final_path = self._quick_mix_audio(
#             vocal_path,
#             instrumental_path,
#             output_path=os.path.join(output_dir, "final_song.wav")
#         )
        
#         end_time = time.time()
#         total_time = end_time - start_time
        
#         if final_path:
#             print(f"\n✅ Quick song generation complete in {total_time:.1f} seconds!")
#             print(f"Final song saved to: {final_path}")
#         else:
#             print("\n❌ Song generation failed")
            
#         return final_path

# # Example usage
# def main():
#     lyrics = """
#     [Verse]
#     Walking through the empty streets
#     Memories of you are all I keep
    
#     [Chorus]
#     But I'm letting go tonight
#     Spreading my wings to fly
#     """
    
#     # Initialize generator with small model for speed
#     generator = OptimizedLyricsToMusicGenerator(model_size="small")
    
#     # Generate song with shorter duration
#     output_path = generator.generate_song_quick(lyrics, duration=8)
    
#     print(f"Song generation process completed. Output: {output_path}")

# if __name__ == "__main__":
#     main()











































from transformers import pipeline, AutoProcessor, AutoModel
import scipy.io.wavfile
import numpy as np
import torch
import soundfile as sf

# =============================
# 1️⃣ Pipeline-based TTS
# =============================
try:
    print("Initializing text-to-speech pipeline...")
    synthesiser = pipeline("text-to-speech", model="suno/bark-small")

    # Generate speech
    print("Generating speech with pipeline...")
    speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})

    # Fix 1: Check the structure of the speech output
    if isinstance(speech, dict) and "audio" in speech and "sampling_rate" in speech:
        audio_data = speech["audio"]
        sampling_rate = speech["sampling_rate"]
    else:
        # Handle unexpected output structure
        if isinstance(speech, dict):
            print(f"Unexpected speech output structure. Keys: {list(speech.keys())}")
        else:
            print(f"Unexpected speech output type: {type(speech)}")
        audio_data = speech if isinstance(speech, np.ndarray) else np.array([])
        sampling_rate = 24000  # Bark's default sampling rate
    
    # Fix 2: Ensure valid sampling rate (must be an integer within range)
    sampling_rate = int(sampling_rate)
    if not (8000 <= sampling_rate <= 48000):
        print(f"Invalid sampling rate: {sampling_rate}, using default 24000")
        sampling_rate = 24000

    # Fix 3: Use soundfile instead of scipy for more robust saving
    if len(audio_data) > 0:
        # Normalize to float in [-1, 1] range for soundfile
        audio_data = np.array(audio_data, dtype=np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        print(f"Saving pipeline output with sampling rate {sampling_rate}Hz...")
        sf.write("bark_out_pipeline.wav", audio_data, sampling_rate)
        print("Pipeline audio saved successfully!")
    else:
        print("No audio data generated from pipeline")

except Exception as e:
    print(f"Error in pipeline-based TTS: {e}")

# =============================
# 2️⃣ AutoModel-based TTS
# =============================
try:
    print("\nInitializing AutoModel...")
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")

    # Prepare input
    print("Processing text input...")
    inputs = processor(
        text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
        return_tensors="pt",
    )

    # Generate speech values
    print("Generating speech with model...")
    with torch.no_grad():
        speech_values = model.generate(**inputs, do_sample=True)

    # Fix 4: Get sampling rate from model config safely
    sampling_rate_model = getattr(model.config, "sample_rate", 24000)
    sampling_rate_model = int(sampling_rate_model)  # Ensure it's an integer
    
    # Fix 5: Check valid range
    if not (8000 <= sampling_rate_model <= 48000):
        print(f"Invalid sampling rate from model: {sampling_rate_model}, using default 24000")
        sampling_rate_model = 24000

    # Fix 6: Convert generated speech to float32 for soundfile
    print(f"Processing audio data...")
    audio_data_model = speech_values.cpu().numpy().squeeze()
    
    # Ensure it's the right shape
    if len(audio_data_model.shape) > 1:
        print(f"Reshaping audio from {audio_data_model.shape}...")
        audio_data_model = audio_data_model.mean(axis=0) if audio_data_model.shape[0] <= 2 else audio_data_model.squeeze(0)
    
    # Normalize to [-1, 1]
    audio_data_model = np.array(audio_data_model, dtype=np.float32)
    audio_data_model = audio_data_model / np.max(np.abs(audio_data_model)) if np.max(np.abs(audio_data_model)) > 0 else audio_data_model
    
    # Save using soundfile
    print(f"Saving model output with sampling rate {sampling_rate_model}Hz...")
    sf.write("bark_out_model.wav", audio_data_model, sampling_rate_model)
    print("Model audio saved successfully!")

except Exception as e:
    print(f"Error in AutoModel-based TTS: {e}")

print("\n✅ Script execution completed!")