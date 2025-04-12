# from flask import Flask, request, render_template, jsonify, send_file
# import torch
# import numpy as np
# import scipy.io.wavfile
# import os
# import re
# import random
# import warnings
# import threading
# import time
# import uuid
# from functools import wraps
# import traceback
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from collections import Counter, deque
# import re
# import random
# import nltk
# from nltk.corpus import cmudict
# import syllables
# import json
# import os

# # Suppress unnecessary warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# # This would need to be installed first: pip install audiocraft flask
# # Import audiocraft after Flask is set up
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write




# app = Flask(__name__)





# JOBS = {}


# # ========= ENVIRONMENT SETUP =========
# # Try to download required NLTK data if needed
# try:
#     nltk.data.find('corpora/cmudict')
# except LookupError:
#     nltk.download('cmudict')

# # Initialize pronunciation dictionary
# pronounce_dict = cmudict.dict()

# # Set up device with proper error handling
# if torch.cuda.is_available():
#     try:
#         # Test CUDA capability before fully committing
#         test_tensor = torch.zeros(1).cuda()
#         del test_tensor
#         device = torch.device("cuda")
#         print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        
#         # Set memory management for better stability
#         torch.cuda.empty_cache()
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#     except Exception as e:
#         print(f"⚠️ CUDA initialization failed: {e}")
#         print("Falling back to CPU")
#         device = torch.device("cpu")
# else:
#     device = torch.device("cpu")
#     print("✅ Using CPU")

# # ======= HYPERPARAMETERS =======
# CONFIG = {
#     "vocab_size": 10000,
#     "max_seq_len": 64,
#     "embed_dim": 256,        # Reduced from 512
#     "hidden_dim": 512,       # Reduced from 1024
#     "num_layers": 2,         # Reduced from 4
#     "dropout": 0.2,
#     "num_heads": 4,          # Reduced from 8
#     "batch_size": 64,        # Reduced from 128
#     "learning_rate": 0.001,
#     "epochs": 30,
#     "patience": 7,
#     "temperature": 0.8,
#     "top_k": 50,
#     "top_p": 0.9,
#     "repetition_penalty": 1.2,
#     "max_song_length": 300,
#     "min_line_length": 3,
#     "max_line_length": 12,
#     "rhyme_boost": 3.0,
#     "gradient_accumulation_steps": 2,  # New: accumulate gradients over multiple batches
#     "structure_weights": {
#         "verse": 1.0,
#         "chorus": 1.2,
#         "bridge": 1.1,
#         "pre-chorus": 1.0,
#         "outro": 0.9
#     }
# }

# # ======= DATA PREPROCESSING =======
# class LyricsPreprocessor:
#     def __init__(self, config):
#         self.config = config
#         self.word_to_index = {}
#         self.index_to_word = {}
#         self.emotion_to_idx = {}
#         self.idx_to_emotion = {}
#         self.structure_markers = [
#             "verse", "chorus", "bridge", "pre-chorus", "hook", 
#             "intro", "outro", "refrain", "interlude"
#         ]
#         self.rhyme_cache = {}
        
        
#     def load_data(self, filepath):
#         """Load and preprocess the dataset"""
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
#         df = pd.read_csv(filepath)
#         # Handle missing values
#         df = df.dropna(subset=["lyrics", "emotion"])
#         df["lyrics"] = df["lyrics"].str.lower()
        
#         # Enhanced cleaning
#         df["lyrics"] = df["lyrics"].apply(self.clean_text)
#         return df
    
#     def clean_text(self, text):
#         """Enhanced text cleaning"""
#         if not isinstance(text, str):
#             return ""
            
#         # Standardize structure markers
#         for marker in self.structure_markers:
#             pattern = rf'\[{marker}\s*\d*\]|\({marker}\s*\d*\)'
#             replacement = f"[{marker}]"
#             text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
#         # Remove unwanted characters but keep basic punctuation
#         text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"\[\]]", "", text)
        
#         # Standardize whitespace
#         text = re.sub(r"\s+", " ", text).strip()
#         return text
    
#     def build_vocabulary(self, df):
#         """Build enhanced vocabulary with special tokens"""
#         if "lyrics" not in df.columns:
#             raise ValueError("DataFrame must contain a 'lyrics' column")
            
#         word_counter = Counter()
#         for text in df["lyrics"]:
#             if isinstance(text, str):
#                 word_counter.update(text.split())

#         # Keep most common words
#         most_common_words = [word for word, _ in word_counter.most_common(self.config["vocab_size"] - 6)]
        
#         # Special tokens
#         self.word_to_index = {
#             "<PAD>": 0,
#             "<UNK>": 1,
#             "<EOS>": 2,
#             "<NEWLINE>": 3,
#             "<START>": 4,
#             "<END>": 5
#         }
        
#         # Add words to vocabulary
#         for idx, word in enumerate(most_common_words):
#             self.word_to_index[word] = idx + 6
            
#         # Reverse mapping
#         self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
#         # Emotion mappings
#         if "emotion" in df.columns:
#             self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(df["emotion"].unique()))}
#             self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
#         else:
#             print("⚠️ Warning: No 'emotion' column found in DataFrame")
#             self.emotion_to_idx = {"neutral": 0}
#             self.idx_to_emotion = {0: "neutral"}
    
#     def encode_text(self, text):
#         """Enhanced text encoding with structure markers"""
#         if not isinstance(text, str):
#             return [self.word_to_index["<UNK>"]]
            
#         # Replace newlines with special token
#         text = text.replace("\n", " <NEWLINE> ")
        
#         # Enhanced structure marker handling
#         for marker in self.structure_markers:
#             pattern = rf'\[{marker}\]'
#             if re.search(pattern, text, re.IGNORECASE):
#                 text = re.sub(pattern, f" <{marker.upper()}> ", text, flags=re.IGNORECASE)
        
#         tokens = text.split()
#         return [self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in tokens]
    
#     def prepare_training_data(self, df):
#         """Prepare training data with enhanced features"""
#         print("Preparing training data...")
        
#         df["tokenized"] = df["lyrics"].apply(self.encode_text)
        
#         input_sequences = []
#         target_words = []
#         emotion_labels = []
#         line_positions = []
#         line_lengths = []
        
#         sequence_count = 0
        
#         for idx, row in df.iterrows():
#             tokens = row["tokenized"]
#             emotion = self.emotion_to_idx[row["emotion"]]
            
#             current_position = 0  # 0: start, 1: middle, 2: end
#             current_line_length = 0
            
#             for i in range(1, len(tokens)):
#                 # Update line position tracking
#                 if tokens[i-1] == self.word_to_index["<NEWLINE>"]:
#                     current_position = 0
#                     current_line_length = 0
#                 elif tokens[i] == self.word_to_index["<NEWLINE>"]:
#                     current_position = 2
#                 else:
#                     current_position = 1
#                     current_line_length += 1
                
#                 input_seq = tokens[:i][-self.config["max_seq_len"]:]
#                 target_word = tokens[i]
                
#                 # Pad the input sequence
#                 if len(input_seq) < self.config["max_seq_len"]:
#                     padding = [self.word_to_index["<PAD>"]] * (self.config["max_seq_len"] - len(input_seq))
#                     input_seq = padding + input_seq
                
#                 input_sequences.append(input_seq)
#                 target_words.append(target_word)
#                 emotion_labels.append(emotion)
#                 line_positions.append(current_position)
#                 line_lengths.append(min(current_line_length, 19))  # Cap at 19 for embedding
                
#                 sequence_count += 1
#                 if sequence_count % 10000 == 0:
#                     print(f"Processed {sequence_count} sequences...")
        
#         print(f"Finished preparing {len(input_sequences)} training sequences")
        
#         return {
#             "input_sequences": torch.tensor(input_sequences, dtype=torch.long),
#             "target_words": torch.tensor(target_words, dtype=torch.long),
#             "emotions": torch.tensor(emotion_labels, dtype=torch.long),
#             "positions": torch.tensor(line_positions, dtype=torch.long),
#             "lengths": torch.tensor(line_lengths, dtype=torch.long)
#         }

# # ======= SIMPLIFIED MODEL ARCHITECTURE =======
# class SimplifiedLyricsGenerator(nn.Module):
#     def __init__(self, config, vocab_size, num_emotions):
#         super(SimplifiedLyricsGenerator, self).__init__()
#         self.config = config
#         self.device = device
        
#         # Simplified embeddings
#         self.word_embedding = nn.Embedding(vocab_size, config["embed_dim"], padding_idx=0)
#         self.emotion_embedding = nn.Embedding(num_emotions, config["embed_dim"])
#         self.position_embedding = nn.Embedding(3, config["embed_dim"])  # line position
#         self.length_embedding = nn.Embedding(20, config["embed_dim"])  # line length (0-19)
        
#         # Embedding combiner
#         self.embed_combiner = nn.Sequential(
#             nn.Linear(config["embed_dim"] * 4, config["embed_dim"]),
#             nn.LayerNorm(config["embed_dim"]),
#             nn.GELU()
#         )
        
#         # Use a unidirectional LSTM (less memory intensive)
#         self.lstm = nn.LSTM(
#             config["embed_dim"],
#             config["hidden_dim"],
#             num_layers=config["num_layers"],
#             batch_first=True,
#             dropout=config["dropout"] if config["num_layers"] > 1 else 0,
#             bidirectional=False  # Changed to unidirectional
#         )
        
#         # Output layer
#         self.output_layer = nn.Sequential(
#             nn.LayerNorm(config["hidden_dim"]),
#             nn.Linear(config["hidden_dim"], config["hidden_dim"]),
#             nn.GELU(),
#             nn.Dropout(config["dropout"]),
#             nn.Linear(config["hidden_dim"], vocab_size)
#         )
        
#         # Initialize weights
#         self.init_weights()
        
#     def init_weights(self):
#         """Initialize weights for better training stability"""
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 if 'lstm' in name:
#                     nn.init.orthogonal_(param)
#                 elif 'embedding' in name:
#                     nn.init.normal_(param, mean=0, std=0.1)
#                 else:
#                     # Check dimensions before applying Xavier initialization
#                     if len(param.shape) >= 2:
#                         nn.init.xavier_uniform_(param)
#                     else:
#                         nn.init.uniform_(param, -0.1, 0.1)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0.1)
    
#     def forward(self, x, emotion, position=None, length=None):
#         """Forward pass with error handling for dimensions"""
#         # Get batch size and sequence length
#         batch_size, seq_len = x.size()
        
#         # Default values if not provided
#         if position is None:
#             position = torch.ones(batch_size, dtype=torch.long, device=x.device)
#         if length is None:
#             length = torch.ones(batch_size, dtype=torch.long, device=x.device) * 5
        
#         # Ensure proper dimensions and valid ranges
#         position = position.view(-1)
#         length = length.view(-1)
        
#         if position.size(0) != batch_size:
#             position = position.repeat(batch_size)[:batch_size]
#         if length.size(0) != batch_size:
#             length = length.repeat(batch_size)[:batch_size]
            
#         # Clip values to valid ranges
#         position = torch.clamp(position, 0, 2)  # Valid range: 0-2
#         length = torch.clamp(length, 0, 19)    # Valid range: 0-19
        
#         # Embeddings
#         word_emb = self.word_embedding(x)
#         emotion_emb = self.emotion_embedding(emotion).unsqueeze(1).expand(-1, seq_len, -1)
#         pos_emb = self.position_embedding(position).unsqueeze(1).expand(-1, seq_len, -1)
#         len_emb = self.length_embedding(length).unsqueeze(1).expand(-1, seq_len, -1)
        
#         # Combine embeddings
#         combined = torch.cat([word_emb, emotion_emb, pos_emb, len_emb], dim=-1)
#         combined = self.embed_combiner(combined)
        
#         # LSTM processing with error handling
#         try:
#             lstm_out, _ = self.lstm(combined)
#         except RuntimeError as e:
#             print(f"LSTM error: {e}")
#             print(f"Input shape: {combined.shape}")
#             # Try to recover with a smaller batch
#             if batch_size > 1:
#                 half_batch = batch_size // 2
#                 first_half = self.lstm(combined[:half_batch])[0]
#                 second_half = self.lstm(combined[half_batch:])[0]
#                 lstm_out = torch.cat([first_half, second_half], dim=0)
#             else:
#                 raise  # Cannot recover
        
#         # Output projection
#         output = self.output_layer(lstm_out)
#         return output

# # ======= GENERATION CLASS =======
# class LyricsGenerator:
#     def __init__(self, model, preprocessor, config):
#         self.model = model
#         self.model.to(device)
#         self.preprocessor = preprocessor
#         self.config = config
#         self.rhyme_cache = {}
#         self.syllable_cache = {}
#         self.recent_words = deque(maxlen=10)  # Track recent words
        
#     def get_rhyme_score(self, word1, word2):
#         """Calculate rhyming score between two words"""
#         if word1 == word2:
#             return 0.0  # Don't count same word as rhyme
        
#         # Check cache first
#         cache_key = (word1, word2)
#         if cache_key in self.rhyme_cache:
#             return self.rhyme_cache[cache_key]
        
#         # Clean words
#         word1 = re.sub(r'[^\w\s]', '', word1.lower())
#         word2 = re.sub(r'[^\w\s]', '', word2.lower())
        
#         # Handle empty strings
#         if not word1 or not word2:
#             return 0.0
        
#         # Simple suffix matching (fallback)
#         if word1 not in pronounce_dict or word2 not in pronounce_dict:
#             score = 0.5 if len(word1) > 2 and len(word2) > 2 and word1[-3:] == word2[-3:] else 0.0
#             self.rhyme_cache[cache_key] = score
#             return score
        
#         # Get pronunciations
#         pron1 = pronounce_dict[word1][0]
#         pron2 = pronounce_dict[word2][0]
        
#         # Extract vowel sounds
#         vowel_sounds1 = [sound for sound in pron1 if any(vowel in sound for vowel in '012')]
#         vowel_sounds2 = [sound for sound in pron2 if any(vowel in sound for vowel in '012')]
        
#         # Calculate rhyme score
#         score = 0.0
#         if vowel_sounds1 and vowel_sounds2:
#             # Perfect rhyme if last vowel and following sounds match
#             if vowel_sounds1[-1] == vowel_sounds2[-1]:
#                 score = 1.0
#             # Slant rhyme if vowel sounds are similar
#             elif vowel_sounds1[-1][0] == vowel_sounds2[-1][0]:
#                 score = 0.7
        
#         self.rhyme_cache[cache_key] = score
#         return score
    
#     def generate_line(self, current_words, emotion_idx, line_length, rhyme_word=None, section_type="verse"):
#         """Generate a single line of lyrics with the specified properties"""
#         # Start with the current context
#         if not current_words:
#             context = ["<START>"]
#         else:
#             # Take the last few words as context
#             context = current_words[-self.config["max_seq_len"]+1:] if len(current_words) > 0 else ["<START>"]
        
#         # Initialize line with existing context if appropriate
#         line = []
#         syllable_count = 0
#         target_syllables = line_length * 2  # Approximate target
        
#         # Position tracking
#         line_position = 0  # Start of line
        
#         # Maximum words to generate to prevent infinite loops
#         max_words = self.config["max_line_length"] * 2
        
#         # Generate words until we reach desired length or hit special token
#         for _ in range(max_words):
#             # Convert context to tensor
#             tokens = [self.preprocessor.word_to_index.get(w, self.preprocessor.word_to_index["<UNK>"]) for w in context]
#             input_tensor = torch.tensor([tokens], device=device).long()
            
#             # Prepare position and length tensors
#             position_tensor = torch.tensor([line_position], device=device).long()
#             length_tensor = torch.tensor([len(line)], device=device).long()
#             emotion_tensor = torch.tensor([emotion_idx], device=device).long()
            
#             # Generate logits for next word
#             with torch.no_grad():
#                 logits = self.model(input_tensor, emotion_tensor, position_tensor, length_tensor)
#                 logits = logits[0, -1, :]  # Get the last position
                
#             # Apply temperature
#             logits = logits / self.config["temperature"]
            
#             # Apply repetition penalty
#             for word in self.recent_words:
#                 word_idx = self.preprocessor.word_to_index.get(word, self.preprocessor.word_to_index["<UNK>"])
#                 logits[word_idx] /= self.config["repetition_penalty"]
            
#             # If we have a rhyme word and are near the end of line
#             if rhyme_word and syllable_count >= target_syllables * 0.7:
#                 # Boost probabilities of rhyming words
#                 for word, idx in self.preprocessor.word_to_index.items():
#                     if word not in ["<PAD>", "<UNK>", "<EOS>", "<NEWLINE>", "<START>", "<END>"]:
#                         rhyme_score = self.get_rhyme_score(word, rhyme_word)
#                         if rhyme_score > 0.5:
#                             logits[idx] *= self.config["rhyme_boost"]
            
#             # Apply Top-K filtering
#             top_k = min(self.config["top_k"], logits.size(-1))
#             indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#             logits[indices_to_remove] = float('-inf')
            
#             # Apply Top-P filtering (nucleus sampling)
#             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#             cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
#             sorted_indices_to_remove = cumulative_probs > self.config["top_p"]
#             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#             sorted_indices_to_remove[..., 0] = 0
#             indices_to_remove = sorted_indices[sorted_indices_to_remove]
#             logits[indices_to_remove] = float('-inf')
            
#             # Sample next token
#             probs = torch.softmax(logits, dim=-1)
#             next_token = torch.multinomial(probs, 1).item()
#             next_word = self.preprocessor.index_to_word[next_token]
            
#             # Check if we should end the line
#             if next_word == "<NEWLINE>" or next_word == "<EOS>" or next_word == "<END>":
#                 break
                
#             # Add to line if not a special token
#             if not next_word.startswith('<') and next_word != '<PAD>' and next_word != '<UNK>':
#                 line.append(next_word)
#                 self.recent_words.append(next_word)
                
#                 # Update syllable count
#                 syllable_count += self.count_syllables(next_word)
                
#                 # Update position (0: start, 1: middle, 2: end)
#                 if len(line) == 1:
#                     line_position = 0  # Start
#                 else:
#                     line_position = 1  # Middle
            
#             # Check if we've reached target length 
#             if syllable_count >= target_syllables:
#                 break
                
#             # Update context with new word
#             context.append(next_word)
#             if len(context) > self.config["max_seq_len"]:
#                 context = context[-self.config["max_seq_len"]:]
        
#         return line

#     def count_syllables(self, word):
#         """Count syllables in a word"""
#         if not word:
#             return 1
            
#         # Check cache first
#         if word in self.syllable_cache:
#             return self.syllable_cache[word]
        
#         try:
#             count = syllables.estimate(word)
#             self.syllable_cache[word] = count
#             return count
#         except:
#             # Fallback: estimate syllables by vowel groups
#             word = word.lower()
#             count = max(1, len(re.findall(r'[aeiouy]+', word)))
#             self.syllable_cache[word] = count
#             return count
    
#     def generate_song(self, seed="", emotion="happy", song_structure=None):
#         """Generate a complete song with specified structure"""
#         if not seed:
#             seed = "<START>"
            
#         # Default structure if none provided
#         if not song_structure:
#             song_structure = [
#                 "verse", "verse", 
#                 "chorus", 
#                 "verse", 
#                 "chorus",
#                 "bridge",
#                 "chorus", "chorus"
#             ]
        
#         # Convert emotion to index
#         emotion_idx = self.preprocessor.emotion_to_idx.get(emotion.lower(), 0)
        
#         # Initialize song components
#         song_parts = {}
#         current_words = seed.lower().split()
        
#         # Generate each section
#         for section in song_structure:
#             print(f"Generating {section}...")
            
#             # Number of lines per section
#             if section == "chorus":
#                 num_lines = 4
#             elif section == "bridge":
#                 num_lines = 3
#             elif section == "verse":
#                 num_lines = 5
#             else:
#                 num_lines = 4
                
#             lines = []
#             rhyme_pattern = self.get_rhyme_pattern(section, num_lines)
#             rhyme_words = {}
            
#             # Generate each line in the section
#             for i in range(num_lines):
#                 # Line length varies by section type
#                 line_length = random.randint(
#                     self.config["min_line_length"],
#                     self.config["max_line_length"]
#                 )
                
#                 # Adjust by structure weighting
#                 weight = self.config["structure_weights"].get(section, 1.0)
#                 if weight != 1.0:
#                     line_length = int(line_length * weight)
                
#                 # Get rhyme word if needed
#                 rhyme_word = None
#                 if i > 0 and rhyme_pattern[i] in rhyme_pattern[:i]:
#                     # Find the previous line with matching rhyme pattern
#                     for j in range(i):
#                         if rhyme_pattern[j] == rhyme_pattern[i]:
#                             # Get last word of that line
#                             if lines[j]:  # Check if line exists and is not empty
#                                 rhyme_word = lines[j][-1]
#                                 break
                
#                 # Generate the line
#                 line = self.generate_line(
#                     current_words, 
#                     emotion_idx, 
#                     line_length, 
#                     rhyme_word,
#                     section
#                 )
                
#                 lines.append(line)
                
#                 # Update current words with the new line
#                 if line:
#                     current_words.extend(line)
                    
#                     # Keep track of rhyme words
#                     if rhyme_pattern[i] != 'X':
#                         rhyme_words[rhyme_pattern[i]] = line[-1] if line else ""
            
#             # Store the section
#             if section not in song_parts:
#                 song_parts[section] = []
            
#             song_parts[section].append(lines)
        
#         # Format the song
#         formatted_song = self.format_song(song_parts, song_structure)
#         return formatted_song
    
#     def get_rhyme_pattern(self, section_type, num_lines):
#         """Generate appropriate rhyme pattern for section"""
#         if section_type == "chorus":
#             if num_lines == 4:
#                 return ['A', 'B', 'A', 'B']
#             else:
#                 return ['A', 'B', 'A', 'B', 'C']
#         elif section_type == "verse":
#             if num_lines == 4:
#                 return ['A', 'A', 'B', 'B']
#             else:
#                 return ['A', 'A', 'B', 'B', 'C']
#         elif section_type == "bridge":
#             return ['X', 'X', 'X']
#         else:
#             # Default pattern
#             return ['A', 'B'] * (num_lines // 2) + (['C'] if num_lines % 2 else [])
    
#     def format_song(self, song_parts, song_structure):
#         """Format the generated song with section labels"""
#         formatted_song = ""
        
#         for section in song_structure:
#             if section in song_parts and song_parts[section]:
#                 formatted_song += f"[{section.upper()}]\n"
                
#                 # Get the next section of this type
#                 lines = song_parts[section][0]
#                 song_parts[section].pop(0)
                
#                 # Format each line
#                 for line in lines:
#                     if line:
#                         formatted_song += " ".join(line) + "\n"
#                     else:
#                         formatted_song += "\n"
                
#                 formatted_song += "\n"
        
#         return formatted_song.strip()

# # Global variables for the app
# preprocessor = None
# model = None
# generator = None

# def initialize_model():
#     global preprocessor, model, generator
    
#     # Initialize preprocessor
#     preprocessor = LyricsPreprocessor(CONFIG)
    
#     try:
#         # Try to load the dataset
#         dataset_path = "lyrics_with_emotions.csv"
#         df = preprocessor.load_data(dataset_path)
        
#         # Build vocabulary
#         preprocessor.build_vocabulary(df)
        
#         # Initialize model
#         num_emotions = len(preprocessor.emotion_to_idx)
#         model = SimplifiedLyricsGenerator(
#             CONFIG, 
#             len(preprocessor.word_to_index),
#             num_emotions
#         ).to(device)
        
#         # Try to load the enhanced model (changed from lyrics_generator.pth to enhanced_lyrics_generator.pth)
#         try:
#             checkpoint = torch.load('enhanced_lyrics_generator.pth', map_location=device)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             print("✅ Loaded enhanced pretrained model")
#         except (FileNotFoundError, RuntimeError) as e:
#             print(f"⚠️ Error loading enhanced model: {e}")
#             # Try to load the original model as fallback
#             try:
#                 checkpoint = torch.load('lyrics_generator.pth', map_location=device)
#                 model.load_state_dict(checkpoint['model_state_dict'])
#                 print("✅ Loaded original pretrained model as fallback")
#             except (FileNotFoundError, RuntimeError) as e:
#                 print(f"⚠️ Error loading fallback model: {e}")
#                 return False
        
#         # Initialize generator
#         generator = LyricsGenerator(model, preprocessor, CONFIG)
#         return True
        
#     except Exception as e:
#         print(f"❌ Error initializing model: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # Initialize the model when the app starts
# model_initialized = initialize_model()

# @app.route('/')
# def index():
#     # Get available emotions if model is initialized
#     emotions = []
#     if model_initialized and preprocessor:
#         emotions = list(preprocessor.emotion_to_idx.keys())
    
#     return render_template('index.html', emotions=emotions, model_initialized=model_initialized)

# @app.route('/generate', methods=['POST'])
# def generate_lyrics():
#     if not model_initialized:
#         return jsonify({
#             'success': False,
#             'error': 'Model not initialized. Please check server logs.'
#         })
    
#     # Get parameters from request
#     seed = request.form.get('seed', '').strip()
#     emotion = request.form.get('emotion', 'happy').strip().lower()
    
#     # Get song structure
#     structure_string = request.form.get('structure', '').strip()
#     if structure_string:
#         song_structure = [section.strip() for section in structure_string.split(',')]
#     else:
#         song_structure = ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]
    
#     # Check if emotion is valid
#     if emotion not in preprocessor.emotion_to_idx:
#         return jsonify({
#             'success': False,
#             'error': f"Emotion '{emotion}' not recognized. Available emotions: {', '.join(preprocessor.emotion_to_idx.keys())}"
#         })
    
#     try:
#         # Generate the song
#         lyrics = generator.generate_song(
#             seed=seed,
#             emotion=emotion,
#             song_structure=song_structure
#         )
        
#         return jsonify({
# 'success': True,
#             'lyrics': lyrics
#         })
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             'success': False,
#             'error': f'Error generating lyrics: {str(e)}'
#         })

# @app.route('/api/emotions', methods=['GET'])
# def get_emotions():
#     """API endpoint to get available emotions"""
#     if not model_initialized or not preprocessor:
#         return jsonify({
#             'success': False,
#             'error': 'Model not initialized'
#         })
    
#     emotions = list(preprocessor.emotion_to_idx.keys())
#     return jsonify({
#         'success': True,
#         'emotions': emotions
#     })

# @app.route('/api/song_structure', methods=['GET'])
# def get_song_structure():
#     """API endpoint to get recommended song structures"""
#     structures = [
#         ["verse", "chorus", "verse", "chorus", "bridge", "chorus"],
#         ["verse", "verse", "chorus", "verse", "chorus"],
#         ["chorus", "verse", "chorus", "verse", "bridge", "chorus"],
#         ["verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "bridge", "chorus"],
#         ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
#     ]
    
#     return jsonify({
#         'success': True,
#         'structures': structures
#     })

# @app.route('/train', methods=['POST'])
# def train_model():
#     """Endpoint to fine-tune the model with provided lyrics"""
#     if not model_initialized:
#         return jsonify({
#             'success': False,
#             'error': 'Model not initialized'
#         })
    
#     try:
#         # Get training data from request
#         lyrics = request.form.get('lyrics', '').strip()
#         emotion = request.form.get('emotion', 'happy').strip().lower()
        
#         # Check if emotion is valid
#         if emotion not in preprocessor.emotion_to_idx:
#             return jsonify({
#                 'success': False,
#                 'error': f"Emotion '{emotion}' not recognized"
#             })
        
#         # Preprocess the lyrics
#         cleaned_lyrics = preprocessor.clean_text(lyrics)
#         tokenized = preprocessor.encode_text(cleaned_lyrics)
        
#         # Skip if too short
#         if len(tokenized) < 10:
#             return jsonify({
#                 'success': False,
#                 'error': 'Training text too short. Please provide more lyrics.'
#             })
        
#         # Prepare for training
#         model.train()
#         optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"] * 0.1)  # Lower learning rate for fine-tuning
#         criterion = nn.CrossEntropyLoss()
        
#         # Create input sequences
#         input_sequences = []
#         target_words = []
#         positions = []
#         lengths = []
        
#         current_position = 0
#         current_line_length = 0
        
#         for i in range(1, len(tokenized)):
#             # Update position tracking
#             if tokenized[i-1] == preprocessor.word_to_index["<NEWLINE>"]:
#                 current_position = 0
#                 current_line_length = 0
#             elif tokenized[i] == preprocessor.word_to_index["<NEWLINE>"]:
#                 current_position = 2
#             else:
#                 current_position = 1
#                 current_line_length += 1
            
#             input_seq = tokenized[:i][-CONFIG["max_seq_len"]:]
#             target_word = tokenized[i]
            
#             # Pad input sequence
#             if len(input_seq) < CONFIG["max_seq_len"]:
#                 padding = [preprocessor.word_to_index["<PAD>"]] * (CONFIG["max_seq_len"] - len(input_seq))
#                 input_seq = padding + input_seq
            
#             input_sequences.append(input_seq)
#             target_words.append(target_word)
#             positions.append(current_position)
#             lengths.append(min(current_line_length, 19))
        
#         # Convert to tensors
#         input_tensor = torch.tensor(input_sequences, dtype=torch.long).to(device)
#         target_tensor = torch.tensor(target_words, dtype=torch.long).to(device)
#         position_tensor = torch.tensor(positions, dtype=torch.long).to(device)
#         length_tensor = torch.tensor(lengths, dtype=torch.long).to(device)
#         emotion_tensor = torch.tensor([preprocessor.emotion_to_idx[emotion]] * len(input_sequences), 
#                                        dtype=torch.long).to(device)
        
#         # Fine-tune for a few epochs
#         num_epochs = 5
#         batch_size = 16
#         total_loss = 0
        
#         for epoch in range(num_epochs):
#             # Process in batches
#             for i in range(0, len(input_tensor), batch_size):
#                 # Get batch
#                 end_idx = min(i + batch_size, len(input_tensor))
#                 batch_input = input_tensor[i:end_idx]
#                 batch_target = target_tensor[i:end_idx]
#                 batch_position = position_tensor[i:end_idx]
#                 batch_length = length_tensor[i:end_idx]
#                 batch_emotion = emotion_tensor[i:end_idx]
                
#                 # Forward pass
#                 optimizer.zero_grad()
#                 outputs = model(batch_input, batch_emotion, batch_position, batch_length)
#                 loss = criterion(outputs.reshape(-1, len(preprocessor.word_to_index)), 
#                                 batch_target)
                
#                 # Backward pass
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
        
#         # Save the fine-tuned model
#         torch.save({
#             'model_state_dict': model.state_dict(),
#         }, 'enhanced_lyrics_generator.pth')
        
#         # Update the generator with the fine-tuned model
#         generator.model = model
        
#         return jsonify({
#             'success': True,
#             'message': f'Model fine-tuned successfully. Avg loss: {total_loss / (len(input_tensor) / batch_size) / num_epochs:.4f}'
#         })
        
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             'success': False,
#             'error': f'Error fine-tuning model: {str(e)}'
#         })

# @app.route('/download_model', methods=['GET'])
# def download_model():
#     """Endpoint to download the current model"""
#     if not model_initialized:
#         return jsonify({
#             'success': False,
#             'error': 'Model not initialized'
#         })
    
#     try:
#         # Save current model state
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'config': CONFIG,
#             'word_to_index': preprocessor.word_to_index,
#             'index_to_word': preprocessor.index_to_word,
#             'emotion_to_idx': preprocessor.emotion_to_idx,
#             'idx_to_emotion': preprocessor.idx_to_emotion
#         }, 'user_lyrics_generator.pth')
        
#         # Return file path (in production, would stream the file)
#         return jsonify({
#             'success': True,
#             'message': 'Model saved successfully. It can be downloaded from the server directory.'
#         })
        
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': f'Error saving model: {str(e)}'
#         })

# # Health check endpoint
# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint for monitoring"""
#     return jsonify({
#         'status': 'ok',
#         'model_initialized': model_initialized,
#         'device': str(device),
#         'emotions': list(preprocessor.emotion_to_idx.keys()) if model_initialized and preprocessor else []
#     })

# if __name__ == '__main__':
#     # Set debug=False for production
#     app.run(host='0.0.0.0', port=5000, debug=True)







































from flask import Flask, request, render_template, jsonify, send_file
import torch
import numpy as np
import scipy.io.wavfile
import os
import re
import random
import warnings
import threading
import time
import uuid
from functools import wraps
import traceback

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# This would need to be installed first: pip install audiocraft flask
# Import audiocraft after Flask is set up
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

app = Flask(__name__)

# Track ongoing jobs
JOBS = {}

class EnhancedMusicComposer:
    def __init__(self):
        """Fully automated CPU-only music composition system with enhanced musicality"""
        print("🎵 Initializing Advanced Music Composer with melodic focus...")
        
        # Force CPU usage
        self.device = "cpu"
        torch.set_num_threads(4)  # Optimize CPU core usage
        torch.set_grad_enabled(False)
        
        # Load model from AudioCraft - Using small for faster generation
        self.model_name = "small"  # Options: small, medium, melody, large
        print(f"🔄 Loading MusicGen-{self.model_name} for CPU...")
        
        try:
            # Use the AudioCraft implementation directly
            self.model = MusicGen.get_pretrained(self.model_name, device=self.device)
            
            # Set generation parameters for better musical quality but faster processing
            self.model.set_generation_params(
                duration=10,  # Default duration in seconds - shorter for faster processing
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
        
        # More realistic timing calculation - optimized for faster generation
        # Average vocal delivery is about 2-3 words per second in a song
        words_per_second = 2.5
        estimated_lyrics_duration = word_count / words_per_second
        
        # Each section typically has musical interludes
        section_duration = section_count * 3  # Reduced to 3 seconds per section for faster generation
        
        # Calculate total with minimum and maximum constraints
        total_duration = estimated_lyrics_duration + section_duration
        return max(5, min(total_duration, 15))  # Reduced to 5-15 seconds for faster generation

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
            "Ensure all instruments blend naturally like a professional recording."
        )
        
        # Return prompt and settings for continuity
        return base_prompt, {'key': key, 'bpm': bpm, 'genre': genre, 'emotion': emotion}

    def _generate_single_segment(self, lyrics, output_path):
        """Generate a complete song as a single segment with enhanced musicality - optimized for web"""
        try:
            # Get genre and emotion without generating continuity
            genre, emotion, sections = self._analyze_lyrics(lyrics)
            
            # Create optimized prompt for faster generation
            prompt, _ = self._create_enhanced_prompt(lyrics, genre, emotion, sections)
            
            # Set generation parameters optimized for web (shorter duration)
            duration = min(10, self._calculate_duration(lyrics))  # Reduced from 15 to 10
            self.model.set_generation_params(
                duration=duration,
                temperature=0.85,  # Keep temperature
                top_p=0.92,        # Keep top_p
                top_k=250,         # Keep top_k
                cfg_coef=3.0       # Keep cfg_coef
            )
            
            # Generate audio with error handling
            print(f"Generating quick audio of {duration} seconds...")
            try:
                # Try with proper text conditioning
                wav = self.model.generate([prompt])
            except AssertionError:
                # If assertion fails, try with a simpler prompt
                simplified_prompt = f"A {emotion} {genre} song with the lyrics: {lyrics}"
                wav = self.model.generate([simplified_prompt])
            
            # Get numpy array from tensor
            audio = wav[0, 0].cpu().numpy()
            
            # Properly normalize the audio
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak) * 0.95  # 5% headroom to avoid clipping
            
            # Save using fixed audio write function
            if self._write_audio(audio, output_path):
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
                
                # Fallback method
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

    def compose_song_web(self, lyrics, job_id):
        """Web-optimized song composition that updates job status"""
        try:
            # Update job status
            JOBS[job_id]['status'] = 'processing'
            JOBS[job_id]['progress'] = 10
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(app.static_folder, 'songs')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a unique filename
            output_path = os.path.join(output_dir, f"{job_id}.wav")
            
            # Update progress
            JOBS[job_id]['progress'] = 20
            
            # Generate the song using the optimized single segment approach for web
            result = self._generate_single_segment(lyrics, output_path)
            
            if result:
                # Update job status on success
                JOBS[job_id]['status'] = 'completed'
                JOBS[job_id]['progress'] = 100
                JOBS[job_id]['file_path'] = output_path
                return output_path
            else:
                # Update job status on failure
                JOBS[job_id]['status'] = 'failed'
                JOBS[job_id]['error'] = 'Failed to generate song'
                return None
                
        except Exception as e:
            # Update job status on exception
            if job_id in JOBS:
                JOBS[job_id]['status'] = 'failed'
                JOBS[job_id]['error'] = str(e)
            print(f"⚠️ Error in web song composition: {e}")
            traceback.print_exc()
            return None

# Initialize composer as a global variable - will be lazy loaded
composer = None

def get_composer():
    """Lazy load the composer to avoid loading the model until needed"""
    global composer
    if composer is None:
        composer = EnhancedMusicComposer()
    return composer

# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@app.route('/generate', methods=['POST'])
def generate_song():
    """Start asynchronous song generation"""
    if request.method == 'POST':
        try:
            # Get lyrics from form
            lyrics = request.form.get('lyrics', '')
            
            if not lyrics:
                return jsonify({'error': 'No lyrics provided'}), 400
            
            # Validate lyrics format
            if not lyrics.strip():
                return jsonify({'error': 'Empty lyrics provided'}), 400
                
            # Generate a unique job ID
            job_id = str(uuid.uuid4())
            
            # Create job record
            JOBS[job_id] = {
                'status': 'queued',
                'progress': 0,
                'created_at': time.time(),
                'error': None,
                'file_path': None
            }
            
            # Start generation in a background thread
            def generate_in_background(lyrics, job_id):
                try:
                    # Get composer (lazy loading)
                    composer = get_composer()
                    # Generate song
                    composer.compose_song_web(lyrics, job_id)
                except Exception as e:
                    if job_id in JOBS:
                        JOBS[job_id]['status'] = 'failed'
                        JOBS[job_id]['error'] = str(e)
                    print(f"⚠️ Background job error: {e}")
                    traceback.print_exc()
                    
            # Start background thread
            thread = threading.Thread(
                target=generate_in_background,
                args=(lyrics, job_id)
            )
            thread.daemon = True
            thread.start()
            
            # Return job ID for status polling
            return jsonify({
                'job_id': job_id,
                'status': 'queued',
                'message': 'Song generation has been queued'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
@app.route('/status/<job_id>')
def job_status(job_id):
    """Check the status of a generation job"""
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
        
    job = JOBS[job_id]
    
    # Check if job is completed and has a file path
    if job['status'] == 'completed' and job['file_path']:
        # Get just the filename part
        filename = os.path.basename(job['file_path'])
        # Return with download URL
        return jsonify({
            'status': job['status'],
            'progress': job['progress'],
            'download_url': f'/download/{job_id}'
        })
    
    # Return status information
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'error': job['error']
    })

@app.route('/download/<job_id>')
def download_song(job_id):
    """Download the generated song"""
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    job = JOBS[job_id]
    if job['status'] != 'completed' or not job['file_path']:
        return jsonify({'error': 'Song not ready or generation failed'}), 400
    
    # Send the file for download
    return send_file(job['file_path'], as_attachment=True, 
                     download_name=f"generated_song_{job_id}.wav")

# Clean up old jobs periodically (optional)
def cleanup_old_jobs():
    """Remove old jobs and files to free up space"""
    current_time = time.time()
    for job_id in list(JOBS.keys()):
        job = JOBS[job_id]
        # Remove jobs older than 1 hour
        if current_time - job['created_at'] > 3600:
            # Delete the file if it exists
            if job['file_path'] and os.path.exists(job['file_path']):
                try:
                    os.remove(job['file_path'])
                except:
                    pass
            # Remove from jobs dictionary
            del JOBS[job_id]

# Create required directories
def setup_dirs():
    """Create necessary directories on startup"""
    # Create static folder if it doesn't exist
    if not os.path.exists(app.static_folder):
        os.makedirs(app.static_folder)
    # Create songs folder inside static
    songs_dir = os.path.join(app.static_folder, 'songs')
    if not os.path.exists(songs_dir):
        os.makedirs(songs_dir)
        
# Create templates directory and HTML templates


# Call create_templates when the app is initialized

if __name__ == '__main__':
    # Call setup_dirs before running the app
    setup_dirs()
    # Create templates
    # Run with threaded=True for handling concurrent requests
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
































