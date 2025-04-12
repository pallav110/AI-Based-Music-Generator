import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter

import re
import random
import nltk
from nltk.corpus import cmudict

# Download pronunciation dictionary for rhyme detection
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict')



# For Rhyming (Added)
import nltk
try:
    # Try to load the dictionary directly
    from nltk.corpus import cmudict
    pronounce_dict = cmudict.dict()
    print("✅ CMU Pronouncing Dictionary loaded.")
except LookupError:
    # If not found, download it
    print("CMU Pronouncing Dictionary not found. Downloading...")
    nltk.download('cmudict')
    from nltk.corpus import cmudict
    try:
        pronounce_dict = cmudict.dict()
        print("✅ CMU Pronouncing Dictionary downloaded and loaded.")
    except Exception as e:
        print(f"⚠️ Error loading CMU Pronouncing Dictionary after download: {e}")
        print("⚠️ Rhyming functionality may be limited.")
        pronounce_dict = None # Set to None if loading fails
except Exception as e:
     print(f"⚠️ An unexpected error occurred loading CMU Pronouncing Dictionary: {e}")
     print("⚠️ Rhyming functionality may be limited.")
     pronounce_dict = None # Set to None on other errors

# ======= DEVICE SETUP =======
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
# ======= DATA PREPARATION =======

# Load dataset
df = pd.read_csv("lyrics_with_emotions.csv").dropna()

# Convert to lowercase
df["lyrics"] = df["lyrics"].str.lower()

# Get all emotions
all_emotions = sorted(df["emotion"].unique())
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(all_emotions)}
idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
NUM_EMOTIONS = len(all_emotions)

print(f"Found {NUM_EMOTIONS} emotions: {all_emotions}")

# ✅ Build Vocabulary 
word_counter = Counter()
for text in df["lyrics"]:
    word_counter.update(text.split())

# ✅ Keep most common words
VOCAB_SIZE = 5000
most_common_words = [word for word, _ in word_counter.most_common(VOCAB_SIZE - 4)]
word_to_index = {word: idx + 4 for idx, word in enumerate(most_common_words)}  
# Special tokens
word_to_index["<PAD>"] = 0
word_to_index["<UNK>"] = 1
word_to_index["<EOS>"] = 2  # End of sequence
word_to_index["<NEWLINE>"] = 3  # Line break token

# Reverse mapping (Index → Word)
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Extract song structure markers
structure_markers = ["verse", "chorus", "bridge", "pre-chorus", "hook", "intro", "outro"]
structure_to_idx = {marker: idx for idx, marker in enumerate(structure_markers)}
idx_to_structure = {idx: marker for marker, idx in structure_to_idx.items()}

# Enhanced text encoding with structure markers
def encode_text(text):
    # Replace newlines with special token
    text = text.replace("\n", " <NEWLINE> ")
    
    # Identify structure markers
    for marker in structure_markers:
        pattern = rf'\[{marker}\]|\({marker}\)'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, f" <{marker.upper()}> ", text, flags=re.IGNORECASE)
    
    tokens = text.split()
    return [word_to_index.get(token, word_to_index["<UNK>"]) for token in tokens]

# Process lyrics to identify structure
def process_lyrics_with_structure(lyrics):
    # Replace common structure markers
    lyrics = re.sub(r'\[verse\s*\d*\]|\(verse\s*\d*\)', '<VERSE>', lyrics, flags=re.IGNORECASE)
    lyrics = re.sub(r'\[chorus\]|\(chorus\)', '<CHORUS>', lyrics, flags=re.IGNORECASE)
    lyrics = re.sub(r'\[bridge\]|\(bridge\)', '<BRIDGE>', lyrics, flags=re.IGNORECASE)
    lyrics = re.sub(r'\[pre.?chorus\]|\(pre.?chorus\)', '<PRE-CHORUS>', lyrics, flags=re.IGNORECASE)
    return lyrics

# Apply structure processing
df["processed_lyrics"] = df["lyrics"].apply(process_lyrics_with_structure)
df["tokenized"] = df["processed_lyrics"].apply(encode_text)

# ✅ Prepare Training Data with improved context
MAX_SEQ_LEN = 32  # Increased for better context

input_sequences = []
target_words = []
emotion_labels = []
line_positions = []  # Track position in line (start, middle, end)

print("Preparing enhanced training data...")
for idx, row in df.iterrows():
    tokens = row["tokenized"]
    emotion = emotion_to_idx[row["emotion"]]
    
    # Track line position
    current_position = 0  # 0: start, 1: middle, 2: end
    
    for i in range(1, len(tokens)):
        # Update line position tracking
        if tokens[i-1] == word_to_index["<NEWLINE>"]:
            current_position = 0  # Start of line
        elif tokens[i] == word_to_index["<NEWLINE>"]:
            current_position = 2  # End of line
        else:
            current_position = 1  # Middle of line
            
        input_seq = tokens[:i][-MAX_SEQ_LEN:]  # Take last words as input
        target_word = tokens[i]  # Next word to predict
        
        # Pad the input sequence
        if len(input_seq) < MAX_SEQ_LEN:
            input_seq = [word_to_index["<PAD>"]] * (MAX_SEQ_LEN - len(input_seq)) + input_seq
        
        input_sequences.append(input_seq)
        target_words.append(target_word)
        emotion_labels.append(emotion)
        line_positions.append(current_position)

# Convert to PyTorch tensors
X = torch.tensor(input_sequences, dtype=torch.long)
y = torch.tensor(target_words, dtype=torch.long)
emotions = torch.tensor(emotion_labels, dtype=torch.long)
positions = torch.tensor(line_positions, dtype=torch.long)

# # ✅ Split data into train/validation/test
dataset = TensorDataset(X, y, emotions, positions)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(
    dataset, [train_size, val_size, test_size]
)

# # ✅ Use mini-batches
BATCH_SIZE = 128
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ======= ENHANCED MODEL DEFINITION =======

# ======= ENHANCED MODEL DEFINITION (Keep your existing class) =======
class EnhancedLyricsGenerator(nn.Module):
    def __init__(self, vocab_size, num_emotions, embed_dim=256, hidden_dim=512,
                 num_layers=3, dropout=0.3, num_heads=4):
        super(EnhancedLyricsGenerator, self).__init__()

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # Assuming PAD index is 0

        # Emotion embeddings
        self.emotion_embedding = nn.Embedding(num_emotions, embed_dim)

        # Position embeddings (start/middle/end of line)
        self.position_embedding = nn.Embedding(3, embed_dim) # 3 positions: start, middle, end

        # Embedding combiner
        self.embed_combiner = nn.Linear(embed_dim * 3, embed_dim) # word + emotion + position

        # LSTM layers
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True # Using bidirectional
        )

        # Self-attention
        # Ensure embed_dim matches hidden_dim*2 for attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=num_heads, batch_first=True) # Added batch_first=True

        # Semantic coherence gate
        self.semantic_gate = nn.Linear(hidden_dim*2, hidden_dim*2)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim) # From BiLSTM output
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim*2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, emotion, position): # Ensure position is passed
        batch_size, seq_len = x.size()

        # Get word embeddings
        x_emb = self.embedding(x) # (batch_size, seq_len, embed_dim)

        # Get emotion embedding and expand
        emotion_emb = self.emotion_embedding(emotion).unsqueeze(1) # (batch_size, 1, embed_dim)
        emotion_emb = emotion_emb.expand(-1, seq_len, -1) # (batch_size, seq_len, embed_dim)

        # Get position embeddings (needs position for each token in sequence)
        # Note: The original code passed only *one* position per sequence during generation.
        # For training, it likely used positions for each token.
        # Let's assume we pass the position of the *last* token for generation prediction.
        # We might need to adjust this if the model expects sequence-wise positions.
        # For simplicity here, we'll use the single provided position expanded.
        pos_emb = self.position_embedding(position).unsqueeze(1) # (batch_size, 1, embed_dim) - Assuming position is single value per batch item
        pos_emb = pos_emb.expand(-1, seq_len, -1) # (batch_size, seq_len, embed_dim)


        # Combine embeddings
        combined = torch.cat([x_emb, emotion_emb, pos_emb], dim=2) # (batch_size, seq_len, embed_dim * 3)
        combined = torch.relu(self.embed_combiner(combined)) # (batch_size, seq_len, embed_dim) Apply activation? Original didn't, adding ReLU common.

        # Process with LSTM
        lstm_out, _ = self.lstm(combined) # lstm_out: (batch_size, seq_len, hidden_dim*2)

        # Apply self-attention
        # LayerNorm before attention is common practice
        lstm_out_norm = self.layer_norm1(lstm_out)
        # Attention needs query, key, value. Using lstm_out for all.
        attn_out, _ = self.attention(lstm_out_norm, lstm_out_norm, lstm_out_norm) # (batch_size, seq_len, hidden_dim*2)

        # Apply semantic gate (Residual connection style)
        gate = torch.sigmoid(self.semantic_gate(lstm_out)) # (batch_size, seq_len, hidden_dim*2)
        gated_output = gate * lstm_out + (1 - gate) * attn_out # (batch_size, seq_len, hidden_dim*2)

        # Use the output corresponding to the *last* token of the input sequence
        last_token_output = gated_output[:, -1, :] # (batch_size, hidden_dim*2)

        # Final predictions
        out = self.dropout(last_token_output)
        out = self.fc1(out)
        out = self.layer_norm2(out) # LayerNorm before activation
        out = torch.relu(out)
        out = self.fc2(out) # (batch_size, vocab_size)
        return out


# ✅ Initialize Enhanced Model
TOTAL_VOCAB_SIZE = len(word_to_index)
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 3

print(f"Vocab size: {TOTAL_VOCAB_SIZE}")
model = EnhancedLyricsGenerator(
    TOTAL_VOCAB_SIZE, 
    NUM_EMOTIONS,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
).to(device)

# ✅ Loss and optimizer with learning rate scheduling
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.003, 
    epochs=25, 
    steps_per_epoch=len(train_loader)
)

# ======= ENHANCED TRAINING LOOP =======

def train_model(num_epochs=25):
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_y, batch_emotion, batch_position in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_emotion = batch_emotion.to(device)
            batch_position = batch_position.to(device)
            
            optimizer.zero_grad()
            
            output = model(batch_X, batch_emotion, batch_position)
            loss = criterion(output, batch_y)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == batch_y).sum().item()
            total_predictions += batch_y.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_emotion, batch_position in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_emotion = batch_emotion.to(device)
                batch_position = batch_position.to(device)
                
                output = model(batch_X, batch_emotion, batch_position)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping with patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'word_to_index': word_to_index,
                'index_to_word': index_to_word,
                'emotion_to_idx': emotion_to_idx,
                'idx_to_emotion': idx_to_emotion,
            }, "enhanced_lyrics_generator.pth")
            print(f"✅ Model saved (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

# Train the model
print("Starting enhanced training...")
train_model(num_epochs=25)
print("Training complete!")

# ======= IMPROVED LYRICS GENERATION =======

# Function to detect rhymes for better rhyme schemes
# ======= RHYME HELPER FUNCTION =======
def words_rhyme(word1, word2):
    """Check if two words rhyme based on their pronunciation using CMU dict."""
    if not pronounce_dict: # Check if dictionary loaded
        # Fallback to simple suffix matching if CMU dict not available
        return word1[-2:] == word2[-2:] if len(word1) > 2 and len(word2) > 2 else False

    # Strip punctuation and convert to lowercase
    word1 = re.sub(r'[^\w\s]', '', word1).lower()
    word2 = re.sub(r'[^\w\s]', '', word2).lower()

    # Check if both words exist in pronouncing dictionary
    if word1 not in pronounce_dict or word2 not in pronounce_dict:
        # Fallback if words not in dict
        return word1[-2:] == word2[-2:] if len(word1) > 2 and len(word2) > 2 else False

    # Get pronunciations (using the first pronunciation listed)
    pron1 = pronounce_dict[word1][0]
    pron2 = pronounce_dict[word2][0]

    # Find the primary stress vowel sound and compare suffixes from there
    idx1 = -1
    idx2 = -2
    for i in range(len(pron1) -1, -1, -1):
        if any(vowel in pron1[i] for vowel in '12'): # Primary or secondary stress
            idx1 = i
            break
    for i in range(len(pron2) -1, -1, -1):
        if any(vowel in pron2[i] for vowel in '12'):
             idx2 = i
             break

    # Compare the pronunciation suffixes from the stressed vowel onwards
    return pron1[idx1:] == pron2[idx2:]

def load_enhanced_model(model_path="enhanced_lyrics_generator.pth"):
    """Loads the saved model and associated dictionaries."""
    try:
        checkpoint = torch.load(model_path, map_location=device) # Load to the correct device

        # Load parameters needed for model init
        # Ensure these match the keys saved during training
        vocab_size = len(checkpoint['word_to_index'])
        num_emotions = len(checkpoint['emotion_to_idx'])
        # You might need to save/load embed_dim, hidden_dim, num_layers if they vary
        # Using placeholder values assuming they are fixed as below:
        embed_dim = 256 # Or load from checkpoint if saved
        hidden_dim = 512 # Or load from checkpoint if saved
        num_layers = 3   # Or load from checkpoint if saved

        model = EnhancedLyricsGenerator(
            vocab_size,
            num_emotions,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set model to evaluation mode

        print(f"✅ Model loaded successfully from {model_path}")
        # Return model and dictionaries
        return model, checkpoint['word_to_index'], checkpoint['index_to_word'], checkpoint['emotion_to_idx'], checkpoint['idx_to_emotion']

    except FileNotFoundError:
        print(f"⚠️ Error: Model file not found at {model_path}")
        print("⚠️ Please ensure the model has been trained and the path is correct.")
        return None, None, None, None, None
    except KeyError as e:
        print(f"⚠️ Error: Missing key in checkpoint file: {e}")
        print("⚠️ Ensure the checkpoint contains 'model_state_dict', 'word_to_index', etc.")
        return None, None, None, None, None
    except Exception as e:
        print(f"⚠️ An unexpected error occurred while loading the model: {e}")
        return None, None, None, None, None

# ======= **REVISED** LYRICS GENERATION FUNCTION =======
def generate_lyrics_formatted(model, seed_text, emotion, word_to_index, index_to_word, emotion_to_idx,
                              max_words=150, temperature=0.8, top_k=50,
                              rhyme_enforcement_strength=3.0):
    """
    Generates structured lyrics with proper paragraph breaks based on <NEWLINE> token.
    Includes optional rhyme enforcement.
    """
    model.eval() # Ensure model is in eval mode

    # --- Input Preparation ---
    emotion_lower = emotion.lower()
    if emotion_lower not in emotion_to_idx:
        print(f"⚠️ Emotion '{emotion}' not recognized. Using default emotion index 0.")
        emotion_idx = 0
    else:
        emotion_idx = emotion_to_idx[emotion_lower]

    # Tokenize seed text
    words = seed_text.lower().split()
    # Handle empty seed text
    if not words:
        words = ["<NEWLINE>"] # Start with a newline to ensure first line generation

    generated_words = words.copy() # Keep track of the token sequence for model input
    output_lines = [] # Store the formatted lines of the song
    current_line_words = [] # Words in the line currently being built

    # Rhyme tracking
    last_line_end_word = None
    enforce_rhyme_now = False

    # --- Generation Loop ---
    print("Generating...")
    for i in range(max_words):
        # Prepare input sequence for the model
        input_sequence = generated_words[-MAX_SEQ_LEN:]
        input_tokens = [word_to_index.get(word, word_to_index["<UNK>"]) for word in input_sequence]

        # Pad the sequence
        padding_needed = MAX_SEQ_LEN - len(input_tokens)
        if padding_needed > 0:
            input_tokens = [word_to_index["<PAD>"]] * padding_needed + input_tokens

        # Determine current position in line (0: start, 1: middle) - End is handled by newline
        # Note: The model expects a position per *batch item*. Here batch size is 1.
        position = 0 if not current_line_words else 1
        # Position tensor needs to match the input tensor shape expected by the embedding layer in the model
        position_tensor = torch.tensor([position], dtype=torch.long).to(device) # Single value for the batch

        # Convert to tensors
        tokens_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension
        emotion_tensor = torch.tensor([emotion_idx], dtype=torch.long).to(device)

        # --- Model Prediction ---
        with torch.no_grad():
            output_logits = model(tokens_tensor, emotion_tensor, position_tensor) # Pass position

            # Apply Temperature Scaling
            output_logits = output_logits / temperature

            # Apply Top-K Filtering
            top_k_values, top_k_indices = torch.topk(output_logits, top_k)
            # Create a mask for filtering
            filter_mask = torch.full_like(output_logits, float('-inf'))
            filter_mask.scatter_(1, top_k_indices, top_k_values) # Use scatter not gather

            # --- Rhyme Enforcement (Optional) ---
            # Enforce rhyme only when starting a new line (position 0) and we have a previous line end word
            if enforce_rhyme_now and last_line_end_word and pronounce_dict:
                # Iterate through the top-k words only for efficiency
                for k in range(top_k_indices.size(1)):
                    idx = top_k_indices[0, k].item() # Get index from top-k list
                    word = index_to_word.get(idx, "<UNK>")
                    if words_rhyme(word, last_line_end_word):
                         # Boost probability of rhyming words within the top-k
                         # Check if the index exists in the filtered output before boosting
                         if idx < filter_mask.size(1):
                            # Add a boost value (adjust strength as needed)
                            filter_mask[0, idx] += rhyme_enforcement_strength
                enforce_rhyme_now = False # Reset enforcement flag

            # Sample from the filtered distribution
            probabilities = torch.softmax(filter_mask, dim=1).squeeze() # Get probabilities
            # Handle potential NaN/Inf issues if probabilities become zero everywhere after filtering
            if not torch.isfinite(probabilities).all() or torch.sum(probabilities) == 0:
                 print("⚠️ Warning: Probability distribution issue after filtering/rhyming. Sampling from original top-k.")
                 # Fallback: Sample directly from the original top-k logits without rhyme boost
                 probabilities = torch.softmax(top_k_values, dim=1).squeeze()
                 predicted_idx = top_k_indices[0, torch.multinomial(probabilities, 1).item()].item()
            else:
                predicted_idx = torch.multinomial(probabilities, 1).item() # Sample index


        # --- Process Predicted Word ---
        predicted_word = index_to_word.get(predicted_idx, "<UNK>")

        # Stop if EOS token is generated
        if predicted_word == "<EOS>":
            break

        # Skip padding or unknown tokens if they are predicted directly
        if predicted_word in ["<PAD>", "<UNK>"]:
            continue

        # **Handle Newline Token for Paragraphs**
        if predicted_word == "<NEWLINE>":
            if current_line_words: # Only add line if it's not empty
                # Finalize the current line
                line_str = " ".join(current_line_words)
                output_lines.append(line_str)

                # Track end word for rhyming
                last_line_end_word = current_line_words[-1]
                # Decide if the *next* line should rhyme (e.g., for AABB, rhyme on even lines)
                if len(output_lines) % 2 == 1: # If we just added an odd line (1st, 3rd, etc.)
                     enforce_rhyme_now = True # Enforce rhyme for the next line
                else:
                     enforce_rhyme_now = False

                # Reset for the next line
                current_line_words = []

            # Add the <NEWLINE> token itself to the sequence history for the model
            generated_words.append(predicted_word)

        # Handle Structure Tokens (like <VERSE>, <CHORUS> if your model learned them)
        # Example: Check if the predicted word is a structure token from your vocab
        elif predicted_word.startswith('<') and predicted_word.endswith('>') and predicted_word != "<NEWLINE>":
             # If there's a line in progress, finish it first
             if current_line_words:
                  line_str = " ".join(current_line_words)
                  output_lines.append(line_str)
                  last_line_end_word = current_line_words[-1]
                  # Decide rhyme enforcement based on the line just added
                  if len(output_lines) % 2 == 1: enforce_rhyme_now = True
                  else: enforce_rhyme_now = False
                  current_line_words = []

             # Format the structure marker (e.g., <VERSE> -> [Verse])
             structure_name = predicted_word[1:-1].replace('-', ' ').title() # Example formatting
             output_lines.append(f"\n[{structure_name}]") # Add with line breaks for separation
             last_line_end_word = None # Reset rhyme target after structure marker
             enforce_rhyme_now = False
             # Add the token to the model's history
             generated_words.append(predicted_word)

        # Handle Regular Words
        else:
            current_line_words.append(predicted_word)
            # Add the word to the model's history
            generated_words.append(predicted_word)

        # Optional: Print progress
        if i % 25 == 0:
            print(f"  ...generated {i+1} words")


    # --- Finalization ---
    # Add any remaining words in the last line
    if current_line_words:
        output_lines.append(" ".join(current_line_words))

    print("Generation complete.")

    # Join the formatted lines into the final output string
    final_lyrics = "\n".join(output_lines)

    # --- Optional: Post-processing Cleanup ---
    # Fix potential spacing issues around punctuation if needed (model might learn this)
    final_lyrics = re.sub(r'\s+([,.!?])', r'\1', final_lyrics) # Remove space before punctuation
    final_lyrics = re.sub(r'\(\s+', r'(', final_lyrics)       # Remove space after opening parenthesis
    final_lyrics = re.sub(r'\s+\)', r')', final_lyrics)       # Remove space before closing parenthesis
    final_lyrics = re.sub(r'\n\s+\n', r'\n\n', final_lyrics) # Reduce multiple blank lines to one
    final_lyrics = final_lyrics.strip() # Remove leading/trailing whitespace

    return final_lyrics

# Function to test model with user input
# ======= INTERACTIVE TESTING FUNCTION (Modified to use new generation) =======
def test_model_interactive():
    """Test the enhanced model interactively using the formatted generation."""

    # Load the model and dictionaries
    model, word_to_index, index_to_word, emotion_to_idx, idx_to_emotion = load_enhanced_model()

    if model is None:
        print("Exiting due to model loading failure.")
        return

    # Check if pronouncing dictionary loaded for rhyme info
    if pronounce_dict is None:
        print("\n⚠️ Warning: CMU Pronouncing Dictionary not loaded. Rhyme enforcement might be basic (suffix matching) or disabled.")


    print("\n--- FORMATTED LYRICS GENERATOR ---")
    if emotion_to_idx:
      print(f"Available emotions: {list(emotion_to_idx.keys())}")
    else:
      print("⚠️ Emotion list not available (ensure emotion_to_idx loaded correctly).")

    try:
        while True:
            seed = input("\nEnter seed text (e.g., 'sun is shining') (or 'quit'): ").strip()
            if seed.lower() == 'quit':
                break

            emotion = input(f"Enter emotion ({'/'.join(emotion_to_idx.keys()) if emotion_to_idx else 'e.g., happy, sad'}): ").lower().strip()
            if emotion == 'quit':
                break

            if emotion_to_idx and emotion not in emotion_to_idx:
                print(f"⚠️ Emotion '{emotion}' not found! Please choose from the available emotions.")
                continue

            try:
                temp_str = input("Enter temperature (e.g., 0.8, higher=more random, lower=more focused): ").strip()
                temp = float(temp_str) if temp_str else 0.8
            except ValueError:
                print("Invalid temperature, using default 0.8.")
                temp = 0.8

            try:
                max_len_str = input("Enter max words to generate (e.g., 100): ").strip()
                max_len = int(max_len_str) if max_len_str else 100
            except ValueError:
                print("Invalid max length, using default 100.")
                max_len = 100

            print(f"\nGenerating {emotion} lyrics from seed '{seed}' (temp={temp}, max_words={max_len})...\n")

            # === Call the **REVISED** generation function ===
            start_time = time.time()
            generated_lyrics = generate_lyrics_formatted(
                model=model,
                seed_text=seed,
                emotion=emotion,
                word_to_index=word_to_index,
                index_to_word=index_to_word,
                emotion_to_idx=emotion_to_idx,
                max_words=max_len,
                temperature=temp,
                top_k=50 # Keep top_k fixed or make it an input too
            )
            end_time = time.time()

            print("-" * 60)
            print(generated_lyrics)
            print("-" * 60)
            print(f"(Generation took {end_time - start_time:.2f} seconds)")
                
    except Exception as e:
        print(f"⚠️ Error: {e}")
        print("⚠️ Make sure you've trained the enhanced model first!")

# Run interactive testing when script is executed
if __name__ == "__main__":
    test_model_interactive()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import pandas as pd
# import numpy as np
# import scanoss.api
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from collections import Counter
# import re
# import random
# import nltk
# from nltk.corpus import cmudict
# from nltk.tokenize import word_tokenize
# import syllables
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.feature_extraction.text import CountVectorizer
# MAX_SEQ_LEN = 32  # Increased for better context

# # Download required NLTK data
# nltk.download('punkt')
# try:
#     nltk.data.find('corpora/cmudict')
# except LookupError:
#     nltk.download('cmudict')

# # Initialize pronunciation dictionary
# pronounce_dict = cmudict.dict()

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"✅ Using device: {device}")

# # ======= PRE-TRAINED MODEL SETUP =======
# class PretrainedLyricsModel:
#     def __init__(self, model_name="gpt2"):
#         self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#         self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
#         self.model.eval()
        
#     def generate(self, prompt, max_length=50, temperature=0.7):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
#         outputs = self.model.generate(
#             inputs.input_ids,
#             max_length=max_length,
#             temperature=temperature,
#             do_sample=True,
#             top_k=50,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Initialize pre-trained model
# pretrained_model = PretrainedLyricsModel("lyricist-gpt")

# # ======= ENHANCED DATA PREPARATION =======
# # Load dataset
# df = pd.read_csv("lyrics_with_emotions.csv").dropna()

# # Enhanced text cleaning
# def clean_text(text):
#     text = text.lower()
#     # Remove special characters but keep punctuation needed for lyrics
#     text = re.sub(r"[^\w\s'.,!?]", "", text)
#     # Standardize whitespace
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# df["lyrics"] = df["lyrics"].apply(clean_text)

# # Get all emotions
# all_emotions = sorted(df["emotion"].unique())
# emotion_to_idx = {emotion: idx for idx, emotion in enumerate(all_emotions)}
# idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
# NUM_EMOTIONS = len(all_emotions)

# print(f"Found {NUM_EMOTIONS} emotions: {all_emotions}")

# # Enhanced Vocabulary Builder
# word_counter = Counter()
# for text in df["lyrics"]:
#     tokens = word_tokenize(text)
#     word_counter.update(tokens)

# # Keep most common words with enhanced filtering
# VOCAB_SIZE = 10000  # Larger vocabulary for better expression
# most_common_words = [word for word, _ in word_counter.most_common(VOCAB_SIZE) 
#                     if len(word) > 1 or word in ["i", "a"]]

# word_to_index = {word: idx + 4 for idx, word in enumerate(most_common_words)}  
# # Special tokens
# word_to_index["<PAD>"] = 0
# word_to_index["<UNK>"] = 1
# word_to_index["<EOS>"] = 2
# word_to_index["<NEWLINE>"] = 3

# index_to_word = {idx: word for word, idx in word_to_index.items()}

# # Enhanced song structure markers
# structure_markers = ["verse", "chorus", "bridge", "pre-chorus", "hook", "intro", "outro", "refrain"]
# structure_to_idx = {marker: idx for idx, marker in enumerate(structure_markers)}
# idx_to_structure = {idx: marker for marker, idx in structure_to_idx.items()}

# # Enhanced text encoding with better structure handling
# def encode_text(text):
#     text = re.sub(r"\n+", " <NEWLINE> ", text)
#     for marker in structure_markers:
#         pattern = rf'\[{marker}[^\]]*\]|\({marker}[^\)]*\)|\b{marker}\b'
#         text = re.sub(pattern, f" <{marker.upper()}> ", text, flags=re.IGNORECASE)
#     tokens = word_tokenize(text)
#     return [word_to_index.get(token.lower(), word_to_index["<UNK>"]) for token in tokens]

# # Process lyrics to identify structure with better pattern matching
# def process_lyrics_with_structure(lyrics):
#     lyrics = re.sub(r'\[verse[^\]]*\]|\(verse[^\)]*\)', '<VERSE>', lyrics, flags=re.IGNORECASE)
#     lyrics = re.sub(r'\[chorus[^\]]*\]|\(chorus[^\)]*\)', '<CHORUS>', lyrics, flags=re.IGNORECASE)
#     lyrics = re.sub(r'\[bridge[^\]]*\]|\(bridge[^\)]*\)', '<BRIDGE>', lyrics, flags=re.IGNORECASE)
#     lyrics = re.sub(r'\[pre.?chorus[^\]]*\]|\(pre.?chorus[^\)]*\)', '<PRE-CHORUS>', lyrics, flags=re.IGNORECASE)
#     lyrics = re.sub(r'\[refrain[^\]]*\]|\(refrain[^\)]*\)', '<REFRAIN>', lyrics, flags=re.IGNORECASE)
#     lyrics = re.sub(r'\[intro[^\]]*\]|\(intro[^\)]*\)', '<INTRO>', lyrics, flags=re.IGNORECASE)
#     lyrics = re.sub(r'\[outro[^\]]*\]|\(outro[^\)]*\)', '<OUTRO>', lyrics, flags=re.IGNORECASE)
#     return lyrics

# # Apply structure processing
# df["processed_lyrics"] = df["lyrics"].apply(process_lyrics_with_structure)
# df["tokenized"] = df["processed_lyrics"].apply(encode_text)

# # ======= TOPIC MODELING SETUP =======
# # Prepare topic modeling
# vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
# lda = LatentDirichletAllocation(n_components=10, random_state=42)

# # Fit LDA model
# lyrics_texts = df["processed_lyrics"].tolist()
# X = vectorizer.fit_transform(lyrics_texts)
# lda.fit(X)

# def get_topic_distribution(text):
#     """Get topic distribution for a given text"""
#     vec = vectorizer.transform([text])
#     return lda.transform(vec)[0]


# # ======= ENHANCED MODEL ARCHITECTURE =======
# class AdvancedSongWriter(nn.Module):
#     def __init__(self, vocab_size, num_emotions, embed_dim=256, hidden_dim=512, 
#                  num_layers=3, dropout=0.3, num_heads=4):
#         super(AdvancedSongWriter, self).__init__()
        
#         # Initialize with pre-trained embeddings
#         self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word_to_index["<PAD>"])
        
#         # Enhanced embeddings
#         self.emotion_embedding = nn.Embedding(num_emotions, embed_dim)
#         self.position_embedding = nn.Embedding(3, embed_dim)  # line position
#         self.length_embedding = nn.Embedding(20, embed_dim)   # line length
#         self.meter_embedding = nn.Embedding(5, embed_dim)     # meter type
        
#         # Embedding projection
#         self.embed_proj = nn.Linear(embed_dim * 5, embed_dim)
        
#         # Transformer-based architecture
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=embed_dim,
#                 nhead=num_heads,
#                 dim_feedforward=hidden_dim,
#                 dropout=dropout
#             ),
#             num_layers=num_layers
#         )
        
#         # Attention heads
#         self.topic_attention = nn.MultiheadAttention(embed_dim, num_heads)
#         self.rhyme_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
#         # Output layers
#         self.dropout = nn.Dropout(dropout)
#         self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
#         # Auxiliary heads
#         self.rhyme_head = nn.Linear(embed_dim, 8)  # predict rhyme group
#         self.topic_head = nn.Linear(embed_dim, 10) # predict topic
        
#         # Layer normalization
#         self.ln1 = nn.LayerNorm(embed_dim)
#         self.ln2 = nn.LayerNorm(hidden_dim)

#     def forward(self, x, emotion, position=None, length=None, meter=None):
#         batch_size, seq_len = x.size()
        
#         # Word embeddings
#         word_emb = self.word_embedding(x)
        
#         # Emotion embedding
#         emotion_emb = self.emotion_embedding(emotion).unsqueeze(1).expand(-1, seq_len, -1)
        
#         # Position embeddings
#         if position is None:
#             position = torch.ones(batch_size, dtype=torch.long, device=x.device)
#         pos_emb = self.position_embedding(position).unsqueeze(1).expand(-1, seq_len, -1)
        
#         # Line length embeddings
#         if length is None:
#             length = torch.zeros(batch_size, dtype=torch.long, device=x.device)
#         len_emb = self.length_embedding(torch.clamp(length, 0, 19)).unsqueeze(1).expand(-1, seq_len, -1)
        
#         # Meter embeddings
#         if meter is None:
#             meter = torch.zeros(batch_size, dtype=torch.long, device=x.device)
#         meter_emb = self.meter_embedding(meter).unsqueeze(1).expand(-1, seq_len, -1)
        
#         # Combine all embeddings
#         combined = torch.cat([word_emb, emotion_emb, pos_emb, len_emb, meter_emb], dim=-1)
#         combined = self.embed_proj(combined)
        
#         # Transformer processing
#         transformer_out = self.transformer(combined)
        
#         # Topic-aware attention
#         topic_attn, _ = self.topic_attention(
#             transformer_out, transformer_out, transformer_out
#         )
        
#         # Rhyme-aware attention
#         rhyme_attn, _ = self.rhyme_attention(
#             transformer_out, transformer_out, transformer_out
#         )
        
#         # Combine features
#         features = torch.cat([transformer_out, topic_attn], dim=-1)
        
#         # Word prediction
#         out = self.dropout(features[:, -1, :])
#         out = self.fc1(out)
#         out = self.ln2(out)
#         out = torch.relu(out)
#         word_logits = self.fc2(out)
        
#         # Auxiliary predictions
#         rhyme_logits = self.rhyme_head(transformer_out[:, -1, :])
#         topic_logits = self.topic_head(transformer_out[:, -1, :])
        
#         return word_logits, rhyme_logits, topic_logits

# # Initialize Enhanced Model
# TOTAL_VOCAB_SIZE = len(word_to_index)
# EMBED_DIM = 256
# HIDDEN_DIM = 512
# NUM_LAYERS = 3

# print(f"Vocab size: {TOTAL_VOCAB_SIZE}")
# model = AdvancedSongWriter(
#     TOTAL_VOCAB_SIZE, 
#     NUM_EMOTIONS,
#     embed_dim=EMBED_DIM,
#     hidden_dim=HIDDEN_DIM,
#     num_layers=NUM_LAYERS
# ).to(device)

# # Loss functions and optimizer
# word_criterion = nn.CrossEntropyLoss()
# rhyme_criterion = nn.CrossEntropyLoss()
# topic_criterion = nn.KLDivLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer, 
#     max_lr=0.003, 
#     epochs=25, 
#     steps_per_epoch=len(train_loader)
# )

# # ======= ENHANCED GENERATION FUNCTIONS =======
# class SongGenerator:
#     def __init__(self, model, pretrained_model):
#         self.model = model
#         self.pretrained_model = pretrained_model
#         self.chorus_cache = {}  # For storing and modifying choruses
        
#     def count_syllables(self, word):
#         """Enhanced syllable counting with better fallback"""
#         word = re.sub(r'[^\w\s]', '', word.lower())
#         if not word:
#             return 0
#         if word in pronounce_dict:
#             phones = pronounce_dict[word][0]
#             return len([p for p in phones if any(c.isdigit() for c in p)])
#         try:
#             return syllables.estimate(word)
#         except:
#             vowels = "aeiouy"
#             count = 0
#             word = word.lower()
#             if len(word) == 0:
#                 return 0
#             if word[0] in vowels:
#                 count += 1
#             for i in range(1, len(word)):
#                 if word[i] in vowels and word[i-1] not in vowels:
#                     count += 1
#             if word.endswith("e"):
#                 count -= 1
#             if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
#                 count += 1
#             return max(1, count)
    
#     def words_rhyme(self, word1, word2):
#         """Enhanced rhyme detection with better fallback"""
#         word1 = re.sub(r'[^\w\s]', '', word1.lower())
#         word2 = re.sub(r'[^\w\s]', '', word2.lower())
        
#         if not word1 or not word2:
#             return False
#         if word1 == word2:
#             return True
        
#         if word1 in pronounce_dict and word2 in pronounce_dict:
#             pron1 = pronounce_dict[word1][0]
#             pron2 = pronounce_dict[word2][0]
            
#             def get_rhyme_sounds(pron):
#                 vowel_positions = []
#                 for i, sound in enumerate(pron):
#                     if any(c.isdigit() for c in sound):
#                         vowel_positions.append(i)
#                 if not vowel_positions:
#                     return pron[-3:] if len(pron) >= 3 else pron
#                 last_vowel = vowel_positions[-1]
#                 return pron[last_vowel:]
            
#             return get_rhyme_sounds(pron1) == get_rhyme_sounds(pron2)
        
#         if len(word1) > 2 and len(word2) > 2:
#             min_len = min(len(word1), len(word2))
#             compare_len = 3 if min_len <= 4 else 4
#             return word1[-compare_len:] == word2[-compare_len:]
#         return False
    
#     def get_theme_words(self, emotion, n=30):
#         """Get words related to specific emotion with enhanced vocabulary"""
#         emotion = str(emotion).lower()
#         theme_words = {
#             "happy": ["joy", "sunshine", "dancing", "laughter", "smiles"],
#             "sad": ["tears", "lonely", "heartbreak", "pain", "empty"],
#             "angry": ["rage", "fury", "storm", "burn", "scream"],
#             "love": ["heart", "passion", "desire", "romance", "kiss"],
#             "calm": ["peace", "serene", "tranquil", "still", "quiet"]
#         }
#         if emotion not in theme_words:
#             emotion = "happy"
#         poetic_words = ["time", "dream", "sky", "stars", "night"]
#         combined = list(set(theme_words.get(emotion, []) + poetic_words))
#         return random.sample(combined, min(n, len(combined)))
    
#     def get_rhyming_words(self, word, n=10):
#         """Find words in vocabulary that rhyme with given word"""
#         rhyming_words = []
#         for candidate in index_to_word.values():
#             if self.words_rhyme(word, candidate):
#                 rhyming_words.append(candidate)
#                 if len(rhyming_words) >= n:
#                     break
#         return rhyming_words
    
#     def analyze_meter(self, text):
#         """Analyze meter using scansion with fallback"""
#         words = word_tokenize(text)
#         syllable_counts = [self.count_syllables(word) for word in words]
#         return "iambic" if len(syllable_counts) > 0 else None, None
    
#     def calculate_musicality_score(self, lyrics, emotion, topic_dist):
#         """Calculate a comprehensive musicality score"""
#         # Rhyme score
#         lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
#         rhyme_score = 0
#         if len(lines) >= 2:
#             end_words = [line.split()[-1] for line in lines if line.split()]
#             for i in range(len(end_words)-1):
#                 if self.words_rhyme(end_words[i], end_words[i+1]):
#                     rhyme_score += 1
#             rhyme_score /= (len(lines) - 1)
        
#         # Rhythm consistency score
#         rhythm_score = 0
#         syllable_counts = []
#         for line in lines:
#             words = word_tokenize(line)
#             syllable_counts.append(sum(self.count_syllables(word) for word in words))
#         if len(syllable_counts) > 1:
#             rhythm_score = 1 - (np.std(syllable_counts) / max(1, np.mean(syllable_counts)))
        
#         # Topic consistency score
#         current_topics = get_topic_distribution(" ".join(lines))
#         topic_score = np.dot(current_topics, topic_dist)
        
#         # Emotion score (presence of theme words)
#         theme_words = self.get_theme_words(emotion)
#         emotion_score = sum(1 for word in word_tokenize(lyrics) if word.lower() in theme_words) / max(1, len(word_tokenize(lyrics)))
        
#         # Combined score
#         return {
#             'total': 0.4 * rhyme_score + 0.3 * rhythm_score + 0.2 * topic_score + 0.1 * emotion_score,
#             'rhyme': rhyme_score,
#             'rhythm': rhythm_score,
#             'topic': topic_score,
#             'emotion': emotion_score
#         }
    
#     def generate_chorus_variation(self, original_chorus, emotion):
#         """Generate a variation of the chorus while preserving structure"""
#         lines = [line.strip() for line in original_chorus.split('\n') if line.strip()]
#         if not lines:
#             return original_chorus
        
#         # Find rhyming pairs
#         end_words = [line.split()[-1] for line in lines if line.split()]
#         rhyme_groups = {}
#         for i, word in enumerate(end_words):
#             found = False
#             for group in rhyme_groups.values():
#                 if any(self.words_rhyme(word, w) for w in group):
#                     group.append(word)
#                     found = True
#                     break
#             if not found:
#                 rhyme_groups[len(rhyme_groups)] = [word]
        
#         # Generate variations
#         varied_lines = []
#         for line in lines:
#             words = line.split()
#             if len(words) < 2:
#                 varied_lines.append(line)
#                 continue
            
#             # Find which rhyme group the end word belongs to
#             end_word = words[-1]
#             current_group = None
#             for group in rhyme_groups.values():
#                 if any(self.words_rhyme(end_word, w) for w in group):
#                     current_group = group
#                     break
            
#             # Replace 1-2 non-keywords while preserving rhyme
#             if current_group and len(words) > 3:
#                 new_line = words.copy()
#                 replace_pos = random.choice(range(len(words)-2))  # Don't replace last word
#                 theme_words = self.get_theme_words(emotion)
                
#                 # Get candidate replacements
#                 candidates = []
#                 for word in index_to_word.values():
#                     if (word not in theme_words and 
#                         word != new_line[replace_pos] and 
#                         self.count_syllables(word) == self.count_syllables(new_line[replace_pos])):
#                         candidates.append(word)
                
#                 if candidates:
#                     new_word = random.choice(candidates)
#                     new_line[replace_pos] = new_word
#                     varied_lines.append(" ".join(new_line))
#                 else:
#                     varied_lines.append(line)
#             else:
#                 varied_lines.append(line)
        
#         return "\n".join(varied_lines)
    
#     def generate_song(self, emotion, song_length="medium", temperature=0.7, 
#                      top_k=50, rhyme_strength=0.5, num_candidates=3):
#         """Generate a high-quality song with multiple candidate selection"""
#         emotion_idx = emotion_to_idx.get(emotion.lower(), 0)
#         topic_dist = get_topic_distribution(emotion)
        
#         # Generate multiple candidates and select the best
#         best_song = None
#         best_score = -1
#         scores = []
        
#         for _ in range(num_candidates):
#             # Determine song structure
#             if song_length == "short":
#                 structure = ["verse", "chorus"]
#             elif song_length == "medium":
#                 structure = ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]
#             else:  # long
#                 structure = ["intro", "verse", "pre-chorus", "chorus", 
#                             "verse", "pre-chorus", "chorus", 
#                             "bridge", "chorus", "outro"]
            
#             generated_lyrics = []
#             current_part = structure[0]
#             current_line = []
#             current_syllables = 0
#             end_words = []
#             chorus_content = None
            
#             # Target syllables per line
#             target_syllables = {
#                 "verse": random.choice([8, 10]),
#                 "chorus": 8,
#                 "pre-chorus": 6,
#                 "bridge": random.choice([6, 8]),
#                 "intro": 4,
#                 "outro": 4
#             }
            
#             # Generate each part
#             for part in structure:
#                 generated_lyrics.append(f"\n[{part.upper()}]\n")
#                 lines_in_part = 0
#                 max_lines = {
#                     "verse": 4,
#                     "chorus": 4,
#                     "pre-chorus": 2,
#                     "bridge": 2,
#                     "intro": 1,
#                     "outro": 1
#                 }.get(part, 4)
                
#                 # Generate lines for this part
#                 while lines_in_part < max_lines:
#                     # Use pre-trained model for initial generation if available
#                     if not current_line and self.pretrained_model:
#                         prompt = f"Write a {emotion} song {part} about "
#                         generated = self.pretrained_model.generate(
#                             prompt, 
#                             max_length=20,
#                             temperature=temperature
#                         )
#                         current_line = word_tokenize(generated[len(prompt):])[:5]
#                         current_syllables = sum(self.count_syllables(word) for word in current_line)
                    
#                     # Prepare input sequence
#                     input_seq = [word_to_index.get(word, word_to_index["<UNK>"]) 
#                                for word in current_line[-MAX_SEQ_LEN:]]
#                     if len(input_seq) < MAX_SEQ_LEN:
#                         input_seq = [word_to_index["<PAD>"]] * (MAX_SEQ_LEN - len(input_seq)) + input_seq
                    
#                     # Determine position and meter
#                     position = 0 if not current_line else (2 if current_syllables >= target_syllables.get(part, 8) else 1)
#                     meter_type = 0  # Default to iambic
                    
#                     # Convert to tensors
#                     tokens_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
#                     emotion_tensor = torch.tensor([emotion_idx], dtype=torch.long).to(device)
#                     position_tensor = torch.tensor([position], dtype=torch.long).to(device)
#                     length_tensor = torch.tensor([current_syllables], dtype=torch.long).to(device)
#                     meter_tensor = torch.tensor([meter_type], dtype=torch.long).to(device)
                    
#                     # Get prediction
#                     with torch.no_grad():
#                         word_logits, rhyme_logits, topic_logits = self.model(
#                             tokens_tensor, 
#                             emotion_tensor, 
#                             position_tensor, 
#                             length_tensor,
#                             meter_tensor
#                         )
                        
#                         # Apply temperature and top-k
#                         word_logits = word_logits / temperature
#                         top_k_indices = torch.topk(word_logits, top_k).indices
#                         filtered_logits = torch.full_like(word_logits, float('-inf'))
#                         filtered_logits.scatter_(1, top_k_indices, word_logits.gather(1, top_k_indices))
                        
#                         # Apply rhyme boosting
#                         if position == 2 and len(end_words) > 0 and random.random() < rhyme_strength:
#                             rhyme_word = end_words[-1]
#                             rhyming_words = self.get_rhyming_words(rhyme_word)
#                             for idx in range(len(index_to_word)):
#                                 word = index_to_word.get(idx, "")
#                                 if word in rhyming_words:
#                                     filtered_logits[0, idx] += 3.0
                        
#                         # Apply topic boosting
#                         for idx in range(len(index_to_word)):
#                             word = index_to_word.get(idx, "")
#                             if word in self.get_theme_words(emotion):
#                                 filtered_logits[0, idx] += 2.0
                        
#                         # Sample next word
#                         probs = torch.softmax(filtered_logits, dim=1).squeeze()
#                         predicted_idx = torch.multinomial(probs, 1).item()
#                         predicted_word = index_to_word.get(predicted_idx, "<UNK>")
                    
#                     # Skip special tokens
#                     if predicted_word in ["<PAD>", "<UNK>", "<EOS>"]:
#                         continue
                    
#                     # Handle new line
#                     if predicted_word == "<NEWLINE>" or (position == 2 and len(current_line) >= 3):
#                         if current_line:
#                             # Capitalize first word
#                             current_line[0] = current_line[0].capitalize()
                            
#                             # Add punctuation
#                             if random.random() < 0.3:
#                                 punct = random.choice([",", ".", "!", "?"])
#                                 current_line[-1] += punct
                            
#                             # Add line to lyrics
#                             line_text = " ".join(current_line)
                            
#                             # Handle chorus repetition with variation
#                             if part == "chorus":
#                                 if chorus_content is None:
#                                     chorus_content = line_text
#                                 elif random.random() < 0.3:  # 30% chance of variation
#                                     line_text = self.generate_chorus_variation(line_text, emotion)
                            
#                             generated_lyrics.append(line_text + "\n")
#                             lines_in_part += 1
                            
#                             # Record end word for rhyming
#                             end_words.append(current_line[-1].strip(",.!?"))
                            
#                             # Reset line
#                             current_line = []
#                             current_syllables = 0
#                         continue
                    
#                     # Add word to current line
#                     current_line.append(predicted_word)
#                     current_syllables += self.count_syllables(predicted_word)
            
#             # Post-process generated lyrics
#             full_lyrics = "".join(generated_lyrics)
#             full_lyrics = re.sub(r"\n\s+\n", "\n\n", full_lyrics)
#             full_lyrics = re.sub(r" , ", ", ", full_lyrics)
#             full_lyrics = re.sub(r" \. ", ". ", full_lyrics)
#             full_lyrics = re.sub(r" \! ", "! ", full_lyrics)
#             full_lyrics = re.sub(r" \? ", "? ", full_lyrics)
#             full_lyrics = re.sub(r" ' ", "'", full_lyrics)
            
#             # Calculate musicality score
#             score = self.calculate_musicality_score(full_lyrics, emotion, topic_dist)
#             scores.append(score)
            
#             # Keep track of best song
#             if score['total'] > best_score:
#                 best_score = score['total']
#                 best_song = full_lyrics.strip()
        
#         return best_song, scores

# # ======= INTERACTIVE INTERFACE =======
# def interactive_song_generator():
#     """Enhanced interactive interface for song generation"""
#     # Load models
#     base_model = AdvancedSongWriter(
#         TOTAL_VOCAB_SIZE, 
#         NUM_EMOTIONS,
#         embed_dim=EMBED_DIM,
#         hidden_dim=HIDDEN_DIM,
#         num_layers=NUM_LAYERS
#     ).to(device)
    
#     # Try to load trained weights
#     try:
#         checkpoint = torch.load("song_writer_model.pth", map_location=device)
#         base_model.load_state_dict(checkpoint['model_state_dict'])
#         base_model.eval()
#     except:
#         print("Warning: Could not load trained weights. Using untrained model.")
    
#     generator = SongGenerator(base_model, pretrained_model)
    
#     print("\n🎵 Advanced AI Song Writer 🎵")
#     print(f"Available emotions: {list(emotion_to_idx.keys())}")
    
#     while True:
#         try:
#             print("\nOptions:")
#             print("1. Quick song generation")
#             print("2. Custom song creation")
#             print("3. Exit")
#             choice = input("Enter your choice (1-3): ").strip()
            
#             if choice == "3":
#                 break
                
#             if choice == "1":
#                 emotion = input("Enter emotion: ").lower().strip()
#                 if emotion not in emotion_to_idx:
#                     print(f"Unknown emotion. Available: {list(emotion_to_idx.keys())}")
#                     continue
                    
#                 length = input("Song length (short/medium/long): ").lower().strip()
#                 if length not in ["short", "medium", "long"]:
#                     length = "medium"
                    
#                 print("\nGenerating your song...\n")
#                 song, _ = generator.generate_song(emotion, song_length=length)
#                 print("\n" + "="*50)
#                 print(song)
#                 print("="*50 + "\n")
                
#             elif choice == "2":
#                 emotion = input("Enter emotion: ").lower().strip()
#                 if emotion not in emotion_to_idx:
#                     print(f"Unknown emotion. Available: {list(emotion_to_idx.keys())}")
#                     continue
                    
#                 length = input("Song length (short/medium/long): ").lower().strip()
#                 if length not in ["short", "medium", "long"]:
#                     length = "medium"
                    
#                 temp = input("Creativity (0.1-1.0, default 0.7): ").strip()
#                 try:
#                     temp = float(temp)
#                     temp = max(0.1, min(1.0, temp))
#                 except:
#                     temp = 0.7
                    
#                 rhyme = input("Rhyme strength (0.0-1.0, default 0.5): ").strip()
#                 try:
#                     rhyme = float(rhyme)
#                     rhyme = max(0.0, min(1.0, rhyme))
#                 except:
#                     rhyme = 0.5
                    
#                 candidates = input("Number of candidates (1-5, default 3): ").strip()
#                 try:
#                     candidates = int(candidates)
#                     candidates = max(1, min(5, candidates))
#                 except:
#                     candidates = 3
                    
#                 print("\nGenerating your custom song...\n")
#                 song, scores = generator.generate_song(
#                     emotion, 
#                     song_length=length,
#                     temperature=temp,
#                     rhyme_strength=rhyme,
#                     num_candidates=candidates
#                 )
                
#                 print("\n" + "="*50)
#                 print("Generated Song:")
#                 print("="*50)
#                 print(song)
#                 print("="*50)
#                 print("\nQuality Scores:")
#                 print(f"- Rhyme: {scores[0]['rhyme']:.2f}")
#                 print(f"- Rhythm: {scores[0]['rhythm']:.2f}")
#                 print(f"- Topic: {scores[0]['topic']:.2f}")
#                 print(f"- Emotion: {scores[0]['emotion']:.2f}")
#                 print(f"- Overall: {scores[0]['total']:.2f}")
#                 print("="*50 + "\n")
                
#         except Exception as e:
#             print(f"Error: {e}")
#             print("Please try again.")
            
#     print("\nThank you for using Advanced AI Song Writer!")

# if __name__ == "__main__":
#     interactive_song_generator()