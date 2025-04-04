# "===================test1================================================="

# from transformers import pipeline, AutoProcessor, AutoModel
# import scipy.io.wavfile
# import numpy as np
# import torch
# import soundfile as sf

# # =============================
# # 1️⃣ Pipeline-based TTS
# # =============================
# try:
#     print("Initializing text-to-speech pipeline...")
#     synthesiser = pipeline("text-to-speech", model="suno/bark-small")

#     # Generate speech
#     print("Generating speech with pipeline...")
#     speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})

#     # Fix 1: Check the structure of the speech output
#     if isinstance(speech, dict) and "audio" in speech and "sampling_rate" in speech:
#         audio_data = speech["audio"]
#         sampling_rate = speech["sampling_rate"]
#     else:
#         # Handle unexpected output structure
#         if isinstance(speech, dict):
#             print(f"Unexpected speech output structure. Keys: {list(speech.keys())}")
#         else:
#             print(f"Unexpected speech output type: {type(speech)}")
#         audio_data = speech if isinstance(speech, np.ndarray) else np.array([])
#         sampling_rate = 24000  # Bark's default sampling rate
    
#     # Fix 2: Ensure valid sampling rate (must be an integer within range)
#     sampling_rate = int(sampling_rate)
#     if not (8000 <= sampling_rate <= 48000):
#         print(f"Invalid sampling rate: {sampling_rate}, using default 24000")
#         sampling_rate = 24000

#     # Fix 3: Use soundfile instead of scipy for more robust saving
#     if len(audio_data) > 0:
#         # Normalize to float in [-1, 1] range for soundfile
#         audio_data = np.array(audio_data, dtype=np.float32)
#         audio_data = np.clip(audio_data, -1.0, 1.0)
        
#         print(f"Saving pipeline output with sampling rate {sampling_rate}Hz...")
#         sf.write("bark_out_pipeline.wav", audio_data, sampling_rate)
#         print("Pipeline audio saved successfully!")
#     else:
#         print("No audio data generated from pipeline")

# except Exception as e:
#     print(f"Error in pipeline-based TTS: {e}")

# # =============================
# # 2️⃣ AutoModel-based TTS
# # =============================
# try:
#     print("\nInitializing AutoModel...")
#     processor = AutoProcessor.from_pretrained("suno/bark-small")
#     model = AutoModel.from_pretrained("suno/bark-small")

#     # Prepare input
#     print("Processing text input...")
#     inputs = processor(
#         text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
#         return_tensors="pt",
#     )

#     # Generate speech values
#     print("Generating speech with model...")
#     with torch.no_grad():
#         speech_values = model.generate(**inputs, do_sample=True)

#     # Fix 4: Get sampling rate from model config safely
#     sampling_rate_model = getattr(model.config, "sample_rate", 24000)
#     sampling_rate_model = int(sampling_rate_model)  # Ensure it's an integer
    
#     # Fix 5: Check valid range
#     if not (8000 <= sampling_rate_model <= 48000):
#         print(f"Invalid sampling rate from model: {sampling_rate_model}, using default 24000")
#         sampling_rate_model = 24000

#     # Fix 6: Convert generated speech to float32 for soundfile
#     print(f"Processing audio data...")
#     audio_data_model = speech_values.cpu().numpy().squeeze()
    
#     # Ensure it's the right shape
#     if len(audio_data_model.shape) > 1:
#         print(f"Reshaping audio from {audio_data_model.shape}...")
#         audio_data_model = audio_data_model.mean(axis=0) if audio_data_model.shape[0] <= 2 else audio_data_model.squeeze(0)
    
#     # Normalize to [-1, 1]
#     audio_data_model = np.array(audio_data_model, dtype=np.float32)
#     audio_data_model = audio_data_model / np.max(np.abs(audio_data_model)) if np.max(np.abs(audio_data_model)) > 0 else audio_data_model
    
#     # Save using soundfile
#     print(f"Saving model output with sampling rate {sampling_rate_model}Hz...")
#     sf.write("bark_out_model.wav", audio_data_model, sampling_rate_model)
#     print("Model audio saved successfully!")

# except Exception as e:
#     print(f"Error in AutoModel-based TTS: {e}")

# print("\n✅ Script execution completed!")








"====================================================================TEST 2======================================================================================"
# from transformers import pipeline, AutoProcessor, AutoModel
# import numpy as np
# import torch
# import os
# import wave
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def save_audio_robust(audio_data, filename, sample_rate):
#     """
#     A robust function to save audio data to a WAV file using multiple methods
#     """
#     logger.info(f"Attempting to save audio to {filename} with sample rate {sample_rate}Hz")
    
#     # Make sure audio_data is a numpy array and normalize to [-1, 1]
#     if isinstance(audio_data, torch.Tensor):
#         audio_data = audio_data.cpu().numpy()
    
#     audio_data = np.array(audio_data, dtype=np.float32)
    
#     # Check for NaN or Inf values
#     if np.isnan(audio_data).any() or np.isinf(audio_data).any():
#         logger.warning("Audio contains NaN or Inf values. Replacing with zeros.")
#         audio_data = np.nan_to_num(audio_data)
    
#     # Normalize to [-1, 1]
#     max_val = np.max(np.abs(audio_data))
#     if max_val > 0:
#         audio_data = audio_data / max_val
    
#     # Method 1: Try using soundfile (if available)
#     try:
#         import soundfile as sf
#         logger.info("Saving using soundfile...")
#         sf.write(filename, audio_data, sample_rate)
#         if os.path.exists(filename) and os.path.getsize(filename) > 0:
#             logger.info("✓ Audio saved successfully using soundfile!")
#             return True
#     except Exception as e:
#         logger.warning(f"soundfile save failed: {e}")
    
#     # Method 2: Try using scipy.io.wavfile
#     try:
#         import scipy.io.wavfile
#         logger.info("Saving using scipy.io.wavfile...")
#         # Convert to int16 for scipy
#         audio_data_int16 = (audio_data * 32767).astype(np.int16)
#         scipy.io.wavfile.write(filename, sample_rate, audio_data_int16)
#         if os.path.exists(filename) and os.path.getsize(filename) > 0:
#             logger.info("✓ Audio saved successfully using scipy.io.wavfile!")
#             return True
#     except Exception as e:
#         logger.warning(f"scipy.io.wavfile save failed: {e}")
    
#     # Method 3: Try manual WAV writing with wave module
#     try:
#         logger.info("Saving using wave module...")
#         # Scale to int16 range
#         audio_data_int16 = (audio_data * 32767).astype(np.int16)
        
#         with wave.open(filename, 'wb') as wf:
#             wf.setnchannels(1)  # Mono
#             wf.setsampwidth(2)  # 2 bytes for int16
#             wf.setframerate(sample_rate)
#             wf.writeframes(audio_data_int16.tobytes())
        
#         if os.path.exists(filename) and os.path.getsize(filename) > 0:
#             logger.info("✓ Audio saved successfully using wave module!")
#             return True
#     except Exception as e:
#         logger.warning(f"wave module save failed: {e}")
    
#     logger.error("Failed to save audio using all available methods")
#     return False

# # =============================
# # 1️⃣ Pipeline-based TTS
# # =============================
# try:
#     logger.info("Initializing text-to-speech pipeline...")
#     synthesiser = pipeline("text-to-speech", model="suno/bark-small")

#     # Generate speech
#     logger.info("Generating speech with pipeline...")
#     speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})

#     # Fix 1: Check the structure of the speech output
#     logger.info(f"Speech output type: {type(speech)}")
#     if isinstance(speech, dict):
#         logger.info(f"Speech dictionary keys: {list(speech.keys())}")
    
#     if isinstance(speech, dict) and "audio" in speech and "sampling_rate" in speech:
#         audio_data = speech["audio"]
#         sampling_rate = speech["sampling_rate"]
#     else:
#         # Handle unexpected output structure
#         if isinstance(speech, dict):
#             logger.warning(f"Unexpected speech output structure. Keys: {list(speech.keys())}")
#             if "audio_arrays" in speech:
#                 audio_data = speech["audio_arrays"][0]
#             elif len(speech) > 0:
#                 # Try to find any numpy array or list in the dictionary values
#                 for key, value in speech.items():
#                     if isinstance(value, (np.ndarray, list)):
#                         audio_data = value
#                         logger.info(f"Found potential audio data in key: {key}")
#                         break
#                 else:
#                     audio_data = np.array([])
#             else:
#                 audio_data = np.array([])
#         elif isinstance(speech, (np.ndarray, list)):
#             audio_data = speech
#         else:
#             logger.warning(f"Unexpected speech output type: {type(speech)}")
#             audio_data = np.array([])
        
#         sampling_rate = 24000  # Bark's default sampling rate
    
#     # Fix 2: Ensure valid sampling rate (must be an integer within range)
#     sampling_rate = int(sampling_rate)
#     if not (8000 <= sampling_rate <= 48000):
#         logger.warning(f"Invalid sampling rate: {sampling_rate}, using default 24000")
#         sampling_rate = 24000

#     # Fix 3: Use our robust audio saving function
#     if len(audio_data) > 0:
#         logger.info(f"Audio data shape: {np.array(audio_data).shape}")
#         success = save_audio_robust(audio_data, "bark_out_pipeline.wav", sampling_rate)
#         if success:
#             logger.info("Pipeline audio saved successfully!")
#         else:
#             logger.error("Failed to save pipeline audio")
#     else:
#         logger.warning("No audio data generated from pipeline")

# except Exception as e:
#     logger.error(f"Error in pipeline-based TTS: {e}", exc_info=True)

# # =============================
# # 2️⃣ AutoModel-based TTS
# # =============================
# try:
#     logger.info("\nInitializing AutoModel...")
#     processor = AutoProcessor.from_pretrained("suno/bark-small")
#     model = AutoModel.from_pretrained("suno/bark-small")

#     # Prepare input
#     logger.info("Processing text input...")
#     inputs = processor(
#         text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
#         return_tensors="pt",
#     )

#     # Generate speech values
#     logger.info("Generating speech with model...")
#     with torch.no_grad():
#         speech_values = model.generate(**inputs, do_sample=True)

#     logger.info(f"Generated speech values shape: {speech_values.shape}")

#     # Fix 4: Get sampling rate from model config safely
#     sampling_rate_model = getattr(model.config, "sample_rate", 24000)
#     sampling_rate_model = int(sampling_rate_model)  # Ensure it's an integer
    
#     # Fix 5: Check valid range
#     if not (8000 <= sampling_rate_model <= 48000):
#         logger.warning(f"Invalid sampling rate from model: {sampling_rate_model}, using default 24000")
#         sampling_rate_model = 24000

#     # Fix 6: Process audio data
#     logger.info(f"Processing audio data...")
#     audio_data_model = speech_values.cpu().numpy()
    
#     # Ensure it's the right shape for audio (1D array)
#     if len(audio_data_model.shape) > 1:
#         logger.info(f"Reshaping audio from {audio_data_model.shape}...")
#         if audio_data_model.shape[0] == 1:
#             # Single sample, just squeeze
#             audio_data_model = audio_data_model.squeeze(0)
#         elif audio_data_model.shape[0] <= 2:
#             # Could be stereo, convert to mono
#             audio_data_model = audio_data_model.mean(axis=0)
#         else:
#             # Multiple samples, take the first one
#             audio_data_model = audio_data_model[0]
    
#     # Save using our robust function
#     success = save_audio_robust(audio_data_model, "bark_out_model.wav", sampling_rate_model)
#     if success:
#         logger.info("Model audio saved successfully!")
#     else:
#         logger.error("Failed to save model audio")

# except Exception as e:
#     logger.error(f"Error in AutoModel-based TTS: {e}", exc_info=True)

# logger.info("\n✅ Script execution completed!")

































"=================================================================================TEST3====================================================="



from transformers import pipeline, AutoProcessor, AutoModel
import numpy as np
import torch
import os
import wave
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# def save_audio_robust(audio_data, filename, sample_rate):
#     """
#     A robust function to save audio data to a WAV file using multiple methods
#     """
#     logger.info(f"Attempting to save audio to {filename} with sample rate {sample_rate}Hz")
    
#     # Make sure audio_data is a numpy array and normalize to [-1, 1]
#     if isinstance(audio_data, torch.Tensor):
#         audio_data = audio_data.cpu().numpy()
    
#     audio_data = np.array(audio_data, dtype=np.float32)
    
#     # Check for NaN or Inf values
#     if np.isnan(audio_data).any() or np.isinf(audio_data).any():
#         logger.warning("Audio contains NaN or Inf values. Replacing with zeros.")
#         audio_data = np.nan_to_num(audio_data)
    
#     # Normalize to [-1, 1]
#     max_val = np.max(np.abs(audio_data))
#     if max_val > 0:
#         audio_data = audio_data / max_val
    
#     # Method 1: Try using soundfile (if available)
#     try:
#         import soundfile as sf
#         logger.info("Saving using soundfile...")
#         sf.write(filename, audio_data, sample_rate)
#         if os.path.exists(filename) and os.path.getsize(filename) > 0:
#             logger.info("✓ Audio saved successfully using soundfile!")
#             return True
#     except Exception as e:
#         logger.warning(f"soundfile save failed: {e}")
    
#     # Method 2: Try using scipy.io.wavfile
#     try:
#         import scipy.io.wavfile
#         logger.info("Saving using scipy.io.wavfile...")
#         # Convert to int16 for scipy
#         audio_data_int16 = (audio_data * 32767).astype(np.int16)
#         scipy.io.wavfile.write(filename, sample_rate, audio_data_int16)
#         if os.path.exists(filename) and os.path.getsize(filename) > 0:
#             logger.info("✓ Audio saved successfully using scipy.io.wavfile!")
#             return True
#     except Exception as e:
#         logger.warning(f"scipy.io.wavfile save failed: {e}")
    
#     # Method 3: Try manual WAV writing with wave module
#     try:
#         logger.info("Saving using wave module...")
#         # Scale to int16 range
#         audio_data_int16 = (audio_data * 32767).astype(np.int16)
        
#         with wave.open(filename, 'wb') as wf:
#             wf.setnchannels(1)  # Mono
#             wf.setsampwidth(2)  # 2 bytes for int16
#             wf.setframerate(sample_rate)
#             wf.writeframes(audio_data_int16.tobytes())
        
#         if os.path.exists(filename) and os.path.getsize(filename) > 0:
#             logger.info("✓ Audio saved successfully using wave module!")
#             return True
#     except Exception as e:
#         logger.warning(f"wave module save failed: {e}")
    
#     logger.error("Failed to save audio using all available methods")
#     return False

# # =============================
# # 1️⃣ Pipeline-based TTS with SINGING
# # =============================
# try:
#     logger.info("Initializing text-to-speech pipeline for singing...")
#     synthesiser = pipeline("text-to-speech", model="suno/bark-small")

#     # Specify singing with the special format Bark uses
#     # [MUSIC] tag indicates singing mode, and ♪ symbols can help
#     singing_text = """
#     [MUSIC] ♪ Twinkle twinkle little star, 
#     How I wonder what you are. 
#     Up above the world so high, 
#     Like a diamond in the sky. ♪
#     """

#     # Generate singing
#     logger.info("Generating singing with pipeline...")
#     speech = synthesiser(singing_text, forward_params={"do_sample": True})

#     # Fix 1: Check the structure of the speech output
#     logger.info(f"Speech output type: {type(speech)}")
#     if isinstance(speech, dict):
#         logger.info(f"Speech dictionary keys: {list(speech.keys())}")
    
#     if isinstance(speech, dict) and "audio" in speech and "sampling_rate" in speech:
#         audio_data = speech["audio"]
#         sampling_rate = speech["sampling_rate"]
#     else:
#         # Handle unexpected output structure
#         if isinstance(speech, dict):
#             logger.warning(f"Unexpected speech output structure. Keys: {list(speech.keys())}")
#             if "audio_arrays" in speech:
#                 audio_data = speech["audio_arrays"][0]
#             elif len(speech) > 0:
#                 # Try to find any numpy array or list in the dictionary values
#                 for key, value in speech.items():
#                     if isinstance(value, (np.ndarray, list)):
#                         audio_data = value
#                         logger.info(f"Found potential audio data in key: {key}")
#                         break
#                 else:
#                     audio_data = np.array([])
#             else:
#                 audio_data = np.array([])
#         elif isinstance(speech, (np.ndarray, list)):
#             audio_data = speech
#         else:
#             logger.warning(f"Unexpected speech output type: {type(speech)}")
#             audio_data = np.array([])
        
#         sampling_rate = 24000  # Bark's default sampling rate
    
#     # Fix 2: Ensure valid sampling rate (must be an integer within range)
#     sampling_rate = int(sampling_rate)
#     if not (8000 <= sampling_rate <= 48000):
#         logger.warning(f"Invalid sampling rate: {sampling_rate}, using default 24000")
#         sampling_rate = 24000

#     # Fix 3: Use our robust audio saving function
#     if len(audio_data) > 0:
#         logger.info(f"Audio data shape: {np.array(audio_data).shape}")
#         success = save_audio_robust(audio_data, "bark_singing_pipeline.wav", sampling_rate)
#         if success:
#             logger.info("Pipeline singing audio saved successfully!")
#         else:
#             logger.error("Failed to save pipeline singing audio")
#     else:
#         logger.warning("No audio data generated from pipeline")

# except Exception as e:
#     logger.error(f"Error in pipeline-based singing TTS: {e}", exc_info=True)

# # =============================
# # 2️⃣ AutoModel-based TTS with SINGING
# # =============================
# try:
#     logger.info("\nInitializing AutoModel for singing...")
#     processor = AutoProcessor.from_pretrained("suno/bark-small")
#     model = AutoModel.from_pretrained("suno/bark-small")

#     # Prepare input with singing format
#     logger.info("Processing singing text input...")
#     singing_text = """
#     [MUSIC] ♪ Fly me to the moon,
#     Let me play among the stars,
#     Let me see what spring is like on,
#     Jupiter and Mars. ♪
#     """
    
#     # Process the text input
#     inputs = processor(
#         text=singing_text,
#         return_tensors="pt",
#         voice_preset="v2/en_speaker_6"  # Use a specific voice preset
#     )
    
#     logger.info("Generating singing with AutoModel...")
#     # Generate the audio with the model
#     with torch.no_grad():
#         output = model.generate(**inputs, do_sample=True)
    
#     # Extract the audio data from the model output
#     if hasattr(output, "audio_values") and output.audio_values is not None:
#         audio_data = output.audio_values.squeeze().numpy()
#     elif isinstance(output, tuple) and len(output) > 0:
#         # Try to find audio data in the output tuple
#         for item in output:
#             if isinstance(item, torch.Tensor) and len(item.shape) > 0:
#                 audio_data = item.squeeze().cpu().numpy()
#                 logger.info(f"Found audio data tensor of shape {audio_data.shape}")
#                 break
#         else:
#             logger.warning("Could not find audio data in output tuple")
#             audio_data = np.array([])
#     elif isinstance(output, torch.Tensor):
#         audio_data = output.squeeze().cpu().numpy()
#     else:
#         logger.warning(f"Unexpected output type from model.generate(): {type(output)}")
#         audio_data = np.array([])
    
#     # Get the sampling rate (Bark usually uses 24kHz)
#     sampling_rate = getattr(model.config, "sample_rate", 24000)
    
#     # Save the audio
#     if len(audio_data) > 0:
#         logger.info(f"AutoModel audio data shape: {audio_data.shape}")
#         success = save_audio_robust(audio_data, "bark_singing_automodel.wav", sampling_rate)
#         if success:
#             logger.info("AutoModel singing audio saved successfully!")
#         else:
#             logger.error("Failed to save AutoModel singing audio")
#     else:
#         logger.warning("No audio data generated from AutoModel")

# except Exception as e:
#     logger.error(f"Error in AutoModel-based singing TTS: {e}", exc_info=True)

# # =============================
# # 3️⃣ Simple text-to-speech with different voice
# # =============================
# try:
#     logger.info("\nTrying text-to-speech with a different voice preset...")
    
#     # Initialize the pipeline again with different parameters
#     tts = pipeline("text-to-speech", model="suno/bark-small")
    
#     # Normal text (not singing)
#     normal_text = "Hello, this is a demonstration of text to speech conversion using the Bark model."
    
#     # Try with a different voice preset
#     logger.info("Generating speech with different voice preset...")
#     result = tts(normal_text, forward_params={"do_sample": True}, voice_preset="v2/en_speaker_9")
    
#     # Extract audio data
#     if isinstance(result, dict) and "audio" in result:
#         audio_data = result["audio"]
#         sampling_rate = result.get("sampling_rate", 24000)
#     else:
#         logger.warning(f"Unexpected result structure: {type(result)}")
#         # Attempt extraction similar to before
#         if isinstance(result, dict):
#             for key, value in result.items():
#                 if isinstance(value, (np.ndarray, list)):
#                     audio_data = value
#                     break
#             else:
#                 audio_data = np.array([])
#         elif isinstance(result, (np.ndarray, list)):
#             audio_data = result
#         else:
#             audio_data = np.array([])
        
#         sampling_rate = 24000
    
#     # Save the audio
#     if len(audio_data) > 0:
#         success = save_audio_robust(audio_data, "bark_different_voice.wav", sampling_rate)
#         if success:
#             logger.info("Different voice audio saved successfully!")
#         else:
#             logger.error("Failed to save different voice audio")
#     else:
#         logger.warning("No audio data generated for different voice")

# except Exception as e:
#     logger.error(f"Error in different voice TTS: {e}", exc_info=True)

# logger.info("Text-to-speech script completed.")


















































"=======================================================TEST 4================================================================================"
# -*- coding: utf-8 -*-
# Ensure UTF-8 encoding for special characters like ♪
print("starting......")
import numpy as np
import torch
import os
import wave
import logging
import time # For duration calculation

# Import necessary libraries
try:
    import soundfile as sf
except ImportError:
    logging.warning("soundfile library not found. Saving with soundfile will fail.")
    sf = None

try:
    import scipy.io.wavfile
    import scipy.signal
    scipy_available = True
except ImportError:
    logging.warning("scipy library not found. Saving with scipy.io.wavfile and resampling with scipy will fail.")
    scipy = None
    scipy_available = False

# Attempt to import librosa for better resampling
try:
    import librosa
    librosa_available = True
    logging.info("librosa found. Will use it for resampling if needed.")
except ImportError:
    librosa_available = False
    logging.warning("librosa library not found. Resampling quality may be lower if needed (will fallback to scipy if available). Install with 'pip install librosa'")

# Corrected imports based on usage
from transformers import pipeline, AutoProcessor, BarkModel, AutoModel

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

output_dir = "generated_audio"
os.makedirs(output_dir, exist_ok=True)

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("CUDA (GPU) is available. Using GPU.")
else:
    device = torch.device("cpu")
    logger.info("CUDA (GPU) not available. Using CPU.")

# --- Robust Audio Saving Function (Unchanged from previous working version) ---
def save_audio_robust(audio_data, filename, sample_rate):
    """
    A robust function to save audio data to a WAV file using multiple methods
    """
    logger.info(f"Attempting to save audio to {filename} with sample rate {int(sample_rate)}Hz")
    filepath = os.path.join(output_dir, filename)

    if audio_data is None: logger.error("Cannot save audio: audio_data is None."); return False
    if isinstance(audio_data, torch.Tensor): audio_data = audio_data.detach().cpu().numpy()
    audio_data = np.array(audio_data)
    try: audio_data = audio_data.astype(np.float32)
    except ValueError as e: logger.error(f"Could not convert audio data to float32: {e}"); return False

    # Handle different input shapes (try to get mono)
    original_ndim = audio_data.ndim
    if original_ndim == 3: # Shape (batch, channels, samples) or similar
        logger.warning(f"Input audio has 3 dimensions ({audio_data.shape}), attempting to get mono audio [0, 0, :]")
        try: audio_data = audio_data[0, 0, :]
        except IndexError:
             logger.warning("Failed to extract mono using [0,0,:], trying [0, :, 0]")
             try: audio_data = audio_data[0, :, 0] # Fallback
             except IndexError: logger.error(f"Cannot handle 3D shape {audio_data.shape}"); return False

    elif original_ndim == 2: # Shape (channels, samples) or (samples, channels) or (batch, samples)
        # If first dim is small (likely channels or batch=1), take first row/channel
        if audio_data.shape[0] < 5 and audio_data.shape[0] < audio_data.shape[1] :
             logger.warning(f"Input audio has 2 dimensions ({audio_data.shape}), assuming channels/batch first, taking [0, :]")
             audio_data = audio_data[0, :]
        # If second dim is small (likely channels), take first column
        elif audio_data.shape[1] < 5 and audio_data.shape[1] < audio_data.shape[0]:
             logger.warning(f"Input audio has 2 dimensions ({audio_data.shape}), assuming channels last, taking [:, 0]")
             audio_data = audio_data[:, 0]
        # If shape looks like (1, N)
        elif audio_data.shape[0] == 1:
             logger.warning(f"Input audio has 2 dimensions ({audio_data.shape}), looks like (1, N), taking [0, :]")
             audio_data = audio_data[0, :]
        else: # Otherwise, assume it's stereo and attempt simple mean mixdown
             logger.warning(f"Input audio has 2 dimensions ({audio_data.shape}), attempting simple mean mixdown.")
             audio_data = audio_data.mean(axis=1) # Example: simple mixdown if shape is (samples, channels)

    audio_data = audio_data.flatten() # Ensure 1D at the end
    if audio_data.size == 0: logger.error("Cannot save audio: audio_data is empty."); return False
    if np.isnan(audio_data).any() or np.isinf(audio_data).any():
        logger.warning("Audio contains NaN or Inf values. Replacing with zeros.")
        audio_data = np.nan_to_num(audio_data)

    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        # logger.info(f"Normalizing audio data (max abs value was {max_val:.4f})") # Reduce verbosity
        audio_data = audio_data / max_val
    elif max_val < 1e-6: logger.warning("Audio data might be all zeros.")

    saved = False
    # Method 1: soundfile
    if sf:
        try:
            # logger.info("Saving using soundfile...") # Reduce verbosity
            sf.write(filepath, audio_data, int(sample_rate), format='WAV', subtype='PCM_16')
            if os.path.exists(filepath) and os.path.getsize(filepath) > 44: saved = True
            else: logger.warning(f"soundfile wrote an empty or invalid file to {filepath}")
        except Exception as e: logger.warning(f"soundfile save failed: {e}")
    # else: logger.info("soundfile library not available, skipping.") # Reduce verbosity

    # Method 2: scipy
    if not saved and scipy_available:
        try:
            # logger.info("Saving using scipy.io.wavfile...") # Reduce verbosity
            # Scale *after* potential normalization
            audio_data_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            scipy.io.wavfile.write(filepath, int(sample_rate), audio_data_int16)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 44: saved = True
            else: logger.warning(f"scipy.io.wavfile wrote an empty or invalid file to {filepath}")
        except Exception as e: logger.warning(f"scipy.io.wavfile save failed: {e}")
    # elif not saved: logger.info("scipy library not available or soundfile already succeeded.") # Reduce verbosity

    # Method 3: wave
    if not saved:
        try:
            # logger.info("Saving using wave module...") # Reduce verbosity
            audio_data_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(sample_rate))
                wf.writeframes(audio_data_int16.tobytes())
            if os.path.exists(filepath) and os.path.getsize(filepath) > 44: saved = True
            else: logger.warning(f"wave module wrote an empty or invalid file to {filepath}")
        except Exception as e: logger.warning(f"wave module save failed: {e}")

    if saved: logger.info(f"✓ Audio saved successfully to {filepath}")
    else: logger.error(f"Failed to save audio to {filepath} using all available methods")
    return saved

# --- Audio Combination Function (Unchanged from previous working version) ---
def combine_audio(vocals_path, instrumental_path, output_path, target_sr=None, normalize=True):
    """
    Combines vocal and instrumental audio tracks with basic resampling and normalization.
    """
    logger.info(f"Attempting to combine '{vocals_path}' and '{instrumental_path}' into '{output_path}'")

    vocals_filepath = os.path.join(output_dir, vocals_path)
    instrumental_filepath = os.path.join(output_dir, instrumental_path)
    output_filepath = os.path.join(output_dir, output_path)

    if not os.path.exists(vocals_filepath):
        logger.error(f"Vocal file not found: {vocals_filepath}"); return False
    if not os.path.exists(instrumental_filepath):
        logger.error(f"Instrumental file not found: {instrumental_filepath}"); return False

    try:
        # Load audio files using soundfile if available (handles more formats/types)
        if sf:
            # logger.info("Loading audio using soundfile...") # Reduce verbosity
            vocals, sr_vocals = sf.read(vocals_filepath, dtype='float32', always_2d=False) # Read as 1D if possible
            instrumental, sr_instrumental = sf.read(instrumental_filepath, dtype='float32', always_2d=False)
        elif scipy_available: # Fallback to scipy.io.wavfile
             # logger.info("Loading audio using scipy.io.wavfile...") # Reduce verbosity
             sr_vocals, vocals_int = scipy.io.wavfile.read(vocals_filepath)
             sr_instrumental, instrumental_int = scipy.io.wavfile.read(instrumental_filepath)
             # Convert int16 to float32
             vocals = vocals_int.astype(np.float32) / 32768.0
             instrumental = instrumental_int.astype(np.float32) / 32768.0
        else:
            logger.error("No suitable library (soundfile or scipy) found to load audio.")
            return False

        logger.info(f"Vocals loaded: SR={sr_vocals} Hz, Samples={len(vocals)}, Dim={vocals.ndim}")
        logger.info(f"Instrumental loaded: SR={sr_instrumental} Hz, Samples={len(instrumental)}, Dim={instrumental.ndim}")

        # --- Ensure Mono Early ---
        if vocals.ndim > 1:
            logger.warning(f"Input vocals have {vocals.ndim} dimensions ({vocals.shape}), mixing down to mono.")
            vocals = vocals.mean(axis=1) if vocals.shape[1] < vocals.shape[0] else vocals.mean(axis=0) # Handle (N, ch) or (ch, N)
        if instrumental.ndim > 1:
            logger.warning(f"Input instrumental have {instrumental.ndim} dimensions ({instrumental.shape}), mixing down to mono.")
            instrumental = instrumental.mean(axis=1) if instrumental.shape[1] < instrumental.shape[0] else instrumental.mean(axis=0) # Handle (N, ch) or (ch, N)
        vocals = vocals.flatten()
        instrumental = instrumental.flatten()


        # --- Resampling (if needed) ---
        if target_sr is None:
            target_sr = max(sr_vocals, sr_instrumental) # Default to highest SR
            logger.info(f"Target sample rate not specified, using highest found: {target_sr} Hz")

        if sr_vocals != target_sr:
            logger.warning(f"Resampling vocals from {sr_vocals} Hz to {target_sr} Hz...")
            if librosa_available:
                vocals = librosa.resample(y=vocals, orig_sr=sr_vocals, target_sr=target_sr, res_type='kaiser_best')
            elif scipy_available:
                num_samples = int(len(vocals) * float(target_sr) / sr_vocals)
                vocals = scipy.signal.resample(vocals, num_samples)
            else:
                logger.error("Cannot resample vocals: No librosa or scipy available.")
                return False
            # logger.info(f"Vocals resampled to {len(vocals)} samples.") # Reduce verbosity

        if sr_instrumental != target_sr:
            logger.warning(f"Resampling instrumental from {sr_instrumental} Hz to {target_sr} Hz...")
            if librosa_available:
                instrumental = librosa.resample(y=instrumental, orig_sr=sr_instrumental, target_sr=target_sr, res_type='kaiser_best')
            elif scipy_available:
                num_samples = int(len(instrumental) * float(target_sr) / sr_instrumental)
                instrumental = scipy.signal.resample(instrumental, num_samples)
            else:
                 logger.error("Cannot resample instrumental: No librosa or scipy available.")
                 return False
            # logger.info(f"Instrumental resampled to {len(instrumental)} samples.") # Reduce verbosity

        # --- Length Alignment ---
        # Pad the shorter track with silence at the end
        len_vocals = len(vocals)
        len_instrumental = len(instrumental)
        if len_vocals > len_instrumental:
            # logger.warning(f"Vocals are longer ({len_vocals}) than instrumental ({len_instrumental}). Padding instrumental.") # Reduce verbosity
            padding = len_vocals - len_instrumental
            instrumental = np.pad(instrumental, (0, padding), 'constant')
        elif len_instrumental > len_vocals:
            # logger.warning(f"Instrumental is longer ({len_instrumental}) than vocals ({len_vocals}). Padding vocals.") # Reduce verbosity
            padding = len_instrumental - len_vocals
            vocals = np.pad(vocals, (0, padding), 'constant')

        # --- Normalization (Peak) ---
        if normalize:
            # logger.info("Normalizing tracks before mixing...") # Reduce verbosity
            peak_vocals = np.max(np.abs(vocals))
            if peak_vocals > 1e-6 : vocals /= peak_vocals
            peak_instrumental = np.max(np.abs(instrumental))
            if peak_instrumental > 1e-6: instrumental /= peak_instrumental

        # --- Basic Mixing (Overlay) ---
        # logger.info("Mixing tracks...") # Reduce verbosity
        # Reduce volume slightly to prevent clipping on addition
        mixed_audio = (vocals * 0.7) + (instrumental * 0.6) # Maybe make instrumental slightly quieter

        # --- Final Normalization/Clipping ---
        final_peak = np.max(np.abs(mixed_audio))
        if final_peak > 1.0:
            logger.warning(f"Mixed audio peak exceeds 1.0 ({final_peak:.2f}), normalizing final mix.")
            mixed_audio /= final_peak
        # mixed_audio = np.clip(mixed_audio, -1.0, 1.0) # Optional clipping instead

        # --- Save Combined Audio ---
        logger.info("Saving combined audio...")
        return save_audio_robust(mixed_audio, output_path, target_sr)

    except Exception as e:
        logger.error(f"Error during audio combination: {e}", exc_info=True)
        return False


# ================================================
# ========= Main Song Generation Process =========
# ================================================

# --- Configuration ---
song_lyrics = """
[MUSIC]
♪ In circuits deep, where data streams, ♪
♪ A silent song, in coded dreams. ♪
♪ We weave the threads of sound and byte, ♪
♪ To craft a tune in digital light. ♪
[laughs]
♪ From text to tones, the models learn, ♪
♪ A synthesized voice, a hopeful turn. ♪
♪ Can silicon hearts truly sing? ♪
♪ Let's listen close, the bells will ring! ♪
[MUSIC]
"""
music_style_description = "Uplifting electronic pop track with a gentle synth melody, steady beat, and atmospheric pads, optimistic mood, instrumental"
bark_voice_preset = "v2/en_speaker_9" # Choose Bark voice
musicgen_model_id = "facebook/musicgen-large" # Using large as requested
output_vocals_filename = "generated_vocals.wav"
output_instrumental_filename = "generated_instrumental.wav"
output_combined_filename = "generated_song_combined.wav"

vocals_duration_sec = 0.0
vocal_sample_rate = 24000 # Bark's default, will be updated

# --- 1. Generate Vocals with Bark (Using AutoModel approach) ---
# Define variables outside try block for finally clause
processor_bark = None
model_bark = None
inputs_bark = None
speech_output_automodel = None
audio_data_vocals = None
bark_success = False

try:
    logger.info("\n--- 1. Initializing Bark AutoModel for Singing Vocals ---")
    # Use "suno/bark" for potentially better quality if RAM allows, otherwise "suno/bark-small"
    model_id_bark = "suno/bark"
    # model_id_bark = "suno/bark-small"
    logger.info(f"Using Bark model: {model_id_bark}")

    processor_bark = AutoProcessor.from_pretrained(model_id_bark)
    model_bark = BarkModel.from_pretrained(model_id_bark).to(device)

    logger.info(f"Processing singing text input with voice preset: {bark_voice_preset}...")
    inputs_bark = processor_bark(
        text=[song_lyrics], # Pass lyrics as a list
        return_tensors="pt",
        voice_preset=bark_voice_preset
    )
    inputs_bark = {k: v.to(device) for k, v in inputs_bark.items()}

    logger.info("Generating singing with Bark AutoModel (using default temperatures)...")
    # Generate using default temperatures to avoid validation errors with base generate
    speech_output_automodel = model_bark.generate(
        **inputs_bark,
        do_sample=True,
    ).cpu() # Move to CPU

    vocal_sample_rate = model_bark.generation_config.sample_rate # Get actual SR
    audio_data_vocals = speech_output_automodel.squeeze() # Remove batch dim

    logger.info(f"Vocals generated (Sample Rate: {vocal_sample_rate})")

    # Calculate duration
    if audio_data_vocals is not None and hasattr(audio_data_vocals, 'shape'):
        num_samples = audio_data_vocals.shape[-1]
        vocals_duration_sec = num_samples / vocal_sample_rate
        logger.info(f"Calculated vocal duration: {vocals_duration_sec:.2f} seconds")
    else:
        logger.error("Failed to get valid audio data shape for duration calculation.")
        raise ValueError("Could not calculate vocal duration") # Stop if duration unknown

    # Save Vocals
    bark_success = save_audio_robust(audio_data_vocals, output_vocals_filename, vocal_sample_rate)

except Exception as e:
    logger.error(f"Error during Bark vocal generation: {e}", exc_info=True)
finally:
    # Clean up Bark resources
    if 'model_bark' in locals() and model_bark is not None: del model_bark
    if 'processor_bark' in locals() and processor_bark is not None: del processor_bark
    if 'inputs_bark' in locals() and inputs_bark is not None: del inputs_bark
    if 'speech_output_automodel' in locals() and speech_output_automodel is not None: del speech_output_automodel
    if 'audio_data_vocals' in locals() and audio_data_vocals is not None: del audio_data_vocals # Keep if needed later? No, combine loads from file.
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("Cleaned Bark resources.")

# --- 2. Generate Instrumental with MusicGen ---
synthesiser_musicgen = None
music_output = None
audio_data_musicgen = None
musicgen_success = False
sampling_rate_musicgen = 32000 # MusicGen default, will be updated

if bark_success and vocals_duration_sec > 1.0: # Only proceed if vocals were saved and have some length
    try:
        logger.info(f"\n--- 2. Initializing MusicGen ({musicgen_model_id}) for Instrumental ---")
        # Check disk space? Optional. Assume user confirmed space is okay.
        synthesiser_musicgen = pipeline("text-to-audio", model=musicgen_model_id, device=device)

        logger.info(f"Generating instrumental music for description: '{music_style_description}'")
        logger.info(f"Aiming for duration: {vocals_duration_sec:.2f} seconds (matching vocals)")

        # Use duration parameter directly
        music_output = synthesiser_musicgen(music_style_description, forward_params={"do_sample": True, "duration": vocals_duration_sec})

        if isinstance(music_output, dict) and "audio" in music_output and "sampling_rate" in music_output:
            audio_data_musicgen = music_output["audio"]
            sampling_rate_musicgen = music_output["sampling_rate"] # Get actual SR
            logger.info(f"Instrumental audio generated (Sample Rate: {sampling_rate_musicgen})")
            # Save Instrumental
            musicgen_success = save_audio_robust(audio_data_musicgen, output_instrumental_filename, sampling_rate_musicgen)
        else:
            logger.warning(f"Unexpected output format from MusicGen pipeline: {type(music_output)}")

    except Exception as e:
        logger.error(f"Error during MusicGen instrumental generation: {e}", exc_info=True)
    finally:
        # Clean up MusicGen resources
        if 'synthesiser_musicgen' in locals() and synthesiser_musicgen is not None: del synthesiser_musicgen
        if 'music_output' in locals() and music_output is not None: del music_output
        if 'audio_data_musicgen' in locals() and audio_data_musicgen is not None: del audio_data_musicgen
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info("Cleaned MusicGen pipeline resources.")
elif not bark_success:
     logger.error("Skipping MusicGen generation because vocal generation failed.")
else:
     logger.error(f"Skipping MusicGen generation because vocal duration was too short or invalid ({vocals_duration_sec:.2f}s).")


# --- 3. Combine Audio Tracks ---
if bark_success and musicgen_success:
    logger.info("\n--- 3. Combining Vocals and Instrumental ---")
    # Determine target sample rate for combination (use MusicGen's rate)
    target_sr_combine = sampling_rate_musicgen

    combine_success = combine_audio(
        output_vocals_filename,
        output_instrumental_filename,
        output_combined_filename,
        target_sr=target_sr_combine # Resample both to MusicGen's rate if needed
    )

    if combine_success:
        logger.info(f"--- Successfully combined audio saved to '{os.path.join(output_dir, output_combined_filename)}' ---")
    else:
        logger.error("--- Failed to combine audio tracks. ---")
else:
    logger.warning("--- Skipping audio combination because one or both tracks failed to generate. ---")

logger.info("\n--- Song Generation Script Finished ---")
























