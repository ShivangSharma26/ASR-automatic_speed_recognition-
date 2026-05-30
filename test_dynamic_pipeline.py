import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import librosa
import numpy as np
import os
import warnings
from pyannote.audio import Pipeline
import soundfile as sf

warnings.filterwarnings("ignore")

def estimate_snr(y, top_db=20):
    """
    Estimate Signal-to-Noise Ratio (SNR) using RMS energy of speech vs non-speech frames.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return 0.0
    
    speech_mask = np.zeros(len(y), dtype=bool)
    for start, end in intervals:
        speech_mask[start:end] = True
        
    speech_frames = y[speech_mask]
    noise_frames = y[~speech_mask]
    
    if len(noise_frames) == 0 or len(speech_frames) == 0:
        return float('inf') 
        
    rms_speech = np.sqrt(np.mean(speech_frames**2))
    rms_noise = np.sqrt(np.mean(noise_frames**2))
    
    if rms_noise == 0:
        return float('inf')
        
    snr_db = 20 * np.log10(rms_speech / rms_noise)
    return snr_db

def main():
    print("Loading Pyannote Overlapped Speech Detection model...")
    try:
        with open("HF TOKEN.TXT", "r") as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Error: Please create 'HF TOKEN.TXT' with your huggingface token.")
        return

    # Use Pyannote Diarization 3.1 (which supports Pyannote 3.1+)
    import os
    os.environ["HF_TOKEN"] = token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    
    HINDI_PATH = r"test_audio\hindi_all\Call with Ahamad and 4 others-20260415_175024-Meeting Recording.mp3"
    TAMIL_PATH = r"test_audio\tamil_all_overlap\Call with Ahamad and 4 others-20260415_174722-Meeting Recording.mp3"
    
    print("\nLoading audio clips...")
    hindi, sr = librosa.load(HINDI_PATH, sr=16000, mono=True, duration=10.0, offset=5.0)
    tamil, _ = librosa.load(TAMIL_PATH, sr=16000, mono=True, duration=10.0, offset=5.0)
    
    min_len = min(len(hindi), len(tamil))
    hindi = hindi[:min_len]
    tamil = tamil[:min_len]
    
    # Create mixed audio for overlap testing
    mixed = hindi * 1.0 + tamil * 1.0
    
    SNR_THRESHOLD = 15.0  # dB

    # --- Test 1: SNR Routing on Clean Audio ---
    print("\n" + "="*60)
    print("TEST 1: SNR Routing on Clean Audio (Hindi)")
    print("="*60)
    snr_clean = estimate_snr(hindi)
    print(f"Calculated SNR: {snr_clean:.2f} dB")
    if snr_clean < SNR_THRESHOLD:
        print(f"--> Action: Route to NOISE CANCELLATION (SNR < {SNR_THRESHOLD})")
    else:
        print(f"--> Action: Route DIRECTLY to WHISPER (SNR >= {SNR_THRESHOLD})")
        
    # --- Test 2: SNR Routing on Noisy Audio ---
    print("\n" + "="*60)
    print("TEST 2: SNR Routing on Noisy Audio (Simulated with White Noise)")
    print("="*60)
    noise = np.random.normal(0, 0.05, len(hindi))
    noisy_hindi = hindi + noise
    snr_noisy = estimate_snr(noisy_hindi)
    print(f"Calculated SNR: {snr_noisy:.2f} dB")
    if snr_noisy < SNR_THRESHOLD:
        print(f"--> Action: Route to NOISE CANCELLATION (SNR < {SNR_THRESHOLD})")
    else:
        print(f"--> Action: Route DIRECTLY to WHISPER (SNR >= {SNR_THRESHOLD})")

    # --- Test 3: OSD on Mixed Audio ---
    print("\n" + "="*60)
    print("TEST 3: Overlapped Speech Detection (OSD) on Mixed Audio")
    print("="*60)
    
    # Pass audio in memory to bypass torchcodec/FFmpeg bugs on Windows
    import torch
    waveform = torch.tensor(mixed).unsqueeze(0).float()
    audio_in_memory = {"waveform": waveform, "sample_rate": sr}
    
    print("Running Pyannote Diarization Pipeline...")
    # Run Pyannote Pipeline
    diarization = pipeline(audio_in_memory)
    
    print("\n--- Diarization Output ---")
    print(diarization)
        
    print("\nPipeline Test Completed!")

if __name__ == '__main__':
    main()
