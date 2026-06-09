import os
import librosa
import soundfile as sf
import numpy as np

def synthesize_edge_cases():
    audio_dir = r"C:\Users\shiva\Desktop\PINACA\ASR\test_audio_batch"
    out_dir = r"C:\Users\shiva\Desktop\PINACA\ASR\edge_case_testing"
    
    files = [
        "sravan_normal_voicecall3.mp3",
        "sravan_normal_voicecall4.mp3",
        "sravan_uday.new.mp3"
    ]
    
    loaded_arrays = []
    
    for f in files:
        path = os.path.join(audio_dir, f)
        print(f"Loading {f}...")
        # Load exactly 10 seconds starting at 20s
        y, sr = librosa.load(path, sr=16000, mono=True, offset=20.0, duration=10.0)
        loaded_arrays.append(y)
        
    # Find minimum length to pad/truncate
    min_len = min(len(y) for y in loaded_arrays)
    arrays = [y[:min_len] for y in loaded_arrays]
    
    y1, y2, y3 = arrays[0], arrays[1], arrays[2]
    
    # Normalize base arrays to similar energy (RMS)
    def normalize_rms(arr, target_rms=0.1):
        rms = np.sqrt(np.mean(arr**2))
        return arr * (target_rms / (rms + 1e-8))
        
    y1 = normalize_rms(y1)
    y2 = normalize_rms(y2)
    y3 = normalize_rms(y3)
    
    def apply_db_gain(arr, db_gain):
        return arr * (10 ** (db_gain / 20.0))

    # Case 1: Equal energy (0dB, 0dB, 0dB)
    case1 = y1 + y2 + y3
    
    # Case 2: Top-2 dominant (0dB, 0dB, -12dB)
    case2 = y1 + y2 + apply_db_gain(y3, -12.0)
    
    # Case 3: Top-1 dominant (0dB, -12dB, -12dB)
    case3 = y1 + apply_db_gain(y2, -12.0) + apply_db_gain(y3, -12.0)
    
    # Save files
    sf.write(os.path.join(out_dir, "case1_equal.wav"), case1, 16000)
    sf.write(os.path.join(out_dir, "case2_top2.wav"), case2, 16000)
    sf.write(os.path.join(out_dir, "case3_top1.wav"), case3, 16000)
    
    print("Edge case synthesis complete! Saved 3 WAV files in edge_case_testing.")

if __name__ == "__main__":
    synthesize_edge_cases()
