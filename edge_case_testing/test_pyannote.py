import os
import torch
import torchaudio
torchaudio.set_audio_backend = lambda x: None

from pyannote.audio import Pipeline

# Ensure huggingface token is set
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"

import librosa

def test_pyannote():
    print("Loading Pyannote Diarization 3.1 pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    
    test_files = ["case1_equal.wav", "case2_top2.wav", "case3_top1.wav"]
    out_dir = r"C:\Users\shiva\Desktop\PINACA\ASR\edge_case_testing"
    
    with open(os.path.join(out_dir, "pyannote_results.txt"), "w", encoding="utf-8") as f:
        for file in test_files:
            f.write(f"\n{'='*40}\nTesting {file}\n{'='*40}\n")
            print(f"Processing {file}...")
            
            path = os.path.join(out_dir, file)
            # Load with librosa to avoid torchcodec bug
            y, sr = librosa.load(path, sr=16000, mono=True)
            # Pyannote expects [channels, time]
            waveform = torch.from_numpy(y).unsqueeze(0)
            
            # Run inference
            diarization = pipeline({"waveform": waveform, "sample_rate": sr})
            
            # Extract the actual Annotation object
            annotation = diarization.speaker_diarization
            
            # Analyze timestamps
            f.write("\nTimestamps:\n")
            active_speakers = set()
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                f.write(f"[{turn.start:.2f}s -> {turn.end:.2f}s] {speaker}\n")
                active_speakers.add(speaker)
                
            f.write(f"\nTotal Unique Speakers Detected: {len(active_speakers)}\n")
            
            # Check maximum simultaneous overlap depth
            # Pyannote internally limits max simultaneous speakers unless configured otherwise
            # Let's see if it found 3 anywhere.
            # We can check the depth by looking at the overlaps. Since it's exactly the same 10s audio,
            # if it outputs 3 distinct speaker turns covering the same time, depth = 3.

if __name__ == "__main__":
    test_pyannote()
