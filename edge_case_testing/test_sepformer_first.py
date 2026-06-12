import os
import sys
sys.modules['torchcodec'] = None

import torch
import torchaudio
torchaudio.set_audio_backend = lambda x: None

from speechbrain.pretrained import SepformerSeparation
from pyannote.audio import Pipeline
import soundfile as sf
import warnings

# Ensure huggingface token is set
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"

warnings.filterwarnings("ignore")

import shutil
# Patch os.symlink to fix Windows privilege error (WinError 1314)
original_symlink = getattr(os, "symlink", None)
def patched_symlink(src, dst, *args, **kwargs):
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
os.symlink = patched_symlink

def test_sepformer_first():
    print("Loading SepFormer 2-mix model...")
    # Using local folder to avoid hf_hub_download auth issues seen previously if applicable,
    # but let's try direct load first. If it fails, we know it's the HF bug.
    # Actually, earlier the hf_hub bug was because of use_auth_token. Let's patch it.
    import huggingface_hub
    original_download = huggingface_hub.hf_hub_download
    def patched_download(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return original_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = patched_download

    try:
        model = SepformerSeparation.from_hparams(source="sepformer-whamr", savedir="sepformer-whamr")
    except Exception as e:
        print(f"Failed to load SepFormer: {e}")
        return

    print("Loading Pyannote Diarization 3.1 pipeline...")
    try:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        # Force CPU if CUDA not available
        diarization_pipeline.to(torch.device("cpu"))
    except Exception as e:
        print(f"Failed to load Pyannote: {e}")
        return

    input_file = "case2_top2.wav"
    print(f"\nProcessing {input_file} through SepFormer FIRST...")
    
    # 1. Source Separation (Bypass torchaudio.load by using soundfile and separate_batch)
    import soundfile as sf
    import librosa
    mix_waveform, mix_sr = sf.read(input_file)
    if mix_sr != 8000:
        mix_waveform = librosa.resample(mix_waveform, orig_sr=mix_sr, target_sr=8000)
    
    mix_tensor = torch.tensor(mix_waveform).unsqueeze(0).float()
    est_sources = model.separate_batch(mix_tensor)
    
    # Save the isolated tracks
    os.makedirs("sepformer_first_outputs", exist_ok=True)
    track1_path = os.path.join("sepformer_first_outputs", "track1.wav")
    track2_path = os.path.join("sepformer_first_outputs", "track2.wav")
    
    # est_sources is shape [batch, time, channels] -> [1, T, 2]
    t1 = est_sources[0, :, 0].detach().cpu().numpy()
    t2 = est_sources[0, :, 1].detach().cpu().numpy()
    
    sf.write(track1_path, t1, 8000)
    sf.write(track2_path, t2, 8000)
    print("Saved separated tracks to sepformer_first_outputs/")

    # 2. Diarization on the Separated Tracks
    results_file = "sepformer_first_results.txt"
    with open(results_file, "w") as f:
        f.write("========================================\n")
        f.write("Evaluation: SepFormer FIRST -> Pyannote Diarization\n")
        f.write("========================================\n\n")

        for track_num, path in enumerate([track1_path, track2_path], 1):
            print(f"Running Pyannote Diarization on Track {track_num}...")
            f.write(f"--- Diarization on Track {track_num} ---\n")
            
            # Load audio using soundfile
            waveform, sr = sf.read(path)
            
            # Resample to 16000 using librosa
            import librosa
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            
            # Pyannote expects shape (1, time) as torch tensor
            waveform_tensor = torch.tensor(waveform).unsqueeze(0).float()
            
            diarization = diarization_pipeline({"waveform": waveform_tensor, "sample_rate": 16000})
            annotation = diarization.speaker_diarization
            
            active_speakers = set()
            f.write("Timestamps:\n")
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                f.write(f"[{turn.start:.2f}s -> {turn.end:.2f}s] {speaker}\n")
                active_speakers.add(speaker)
                
            f.write(f"\nTotal Unique Speakers Detected in Track {track_num}: {len(active_speakers)}\n\n")

    print(f"Done! Check {results_file} for detailed results.")

if __name__ == "__main__":
    test_sepformer_first()
