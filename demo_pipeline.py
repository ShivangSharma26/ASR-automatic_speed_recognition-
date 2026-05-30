import huggingface_hub
_orig_hf_hub_download = huggingface_hub.hf_hub_download
def _hf_hub_download_wrapper(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    try:
        return _orig_hf_hub_download(*args, **kwargs)
    except Exception as e:
        if 'custom.py' in str(e): raise ValueError(str(e))
        raise
huggingface_hub.hf_hub_download = _hf_hub_download_wrapper
import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import librosa
import torch
import os
import warnings
import shutil
import pathlib
import soundfile as sf
import json

warnings.filterwarnings("ignore")

# Windows symlink fix for SpeechBrain
_original_symlink = pathlib.Path.symlink_to
def _copy_instead(self, target, target_is_directory=False):
    target = pathlib.Path(target)
    self = pathlib.Path(self)
    if self.exists():
        return
    if target.is_dir():
        shutil.copytree(str(target), str(self))
    else:
        shutil.copy2(str(target), str(self))
pathlib.Path.symlink_to = _copy_instead

try:
    from speechbrain.pretrained import SepformerSeparation
except ImportError:
    from speechbrain.inference.separation import SepformerSeparation

from transformers import pipeline as hf_pipeline
from pyannote.audio import Pipeline

# --- Setup ---
RESULTS_DIR = "demo_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
HF_TOKEN = "HIDDEN_TOKEN"

HINDI_PATH = r"test_audio\hindi_all\Call with Ahamad and 4 others-20260415_175024-Meeting Recording.mp3"
TAMIL_PATH = r"test_audio\tamil_all_overlap\Call with Ahamad and 4 output-20260415_174722-Meeting Recording.mp3"

# Note: Using shorter audio for demo to prevent infinite hang
hindi, sr = librosa.load(HINDI_PATH, sr=16000, mono=True, duration=5.0, offset=5.0)
# Wait, tamil path was "Call with Ahamad and 4 others-20260415_174722-Meeting Recording.mp3" in previous script
TAMIL_PATH = r"test_audio\tamil_all_overlap\Call with Ahamad and 4 others-20260415_174722-Meeting Recording.mp3"
tamil, _ = librosa.load(TAMIL_PATH, sr=16000, mono=True, duration=5.0, offset=5.0)

min_len = min(len(hindi), len(tamil))
hindi = hindi[:min_len]
tamil = tamil[:min_len]

# Create overlapped audio mix
mixed = hindi * 1.0 + tamil * 1.0
mixed_path = os.path.join(RESULTS_DIR, "mixed_overlap.wav")
sf.write(mixed_path, mixed, 16000)

print("\n" + "="*80)
print("STEP 1: PYANNOTE DIARIZATION (OVERLAPPED SPEECH DETECTION)")
print("="*80)
try:
    print("Loading Pyannote 3.1 (Downloading if first time)...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    diarization_pipeline.to(torch.device("cpu"))
    
    print("Running Diarization on mixed audio...")
    diarization = diarization_pipeline(mixed_path)
    
    print("\nPyannote Output Timestamps:")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"  [{turn.start:.2f}s - {turn.end:.2f}s] {speaker}")
    
except Exception as e:
    print(f"\n[ERROR] Pyannote failed to run. This is usually due to missing HuggingFace gating agreements. Error: {e}")
    print("\n[MOCKING PYANNOTE OUTPUT FOR PIPELINE CONTINUATION]")
    print("  [0.00s - 5.00s] SPEAKER_00")
    print("  [0.00s - 5.00s] SPEAKER_01")


print("\n" + "="*80)
print("STEP 2: SOURCE SEPARATION (SEPFORMER)")
print("="*80)
print("Loading speechbrain/sepformer-wsj02mix...")
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix", 
    savedir="models_cache/sepformer", 
    run_opts={"device": "cpu"}
)

print(f"Passing the {len(mixed)/16000:.1f}s overlapped audio into SepFormer...")
mixed_8k = librosa.resample(mixed, orig_sr=16000, target_sr=8000)
mixed_tensor = torch.tensor(mixed_8k).unsqueeze(0).float()
est_sources = separator.separate_batch(mixed_tensor)

source1_8k = est_sources[0, :, 0].detach().cpu().numpy()
source2_8k = est_sources[0, :, 1].detach().cpu().numpy()

s1_path = os.path.join(RESULTS_DIR, "separated_track_1.wav")
s2_path = os.path.join(RESULTS_DIR, "separated_track_2.wav")

# Resample back to 16k for Whisper
source1_16k = librosa.resample(source1_8k, orig_sr=8000, target_sr=16000)
source2_16k = librosa.resample(source2_8k, orig_sr=8000, target_sr=16000)

sf.write(s1_path, source1_16k, 16000)
sf.write(s2_path, source2_16k, 16000)

print(f"Success! Separated audio chunks saved to:")
print(f"  - {s1_path}")
print(f"  - {s2_path}")


print("\n" + "="*80)
print("STEP 3: WHISPER TRANSCRIPTION ON SEPARATED TRACKS")
print("="*80)
print("Loading openai/whisper-tiny...")
whisper_pipe = hf_pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device="cpu"
)

print("\nTranscribing Track 1...")
s1_audio, _ = librosa.load(s1_path, sr=16000); res1 = whisper_pipe({'array': s1_audio, 'sampling_rate': 16000})
print(f"  Track 1 Output: \"{res1['text'].strip()}\"")

print("\nTranscribing Track 2...")
s2_audio, _ = librosa.load(s2_path, sr=16000); res2 = whisper_pipe({'array': s2_audio, 'sampling_rate': 16000})
print(f"  Track 2 Output: \"{res2['text'].strip()}\"")

print("\n" + "="*80)
print("PIPELINE DEMO COMPLETE!")
print("="*80)
