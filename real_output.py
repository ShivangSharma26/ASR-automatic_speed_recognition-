import warnings
warnings.filterwarnings('ignore')

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

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

import os
import torch
import librosa
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline

try:
    from speechbrain.pretrained import SepformerSeparation
except ImportError:
    from speechbrain.inference.separation import SepformerSeparation

try:
    with open('HF TOKEN.TXT', 'r') as f:
        os.environ['HF_TOKEN'] = f.read().strip()
except: pass

print('\n' + '='*80)
print('Loading Audio')
print('='*80)
HINDI_PATH = r'test_audio\hindi_all\Call with Ahamad and 4 others-20260415_175024-Meeting Recording.mp3'
TAMIL_PATH = r'test_audio\tamil_all_overlap\Call with Ahamad and 4 others-20260415_174722-Meeting Recording.mp3'
hindi, sr = librosa.load(HINDI_PATH, sr=16000, mono=True, duration=5.0, offset=5.0)
tamil, _ = librosa.load(TAMIL_PATH, sr=16000, mono=True, duration=5.0, offset=5.0)
min_len = min(len(hindi), len(tamil))
mixed = hindi[:min_len] + tamil[:min_len]
waveform = torch.tensor(mixed).unsqueeze(0).float()
audio_in_memory = {'waveform': waveform, 'sample_rate': 16000}

print('\n' + '='*80)
print('1. REAL PYANNOTE DIARIZATION')
print('='*80)
pyannote_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
diarization = pyannote_pipe(audio_in_memory)
if False:
    pass
print(diarization)

print('\n' + '='*80)
print('2. REAL SOURCE SEPARATION (SEPFORMER)')
print('='*80)
separator = SepformerSeparation.from_hparams(
    source='speechbrain/sepformer-wsj02mix', 
    savedir='models_cache/sepformer', 
    run_opts={'device': 'cpu'}
)
mixed_8k = librosa.resample(mixed, orig_sr=16000, target_sr=8000)
mixed_tensor = torch.tensor(mixed_8k).unsqueeze(0).float()
est_sources = separator.separate_batch(mixed_tensor)
source1_8k = est_sources[0, :, 0].detach().cpu().numpy()
source2_8k = est_sources[0, :, 1].detach().cpu().numpy()
source1_16k = librosa.resample(source1_8k, orig_sr=8000, target_sr=16000)
source2_16k = librosa.resample(source2_8k, orig_sr=8000, target_sr=16000)
print('Successfully separated into Track 1 and Track 2.')

print('\n' + '='*80)
print('3. REAL WHISPER TRANSCRIPTION')
print('='*80)
try:
    whisper_pipe = hf_pipeline('automatic-speech-recognition', model='openai/whisper-tiny', device='cpu')
    res1 = whisper_pipe({'array': source1_16k, 'sampling_rate': 16000})
    print(f'Track 1: "{res1["text"].strip()}"')
    res2 = whisper_pipe({'array': source2_16k, 'sampling_rate': 16000})
    print(f'Track 2: "{res2["text"].strip()}"')
except Exception as e:
    print('Whisper crashed natively, falling back to basic requests API if needed:', str(e))
