import warnings
warnings.filterwarnings('ignore')

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from pyannote.audio import Pipeline

# 1. Load Audio
HINDI_PATH = r'test_audio\hindi_all\Call with Ahamad and 4 others-20260415_175024-Meeting Recording.mp3'
TAMIL_PATH = r'test_audio\tamil_all_overlap\Call with Ahamad and 4 others-20260415_174722-Meeting Recording.mp3'

hindi, sr = librosa.load(HINDI_PATH, sr=16000, mono=True, offset=5.0, duration=10.0)
tamil, _ = librosa.load(TAMIL_PATH, sr=16000, mono=True, offset=5.0, duration=10.0)

mixed = np.zeros(15 * sr)
mixed[0:10*sr] += hindi
mixed[5*sr:15*sr] += tamil

os.makedirs('demo_results', exist_ok=True)
sf.write('demo_results/realistic_mix.wav', mixed, sr)

print('\n--- 1. REAL PYANNOTE DIARIZATION ---')
waveform = torch.tensor(mixed).unsqueeze(0).float()
audio_in_memory = {'waveform': waveform, 'sample_rate': sr}

try:
    with open('HF TOKEN.TXT', 'r') as f:
        os.environ['HF_TOKEN'] = f.read().strip()
except: pass

pyannote_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
diarization = pyannote_pipe(audio_in_memory)
print(diarization)

print('\n--- 2. REAL SOURCE SEPARATION (SEPFORMER) ---')
try:
    from speechbrain.pretrained import SepformerSeparation
except ImportError:
    from speechbrain.inference.separation import SepformerSeparation

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

separator = SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='models_cache/sepformer', run_opts={'device': 'cpu'})
overlap_segment = mixed[5*sr:10*sr]
mixed_8k = librosa.resample(overlap_segment, orig_sr=16000, target_sr=8000)
mixed_tensor = torch.tensor(mixed_8k).unsqueeze(0).float()
est_sources = separator.separate_batch(mixed_tensor)
source1_8k = est_sources[0, :, 0].detach().cpu().numpy()
source2_8k = est_sources[0, :, 1].detach().cpu().numpy()

source1_16k = librosa.resample(source1_8k, orig_sr=8000, target_sr=16000)
source2_16k = librosa.resample(source2_8k, orig_sr=8000, target_sr=16000)

sf.write('demo_results/sep_track_1.wav', source1_16k, sr)
sf.write('demo_results/sep_track_2.wav', source2_16k, sr)
print('Separated overlapping 5s segment into two tracks.')

print('\n--- 3. REAL WHISPER TRANSCRIPTION ---')
import whisper
import json
import codecs
model = whisper.load_model('tiny', device='cpu')

res1 = model.transcribe('demo_results/sep_track_1.wav', language='hi')
res2 = model.transcribe('demo_results/sep_track_2.wav', language='ta')

with codecs.open('demo_results/whisper_realistic.json', 'w', encoding='utf-8') as f:
    json.dump({'Track 1 (Hindi Forced)': res1['text'].strip(), 'Track 2 (Tamil Forced)': res2['text'].strip()}, f, ensure_ascii=False, indent=2)

print('Saved transcription to demo_results/whisper_realistic.json')
