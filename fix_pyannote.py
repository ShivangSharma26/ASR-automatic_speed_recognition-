import warnings
warnings.filterwarnings('ignore')

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import os
from pyannote.audio import Pipeline

try:
    with open('HF TOKEN.TXT', 'r') as f:
        os.environ['HF_TOKEN'] = f.read().strip()
except: pass

print('Loading Pyannote and running on file...')
pyannote_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
diarization = pyannote_pipe('demo_results/realistic_mix.wav')

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f" -> SPEAKER DETECTED: {speaker} | Active: {turn.start:.2f}s to {turn.end:.2f}s")
