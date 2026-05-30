import warnings
warnings.filterwarnings('ignore')

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import os
import torch
import numpy as np
import librosa
from pyannote.audio import Pipeline

HINDI_PATH = r'test_audio\hindi_all\Call with Ahamad and 4 others-20260415_175024-Meeting Recording.mp3'
TAMIL_PATH = r'test_audio\tamil_all_overlap\Call with Ahamad and 4 others-20260415_174722-Meeting Recording.mp3'

hindi, sr = librosa.load(HINDI_PATH, sr=16000, mono=True, offset=5.0, duration=10.0)
tamil, _ = librosa.load(TAMIL_PATH, sr=16000, mono=True, offset=5.0, duration=10.0)

mixed = np.zeros(15 * sr)
mixed[0:10*sr] += hindi
mixed[5*sr:15*sr] += tamil

waveform = torch.tensor(mixed).unsqueeze(0).float()
audio_in_memory = {'waveform': waveform, 'sample_rate': sr}

pyannote_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
diarization = pyannote_pipe(audio_in_memory)

print('Type:', type(diarization))
print('Dir:', dir(diarization))
print('String Output:')
print(diarization)
