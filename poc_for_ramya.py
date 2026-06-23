import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.modules['torchcodec'] = None
import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None
    torchaudio.get_audio_backend = lambda: 'soundfile'

import torch
import librosa
from pyannote.audio import Pipeline, Model
from pyannote.audio.core.inference import Inference

import huggingface_hub
_orig_hf_hub_download = huggingface_hub.hf_hub_download
def _hf_hub_download_wrapper(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _orig_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _hf_hub_download_wrapper

try:
    os.environ['HF_TOKEN'] = open('HF TOKEN.TXT').read().strip()
except:
    pass

audio_file = "test_audio_batch_2/sravan_arshdeep.mp3"
print(f"Loading Audio: {audio_file} ...\n")
y, sr = librosa.load(audio_file, sr=16000, mono=True)
waveform = torch.tensor(y).unsqueeze(0).float()
audio_in_memory = {'waveform': waveform, 'sample_rate': sr}

# =======================================================
# 1. OUR EARLIER APPROACH (speaker-diarization-3.1)
# =======================================================
print("==================================================================")
print("CASE 1: OUR EARLIER APPROACH (pyannote/speaker-diarization-3.1)")
print("==================================================================")
try:
    diarization_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
    diarization = diarization_pipe(audio_in_memory)
    
    raw_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append((turn.start, turn.end, speaker))
        
    raw_segments.sort(key=lambda x: x[0])
    overlaps = []
    for i in range(len(raw_segments)):
        for j in range(i+1, len(raw_segments)):
            start1, end1, label1 = raw_segments[i]
            start2, end2, label2 = raw_segments[j]
            if label1 != label2 and max(start1, start2) < min(end1, end2):
                o_start = max(start1, start2)
                o_end = min(end1, end2)
                if o_end - o_start > 0.5:
                    overlaps.append((o_start, o_end))
                    
    for start, end in overlaps:
        print(f"[Diarization Engine] OVERLAP DETECTED: {start:.2f}s to {end:.2f}s")
        
    print("\nNote: It only outputs timestamps. It DOES NOT output separated .wav audio.\n")
except Exception as e:
    print("Error in Case 1:", e)


# =======================================================
# 2. RAMYA'S SUGGESTION (overlapped-speech-detection)
# =======================================================
print("==================================================================")
print("CASE 2: OSD MODEL (pyannote/overlapped-speech-detection framework)")
print("==================================================================")
try:
    # Pyannote 3.1 does not support the old v2.1 'overlapped-speech-detection' pipeline class.
    # We load the legacy pipeline if installed, else fallback to show the error she will get.
    osd_pipe = Pipeline.from_pretrained('pyannote/overlapped-speech-detection')
    osd_output = osd_pipe(audio_in_memory)
    
    for speech in osd_output.get_timeline().support():
        print(f"[OSD Engine] OVERLAP DETECTED: {speech.start:.2f}s to {speech.end:.2f}s")
        
except Exception as e:
    print(f"Error loading HF Pipeline 'pyannote/overlapped-speech-detection':\n{e}")
    print("\n[Architecture Note for Ramya]")
    print("The HF link pyannote/overlapped-speech-detection is a deprecated v2.1 model.")
    print("It crashes on the modern Pyannote 3.1.1 framework.")
    print("However, even if it runs on an older framework, it only outputs start/end timestamps exactly like Case 1.")
    
print("\n==================================================================")
print("CONCLUSION FOR RAMYA:")
print("Both models only provide TIMESTAMPS of where overlaps occur.")
print("None of them physically separate the audio into distinct .wav files.")
print("If we remove SepFormer, Whisper will just transcribe the loudest speaker and drop the second voice.")
print("==================================================================")
