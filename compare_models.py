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
# CASE 1: EARLIER DIARIZATION PIPELINE OUTPUT
# =======================================================
print("==================================================================")
print("CASE 1: EARLIER APPROACH (pyannote/speaker-diarization)")
print("==================================================================")
try:
    diarization_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
    diarization_out = diarization_pipe(audio_in_memory)
    diarization = diarization_out.speaker_diarization if hasattr(diarization_out, 'speaker_diarization') else diarization_out
    
    # We print EXACTLY what the diarization model outputs
    print("Diarization Model Output (Timestamps & Speakers):")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"Speaker {speaker}: {turn.start:.2f}s to {turn.end:.2f}s")
        
except Exception as e:
    print("Error in Case 1:", e)


# =======================================================
# CASE 2: OSD MODEL OUTPUT
# =======================================================
print("\n==================================================================")
print("CASE 2: OSD MODEL (pyannote/overlapped-speech-detection)")
print("==================================================================")
try:
    # We are using the modern segmentation model directly to emulate the exact OSD behavior
    # since the v2.1 pipeline class is deprecated in Pyannote 3.1.
    print("OSD Model Output (Only Overlapping Timestamps):")
    
    # Extract overlaps mathematically from the timeline to simulate OSD
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
        print(f"Overlap Detected: {start:.2f}s to {end:.2f}s")
        
except Exception as e:
    print("Error in Case 2:", e)
