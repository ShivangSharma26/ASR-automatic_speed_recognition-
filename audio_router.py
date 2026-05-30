import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

import librosa
import numpy as np
import os
import warnings
import torch
from pyannote.audio import Pipeline
import re

warnings.filterwarnings("ignore")

try:
    with open("HF TOKEN.TXT", "r") as f:
        token = f.read().strip()
    os.environ["HF_TOKEN"] = token
except FileNotFoundError:
    pass

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
SNR_THRESHOLD = 15.0

def estimate_snr(y, top_db=20):
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0: return 0.0
    speech_mask = np.zeros(len(y), dtype=bool)
    for start, end in intervals: speech_mask[start:end] = True
    speech_frames = y[speech_mask]
    noise_frames = y[~speech_mask]
    if len(noise_frames) == 0 or len(speech_frames) == 0: return float('inf') 
    rms_speech = np.sqrt(np.mean(speech_frames**2))
    rms_noise = np.sqrt(np.mean(noise_frames**2))
    if rms_noise == 0: return float('inf')
    return 20 * np.log10(rms_speech / rms_noise)

def parse_time(t):
    h, m, s = t.split(':')
    return float(h)*3600 + float(m)*60 + float(s)

def extract_overlaps(diarization):
    lines = str(diarization).strip().split('\n')
    segments = []
    for line in lines:
        match = re.search(r'\[\s*(.*?)\s*-->\s*(.*?)\s*\]\s*(.*)', line)
        if match:
            start_str, end_str, label = match.groups()
            segments.append((parse_time(start_str), parse_time(end_str), label.strip()))
            
    overlaps = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            start1, end1, label1 = segments[i]
            start2, end2, label2 = segments[j]
            if label1 != label2 and max(start1, start2) < min(end1, end2):
                overlaps.append({
                    "start": round(max(start1, start2), 2),
                    "end": round(min(end1, end2), 2),
                    "speakers": [label1, label2]
                })
    return overlaps

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    snr = estimate_snr(y)
    routing_decision = "whisper" if snr >= SNR_THRESHOLD else "noise_cancellation"
    waveform = torch.tensor(y).unsqueeze(0).float()
    audio_in_memory = {"waveform": waveform, "sample_rate": sr}
    diarization = pipeline(audio_in_memory)
    overlaps = extract_overlaps(diarization)
    return {
        "snr_db": float(round(snr, 2)),
        "routing_decision": routing_decision,
        "overlapping_segments": overlaps
    }
