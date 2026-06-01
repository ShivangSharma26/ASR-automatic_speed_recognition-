import warnings
warnings.filterwarnings('ignore')

import sys
import os
os.environ['PATH'] += os.pathsep + r'c:\Users\shiva\Desktop\ASR'
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None
    torchaudio.get_audio_backend = lambda: 'soundfile'

import os
import torch
import numpy as np
import librosa
import soundfile as sf
import json
import codecs
import glob

from pyannote.audio import Pipeline
import whisper

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

FILES = glob.glob(r"test_audio_batch_2\*.mp3")

print("Initializing Pyannote...")
try:
    with open('HF TOKEN.TXT', 'r') as f:
        os.environ['HF_TOKEN'] = f.read().strip()
except: pass
pyannote_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')

print("Initializing SepFormer...")
separator = SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='models_cache/sepformer', run_opts={'device': 'cpu'})

print("Initializing Whisper...")
model = whisper.load_model('tiny', device='cpu')

for file_path in FILES:
    filename = os.path.basename(file_path).replace('.mp3', '')[:40]
    print(f"\n{'='*50}\nProcessing: {filename}\n{'='*50}")
    
    base_dir = f"{filename}_results"
    dir_speakers = os.path.join(base_dir, "1_speaker_chunks")
    dir_overlap_before = os.path.join(base_dir, "2_overlap_before")
    dir_overlap_after = os.path.join(base_dir, "3_overlap_after_sepformer")
    
    os.makedirs(dir_speakers, exist_ok=True)
    os.makedirs(dir_overlap_before, exist_ok=True)
    os.makedirs(dir_overlap_after, exist_ok=True)
    
    print("Loading audio (first 45 seconds for rapid testing)...")
    y, sr = librosa.load(file_path, sr=16000, mono=True, duration=45.0)
    
    waveform = torch.tensor(y).unsqueeze(0).float()
    audio_in_memory = {'waveform': waveform, 'sample_rate': sr}
    
    print("Running Diarization...")
    diarization = pyannote_pipe(audio_in_memory)
    
    raw_segments = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        raw_segments.append((turn.start, turn.end, speaker))
        
    print("Merging Micro-Chunks...")
    segments = []
    if raw_segments:
        raw_segments.sort(key=lambda x: x[0])
        current = raw_segments[0]
        for seg in raw_segments[1:]:
            if seg[2] == current[2] and (seg[0] - current[1]) < 1.0:
                current = (current[0], seg[1], current[2])
            else:
                segments.append(current)
                current = seg
        segments.append(current)

    print(f"Reduced {len(raw_segments)} raw chunks to {len(segments)} merged chunks.")
    
    print("Extracting Speaker Chunks (Req 1)...")
    for start, end, speaker in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]
        
        chunk_name = f"{speaker}_{start:.2f}s_to_{end:.2f}s.wav"
        sf.write(os.path.join(dir_speakers, chunk_name), chunk, sr)
        
    print("Finding Overlaps (Req 2)...")
    overlaps = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            start1, end1, label1 = segments[i]
            start2, end2, label2 = segments[j]
            if label1 != label2 and max(start1, start2) < min(end1, end2):
                o_start = max(start1, start2)
                o_end = min(end1, end2)
                if o_end - o_start > 0.5:
                    overlaps.append((o_start, o_end, [label1, label2]))
                    
    transcriptions = {"speaker_chunks": {}, "overlaps_before": {}, "overlaps_after": {}}
    
    print("Transcribing isolated speaker chunks...")
    for f in os.listdir(dir_speakers):
        res = model.transcribe(os.path.join(dir_speakers, f))
        transcriptions["speaker_chunks"][f] = res['text'].strip()

    print(f"Found {len(overlaps)} overlaps (>0.5s). Processing (Req 2 & 3)...")
    for idx, (o_start, o_end, speakers) in enumerate(overlaps):
        o_start_samp = int(o_start * sr)
        o_end_samp = int(o_end * sr)
        overlap_audio = y[o_start_samp:o_end_samp]
        
        chunk_name = f"overlap_{idx+1}_{o_start:.2f}s_to_{o_end:.2f}s.wav"
        overlap_path = os.path.join(dir_overlap_before, chunk_name)
        sf.write(overlap_path, overlap_audio, sr)
        
        res_before = model.transcribe(overlap_path)
        transcriptions["overlaps_before"][chunk_name] = res_before['text'].strip()
        
        mixed_8k = librosa.resample(overlap_audio, orig_sr=16000, target_sr=8000)
        mixed_tensor = torch.tensor(mixed_8k).unsqueeze(0).float()
        est_sources = separator.separate_batch(mixed_tensor)
        
        t1_16k = librosa.resample(est_sources[0, :, 0].detach().cpu().numpy(), orig_sr=8000, target_sr=16000)
        t2_16k = librosa.resample(est_sources[0, :, 1].detach().cpu().numpy(), orig_sr=8000, target_sr=16000)
        
        t1_path = os.path.join(dir_overlap_after, chunk_name.replace('.wav', '_track1.wav'))
        t2_path = os.path.join(dir_overlap_after, chunk_name.replace('.wav', '_track2.wav'))
        
        sf.write(t1_path, t1_16k, sr)
        sf.write(t2_path, t2_16k, sr)
        
        res_t1 = model.transcribe(t1_path)
        res_t2 = model.transcribe(t2_path)
        transcriptions["overlaps_after"][os.path.basename(t1_path)] = res_t1['text'].strip()
        transcriptions["overlaps_after"][os.path.basename(t2_path)] = res_t2['text'].strip()

    print("Saving Final Transcription (Req 4)...")
    with codecs.open(os.path.join(base_dir, '4_final_transcription.json'), 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=4)

print("\nALL FILES PROCESSED SUCCESSFULLY!")
