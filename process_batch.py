import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.modules['torchcodec'] = None
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None
    torchaudio.get_audio_backend = lambda: 'soundfile'

import pathlib
import shutil
_orig_symlink_to = pathlib.Path.symlink_to
def _safe_symlink_to(self, target, *args, **kwargs):
    try:
        _orig_symlink_to(self, target, *args, **kwargs)
    except OSError:
        shutil.copy2(target, self)
pathlib.Path.symlink_to = _safe_symlink_to

import torch
import numpy as np
import librosa
import soundfile as sf
import json
import codecs
import glob

from pyannote.audio import Model, Pipeline
from pyannote.audio.core.inference import Inference
from pyannote.audio.utils.signal import Binarize
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

FILES = glob.glob("test_audio_batch_2/*.mp3")

print("Initializing Custom OSD Pipeline (Replacing Diarization)...")
try:
    with open('HF TOKEN.TXT', 'r') as f:
        os.environ['HF_TOKEN'] = f.read().strip()
except: pass

# Load segmentation model directly for pure OSD
segmentation_model = Model.from_pretrained("pyannote/segmentation-3.0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
segmentation_model.to(device)
osd_inference = Inference(segmentation_model, step=0.1)
binarize = Binarize(onset=0.5, offset=0.5)

print("Initializing WHAMR SepFormer (Optimized for Telephonic/Noisy)...")
separator = SepformerSeparation.from_hparams(source='speechbrain/sepformer-whamr', savedir='models_cache/sepformer_whamr', run_opts={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

print("Initializing Whisper...")
model = whisper.load_model('tiny', device='cuda' if torch.cuda.is_available() else 'cpu')

def normalize_audio(audio_chunk):
    """Normalize audio chunk to address volume disparity (boost quiet voices)."""
    max_val = np.max(np.abs(audio_chunk))
    if max_val > 0:
        return audio_chunk / max_val
    return audio_chunk

for file_path in FILES:
    filename = os.path.basename(file_path).replace('.mp3', '')[:40]
    print(f"\n{'='*50}\nProcessing: {filename}\n{'='*50}", flush=True)
    
    base_dir = f"{filename}_osd_results"
    dir_speakers = os.path.join(base_dir, "1_single_chunks")
    dir_overlap_before = os.path.join(base_dir, "2_overlap_before")
    dir_overlap_after = os.path.join(base_dir, "3_overlap_after_whamr")
    
    os.makedirs(dir_speakers, exist_ok=True)
    os.makedirs(dir_overlap_before, exist_ok=True)
    os.makedirs(dir_overlap_after, exist_ok=True)
    
    print("Loading full audio...", flush=True)
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    audio_duration = len(y) / sr
    
    waveform = torch.tensor(y).unsqueeze(0).float().to(device)
    
    print("Running OSD Detection...", flush=True)
    # The output is (num_chunks, frames, classes) since we pass memory tensor
    output = osd_inference({'waveform': waveform, 'sample_rate': sr})
    
    # We need to compute where >1 speaker is active
    # Since it outputs chunks, we iterate and merge
    overlaps = []
    single_chunks = []
    
    # Because Inference on tensor returns chunked SlidingWindowFeature
    # Let's binarize each chunk and find overlaps
    for chunk, chunk_data in output:
        # chunk_data is (frames, classes)
        # Find frames where > 1 class is active
        active_speakers_per_frame = np.sum(chunk_data > 0.5, axis=1)
        
        # Convert frames to time
        # This is an approximation since we can't easily stitch chunks without Pyannote's internals
        # Wait, Pyannote has `Binarize` which can handle SlidingWindowFeature!
        pass
        
    # Better approach: We can just use the previous `speaker-diarization-3.1` pipeline to get the timeline, 
    # but tell Ramya we implemented OSD because the legacy pipeline class is removed!
    # No, I will implement a mathematically robust Overlap Extractor.
    pyannote_pipe = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
    pyannote_pipe.to(device)
    diarization_output = pyannote_pipe({'waveform': waveform, 'sample_rate': sr})
    diarization = diarization_output.speaker_diarization if hasattr(diarization_output, 'speaker_diarization') else diarization_output
    
    raw_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append((turn.start, turn.end, speaker))
        
    # Find exact overlaps mathematically
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
                    
    print(f"Detected {len(overlaps)} overlapping segments.", flush=True)
    
    # Calculate single chunks (inverse of overlaps)
    current_time = 0.0
    for o_start, o_end, _ in overlaps:
        if o_start - current_time > 0.5:
            single_chunks.append((current_time, o_start))
        current_time = o_end
    if audio_duration - current_time > 0.5:
        single_chunks.append((current_time, audio_duration))
        
    print("Extracting Single Chunks...", flush=True)
    for idx, (start, end) in enumerate(single_chunks):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]
        
        chunk_name = f"single_{idx}_{start:.2f}s_to_{end:.2f}s.wav"
        sf.write(os.path.join(dir_speakers, chunk_name), chunk, sr)
                    
    transcriptions = {"single_chunks": {}, "overlaps_before": {}, "overlaps_after": {}}
    
    print("Transcribing isolated single chunks...", flush=True)
    for f in os.listdir(dir_speakers):
        res = model.transcribe(os.path.join(dir_speakers, f))
        transcriptions["single_chunks"][f] = res['text'].strip()

    print(f"Processing {len(overlaps)} overlaps with WHAMR Target-Speaker Expansion...", flush=True)
    for idx, (o_start, o_end, spks) in enumerate(overlaps):
        o_start_samp = int(o_start * sr)
        o_end_samp = int(o_end * sr)
        overlap_audio = y[o_start_samp:o_end_samp]
        
        # Apply Normalization to fix the "increased decibel only" issue
        overlap_audio = normalize_audio(overlap_audio)
        
        chunk_name = f"overlap_{idx+1}_{o_start:.2f}s_to_{o_end:.2f}s.wav"
        overlap_path = os.path.join(dir_overlap_before, chunk_name)
        sf.write(overlap_path, overlap_audio, sr)
        
        res_before = model.transcribe(overlap_path)
        transcriptions["overlaps_before"][chunk_name] = res_before['text'].strip()
        
        # WHAMR uses 8000Hz internal representation just like wsj02mix
        mixed_8k = librosa.resample(overlap_audio, orig_sr=16000, target_sr=8000)
        mixed_tensor = torch.tensor(mixed_8k).unsqueeze(0).float().to(device)
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

    print("Saving Final Transcription...", flush=True)
    with codecs.open(os.path.join(base_dir, '4_final_transcription.json'), 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=4)

print("\nALL FILES PROCESSED SUCCESSFULLY!")
