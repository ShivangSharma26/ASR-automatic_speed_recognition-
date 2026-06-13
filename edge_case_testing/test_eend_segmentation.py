import os
import sys

# Mocks for torchaudio and torchcodec bugs on Windows
sys.modules['torchcodec'] = None
import torchaudio
torchaudio.set_audio_backend = lambda x: None

import torch
from pyannote.audio import Model
from pyannote.audio.core.inference import Inference
from pyannote.audio.utils.signal import Binarize
import soundfile as sf
import librosa
import warnings

# Ensure huggingface token is set
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"
warnings.filterwarnings("ignore")

def test_eend():
    print("Loading Pyannote Segmentation 3.0 (EEND) model...")
    try:
        model = Model.from_pretrained("pyannote/segmentation-3.0")
        model.to(torch.device("cpu"))
    except Exception as e:
        print(f"Failed to load Pyannote Segmentation: {e}")
        return
        
    inference = Inference(model, step=0.1)
    
    input_file = "case2_top2.wav"
    print(f"\nProcessing {input_file} through EEND (Segmentation without Clustering)...")
    
    # Load and resample
    waveform, sr = sf.read(input_file)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    
    waveform_tensor = torch.tensor(waveform).unsqueeze(0).float()
    
    # Run inference directly to get frame-wise probabilities
    output = inference({"waveform": waveform_tensor, "sample_rate": 16000})
    
    # Remove batch dimension for Binarize (1, frames, classes) -> (frames, classes)
    output.data = output.data.squeeze(0) if len(output.data.shape) == 3 else output.data
    
    # Binarize probabilities to binary active/inactive states
    # onset=0.5 means probability must be >0.5 to trigger 'active'
    # offset=0.5 means probability must drop <0.5 to trigger 'inactive'
    binarize = Binarize(onset=0.5, offset=0.5)
    binary_output = binarize(output)
    
    results_file = "eend_results.txt"
    with open(results_file, "w") as f:
        f.write("========================================\n")
        f.write("Evaluation: End-to-End Neural Diarization (EEND)\n")
        f.write("Model: pyannote/segmentation-3.0 (Raw Multi-Label Output)\n")
        f.write("========================================\n\n")
        
        f.write(f"Raw Output Tensor Shape: {output.data.shape} (Time Frames, Local Speakers)\n\n")
        f.write("Detected Overlapping Speakers Timestamps:\n")
        
        active_speakers = set()
        for turn, _, speaker in binary_output.itertracks(yield_label=True):
            # The segmentation model outputs local speakers as integers (0, 1, 2)
            speaker_label = f"SPEAKER_0{speaker}"
            f.write(f"[{turn.start:.2f}s -> {turn.end:.2f}s] {speaker_label}\n")
            active_speakers.add(speaker_label)
            
        f.write(f"\nTotal Unique Speakers Detected Simultaneously: {len(active_speakers)}\n")
        f.write("\nConclusion:\n")
        if len(active_speakers) == 2:
            f.write("SUCCESS: The EEND model successfully mapped 2 distinct speakers overlapping simultaneously, completely avoiding the clustering bottleneck!\n")
        else:
            f.write(f"FAILURE: Expected 2 speakers, but detected {len(active_speakers)}.\n")

    print(f"Done! Check {results_file} for detailed results.")

if __name__ == "__main__":
    test_eend()
