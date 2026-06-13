import os
import sys

# Mocks for torchaudio and torchcodec bugs on Windows
sys.modules['torchcodec'] = None
import torchaudio
torchaudio.set_audio_backend = lambda x: None

import torch
from pyannote.audio import Model
from pyannote.audio.core.inference import Inference
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings("ignore")

# Ensure huggingface token is set
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"

def run_eend():
    print("Loading Pyannote Segmentation 3.0 (EEND) model...")
    # The segmentation model natively outputs multi-label frame-wise probabilities!
    model = Model.from_pretrained("pyannote/segmentation-3.0")
    model.to(torch.device("cpu"))
    
    # We use the Inference wrapper to apply the model over the audio
    inference = Inference(model, step=0.1)
    
    input_file = "case2_top2.wav"
    print(f"Running EEND on {input_file}...")
    
    # Load and resample
    waveform, sr = sf.read(input_file)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    
    waveform_tensor = torch.tensor(waveform).unsqueeze(0).float()
    
    # Run inference directly
    # Pyannote expects dict or path
    output = inference({"waveform": waveform_tensor, "sample_rate": 16000})
    
    # output is a SlidingWindowFeature. Let's inspect it.
    print(f"Output shape: {output.data.shape}")
    print(f"Labels: {output.labels}")
    
    # We can just binarize the output with a threshold
    from pyannote.audio.utils.signal import Binarize
    binarize = Binarize(offset=0.5, onset=0.5, log_scale=False)
    
    binary_output = binarize(output)
    
    # Print timestamps
    print("\nDetected Overlapping Speakers:")
    for turn, _, speaker in binary_output.itertracks(yield_label=True):
        print(f"[{turn.start:.2f}s -> {turn.end:.2f}s] {speaker}")

if __name__ == "__main__":
    run_eend()
