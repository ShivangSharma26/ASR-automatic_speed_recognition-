import os
import torch
import torchaudio
torchaudio.set_audio_backend = lambda x: None

from speechbrain.pretrained import SepformerSeparation

def test_sepformer():
    print("Loading SepFormer 2-mix model...")
    # Load the pretrained model
    model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr", savedir="pretrained_models")
    
    test_files = ["case1_equal.wav", "case2_top2.wav", "case3_top1.wav"]
    in_dir = r"C:\Users\shiva\Desktop\PINACA\ASR\edge_case_testing"
    out_dir = r"C:\Users\shiva\Desktop\PINACA\ASR\edge_case_testing\sepformer_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(in_dir, "sepformer_results.txt"), "w", encoding="utf-8") as f:
        for file in test_files:
            f.write(f"\n{'='*40}\nTesting {file}\n{'='*40}\n")
            print(f"Processing {file}...")
            
            path = os.path.join(in_dir, file)
            # Separate
            est_sources = model.separate_file(path=path) 
            # est_sources is shape [1, time, 2]
            
            track1 = est_sources[:, :, 0].detach().cpu()
            track2 = est_sources[:, :, 1].detach().cpu()
            
            # Save the tracks for manual review
            base_name = file.replace(".wav", "")
            t1_path = os.path.join(out_dir, f"{base_name}_track1.wav")
            t2_path = os.path.join(out_dir, f"{base_name}_track2.wav")
            
            torchaudio.save(t1_path, track1, 8000) # whamr operates at 8kHz natively usually, or we can save at 8kHz
            torchaudio.save(t2_path, track2, 8000)
            
            # Calculate RMS energy to see distribution
            rms1 = torch.sqrt(torch.mean(track1**2)).item()
            rms2 = torch.sqrt(torch.mean(track2**2)).item()
            
            f.write(f"Track 1 Energy (RMS): {rms1:.4f}\n")
            f.write(f"Track 2 Energy (RMS): {rms2:.4f}\n")
            
            # Simple conclusion based on energy
            if abs(rms1 - rms2) < 0.05:
                f.write("Result: Energy is balanced between the two tracks.\n")
            else:
                f.write("Result: One track is significantly dominant over the other.\n")

if __name__ == "__main__":
    test_sepformer()
