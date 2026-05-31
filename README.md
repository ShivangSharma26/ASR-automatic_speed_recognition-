

---

## ??? Usage & Execution

**Running the Batch Pipeline**
Place your .mp3 or .wav files inside a directory (e.g., 	est_audio_batch/) and execute the main processing script:
`ash
python process_batch.py
`

**Pipeline Execution Flow:**
1. Loads audio dynamically via librosa.
2. Pyannote segments the audio into speaker turns.
3. The *Chunk-Merging Algorithm* fuses segments < 1.0s to prevent Whisper degradation.
4. Overlaps are detected and passed to SepFormer for track isolation.
5. Whisper transcribes the clean isolated tracks.
6. Results are saved in highly structured semantic output folders.

---

## ?? Output Artifacts Structure

For each input file (e.g., sample.mp3), the script generates a sample_results/ directory containing:
- **1_speaker_chunks/**: Individual cleanly separated .wav chunks per speaker.
- **2_overlap_before/**: Raw overlapping audio segments before separation.
- **3_overlap_after_sepformer/**: Clean tracks separated by SepFormer.
- **4_final_transcription.json**: The complete mapped JSON transcription of the entire file.
