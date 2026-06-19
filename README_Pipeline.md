# ASR Pipeline: EEND + Target-Speaker Separation

This folder contains the core scripts for the upgraded ASR pipeline.

## Files Included:

1. **`process_batch.py` (The Master Script)**
   - This is the main engine you should look at. 
   - **Line 37-43:** Initializes the models (Pyannote 3.1 EEND, SpeechBrain SepFormer, OpenAI Whisper).
   - **Line 81:** Runs the global EEND diarization which scans the continuous timeline for overlaps.
   - **Line 115:** The "Router" logic mathematically detects the exact overlapping timestamps.
   - **Line 139:** The Target-Speaker VAD loop passes ONLY the cropped overlap chunks into SepFormer to cleanly split them without acoustic bleed/hallucinations.
   - **Line 150:** The cleanly separated overlapping tracks are fed into Whisper.

2. **`requirements.txt`**
   - Contains the exact dependencies required to run the pipeline. 
   - Key libraries: `pyannote.audio==3.1.1`, `speechbrain`, `openai-whisper`, `torch`.

3. **`HF TOKEN.TXT`**
   - Pyannote 3.1 models are gated. This file contains the HuggingFace authentication token needed to download and run the EEND model weights. The master script automatically reads this file on initialization.

## How it works (Data Flow):
Audio -> Pyannote 3.1 (EEND Segmentation) -> Python Router (Extract Overlaps) -> SepFormer (2-mix track split) -> Whisper (Text Generation) -> JSON Output.
