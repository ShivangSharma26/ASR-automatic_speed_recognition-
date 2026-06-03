cd c:\Users\shiva\Desktop\ASR

# Commit 1
$readme1 = @"
# ??? End-to-End Dynamic ASR Pipeline

A robust, local, and dynamic Automatic Speech Recognition (ASR) pipeline designed to effectively process overlapping voices, code-switched languages (Hindi/English), and artifact-heavy audio directly on CPU architectures.

## ?? Architecture Overview

This pipeline orchestrates three state-of-the-art AI components to solve the "Cocktail Party Problem" in speech recognition:
1. **Pyannote Audio 3.1**: Accurately identifies exact speaker timestamps (Speaker Diarization) and tracks "who spoke when".
2. **SpeechBrain SepFormer**: Isolates overlapping voices from mixed audio channels into distinct, clean background tracks.
3. **OpenAI Whisper (Tiny)**: Transcribes the resulting isolated audio tracks natively into text.
"@
Set-Content -Path "README.md" -Value $readme1 -Encoding UTF8
git add README.md
git commit -m "docs: add comprehensive project overview and architecture components"

# Commit 2
$readme2 = $readme1 + @"


### ?? Core Innovations
- **Chunk-Merging Algorithm**: Mitigates Whisper micro-segment hallucinations by intelligently merging contiguous audio chunks natively inside the router.
- **Dynamic Memory Processing**: Circumvents traditional `torchcodec` and `sox_io` limitations on Windows by dynamically streaming memory-resident `librosa` and `soundfile` tensors into the pipeline without heavy intermediate I/O operations.
- **Automated Batch Processing**: Handles an entire folder of arbitrary MP3/WAV files and outputs them into strictly organized semantic directories.
"@
Set-Content -Path "README.md" -Value $readme2 -Encoding UTF8
git add README.md
git commit -m "docs: detail core algorithmic innovations and memory processing features"

# Commit 3
$readme3 = $readme2 + @"


---

## ?? Directory & File Structure

```text
ASR-automatic_speed_recognition/
¦
+-- ?? README.md                    # Comprehensive project documentation
+-- ?? requirements.txt             # Core Python dependencies
+-- ?? .gitignore                   # Excludes caches, venvs, and sensitive tokens
¦
+-- ?? asr_env/                     # (Ignored) Virtual environment
+-- ?? models_cache/                # (Ignored) Cached huggingface & speechbrain models
+-- ?? test_audio_batch/            # Source directory for batch input audio files (.mp3, .wav)
¦
+-- ?? Pipeline Scripts
¦   +-- process_batch.py            # MAIN ENTRY POINT: End-to-end automated batch processing script
¦   +-- demo_pipeline.py            # Single-file demonstration of the ASR pipeline
¦   +-- audio_router.py             # Core routing logic for diarization tracking
¦   +-- create_realistic_mix.py     # Utility to generate synthetic overlapped audio mixes
¦   +-- generate_report.py          # Generates JSON reporting for pipeline verification
¦
+-- ?? Transcription Modules
¦   +-- hf_whisper.py               # HuggingFace-based Whisper implementation
¦   +-- native_whisper.py           # Native OpenAI Whisper implementation
¦
+-- ??? Utilities & Fixes
    +-- fix_pyannote.py             # Patches Windows torchaudio Pyannote 3.1 attribute errors
    +-- debug_pyannote.py           # Debugging script for diarization steps
    +-- test_dynamic_parser.py      # Validates parsing logic for raw audio waveforms
    +-- test_dynamic_pipeline.py    # Integration test for the end-to-end pipeline
    +-- real_output.py              # Output formatter for transcriptions
```
"@
Set-Content -Path "README.md" -Value $readme3 -Encoding UTF8
git add README.md
git commit -m "docs: document comprehensive directory and script file structure"

# Commit 4
$readme4 = $readme3 + @"


---

## ?? Environment Setup & Installation

**1. Create Virtual Environment**
```bash
python -m venv asr_env
asr_env\Scripts\activate
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure Authentication**
Pyannote Audio requires a HuggingFace API Token to download the diarization models.
- Generate a token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Create a file named `HF TOKEN.TXT` in the root directory and paste your token inside.
"@
Set-Content -Path "README.md" -Value $readme4 -Encoding UTF8
git add README.md
git commit -m "docs: add environment setup, installation and authentication guides"

# Commit 5
$readme5 = $readme4 + @"


---

## ??? Usage & Execution

**Running the Batch Pipeline**
Place your `.mp3` or `.wav` files inside a directory (e.g., `test_audio_batch/`) and execute the main processing script:
```bash
python process_batch.py
```

**Pipeline Execution Flow:**
1. Loads audio dynamically via `librosa`.
2. Pyannote segments the audio into speaker turns.
3. The *Chunk-Merging Algorithm* fuses segments `< 1.0s` to prevent Whisper degradation.
4. Overlaps are detected and passed to SepFormer for track isolation.
5. Whisper transcribes the clean isolated tracks.
6. Results are saved in highly structured semantic output folders.

---

## ?? Output Artifacts Structure

For each input file (e.g., `sample.mp3`), the script generates a `sample_results/` directory containing:
- **`1_speaker_chunks/`**: Individual cleanly separated `.wav` chunks per speaker.
- **`2_overlap_before/`**: Raw overlapping audio segments before separation.
- **`3_overlap_after_sepformer/`**: Clean tracks separated by SepFormer.
- **`4_final_transcription.json`**: The complete mapped JSON transcription of the entire file.
"@
Set-Content -Path "README.md" -Value $readme5 -Encoding UTF8
git add README.md
git commit -m "docs: document usage execution flow and output artifacts structure"

git push origin main
