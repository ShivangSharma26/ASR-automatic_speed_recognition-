# ??? End-to-End Dynamic ASR Pipeline

A robust, local, and dynamic Automatic Speech Recognition (ASR) pipeline designed to effectively process overlapping voices, code-switched languages (Hindi/English), and artifact-heavy audio directly on CPU architectures.

## ?? Architecture Overview

This pipeline orchestrates three state-of-the-art AI components to solve the "Cocktail Party Problem" in speech recognition:
1. **Pyannote Audio 3.1**: Accurately identifies exact speaker timestamps (Speaker Diarization) and tracks "who spoke when".
2. **SpeechBrain SepFormer**: Isolates overlapping voices from mixed audio channels into distinct, clean background tracks.
3. **OpenAI Whisper (Tiny)**: Transcribes the resulting isolated audio tracks natively into text.

### ?? Core Innovations
- **Chunk-Merging Algorithm**: Mitigates Whisper micro-segment hallucinations by intelligently merging contiguous audio chunks natively inside the router.
- **Dynamic Memory Processing**: Circumvents traditional 	orchcodec and sox_io limitations on Windows by dynamically streaming memory-resident librosa and soundfile tensors into the pipeline without heavy intermediate I/O operations.
- **Automated Batch Processing**: Handles an entire folder of arbitrary MP3/WAV files and outputs them into strictly organized semantic directories.
