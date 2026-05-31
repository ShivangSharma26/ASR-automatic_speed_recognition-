# ??? End-to-End Dynamic ASR Pipeline

A robust, local, and dynamic Automatic Speech Recognition (ASR) pipeline designed to effectively process overlapping voices, code-switched languages (Hindi/English), and artifact-heavy audio directly on CPU architectures.

## ?? Architecture Overview

This pipeline orchestrates three state-of-the-art AI components to solve the "Cocktail Party Problem" in speech recognition:
1. **Pyannote Audio 3.1**: Accurately identifies exact speaker timestamps (Speaker Diarization) and tracks "who spoke when".
2. **SpeechBrain SepFormer**: Isolates overlapping voices from mixed audio channels into distinct, clean background tracks.
3. **OpenAI Whisper (Tiny)**: Transcribes the resulting isolated audio tracks natively into text.
