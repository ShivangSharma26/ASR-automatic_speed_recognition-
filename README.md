# ASR Pipeline with Source Separation

This repository contains a state-of-the-art Automatic Speech Recognition (ASR) pipeline built to handle highly complex, overlapping, and code-switched audio.

## Architecture
- **Diarization:** Pyannote Audio
- **Source Separation:** SpeechBrain SepFormer
- **Inference:** OpenAI Whisper
- **Acceleration:** Optimized for NVIDIA A100 GPU clusters

## Features
- Eliminates multi-speaker hallucinations in Whisper.
- Dynamically isolates overlapping audio chunks.
- Provides unified JSON transcriptions and separated audio tracks.
