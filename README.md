# Automatic Speech Recognition (ASR) Pipeline

An end-to-end local dynamic ASR pipeline that effectively processes overlapping, code-switched, and artifact-heavy audio locally on CPU architectures. 

## ?? Architecture
This pipeline orchestrates three major AI components:
1. **Pyannote Audio 3.1**: Identifies exact speaker timestamps and handles speaker diarization.
2. **SpeechBrain SepFormer**: Isolates overlapping voices into distinct clean audio tracks.
3. **OpenAI Whisper (Tiny)**: Transcribes the resulting isolated tracks natively.

## ?? Key Features
- **Chunk-Merging Algorithm**: Mitigates Whisper micro-segment hallucinations by intelligently merging contiguous audio chunks natively inside the router.
- **Automated Batching**: Handles an entire folder of arbitrary MP3/WAV files and outputs them into strictly organized semantic directories (1_speaker_chunks, 2_overlap_before, 3_overlap_after, 4_final_transcription.json).
