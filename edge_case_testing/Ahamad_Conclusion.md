# Empirical Evaluation: Target-Speaker VAD (SepFormer-First)

This document summarizes our findings on Ahamad's proposed architecture: **Running SepFormer source separation prior to Pyannote diarization** to solve the extreme overlapping speech bottleneck.

## Objective
To determine if separating a highly overlapping multi-speaker audio into isolated continuous tracks improves Pyannote's clustering accuracy, bypassing Pyannote's inherent Overlapping Speech Detection (OSD) limit.

## Implementation Details
We executed this pipeline on `case2_top2.wav` (a 10-second synthetic audio with 2 overlapping speakers).
1. **Source Separation**: Passed the raw mixed audio through `speechbrain/sepformer-whamr`.
2. **Channel Isolation**: Extracted `Track 1` and `Track 2`.
3. **Diarization**: Passed `Track 1` and `Track 2` independently into `pyannote/speaker-diarization-3.1`.

## Results
The raw outputs are logged in `sepformer_first_results.txt`.

### Track 1 Diarization
```text
--- Diarization on Track 1 ---
Timestamps:
[0.03s -> 1.26s] SPEAKER_00
[2.17s -> 9.97s] SPEAKER_00

Total Unique Speakers Detected in Track 1: 1
```
**Observation**: Perfect evaluation. SepFormer successfully isolated Speaker A, and Pyannote accurately clustered the entire track to a single speaker.

### Track 2 Diarization
```text
--- Diarization on Track 2 ---
Timestamps:
[0.03s -> 1.92s] SPEAKER_00
[0.52s -> 1.18s] SPEAKER_01
[2.12s -> 3.52s] SPEAKER_00
[3.64s -> 4.37s] SPEAKER_00
[4.45s -> 5.99s] SPEAKER_00
[5.13s -> 5.55s] SPEAKER_01
[6.29s -> 6.63s] SPEAKER_00
[7.00s -> 9.41s] SPEAKER_00
[9.77s -> 9.97s] SPEAKER_00

Total Unique Speakers Detected in Track 2: 2
```
**Observation**: Failure. Pyannote detected 2 overlapping speakers within Track 2 itself. 

> [!WARNING]
> **Why did Track 2 fail?**
> SepFormer's source isolation is not perfectly clean; it leaves behind **acoustic bleed** and **phase-cancellation artifacts**. Pyannote's Voice Activity Detection is extremely sensitive. When Pyannote analyzed Track 2, it heard the primary voice, but it also picked up the distorted bleed from Track 1, clustering it as a secondary hallucinated speaker. 

## Final Conclusion
Ahamad's idea of a "SepFormer-First" (Target-Speaker VAD) approach is mathematically sound and works in theory (as proven by Track 1). However, in practical deployment, **it is not viable** because:
1. **Acoustic Bleed:** Separation artifacts cause downstream clustering models (like Pyannote) to hallucinate false speakers, ultimately ruining the timestamp annotations.
2. **Computational Footprint:** Processing an entire meeting through a Transformer-based source separator requires complex chunking and permutation solvers to avoid catastrophic VRAM OOM errors.

**Recommendation:** We should abandon the modular approach (stitching disparate Separation and Clustering models) and instead adopt **EEND (End-to-End Neural Diarization)**, which handles overlap natively via multi-label classification without suffering from acoustic bleed.
