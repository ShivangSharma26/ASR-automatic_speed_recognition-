cd c:\Users\shiva\Desktop\ASR

Remove-Item .git -Recurse -Force -ErrorAction SilentlyContinue

git init
git checkout -b main

# Commit 1
git add .gitignore README.md requirements.txt
git commit -m "chore: initial commit, setup project structure and dependencies"

# Commit 2
git add server.py
git commit -m "feat: initialize base inference API server architecture"

# Commit 3
git add audio_router.py
git commit -m "feat: add initial audio routing for diarization tracking"

# Commit 4
git add hf_whisper.py native_whisper.py
git commit -m "feat: implement local whisper transcription modules"

# Commit 5
git add fix_pyannote.py debug_pyannote.py
git commit -m "fix: resolve Windows torchaudio Pyannote 3.1 attribute errors"

# Commit 6
git add test_dynamic_parser.py
git commit -m "test: validate dynamic parsing logic for raw audio waveform arrays"

# Commit 7
git add test_dynamic_pipeline.py demo_pipeline.py real_output.py
git commit -m "feat: build full end-to-end Pyannote and Whisper pipeline"

# Commit 8
git add create_realistic_mix.py
git commit -m "chore: add scripts to generate synthetic overlapped audio mixes"

# Commit 9
git add generate_report.py
git commit -m "docs: implement json reporting for verification and analysis"

# Commit 10
git add process_batch.py
git commit -m "feat: implement batch processing script with robust folder management"

# Commit 11
git add process_batch.py
git commit -m "feat: integrate chunk-merging algorithm to mitigate Whisper micro-segment hallucinations"

# Push
git remote add origin https://github.com/ShivangSharma26/ASR-automatic_speed_recognition-.git
git push -u origin main --force
