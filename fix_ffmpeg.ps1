Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile "c:\Users\shiva\Desktop\ASR\ffmpeg.zip"
Expand-Archive -Path "c:\Users\shiva\Desktop\ASR\ffmpeg.zip" -DestinationPath "c:\Users\shiva\Desktop\ASR\ffmpeg_dir" -Force
$ffmpegPath = Get-ChildItem -Path "c:\Users\shiva\Desktop\ASR\ffmpeg_dir" -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
Copy-Item $ffmpegPath.FullName "c:\Users\shiva\Desktop\ASR\ffmpeg.exe" -Force

(Get-Content c:\Users\shiva\Desktop\ASR\process_batch.py) -replace "import sys", "import sys`nimport os`nos.environ['PATH'] += os.pathsep + r'c:\Users\shiva\Desktop\ASR'" | Set-Content c:\Users\shiva\Desktop\ASR\process_batch.py

c:\Users\shiva\Desktop\ASR\asr_env\Scripts\python c:\Users\shiva\Desktop\ASR\process_batch.py
