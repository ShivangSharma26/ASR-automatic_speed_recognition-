import zipfile
import os

os.chdir(r"c:\Users\shiva\Desktop\ASR")
with zipfile.ZipFile("remote_deploy_fixed.zip", "w") as z:
    # Add files
    for f in ['process_batch.py', 'hf_whisper.py', 'native_whisper.py', 'audio_router.py', 'requirements.txt']:
        z.write(f, f)
    # Add directory
    for root, dirs, files in os.walk("test_audio_batch_remote"):
        for f in files:
            filepath = os.path.join(root, f)
            # Force forward slash
            arcname = filepath.replace("\\", "/")
            z.write(filepath, arcname)
print("Fixed zip created.")