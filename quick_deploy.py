import paramiko
import sys
import time
import os

sys.stdout.reconfigure(encoding="utf-8")

host = "10.99.0.56"
username = "shivang"
password = "Pinaca@Welcome2026"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("Connecting...")
ssh.connect(host, username=username, password=password, timeout=10)

print("Uploading updated process_batch.py...")
sftp = ssh.open_sftp()
sftp.put(r"c:\Users\shiva\Desktop\ASR\process_batch.py", "process_batch_tmp.py")
sftp.close()

cmds = [
    "mv process_batch_tmp.py ~/ASR_Pipeline/process_batch.py",
    "cd ~/ASR_Pipeline && sed -i 's/test_audio_batch_2/test_audio_batch_remote/g' process_batch.py",
    "cd ~/ASR_Pipeline && sed -i \"s/Pipeline.from_pretrained('pyannote\/speaker-diarization-3.1')/Pipeline.from_pretrained('pyannote\/speaker-diarization-3.1'); pyannote_pipe.to(torch.device('cuda'))/g\" process_batch.py",
    "cd ~/ASR_Pipeline && source asr_env/bin/activate && export HF_TOKEN=YOUR_HF_TOKEN_HERE && python process_batch.py",
    "cd ~/ASR_Pipeline && zip -r ASR_Pipeline_Results_Final.zip *_results"
]

for cmd in cmds:
    print(f"Running: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    while True:
        line = stdout.readline()
        if not line:
            break
        try:
            print(line, end="")
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), end="")
        sys.stdout.flush()
        
    err = stderr.read().decode("utf-8", "replace")
    if err:
        print("STDERR:", err)
    
    status = stdout.channel.recv_exit_status()
    if status != 0:
        print(f"Failed with {status}")
        sys.exit(1)

print("Downloading results...")
sftp = ssh.open_sftp()
sftp.get("/home/shivang/ASR_Pipeline/ASR_Pipeline_Results_Final.zip", r"c:\Users\shiva\Desktop\ASR\ASR_Pipeline_Results_Final.zip")
sftp.close()
ssh.close()
print("Done!")