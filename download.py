"""SFTP script to securely download processing results."""
import paramiko
import sys

host = "10.99.0.56"
username = "shivang"
password = "Pinaca@Welcome2026"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=username, password=password, timeout=10)

print("Zipping results...")
cmd = "cd ~/ASR_Pipeline/test_audio_batch_remote && zip -r ../ASR_Pipeline_Results_Final.zip *_results"
stdin, stdout, stderr = ssh.exec_command(cmd)
status = stdout.channel.recv_exit_status()
if status != 0:
    print("Zip failed:", stderr.read().decode())
else:
    print("Zipped successfully.")

print("Downloading results...")
sftp = ssh.open_sftp()
sftp.get("/home/shivang/ASR_Pipeline/ASR_Pipeline_Results_Final.zip", r"c:\Users\shiva\Desktop\ASR\ASR_Pipeline_Results_Final.zip")
sftp.close()
ssh.close()
print("Download complete!")