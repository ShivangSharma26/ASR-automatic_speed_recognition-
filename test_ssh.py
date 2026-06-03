import paramiko
import sys

host = "10.99.0.56"
username = "shivang"
password = "Pinaca@Welcome2026"

print(f"Connecting to {host}...")
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=username, password=password, timeout=10)
    print("Connection successful!")
    
    stdin, stdout, stderr = ssh.exec_command("uname -a; lscpu | grep 'Model name'; nvidia-smi -L")
    print("--- Remote System Info ---")
    print(stdout.read().decode())
    
    ssh.close()
except Exception as e:
    print(f"Failed to connect: {e}")
    sys.exit(1)
