import os
import requests

try:
    with open('HF TOKEN.TXT', 'r') as f:
        token = f.read().strip()
except:
    token = 'HIDDEN_TOKEN'

API_URL = 'https://api-inference.huggingface.co/models/openai/whisper-tiny'
headers = {'Authorization': f'Bearer {token}'}

def query(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

print('Track 1:')
print(query('demo_results/separated_track_1.wav'))
print('Track 2:')
print(query('demo_results/separated_track_2.wav'))
