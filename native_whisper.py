import warnings
warnings.filterwarnings('ignore')

import whisper
import json
import codecs

model = whisper.load_model('tiny', device='cpu')

res1 = model.transcribe('demo_results/separated_track_1.wav')
res2 = model.transcribe('demo_results/separated_track_2.wav')

with codecs.open('demo_results/whisper_results.json', 'w', encoding='utf-8') as f:
    json.dump({'Track 1': res1['text'].strip(), 'Track 2': res2['text'].strip()}, f, ensure_ascii=False)

print('Whisper transcription saved to demo_results/whisper_results.json')
