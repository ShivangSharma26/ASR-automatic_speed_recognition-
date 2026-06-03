import os
import json
import glob

results_dir = r"c:\Users\shiva\Desktop\ASR\Remote_Results"
json_files = glob.glob(os.path.join(results_dir, "**", "*_results", "4_final_transcription.json"), recursive=True)

if not json_files:
    json_files = glob.glob(os.path.join(results_dir, "*_results", "4_final_transcription.json"))

total_files = len(json_files)
total_overlaps = 0
examples = []

for jf in json_files:
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            overlaps = data.get("overlaps_before", {})
            total_overlaps += len(overlaps)
            
            for k, before_text in overlaps.items():
                track1_key = k.replace(".wav", "_track1.wav")
                track2_key = k.replace(".wav", "_track2.wav")
                after = data.get("overlaps_after", {})
                t1_text = after.get(track1_key, "")
                t2_text = after.get(track2_key, "")
                
                if len(before_text) > 1 and (len(t1_text) > 1 or len(t2_text) > 1):
                    examples.append({
                        "file": os.path.basename(os.path.dirname(jf)),
                        "overlap": k,
                        "before": before_text,
                        "track1": t1_text,
                        "track2": t2_text
                    })
    except Exception as e:
        pass

with open(r"c:\Users\shiva\Desktop\ASR\analysis_data.md", "w", encoding="utf-8") as out:
    out.write(f"Analyzed {total_files} processed sessions.\n")
    out.write(f"Total overlapping segments identified: {total_overlaps}\n\n")
    out.write("## Qualitative Examples of Source Separation\n")
    for ex in examples:
        out.write(f"**File:** {ex['file']}  \n")
        out.write(f"- **Before (Mixed):** {ex['before']}  \n")
        out.write(f"- **Track 1 (Clean):** {ex['track1']}  \n")
        out.write(f"- **Track 2 (Clean):** {ex['track2']}  \n\n")