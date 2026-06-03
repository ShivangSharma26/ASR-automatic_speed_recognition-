import os
import zipfile
import glob

output_zip = r"c:\Users\shiva\Desktop\ASR\ASR_Pipeline_Master_Results.zip"
print(f"Creating {output_zip}...")

folders_to_zip = []
folders_to_zip.extend(glob.glob(r"c:\Users\shiva\Desktop\ASR\*_results"))
folders_to_zip.extend(glob.glob(r"c:\Users\shiva\Desktop\ASR\Remote_Results\*_results"))

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in folders_to_zip:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Ensure the folder structure inside the zip is just the base folder name
                arcname = os.path.relpath(file_path, os.path.dirname(folder))
                zf.write(file_path, arcname)

print(f"Successfully zipped {len(folders_to_zip)} results folders into Master Zip!")