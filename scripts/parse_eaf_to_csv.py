import os
import pympi
import pandas as pd

# Set your local DecepTech project path here
BASE_DIR = r'C:\Users\barrettm5\OneDrive - Wentworth Institute of Technology\Desktop\DecepTech_Structured'
ANNOTATION_DIR = os.path.join(BASE_DIR, 'annotations')
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, 'csv', 'utterances.csv')

utterances = []

# Loop through all .eaf files
for fname in os.listdir(ANNOTATION_DIR):
    if not fname.endswith('.eaf'):
        continue

    eaf_path = os.path.join(ANNOTATION_DIR, fname)
    eaf = pympi.Elan.Eaf(eaf_path)

    # Veracity and transcript tiers (ensure your tier names match exactly)
    veracity = eaf.get_annotation_data_for_tier("Veracity")
    transcript = eaf.get_annotation_data_for_tier("Transcript")

    for start, end, label in veracity:
        text = next((t for s, e, t in transcript if s == start and e == end), "")
        video_index = fname.split('.')[0]
        video_filename = f"{int(video_index):02d}_BoL.mp4"

        utterances.append({
            'eaf_file': fname,
            'video_file': video_filename,
            'start_time_ms': start,
            'end_time_ms': end,
            'veracity': label,
            'transcript': text.strip()
        })

# Save to CSV
df = pd.DataFrame(utterances)
os.makedirs(os.path.join(BASE_DIR, 'csv'), exist_ok=True)
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Saved to: {CSV_OUTPUT_PATH}")
