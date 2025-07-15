import os
import pandas as pd
import pympi

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level above /scripts
ANNOTATION_DIR = os.path.join(BASE_DIR, 'annotations')
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, 'csv', 'utterances.csv')

utterances = []

# Loop through all .eaf files in the annotations folder
for filename in os.listdir(ANNOTATION_DIR):
    if filename.endswith('.eaf'):
        eaf_path = os.path.join(ANNOTATION_DIR, filename)
        eaf = pympi.Elan.Eaf(eaf_path)

        # Adjust tier names if necessary
        try:
            veracity_tier = eaf.get_annotation_data_for_tier("Veracity_description")
            transcript_tier = eaf.get_annotation_data_for_tier("Guest_verbal")
        except KeyError as e:
            print(f"Tier missing in {filename}: {e}")
            continue

        for start, end, veracity in veracity_tier:
            transcript = ""
            for s2, e2, t2 in transcript_tier:
                if s2 == start and e2 == end:
                    transcript = t2.strip()
                    break

            # Extract file index to match with video filename
            eaf_index = filename.split('.')[0].split('BoL')[0].strip()
            if eaf_index.isdigit():
                video_file = f"{int(eaf_index):02d}_BoL.mp4"
            else:
                video_file = "UNKNOWN"

            utterances.append({
                'eaf_file': filename,
                'video_file': video_file,
                'start_time_ms': start,
                'end_time_ms': end,
                'veracity': veracity,
                'transcript': transcript
            })

# Convert to DataFrame and save as CSV
os.makedirs(os.path.join(BASE_DIR, 'csv'), exist_ok=True)
df = pd.DataFrame(utterances)
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Completed: CSV file saved to {CSV_OUTPUT_PATH}")
