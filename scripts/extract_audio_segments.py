import os
import pandas as pd
import subprocess

# === CONFIG ===
CSV_PATH = os.path.join('csv', 'utterances.csv')
VIDEO_DIR = 'VideoClips'
AUDIO_DIR = 'audio_clips'

os.makedirs(AUDIO_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[df['veracity'].isin(['lie', 'truthful']) & df['video_file'].notna()]

for _, row in df.iterrows():
    video_path = os.path.join(VIDEO_DIR, row['video_file'])

    if not os.path.exists(video_path):
        print(f"Missing video: {video_path}")
        continue

    start_sec = row['start_time_ms'] / 1000
    duration_sec = (row['end_time_ms'] - row['start_time_ms']) / 1000

    output_filename = f"{row['video_file'].replace('.mp4','')}_{int(row['start_time_ms'])}_{int(row['end_time_ms'])}.wav"
    output_path = os.path.join(AUDIO_DIR, output_filename)

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-ar", "22050",           # Downsample to reduce size (optional)
        "-ac", "1",               # Mono channel (optional)
        "-loglevel", "quiet",     # Suppress verbose logs
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f" SAVED: {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {video_path}: {e}")
