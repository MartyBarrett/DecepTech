# DecepTech

**DecepTech** is a multimodal deception detection system that analyzes short video segments to determine whether a speaker is being truthful or deceptive. This project is based on annotated `.eaf` files and associated video clips from the Bag of Lies dataset.

# Project Structure 
├── annotations/ # Contains ELAN .eaf annotation files
├── csv/ # Output directory for parsed utterances (utterances.csv)
├── scripts/ # Utility scripts for parsing and inspection
│ ├── parse_eaf_to_csv.py # Parses .eaf files into CSV
│ └── list_tiers.py # Lists available tiers in EAF files
├── VideoClips/ # MP4 video clips matching each .eaf file
├── audio_clips/ # (to be created) Extracted audio segments for ML