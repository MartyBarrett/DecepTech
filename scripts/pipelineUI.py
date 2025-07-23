#!/usr/bin/env python3
"""
Standalone UI prediction pipeline:
- Automatically load all videos, audios, and transcripts
- Extract face-bbox, audio MFCC, and text TF-IDF features
- Load a full sklearn pipeline or fallback to vectorizer+classifier
- Display lie probability for each clip in the UI and allow manual refresh
"""
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import cv2
import librosa

# Predeclare REPO_ROOT for type checking (actual value set below)
REPO_ROOT = ''  # type: ignore

# Extraction functions: try loading from VideoExtract.ipynb, else define locally
try:
    import import_ipynb  # pip install import-ipynb
    import VideoExtract  # type: ignore[reportMissingImports]
    extract_frames = VideoExtract.extract_frames
    extract_facial_features = VideoExtract.extract_facial_features
    print("✅ Loaded extraction functions from VideoExtract.ipynb")
except Exception:
    print("⚠️ Could not import VideoExtract; using local definitions")
    # Local extraction definitions
    def extract_frames(video_path, num_frames=20):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def extract_facial_features(frames):
        feats = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            feats.append(np.array([x, y, w, h], dtype=float))
        return np.mean(feats, axis=0) if feats else np.zeros(4)

# Resolve paths
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Model & Vectorizer — look in project root then script directory
PIPELINE_PATHS   = [
    os.path.join(REPO_ROOT, 'stacked_model.pkl'),
    os.path.join(SCRIPT_DIR, 'stacked_model.pkl')
]
VECTORIZER_PATHS = [
    os.path.join(REPO_ROOT, 'rf_face_model.pkl'),
    os.path.join(SCRIPT_DIR, 'rf_face_model.pkl')
]
CLASSIFIER_PATHS = [
    os.path.join(REPO_ROOT, 'RandomForest.pkl'),
    os.path.join(SCRIPT_DIR, 'RandomForest.pkl')
]

# Load pipeline or fallback
pipeline = None
vectorizer = None
classifier = None
# Try full pipeline files
for path in PIPELINE_PATHS:
    if os.path.exists(path):
        try:
            pipeline = joblib.load(path)
            print(f"Loaded full pipeline from {path}")
            break
        except Exception as e:
            print(f"Failed to load pipeline at {path}: {e}")

# Fallback to separate vectorizer+classifier
if pipeline is None:
    for v_path, c_path in zip(VECTORIZER_PATHS, CLASSIFIER_PATHS):
        if os.path.exists(v_path) and os.path.exists(c_path):
            try:
                vectorizer = joblib.load(v_path)
                classifier = joblib.load(c_path)
                print(f"Loaded vectorizer from {v_path} and classifier from {c_path}")
                break
            except Exception as e:
                print(f"Failed to load components at {v_path} and {c_path}: {e}")

# If still missing
if pipeline is None and (vectorizer is None or classifier is None):
    raise RuntimeError(
        f"Could not load any model. Tried pipelines: {PIPELINE_PATHS}, "
        f"vectorizers: {VECTORIZER_PATHS}, classifiers: {CLASSIFIER_PATHS}"
    )

# Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


def extract_text_features(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    transcript_dir = os.path.join(REPO_ROOT, 'transcripts')
    # find transcript matching base
    for fname in os.listdir(transcript_dir):
        if fname.startswith(base) and fname.lower().endswith('.txt'):
            text = open(os.path.join(transcript_dir, fname), 'r', encoding='utf-8').read()
            return vectorizer.transform([text]).toarray()[0]
    raise FileNotFoundError(f"No transcript for {base}")


def get_latest_files(directory, exts):
    """Return sorted list of full paths to files matching exts in directory"""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.lower().endswith(exts)]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return [os.path.join(directory, f) for f in files]


class PipelineApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        master.title("DecepTech: Video+Audio+Text Pipeline")
        self.pack(padx=10, pady=10)

        self.refresh_btn = tk.Button(self, text="Refresh Scores", command=self.update_scores)
        self.refresh_btn.pack(fill='x', pady=5)

        self.result_label = tk.Label(self, text="Lie probabilities:", font=('Arial', 12), justify='left')
        self.result_label.pack(pady=10)

                # Extraction functions are already defined above; no further import needed
        self.update_scores()

    def compute_scores(self):
        vids = get_latest_files(os.path.join(REPO_ROOT, 'CroppedClips'), ('.avi', '.mp4', '.mov'))
        if not vids:
            raise FileNotFoundError("No video files in CroppedClips")

        results = []
        for vid in vids:
            # features
            frames = extract_frames(vid)
            vf = extract_facial_features(frames)
            # audio
            af = librosa.feature.mfcc(
                y=librosa.load(os.path.join(REPO_ROOT, 'CroppedAudio', os.path.basename(vid).replace('.avi','.wav')), sr=16000)[0],
                sr=16000, n_mfcc=16
            ).mean(axis=1)
            # text
            tf = extract_text_features(vid)
            feat = np.concatenate([vf, af, tf])
            # predict
            if pipeline:
                p = pipeline.predict_proba([feat])[0][1]
            else:
                p = classifier.predict_proba([feat])[0][1]
            results.append((os.path.basename(vid), p))
        return results

    def update_scores(self):
        try:
            scores = self.compute_scores()
            lines = [f"{name}: {prob:.2%}" for name, prob in scores]
            display_text = "\n".join(lines)
            self.result_label.config(text=display_text)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_label.config(text="Lie probabilities: N/A")


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("400x300")
    app = PipelineApp(master=root)
    app.mainloop()
