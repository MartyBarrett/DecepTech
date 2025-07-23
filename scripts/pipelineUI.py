#!/usr/bin/env python3
"""
Standalone UI prediction pipeline without dlib, transcript, or moviepy/ffmpeg:
- Load video file via UI
- Extract face-bbox and audio MFCC features (using pre-extracted WAV)
- Load scikit-learn model trained on these features
- Display lie probability in the UI
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import joblib
import cv2
import librosa

# Resolve paths
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'RandomForest.pkl')  # ensure this file exists here

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


def extract_frames(video_path, num_frames=20):
    """Grab `num_frames` evenly spaced frames from the video."""
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
    """Detect face bounding boxes and return average [x, y, w, h]."""
    feats = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        feats.append(np.array([x, y, w, h], dtype=float))
    return np.mean(feats, axis=0) if feats else np.zeros(4)


def extract_audio_features(video_path, n_mfcc=16, sr=16000):
    """Extract MFCC features from pre-extracted WAV file in CroppedAudio folder."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(REPO_ROOT, 'CroppedAudio', f"{base}.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

class PipelineApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("DecepTech: Video+Audio Pipeline")
        self.pack(padx=10, pady=10)
        self.video_path = None

        tk.Button(self, text="Load Video", command=self.load_video).pack(fill='x', pady=5)
        self.predict_btn = tk.Button(self, text="Predict", state='disabled', command=self.on_predict)
        self.predict_btn.pack(fill='x', pady=5)
        self.result_label = tk.Label(self, text="Lie probability: N/A", font=('Arial', 14))
        self.result_label.pack(pady=10)

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.avi;*.mp4;*.mov")]
        )
        if path:
            self.video_path = path
            self.predict_btn.config(state='normal')

    def on_predict(self):
        if not self.video_path:
            messagebox.showwarning("Missing Input", "Please load a video first.")
            return

        frames = extract_frames(self.video_path)
        vid_feat = extract_facial_features(frames)
        aud_feat = extract_audio_features(self.video_path)
        features = np.concatenate([vid_feat, aud_feat])

        try:
            proba = model.predict_proba(features.reshape(1, -1))[0][1]
            self.result_label.config(text=f"Prediction Accuracy: {proba:.2%}")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"{e}")

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("400x200")
    app = PipelineApp(master=root)
    app.mainloop()
