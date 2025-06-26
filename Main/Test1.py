import os
import numpy as np
import librosa
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === PHASE 1: Preparation / Extraction ===

def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if i in frame_indices and ret:
            frames.append(frame)
    cap.release()
    return frames

def extract_facial_features(frames):
    # Placeholder for real emotion detection model (e.g., DeepFace/FER)
    return np.mean([np.random.rand(7) for _ in frames], axis=0)  # Example: 7-dim emotion vector

def extract_audio_features(video_path):
    audio_path = "temp.wav"
    import moviepy.editor as mp
    mp.VideoFileClip(video_path).audio.write_audiofile(audio_path, verbose=False, logger=None)
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)  # 13-dim MFCC average

def process_video(video_path):
    frames = extract_frames(video_path)
    facial_vec = extract_facial_features(frames)
    audio_vec = extract_audio_features(video_path)
    return np.concatenate((facial_vec, audio_vec))  # Final shape: (20,)

# === PHASE 2: Model Development ===

def load_dataset(video_folder):
    X, y = [], []
    for file in os.listdir(video_folder):
        if file.endswith(".mp4"):
            label = 1 if "lie" in file else 0  # Simple labeling logic
            vec = process_video(os.path.join(video_folder, file))
            X.append(vec)
            y.append(label)
    return np.array(X), np.array(y)

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# === PHASE 3: Evaluation ===

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Evaluation Report:")
    print(classification_report(y_test, preds))

# === PIPELINE EXECUTION ===

if __name__ == "__main__":
    video_dir = "videos"  # Folder with .mp4 files labeled in filename
    X, y = load_dataset(video_dir)

    # Split dataset 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
