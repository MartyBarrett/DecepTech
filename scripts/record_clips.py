import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pydub import AudioSegment, silence
import cv2
import threading
import pyaudio
import wave
import random
import speech_recognition as sr
import os
import glob
from datetime import datetime

# === Config ===
AUDIO_FILENAME = "audio_clips/recorded_audio.wav"
TRANSCRIPT_FILENAME = "transcripts/transcript.txt"
VIDEO_OUTPUT_DIR = "CroppedClips"
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAME_RATE = 20.0

# Ensure output folders exist
os.makedirs("audio_clips", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Image selection
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
all_files = glob.glob("RandomImages/*")
IMAGES = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]

class TruthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Truth Game")

        self.recording = False
        self.frames = []
        self.video_frames = []
        self.image_label = None
        self.transcript_box = None
        self.spinner = None

        self.init_ui()

    def init_ui(self):
        self.start_button = tk.Button(self.root, text="Start Recording", command=self.start_recording, bg="#4caf50", font = "bold")
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED, bg="#f44336", fg="white", font = "bold")
        self.stop_button.pack(pady=10)

        self.spinner = ttk.Label(self.root, text="")
        self.spinner.pack()

        self.new_image_button = tk.Button(self.root, text="New Random Image", command=self.show_random_image,   bg="#2196f3", fg="white", font=("Arial", 12))
        self.new_image_button.pack(pady=5)

        self.transcript_box = tk.Text(self.root, height=6, width=50,  font=("Courier", 10))
        self.transcript_box.pack(pady=10)

        self.image_label = tk.Label(self.root,  bg="#f0f2f5")
        self.image_label.pack(pady=10)

        self.show_random_image()

    def show_random_image(self):
        if not IMAGES:
            self.image_label.config(text="No dog images found!")
            return
        image_path = random.choice(IMAGES)
        img = Image.open(image_path)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.video_frames = []
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.spinner.config(text="Recording... 🎙️")

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                                      input=True, frames_per_buffer=CHUNK)

        self.cap = cv2.VideoCapture(0)

        threading.Thread(target=self.record_audio).start()
        threading.Thread(target=self.record_video).start()

    def record_audio(self):
        while self.recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

    def record_video(self):
        while self.recording:
            ret, frame = self.cap.read()
            if ret:
                self.video_frames.append(frame)
            cv2.imshow('Recording - Press Q to stop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_recording()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.spinner.config(text="Transcribing... ⏳")

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        with wave.open(AUDIO_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))

        self.cap.release()
        cv2.destroyAllWindows()

        if self.video_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(VIDEO_OUTPUT_DIR, f"clip_{timestamp}.avi")
            height, width, _ = self.video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (width, height))
            for frame in self.video_frames:
                out.write(frame)
            out.release()

        threading.Thread(target=self.transcribe_audio).start()

    def transcribe_audio(self):
        sound = AudioSegment.from_wav(AUDIO_FILENAME)
        silence_ranges = silence.detect_silence(sound, min_silence_len=1000, silence_thresh=-40)
        silence_ranges = [(start / 1000.0, stop / 1000.0) for start, stop in silence_ranges]

        recognizer = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILENAME) as source:
            audio_data = recognizer.record(source)
            try:
                raw_text = recognizer.recognize_google(audio_data)
                words = raw_text.split()
                pause_count = len(silence_ranges)
                interval = len(words) // (pause_count + 1) if pause_count else len(words)
                for i in range(pause_count):
                    insert_at = (i + 1) * interval
                    filler = random.choice(["uh", "um"])
                    words.insert(insert_at, filler)
                final_text = ' '.join(words)
                with open(TRANSCRIPT_FILENAME, 'w') as f:
                    f.write(final_text)
                self.display_transcript(final_text)
            except sr.UnknownValueError:
                self.display_transcript("❌ Could not understand the audio.")
            except sr.RequestError as e:
                self.display_transcript(f"❌ API Error: {e}")

    def display_transcript(self, text):
        self.spinner.config(text="✅ Done")
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TruthApp(root)
    root.mainloop()
