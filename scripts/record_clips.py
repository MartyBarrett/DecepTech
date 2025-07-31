import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageStat
from pydub import AudioSegment, silence
import cv2
import threading
import pyaudio
import wave
import random
import speech_recognition as sr
import os
import glob
import csv
from datetime import datetime
import subprocess

# === Config ===
AUDIO_FILENAME = "CroppedAudio/recorded_audio.wav"
TRANSCRIPT_FILENAME = "transcripts/transcript.txt"
VIDEO_OUTPUT_DIR = "CroppedClips"
CSV_LOG_PATH = "recordings_log.csv"
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAME_RATE = 20.0

# Ensure output folders exist
os.makedirs("CroppedAudio", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Create CSV log with headers if it doesn't exist
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Audio Filename', 'Video Filename', 'Outcome'])

# Image selection
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
all_files = glob.glob("RandomImages/*")
IMAGES = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]

class TruthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DecepTech")

        # Load and set background image
        self.bg_image_path = "background.png"  # Ensure this file exists
        bg_img_pil = Image.open(self.bg_image_path).resize((1600, 1500), Image.Resampling.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(bg_img_pil)

        # Place background label
        self.bg_label = tk.Label(self.root, image=self.bg_image)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Load and process logo image
        self.logo_path = "decepticon_logo.png"
        self.logo_img_pil = Image.open(self.logo_path).resize((300, 300), Image.Resampling.LANCZOS)
        r, g, b = 14, 28, 43
        self.BG_COLOR = f'#{r:02x}{g:02x}{b:02x}'
        self.root.configure(bg=self.BG_COLOR)
        self.logo_img = ImageTk.PhotoImage(self.logo_img_pil)

        # Recording state
        self.recording = False
        self.frames = []
        self.video_frames = []

        # Label state for Outcome
        self.label_value = None

        # Bind keys 1 and 0 to set outcome
        self.root.bind('1', self.on_key_1)
        self.root.bind('0', self.on_key_0)

        # Top Frame for Logo
        self.logo_label = tk.Label(self.root, image=self.logo_img, bg=self.BG_COLOR)
        self.logo_label.place(x=10, y=10)

        # Main Frame for UI
        self.main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        self.main_frame.pack(pady=20, expand=True)

        # Frame for holding image display
        self.image_frame = tk.Frame(self.main_frame, bg=self.BG_COLOR)
        self.image_frame.pack(pady=10, expand=True)

        # Camera location
        self.camera_label = tk.Label(self.root, bg="black")
        self.camera_label.place(relx=1.0, y=10, anchor='ne')

        self.init_ui()

    def init_ui(self):
        self.start_button = tk.Button(
            self.root, text="Start Recording",
            command=self.start_recording,
            bg="#4caf50", font="bold"
        )
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(
            self.root, text="Stop Recording",
            command=self.stop_recording,
            state=tk.DISABLED, bg="#f44336",
            fg="white", font="bold"
        )
        self.stop_button.pack(pady=10)

        self.spinner = ttk.Label(self.root, text="")
        self.spinner.pack()

        self.new_image_button = tk.Button(
            self.root, text="New Random Image",
            command=self.show_random_image,
            bg="#2196f3", fg="white",
            font=("Arial", 12)
        )
        self.new_image_button.pack(pady=5)

        self.transcript_box = tk.Text(
            self.root, height=6, width=50, font=("Courier", 10)
        )
        self.transcript_box.pack(pady=10)

        self.image_label = tk.Label(self.root, bg="#f0f2f5")
        self.image_label.pack(pady=10)

        self.show_random_image()

    def show_random_image(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        if not IMAGES:
            label = tk.Label(self.image_frame, text="No images found!", bg=self.BG_COLOR)
            label.pack()
            return

        image_path = random.choice(IMAGES)
        img = Image.open(image_path)
        fixed_size = (400, 400)
        img = img.resize(fixed_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        self.image_label = tk.Label(self.image_frame, image=photo, bg=self.BG_COLOR)
        self.image_label.image = photo
        self.image_label.pack(expand=True)

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.video_frames = []
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.spinner.config(text="Recording... 🎙️")

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT, channels=CHANNELS,
            rate=SAMPLE_RATE, input=True,
            frames_per_buffer=CHUNK
        )
        self.cap = cv2.VideoCapture(0)

        threading.Thread(target=self.record_audio, daemon=True).start()
        threading.Thread(target=self.record_video, daemon=True).start()

    def record_audio(self):
        while self.recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

    def record_video(self):
        def update_frame():
            if self.recording:
                ret, frame = self.cap.read()
                if ret:
                    self.video_frames.append(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    img_pil = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img_pil)
                    self.camera_label.imgtk = imgtk
                    self.camera_label.config(image=imgtk)
                self.root.after(50, update_frame)
        update_frame()

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.spinner.config(text="Transcribing... ⏳")

        # Stop audio
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save audio file
        audio_path = os.path.join("CroppedAudio", f"audio_{timestamp}.wav")
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))

        # Save video file
        video_path = ""
        if self.video_frames:
            video_path = os.path.join(VIDEO_OUTPUT_DIR, f"clip_{timestamp}.avi")
            height, width, _ = self.video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (width, height))
            for frame in self.video_frames:
                out.write(frame)
            out.release()

        # Append to CSV with label
        with open(CSV_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            outcome = self.label_value if self.label_value is not None else ''
            writer.writerow([timestamp, audio_path, video_path, outcome])
            self.label_value = None  # reset for next recording

        # Clear camera display
        self.camera_label.config(image='')

        # Run model inference
        self.run_prediction_inference(audio_path, video_path)

    def run_prediction_inference(self, audio_path, video_path):
        python_executable = r"C:\Users\connellj2\OneDrive - Wentworth Institute of Technology\Documents\DecepTech\decep_env\Scripts\python.exe"
        inference_script = r"C:\Users\connellj2\OneDrive - Wentworth Institute of Technology\Documents\DecepTech\decep_env\Scripts\predictor.py"

        try:
            result = subprocess.run(
<<<<<<< HEAD
                [python_executable, inference_script, "--video", video_path, "--audio", audio_path, "--visualize"],
                capture_output=True,
                text=True,
                check=True
=======
                [python_executable, inference_script, "--video", video_path, "--audio", audio_path],
                capture_output=True, text=True, check=True
>>>>>>> ec3a49f476891e2a41d0209b0c0a1b88a499b2fc
            )
            output = result.stdout.strip()
            self.root.after(0, lambda: self.transcript_box.insert(
                tk.END, f"\n🧠 Model Verdict: {output}\n"))
        except subprocess.CalledProcessError as e:
            err = e.stderr or str(e)
            self.root.after(0, lambda: self.transcript_box.insert(
                tk.END, f"❌ Inference Error:\n{err}"))

    def transcribe_audio(self, audio_path, transcript_path):
        sound = AudioSegment.from_wav(audio_path)
        silence_ranges = silence.detect_silence(sound, min_silence_len=1300, silence_thresh=-40)
        silence_ranges = [(start/1000.0, stop/1000.0) for start, stop in silence_ranges]

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
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
                self.root.after(0, lambda: self.display_transcript(final_text))
            except sr.UnknownValueError:
                self.display_transcript("❌ Could not understand the audio.")
            except sr.RequestError as e:
                self.display_transcript(f"❌ API Error: {e}")

    def display_transcript(self, text):
        self.spinner.config(text="✅ Done")
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert(tk.END, text)

    def on_key_1(self, event):
        """User pressed 1 → mark Outcome=1"""
        self.label_value = 1
        print("[Outcome] set to 1")

    def on_key_0(self, event):
        """User pressed 0 → mark Outcome=0"""
        self.label_value = 0
        print("[Outcome] set to 0")


if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = TruthApp(root)
    root.mainloop()
