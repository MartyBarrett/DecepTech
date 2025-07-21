import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pydub import AudioSegment, silence
import threading
import pyaudio
import wave
import random
import speech_recognition as sr
import glob

# Config
AUDIO_FILENAME = "audio_clips/recorded_audio.wav"
TRANSCRIPT_FILENAME = "transcripts/transcript.txt"
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

#Image selection
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
all_files = glob.glob("RandomImages/*")
IMAGES = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]

class DogTruthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog Truth Game 🐶")

        self.recording = False
        self.frames = []
        self.image_label = None
        self.transcript_box = None
        self.spinner = None

        self.init_ui()

    def init_ui(self):
        # Buttons
        self.start_button = tk.Button(self.root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        # Spinner (loading label)
        self.spinner = ttk.Label(self.root, text="")
        self.spinner.pack()

        self.new_image_button = tk.Button(self.root, text="New Random Image", command=self.show_random_image)
        self.new_image_button.pack(pady=5)


        # Transcript display
        self.transcript_box = tk.Text(self.root, height=6, width=50)
        self.transcript_box.pack(pady=10)

        # Dog image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Load a random dog image
        self.show_random_image()

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.spinner.config(text="Recording... 🎙️")

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                                      input=True, frames_per_buffer=CHUNK)

        threading.Thread(target=self.record).start()

    def record(self):
        while self.recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

    def stop_recording(self):
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

        threading.Thread(target=self.transcribe_audio).start()

    def show_random_image(self):
        if not IMAGES:
            self.image_label.config(text="No dog images found!")
            return
        image_path = random.choice(IMAGES)  # Pick random image each time!
        img = Image.open(image_path)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep reference to avoid GC

    def transcribe_audio(self):
        self.spinner.config(text="Transcribing... ⏳")

        # Step 1: Detect pauses
        sound = AudioSegment.from_wav(AUDIO_FILENAME)
        silence_ranges = silence.detect_silence(sound, min_silence_len=1000, silence_thresh=-40)
        silence_ranges = [(start / 1000.0, stop / 1000.0) for start, stop in silence_ranges]  # ms to seconds

        # Step 2: Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILENAME) as source:
            audio_data = recognizer.record(source)
            try:
                raw_text = recognizer.recognize_google(audio_data)

                # Step 3: Split into words
                words = raw_text.split()

                # Step 4: Insert "uh"/"um" based on silence count
                pause_count = len(silence_ranges)
                interval = len(words) // (pause_count + 1) if pause_count else len(words)

                for i in range(pause_count):
                    insert_at = (i + 1) * interval
                    filler = random.choice(["uh", "um"])
                    words.insert(insert_at, filler)

                final_text = ' '.join(words)

                # Step 5: Save and display
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
    app = DogTruthApp(root)
    root.mainloop()
