import cv2
import os
from datetime import datetime

# === Step 1: Set up video capture ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

# === Step 2: Define codec and video writer placeholder ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording_frames = []
print("Press 'q' to stop recording...")

# === Step 3: Start recording loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow('Recording - Press q to stop', frame)
    recording_frames.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Step 4: Release camera ===
cap.release()
cv2.destroyAllWindows()

# === Step 5: Create CroppedClips folder if it doesn't exist ===
output_dir = r"C:\Users\barrettm5\git\DecepTech-2\CroppedClips"
os.makedirs(output_dir, exist_ok=True)

# === Step 6: Save the recorded video ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"clip_{timestamp}.avi")

height, width, _ = recording_frames[0].shape
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

for frame in recording_frames:
    out.write(frame)

out.release()
print(f"Recording saved to: {output_path}")
