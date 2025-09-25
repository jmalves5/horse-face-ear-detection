from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm

# Load model
model = YOLO("/home/joao/horse-face-ear-detection/horse_face_detection/yolov8l_horse_face_detection.pt")

# Input video
video_path = "/home/joao/Downloads/horse.mp4"
my_file = Path(video_path)

if not my_file.is_file():
    print("File does not exist")

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
out = cv2.VideoWriter(
    "annotated_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Silence ultralytics
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Loop through frames with progress bar
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for _ in tqdm(range(frame_count), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Write annotated frame to output video
    out.write(annotated_frame)

# Release everything
cap.release()
out.release()
