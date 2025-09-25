from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from pathlib import Path
from math import floor

# Annotations drawing function
def draw_results(res, names, color_offset=0):
    if not hasattr(res, "boxes") or res.boxes is None:
        return
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    for i, box in enumerate(xyxy):
        cls = clss[i]
        conf = confs[i]
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        annotator.box_label(box, label, color=colors(cls + color_offset, True))

# Load models
model = YOLO("/home/joao/horse-face-ear-detection/horse_face_detection/yolov8l_horse_face_detection.pt")
ear_model = YOLO("/home/joao/horse-face-ear-detection/horse_ear_detection/yolov8l_horse_ear_detection.pt")

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
    "face_ear_annotated_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Loop through video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO face detection
    face_results = model(frame, conf=0.5, imgsz=640)
    ear_results = ear_model(frame, conf=0.3, imgsz=640)
    
    # Annotate frame with face detections
    annotator = Annotator(frame)
    for res in face_results:
        draw_results(res, model.names, color_offset=0)
    for res in ear_results:
        draw_results(res, ear_model.names, color_offset=len(model.names))
    
    annotated_frame = annotator.result()

    # Write combined annotation to video
    out.write(annotated_frame)

# Release everything
cap.release()
out.release()
