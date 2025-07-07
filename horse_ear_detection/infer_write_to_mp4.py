from ultralytics import YOLO
import cv2
from pathlib import Path
import os
import math
import numpy as np

def resize_with_bars(img, desired_shape, color):
    border_v = 0
    border_h = 0
    IMG_COL = desired_shape[0]
    IMG_ROW = desired_shape[1]

    if (IMG_COL / IMG_ROW) >= (img.shape[0] / img.shape[1]):
        border_v = int((((IMG_COL / IMG_ROW) * img.shape[1]) - img.shape[0]) / 2)
    else:
        border_h = int((((IMG_ROW / IMG_COL) * img.shape[0]) - img.shape[1]) / 2)
    aux_img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, borderType=cv2.BORDER_CONSTANT, value=0)
    bar_resized_img = cv2.resize(aux_img, (IMG_ROW, IMG_COL))
    return bar_resized_img

def smooth_box(previous_box, current_box, smoothness):
    # Smoothly interpolate the previous and current box positions using the smoothness factor.
    return [int(current_box[0] + smoothness * previous_box[0])//2,
            int(current_box[1] + smoothness * previous_box[1])//2,
            int(current_box[2] + smoothness * previous_box[2])//2,
            int(current_box[3] + smoothness * previous_box[3])//2]

# Load a pretrained YOLOv8n model
model = YOLO("/home/joao/workspace/extract-ear-frames/horse_ear_detection/yolov8n_horse_ear_detection.pt")

# Set smoothing parameters (you can adjust these values)
movement_smoothness = 1  # A bigger value makes the movement slower and smoother
size_smoothness = 1      # A bigger value makes the size change slower and smoother

# Run inference
clips_path = "/home/joao/workspace/EquinePainFaceDataset/dataset/clips/25FPS/original"
# find files in clips path that are mp4
for clip_file in os.listdir(clips_path):
    if clip_file.endswith(".mp4"):
        print(f"{clips_path}/{clip_file}")
        video_path = f"{clips_path}/{clip_file}"
        my_file = Path(video_path)
        filepath, filename = os.path.split(video_path)

        input_size = (736, 480)
        input_fps = 25

        # Set desired fps and video size
        output_fps = 25
        video_size = (1280, 720)

        if not my_file.is_file():
            print("File does not exist")

        # Set VideoCaptura and VideoWriter properties
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, input_fps)
        out = cv2.VideoWriter(f"/home/joao/workspace/EquinePainFaceDataset/dataset/clips/25FPS/masked_ears/{clip_file}", cv2.VideoWriter_fourcc(*'MP4V'), output_fps, video_size)

        last_box = []

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = model(frame, conf=0.65, imgsz=input_size[1])

                # Find best detection - if no detection we use last detection box from previous frame
                best_conf = 0
                best_idx = 0
                for idx, result in enumerate(results):
                    if result and result.boxes[idx].conf > best_conf:
                        best_conf = result.boxes.conf
                        best_idx = idx

                # Handle the case when no object is detected in the current frame
                if results[best_idx].boxes.xyxy.numel() != 0:
                    xstart = math.floor(results[best_idx].boxes.xyxy[best_idx].cpu().numpy()[0])
                    ystart = math.floor(results[best_idx].boxes.xyxy[best_idx].cpu().numpy()[1])
                    xstop = math.floor(results[best_idx].boxes.xyxy[best_idx].cpu().numpy()[2])
                    ystop = math.floor(results[best_idx].boxes.xyxy[best_idx].cpu().numpy()[3])
                    current_box = [xstart, xstop, ystart, ystop]
                    last_box = current_box
                else:
                    print(f"No detection. Using box from last detection: {last_box}")
                    if last_box == []:
                        # skip frame if no box exists from previous frame
                        continue
                    current_box = last_box

                # Smooth the movement of the box
                smoothed_box = smooth_box(last_box, current_box, movement_smoothness)
                
                # Crop the image based on the smoothed box
                xstart, xstop, ystart, ystop = smoothed_box
                face_cropped_image = frame[ystart:ystop, xstart:xstop]
                out_img = resize_with_bars(face_cropped_image, (video_size[1], video_size[0]), 0)

                # Write to output video
                out.write(out_img)

            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
