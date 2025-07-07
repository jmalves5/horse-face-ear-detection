from ultralytics import YOLO
import cv2
from pathlib import Path
from math import floor


# Load a pretrained YOLOv8n model
model = YOLO("/home/joaoalves/Documents/PhD/workspace/horse-face-detection/yolov8n_horse_face_detection.pt")
eye_model = YOLO("/home/joaoalves/Documents/PhD/workspace/eye-detector/yolov8n_eye_detector.pt")


# Run inference
# Open the video file
video_path = "/home/joaoalves/Documents/PhD/Datasets/AnEquinePainFaceDataset/CleanAnEquinePainFaceDataset/videos/S5.mp4"

my_file = Path(video_path)

if not my_file.is_file():
    print("File does not exist")


cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results= model(frame, conf=0.5, imgsz=640)
        annotated_frame = results[0].plot()
        
        for result in results:
            if result.boxes.xyxy.numel()!=0:
                xstart = floor(result.boxes.xyxy[0].cpu().numpy()[0])
                ystart = floor(result.boxes.xyxy[0].cpu().numpy()[1])
                xstop = floor(result.boxes.xyxy[0].cpu().numpy()[2])
                ystop = floor(result.boxes.xyxy[0].cpu().numpy()[3])

                face_cropped_image = frame[ystart:ystop, xstart:xstop]

                # Do eye detection on cropped result
                results_eye = eye_model(face_cropped_image, conf=0.3, imgsz=640)
                
                annotated_frame_eye = results_eye[0].plot()
                
                cv2.imshow("Horse Eye YOLOv8n Inference", annotated_frame_eye)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()