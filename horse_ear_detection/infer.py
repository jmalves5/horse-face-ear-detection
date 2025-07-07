from ultralytics import YOLO
import cv2
from pathlib import Path


# Load a pretrained YOLOv8n model
model = YOLO("/home/joao/workspace/horse-ear-detection/yolov8n_horse_ear_detection.pt")

# Run inference
# Open the video file
video_path = "/home/joao/workspace/EquinePainFaceDataset/CleanAnEquinePainFaceDataset/videos/original/S6.mp4"

my_file = Path(video_path)

if not my_file.is_file():
    print("File does not exist")


cap = cv2.VideoCapture(video_path)
i=0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.4, imgsz=640)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Horse Faces YOLOv8n Inference", annotated_frame)
        # i+=1
        # # write frame to file
        # cv2.imwrite(f"out/{i}.jpg", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()