from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse
import re

def main(images_dir = "/home/joao/workspace/EquinePainFaceDataset/CleanAnEquinePainFaceDataset/videos/frames/S5"):
    # Load a pretrained YOLOv8n model
    model = YOLO("/home/joao/workspace/horse-ear-detection/yolov8n_horse_ear_detection.pt")

    my_file = Path(images_dir)

    if not my_file.is_dir():
        print("Path does not exist")


    cap = cv2.VideoCapture(f"{images_dir}/%5d.jpg")

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.7, imgsz=640)
            
            box = str(results[0].boxes.xyxy[0].cpu().numpy())
            box = re.sub(r'^([^\s]*)\s+', r'\1', box)
            box = re.sub("\s+", ",", box.strip())
            print(box)
            
            break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", help="full path to images to track object", type=str)
    args = parser.parse_args()
    main(args.images_dir)