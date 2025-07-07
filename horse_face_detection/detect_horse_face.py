import math
import argparse

def detect_horse_face(frame, model):
    # Run YOLOv8 inference on the frame
    results = model(frame, conf=0.5, imgsz=640)

    # Find the detection with the highest confidence
    if not results[0].boxes:
        print("No detections found.")
        return 0, 0, 0, 0

    # Extract bounding boxes and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    # Get the best detection
    best_idx = confs.argmax()
    best_box = boxes[best_idx]

    # Extract coordinates
    xstart, ystart, xstop, ystop = map(math.floor, best_box)

    return xstart, ystart, xstop, ystop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", help="image to detect")
    args = parser.parse_args()
    detect_horse_face(args.frame)
