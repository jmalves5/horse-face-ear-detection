import cv2
import sys
from ultralytics import YOLO

def scale_boxes(boxes, orig_size, resized_size):
    orig_w, orig_h = orig_size
    resized_w, resized_h = resized_size
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    scaled = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        scaled.append([int(x1), int(y1), int(x2), int(y2)])
    return scaled

def main():
    # Load models
    face_model = YOLO("horse_face_detection/yolov8l_horse_face_detection.pt")
    ear_model = YOLO("horse_ear_detection/yolov8n_horse_ear_detection.pt")  # <-- updated here

    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    window_name = "Horse Face + Ear Detection (2x display)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Read first frame to get size
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    orig_h, orig_w = frame.shape[:2]
    cv2.resizeWindow(window_name, orig_w * 2, orig_h * 2)

    resized_size = (640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to grab frame.")
            break

        resized_frame = cv2.resize(frame, resized_size)

        # Run both models
        face_results = face_model(resized_frame)
        ear_results = ear_model(resized_frame)

        annotated_frame = frame.copy()

        # Face detections
        face_dets = face_results[0].boxes
        if face_dets:
            face_boxes = face_dets.xyxy.cpu().numpy()
            face_probs = face_dets.conf.cpu().numpy()
            face_boxes_scaled = scale_boxes(face_boxes, (orig_w, orig_h), resized_size)

            for box, prob in zip(face_boxes_scaled, face_probs):
                if prob < 0.5:
                    continue
                x1, y1, x2, y2 = box
                label = f"Face {prob:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2
                )

        # Ear detections
        ear_dets = ear_results[0].boxes
        if ear_dets:
            ear_boxes = ear_dets.xyxy.cpu().numpy()
            ear_probs = ear_dets.conf.cpu().numpy()
            ear_boxes_scaled = scale_boxes(ear_boxes, (orig_w, orig_h), resized_size)

            for box, prob in zip(ear_boxes_scaled, ear_probs):
                if prob < 0.5:
                    continue
                x1, y1, x2, y2 = box
                label = f"Ear {prob:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2
                )

        # Display enlarged window
        display_frame = cv2.resize(annotated_frame, (orig_w * 2, orig_h * 2))
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()