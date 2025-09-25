from ultralytics import YOLO
import os

project_folder = os.path.dirname(os.path.abspath(__file__))

# Load a model
model = YOLO(f"{project_folder}/yolov8l_horse_face_detection.pt")  # load a custom model

# Validate the model
metrics = model.val(
    data=f'{project_folder}/Horse Face Detection.v1i.yolov8/data.yaml',
    imgsz=640,
    project=project_folder,
    split="val"
)

print(" ")
print("***TEST RESULTS***")
print(" ")
print("mAP50-95: " + str(metrics.box.map))     # Mean AP @ IoU=0.5:0.95
print("mAP50: " + str(metrics.box.map50))     # Mean AP @ IoU=0.5
print("mAP75: " + str(metrics.box.map75))     # Mean AP @ IoU=0.75
print("Precision: " + str(metrics.box.p))     # Precision
print("Recall: " + str(metrics.box.r))        # Recall
print("F1-Score: " + str(metrics.box.f1))     # F1 Score
print(" ")