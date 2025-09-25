from ultralytics import YOLO
import os

# Get project folder path
project_folder = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model = YOLO("yolov8l_horse_ear_detection.pt")

# Evaluate on the test split instead of val
metrics = model.val(
    data=f'{project_folder}/Horse Ear Detection.v2i.yolov8/data.yaml',
    imgsz=640,
    project=project_folder,
    split="test",
    # change output folder name
    name="test"    
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