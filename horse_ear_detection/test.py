from ultralytics import YOLO
import os

project_folder = os.path.dirname(os.path.abspath(__file__))

# Load a model
model = YOLO("yolov8l_horse_ear_detection.pt")  # load a custom model

# Test the model
metrics = model.val(
    data=f'{project_folder}/HorseEarDetection/data_test.yaml',
    imgsz=640,
    project=project_folder
)

print(" ")
print("***TEST RESULTS***")
print(" ")
print("map50-95: " + str(metrics.box.map))
print("map50: " + str(metrics.box.map50))
print("map75: " + str(metrics.box.map75))
print("Precision: " + str(metrics.box.p))
print("Recall: " + str(metrics.box.r))
print("F1-Score: " + str(metrics.box.f1))

print(" ")

