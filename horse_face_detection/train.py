import os, shutil
from ultralytics import YOLO
 
project_folder = os.path.dirname(os.path.abspath(__file__))

# Load the model.
model = YOLO('yolov8l.pt')

# Specify the save directory for training runs
os.makedirs(project_folder, exist_ok=True)
 
# Training.
results = model.train(
   data=f'{project_folder}/HorseFaceDetection/data.yaml',
   imgsz=640,
   epochs=150,
   batch=16,
   name='yolov8l_custom_train',
   project="face_detection_custom_train",
   patience=15
)

best_model = project_folder + "/face_detection_custom_train/yolov8l_custom_train/weights/best.pt"

shutil.copyfile(best_model, f"{project_folder}/yolov8l_horse_face_detection.pt")