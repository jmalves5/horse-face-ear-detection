# Training and inference code for horse face detection using YOLOv8

This repo's purpose is to easily reproduce building a horse face and ear detection system.

## Getting the data

### Face
Download yolov8 version of dataset (`horse-face-detection`) from:

https://universe.roboflow.com/jmalves5/horse-face-detection-qs4zj/dataset/1

### Ear
Download yolov8 version of dataset (`horse-ear-detection`) from:

https://universe.roboflow.com/jmalves5/horse-ear-detection/dataset/2

### Folder structure

Folder structure to run our code should look like this:

```
├── workspace
│   ├── horse-face-ear-detection
│     ├── horse-face-detection
│     │   ├── train.py
│     │   ├── test.py
│     │   ├── val.py
│     │   ├── ...
│     │   ├── HorseFaceDetection.yolov8
│     │     ├──── train
│     │     ├──── test
│     │     ├──── val
│     │     ├──── data.yaml
│     ├── horse-ear-detection
│         ├── train.py
│         ├── test.py
│         ├── val.py
│         ├── ...
│         ├── HorseEarDetection.yolov8
│           ├──── train
│           ├──── test
│           ├──── val
│           ├──── data.yaml
```

## Running training and inference

Once you have setup you folders as described above you can run from each of the face or ear folders

NOTE: You need to adapt the paths to you environment.

### Infer both face and ear
```
python3 infer_face_then_ear.py
```

This script uses both models and writes results to mp4

### Train

```
cd horse_X_detection
python3 train.py
```

This command will perform the training on `horse-X-detection` dataset and copy the final model weights to the repo's root folder (see `train.py` for more details)


### Infer
We supply the model weights for detection using YOLOv8n in `yolov8l_horse_X_detection.pt`.

Then to perform inference you can run the following adapting the paths to where you have setup you inference data:
```
cd horse_X_detection
python3 infer.py
```


### For AMD gpus run after installing requirements.txt to get ROCM compatible torch/vision/audio: 
- `pip uninstalling torch torchvision torchaudio` 
- `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0/` 