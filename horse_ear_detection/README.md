# Training and inference code for horse face detection using YOLOv8

This repo's purpose is to easily reproduce building a horse-face-detection system.

## Getting the data
Training dataset (`horse-face-detection`):

https://universe.roboflow.com/jjb-object-detection-projects/horse-face-detection

We perform inference on AnEquinePainFace dataset that can be found here: 

https://github.com/jmalves5/EquinePainFaceDataset

Folder structure to run our code should look like this:

```
├── workspace
│   ├── HorseFaceDetection.v4-all-rgb.yolov8
│   │   ├── train
│   │   ├── test
│   │   ├── val
│   │   ├── ...
│   ├── horse-face-detection
│   │   ├── train.py
│   │   ├── ...
```

## Running training and inference

Once you have setup you folders as described above you can run

### Train
For AMD gpus run after installing requirements.txt to get ROCM compatible torch/vision/audio: 
- `pip uninstalling torch torchvision torchaudio` 
- `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0/` 

```
python3 train.py
```

This command will perform the training on `horse-face-detection` dataset and copy the final model weights to the repo's root folder (see `train.py` for more details)


### Infer
We supply the model weights for horse detection using YOLOv8n in `yolov8n_horse_face_detection.pt`.

Then to perform inference you can run the following adapting the paths to where you have setup you inference data:
```
python3 infer.py
```