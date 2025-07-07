import cv2
import os

FPS = [25]

for fps in FPS:
    VIDEO_FOLDER = f"/home/joao/workspace/EquinePainFaceDataset/dataset/clips/25FPS/"

    # loop over files in folder
    for file in os.listdir(VIDEO_FOLDER):
        if not file.endswith('.mp4'):
            continue
        # create folder with same_name_frames to store frames next to file, do not care if already exists
        folder_name = file.split('_.mp4')[0] + '_frames'
        os.makedirs(os.path.join(VIDEO_FOLDER, folder_name), exist_ok=True)

        # open video file
        cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, file))
        # Write all frames to folder
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(VIDEO_FOLDER, folder_name, f'{i:05d}.jpg'), frame)
            i += 1
        cap.release()
        print(f'Finished writing frames for {file}')
