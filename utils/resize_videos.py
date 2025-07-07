# Pseudo-code:
# Taking a video path as input
# read each frame
# find max and min indexed of x and y that are different from zero in the entire video
# crop the video based on the max and min indexed of x and y, padding the video with black pixels where needed

import cv2
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def find_video_nonzero_limits(video_path):
    cap = cv2.VideoCapture(video_path)

    # initialize the max and min indexed of x and y that are different from zero
    max_nonzero_x = 0
    min_nonzero_x = sys.maxsize
    max_nonzero_y = 0
    min_nonzero_y = sys.maxsize
    
    # loop over the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # get max and min indexed of x and y that are different from zero for the frame
        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nonzero_indices = cv2.findNonZero(gray_image)

        x_indices = [index[0][0] for index in nonzero_indices]
        y_indices = [index[0][1] for index in nonzero_indices]

        min_x, max_x = min(x_indices), max(x_indices)
        min_y, max_y = min(y_indices), max(y_indices)

        # update the max and min indexes
        max_nonzero_x = max(max_nonzero_x, max_x)
        min_nonzero_x = min(min_nonzero_x, min_x)
        max_nonzero_y = max(max_nonzero_y, max_y)
        min_nonzero_y = min(min_nonzero_y, min_y)

    # close the video
    cap.release()

    return (min_nonzero_x, max_nonzero_x, min_nonzero_y, max_nonzero_y)

def resize_with_padding(frame, desired_size, padding_color=[0,0,0]):
    # Get the desired size
    desired_h, desired_w = desired_size

    # Get the current size
    h, w = frame.shape[:2]

    # Calculate the padding needed
    print(f"desired_h: {desired_h}, desired_w: {desired_w}, h: {h}, w: {w}")
    
    padding_h = desired_h - h
    padding_w = desired_w - w

    # Get the padding needed for the top, bottom, left, and right
    top = padding_h // 2
    bottom = padding_h - top
    left = padding_w // 2
    right = padding_w - left

    print(f"top: {top}, bottom: {bottom}, left: {left}, right: {right}")

    # Create the padding
    padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return padded_frame

def main():
    video_path = f"/home/joao/workspace/EquinePainFaceDataset/CleanAnEquinePainFaceDataset/videos/masked_videos/ear_videos/ear_S5_masked.mp4"

    min_nonzero_x, max_nonzero_x, min_nonzero_y, max_nonzero_y = find_video_nonzero_limits(video_path)
    
    print(f"min_nonzero_x: {min_nonzero_x}, max_nonzero_x: {max_nonzero_x}, min_nonzero_y: {min_nonzero_y}, max_nonzero_y: {max_nonzero_y}")
    
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        padded_frame = resize_with_padding(frame, (max_nonzero_y - min_nonzero_y, max_nonzero_x - min_nonzero_x))
        cv2.imshow("frame", padded_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()