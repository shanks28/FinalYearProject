import cv2
import os
def extract_frames(url_path,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_count=0
    cap=cv2.VideoCapture(url_path)
    while cap.isOpened() and frame_count<10:
        ret,frame=cap.read()
        if not ret:
            break
        frame_name=f"{frame_count}.png"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_count+=1
    cap.release()
extract_frames("C:/Users/BRIDGES/Downloads/Video1.mp4","output")
# this is a test change to merge later