import shutil
import cv2
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_frames(url_path, output_dir) -> int :
    '''
    Acts as initial feed into the SuperSlomo Model
    The Frames are stored in an output directory which is then loaded into the SuperSlomo Model.
    :param url_path:
    :param output_dir:
    :return: None
    '''
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        frame_count = 0
        cap = cv2.VideoCapture(url_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        while cap.isOpened():
            ret, frame = cap.read()  # frame is a numpy array
            if not ret:
                break
            frame_name = f"frame_{frame_count}.png"
            frame_count += 1
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        cap.release()
        return fps
    except Exception as e:
        print(e)


def downsample(video_path, output_dir, target_fps):
    pass



if __name__ == "__main__":  # sets the __name__ variable to __main__ for this script

    print(extract_frames("Test2-15fps.mp4", "output"))
