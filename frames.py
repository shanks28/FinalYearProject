import cv2
import os
from PIL import Image
from torchvision.transforms import transforms, ToTensor
from torch import tensor
from torchvision.transforms import ToPILImage,Resize
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
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    cap = cv2.VideoCapture(url_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while cap.isOpened():
        ret, frame = cap.read()  # frame is a numpy array
        if not ret:
            break
        frame_name = f"{frame_count}.png"
        frame_count += 1
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
    cap.release()
    return fps


def downsample(video_path, output_dir, target_fps):
    pass


def load_frames(path,size=(128,128)) -> tensor: # converts PIL image to tensor on the GPU
    image = Image.open(path).convert('RGB')
    tensor = ToTensor()
    resized_image=Resize(size)(image)
    return tensor(resized_image).unsqueeze(0).to(device)



if __name__ == "__main__":  # sets the __name__ variable to __main__ for this script

    extract_frames("Test.mp4", "output")
