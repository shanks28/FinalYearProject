import cv2
import torch
from model import UNet
from PIL import Image
from torchvision.transforms import transforms, ToTensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
import argparse
import shutil
import os
import subprocess
import frames
from torchvision.transforms import Resize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_frames(tensor, out_path) -> None:
    image = normalize_frames(tensor)
    image = Image.fromarray(image)
    image.save(out_path)

def normalize_frames(tensor):
    '''
    Normalize the tensor image to [0, 255] Bell Shaped Curve
    :param tensor:
    :return:
    '''
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = torch.clamp(tensor, 0.0, 1.0)  # Ensure values are in [0, 1]
    tensor = (tensor * 255).byte()  # Scale to [0, 255]
    tensor = tensor.permute(1, 2, 0).numpy()  # Convert to [H, W, C] height width channels
    return tensor
def laod_allframes(frame_dir):
    '''
    Loads all of the frames in the original video frame directory
    :param frame_dir: Input Frame Directory to the model
    :return: Iterator for batch processing of frames
    '''
    frames_path = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    )
    for frame_path in frames_path:
        yield load_frames(frame_path)
def load_frames(image_path)->torch.Tensor:
    '''
    Converts the PIL image(RGB) to a pytorch Tensor and loads into GPU
    :params image_path
    :return: pytorch tensor
    '''
    transform = transforms.Compose([
        Resize((720,1280)),
        ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor

def time_steps(input_fps, output_fps) -> list[float]:
    '''
    Generates Time intervals to interpolate between frames A and B
    :param input_fps: Video FPS(Original)
    :param output_fps: Target FPS(Output)
    :return: List of intermediate FPS required between 2 Frames A and B
    '''
    if output_fps <= input_fps:
        return []
    k = output_fps // input_fps
    n = k - 1
    return [i / (n + 1) for i in range(1, n + 1)]
def interpolate_video(frames_dir,model_fc,input_fps,ouput_fps,output_dir)->None:
    '''
    Accumalates Frame A and Frame B and sends to interpolate function along with input and output FPS
    :param frames_dir: Original Frame Directory to the model
    :param model_fc: Compute Model with weights and Unet Architecture
    :param input_fps: input fps
    :param ouput_fps: target fps
    :param output_dir: interpolate and store into this directory
    :return:None
    '''
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    count=0
    iterator=laod_allframes(frames_dir)
    try:
        prev_frame=next(iterator)
        for curr_frame in iterator:
            interpolated_frames=interpolate(model_fc,prev_frame,curr_frame,input_fps,ouput_fps)
            save_frames(prev_frame,os.path.join(output_dir,"frame_{}.png".format(count)))
            count+=1
            for frame in interpolated_frames:
                save_frames(frame[:,:3,:,:],os.path.join(output_dir,"frame_{}.png".format(count)))
                count+=1
            prev_frame=curr_frame
        save_frames(prev_frame,os.path.join(output_dir,"frame_{}.png".format(count)))
    except StopIteration:
        print("no more Frames")


def interpolate(model_FC, A, B, input_fps, output_fps)-> list[torch.Tensor]:
    '''
    Calculates time steps using the input and output fps
    Concatenates the Frame A and Frame B together in 1 tensor to identify change in frames
    :param model_FC: Computer Model with weights and Unet Architecture
    :param A: Frame A
    :param B: Frame B
    :param input_fps:
    :param output_fps:
    :return: Pytorch tensor
    '''
    interval = time_steps(input_fps, output_fps)
    input_tensor = torch.cat((A, B), dim=1) # Concatenate Frame A and B to Compare difference
    with torch.no_grad():
        flow_output = model_FC(input_tensor) # outputs 4 channels
        flow_forward = flow_output[:, :2, :, :]  # Forward flow A to B 0,1
        flow_backward = flow_output[:, 2:4, :, :]  # Backward flow B to A 2,3
    generated_frames = []
    with torch.no_grad(): # only during inference
        for t in interval:
            t_tensor = torch.tensor([t], dtype=torch.float32).view(1, 1, 1, 1).to(device)
            with autocast(): # for tradeoff between memory and accuracy
                warped_A = warp_frames(A, flow_forward * t_tensor) # shifting pixels according to time step
                warped_B = warp_frames(B, flow_backward * (1 - t_tensor))
                interpolated_frame = warped_A * (1 - t_tensor) + warped_B * t_tensor
            generated_frames.append(interpolated_frame)
    return generated_frames

def warp_frames(frame, flow)->torch.Tensor:
    '''
    Used to align the frame with the computed optical flow vectors
    :param frame: Input frame
    :param flow: Optical flow vectors
    :return: pytorch tensor aligned with the frame A
    '''
    b, c, h, w = frame.size() # dimensions of Frame A
    i,j,flow_h, flow_w = flow.size() # dimensions of optical vector
    if h != flow_h or w != flow_w:
        frame = F.interpolate(frame, size=(flow_h, flow_w), mode='bilinear', align_corners=True) # resize vectors if they dont match in dimensions
    grid_y, grid_x = torch.meshgrid(torch.arange(0, flow_h), torch.arange(0, flow_w), indexing="ij")
    grid_x = grid_x.float().to(device)
    grid_y = grid_y.float().to(device)
    flow_x = flow[:, 0, :, :] # extract x coordinate
    flow_y = flow[:, 1, :, :]# extract y coordinate
    x = grid_x.unsqueeze(0) + flow_x
    y = grid_y.unsqueeze(0) + flow_y
    x = 2.0 * x / (flow_w - 1) - 1.0
    y = 2.0 * y / (flow_h - 1) - 1.0
    grid = torch.stack((x, y), dim=-1)

    warped_frame = F.grid_sample(frame, grid, align_corners=True,mode='bilinear', padding_mode='border')
    return warped_frame
def frames_to_video(frame_dir,output_video,fps)-> None:
    '''
    Restitch the video from the generated Frames
    :param frame_dir: Interpolated Frame Directory
    :param output_video: Name of the  output Video
    :param fps: Desired FPS
    :return: None
    '''
    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
    )
    for i, frame in enumerate(frame_files):
        os.rename(os.path.join(frame_dir, frame), os.path.join(frame_dir, f"frame_{i}.png"))
    frame_pattern = os.path.join(frame_dir, "frame_%d.png")
    subprocess.run([ #  run shell command
        "ffmpeg", "-framerate", str(fps), "-i", frame_pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video
    ],check=True)
def solve(frames_dir,input_fps,output_fps,final_video="result.mp4"):
    checkpoint = torch.load("SuperSloMo.ckpt")
    model_FC = UNet(6, 4).to(device)  # Initialize flow computation model
    model_FC.load_state_dict(checkpoint["state_dictFC"])  # Load weights
    model_FC.eval()
    # model_AT = UNet(20, 5).to(device)  # Initialize auxiliary task model
    # model_AT.load_state_dict(checkpoint["state_dictAT"], strict=False)  # Load weights
    # model_AT.eval()
    interpolate_dir="interpolated_frames"
    interpolate_video(frames_dir,model_FC,input_fps,output_fps,interpolate_dir)
    frames_to_video(interpolate_dir,final_video,output_fps)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ofps",type=int,help="Desired FPS",default=60)
    parser.add_argument("--fdir",type=str,help="Store initial video frames",default="output")
    parser.add_argument("--ivideo",type=str,help="Input video path ",default="Test.mp4")
    parser.add_argument("--ovideo",type=str,help="Output video name ",default="result.mp4")
    args=parser.parse_args()
    input_fps=frames.extract_frames(args.ivideo,args.fdir)
    solve(args.fdir,input_fps,args.ofps,args.ovideo)

if __name__ == "__main__":
    main()
