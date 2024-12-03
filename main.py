import torch
from model import UNet
from PIL import Image
from torchvision.transforms import transforms, ToTensor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_frames(tensor, out_path) -> None:
    image = normalize_frames(tensor)
    image = Image.fromarray(image)
    image.save(out_path)

def normalize_frames(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = torch.clamp(tensor, 0.0, 1.0)  # Ensure values are in [0, 1]
    tensor = (tensor * 255).byte()  # Scale to [0, 255]
    tensor = tensor.permute(1, 2, 0).numpy()  # Convert to [H, W, C]
    return tensor

def load_frames(image_path)->torch.Tensor:
    transform = transforms.Compose([
        ToTensor()  # Converts to [0, 1] range and [C, H, W]
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return tensor

def time_steps(input_fps, output_fps) -> list[float]:
    if output_fps <= input_fps:
        return []
    k = output_fps // input_fps
    n = k - 1
    return [i / (n + 1) for i in range(1, n + 1)]

def expand_channels(tensor, target):
    batch_size, current_channels, height, width = tensor.shape
    if current_channels >= target:
        return tensor
    required = target - current_channels
    extra = torch.zeros(batch_size, required, height, width, device=tensor.device, dtype=tensor.dtype)
    return torch.cat((tensor, extra), dim=1)

def interpolate(model_FC, model_AT, A, B, input_fps, output_fps):
    interval = time_steps(input_fps, output_fps)
    input_tensor = torch.cat((A, B), dim=1)  # Combine frames A and B

    with torch.no_grad():
        flow_output = model_FC(input_tensor)
        flow_forward = flow_output[:, :2, :, :]  # Forward flow
        flow_backward = flow_output[:, 2:4, :, :]  # Backward flow

    generated_frames = []
    with torch.no_grad():
        for t in interval:
            t_tensor = torch.tensor([t], dtype=torch.float32).view(1, 1, 1, 1).to(device)

            warped_A = warp_frames(A, flow_forward * t_tensor)
            warped_B = warp_frames(B, flow_backward * (1 - t_tensor))

            interpolated_frame = warped_A * (1 - t_tensor) + warped_B * t_tensor
            generated_frames.append(interpolated_frame)

    return generated_frames


def warp_frames(frame, flow):
    b, c, h, w = frame.size()
    _, _, flow_h, flow_w = flow.size()

    if h != flow_h or w != flow_w:
        frame = F.interpolate(frame, size=(flow_h, flow_w), mode='bilinear', align_corners=True)

    grid_y, grid_x = torch.meshgrid(torch.arange(0, flow_h), torch.arange(0, flow_w), indexing="ij")
    grid_x = grid_x.float().to(device)
    grid_y = grid_y.float().to(device)

    flow_x = flow[:, 0, :, :]
    flow_y = flow[:, 1, :, :]
    x = grid_x.unsqueeze(0) + flow_x
    y = grid_y.unsqueeze(0) + flow_y

    x = 2.0 * x / (flow_w - 1) - 1.0
    y = 2.0 * y / (flow_h - 1) - 1.0
    grid = torch.stack((x, y), dim=-1)

    warped_frame = F.grid_sample(frame, grid, align_corners=True)
    return warped_frame


def solve():
    checkpoint = torch.load("SuperSloMo.ckpt")
    model_FC = UNet(6, 4).to(device)  # Initialize flow computation model
    model_FC.load_state_dict(checkpoint["state_dictFC"])  # Load weights
    model_FC.eval()

    model_AT = UNet(20, 5).to(device)  # Initialize auxiliary task model
    model_AT.load_state_dict(checkpoint["state_dictAT"], strict=False)  # Load weights
    model_AT.eval()

    A = load_frames("output/1.png")
    B = load_frames("output/10.png")
    interpolated_frames = interpolate(model_FC, model_AT, A, B, 30, 90)

    for index, value in enumerate(interpolated_frames):
        save_frames(value[:, :3, :, :], f"Result_Test/image{index + 1}.png")  # Save only RGB channels

def main():
    solve()

if __name__ == "__main__":
    main()
