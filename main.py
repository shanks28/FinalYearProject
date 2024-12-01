import torch
from model import UNet
from PIL import Image
from torchvision.transforms import transforms,ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_frames(path):
    image=Image.open(path).convert('RGB')
    tensor=ToTensor()
    return tensor(image).unsqueeze(0).to(device)
def solve():
    checkpoint=torch.load("SuperSloMo.ckpt")
    model_FC=UNet(6,4) # initialize ARCH
    model_FC=model_FC.to(device)# reassign model tensors
    model_FC.load_state_dict(checkpoint["state_dictFC"]) # loading all weights from model
    model_AT=UNet(20,5)
    model_AT.load_state_dict(checkpoint["state_dictAT"])
    model_AT=model_AT.to(device)
    model_AT.eval()
    model_FC.eval()

def main():
    solve()
if __name__=="__main__":
    main()