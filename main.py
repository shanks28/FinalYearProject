import torch
from model import UNet
from frames import load_frames,save_frames
from PIL import Image
from torchvision.transforms import transforms,ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def time_steps(input_fps,output_fps)->list[float]:
    if output_fps<=input_fps:
        return []
    k=output_fps//input_fps
    n=k-1
    return [i/n+1 for i in range(1,n+1)]
def expand_channels(tensor,target):
    batch_size,current_channels,height,width=tensor.shape
    if current_channels>=target:
        return tensor
    required=target-current_channels
    extra=torch.zeros(batch_size,required,height,width,device=tensor.device,dtype=tensor.dtype)
    return torch.cat((tensor,extra),dim=1)
def interpolate(model_FC,model_AT,A,B,input_fps,output_fps)-> list[float]:
    interval=time_steps(input_fps,output_fps)
    input_tensor=torch.cat((A,B),dim=1)
    with torch.no_grad():
        flow_output=model_FC(input_tensor)
        flow_output=expand_channels(flow_output,20)
    generated_frames=[]
    with torch.no_grad():
        for i in interval:
            inter_tensor=torch.tensor([i],dtype=torch.float32).unsqueeze(0).to(device)
            interpolated_frame=model_AT(flow_output,inter_tensor)
            generated_frames.append(interpolated_frame)
    return generated_frames

def solve():
    checkpoint=torch.load("SuperSloMo.ckpt")
    model_FC=UNet(6,4) # initialize ARCH
    model_FC=model_FC.to(device)# reassign model tensors
    model_FC.load_state_dict(checkpoint["state_dictFC"]) # loading all weights from model
    model_AT=UNet(20,5)
    model_AT.load_state_dict(checkpoint["state_dictAT"],strict=False)
    model_AT=model_AT.to(device)
    model_AT.eval()
    model_FC.eval()
    A=load_frames("output/1.png")
    B=load_frames("output/69.png")
    interpolated_frames=interpolate(model_FC,model_AT,A,B,60,120)
    print(interpolated_frames)
    for index,value in enumerate(interpolated_frames):
        save_frames(value[:,:3,:,:],"Result_Test/image{}.png".format(index+1))


def main():
    solve()
if __name__=="__main__":
    main()