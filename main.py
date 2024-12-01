import torch
def solve():
    checkpoint=torch.load("SuperSloMo.ckpt")
    checkpoint.eval()
    print(checkpoint)
def main():
    solve()
if __name__=="__main__":
    main()