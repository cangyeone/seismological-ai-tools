import torch  
import torch.nn as nn 
import torch.nn.functional as F 

class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=[7, 1], st=[4, 1], padding=[3, 0]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, st, padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class Conv2dT(nn.Module):
    def __init__(self, nin=8, nout=11, ks=[7, 1], st=[4, 1], padding=[3, 0]):
        super().__init__()
        # 这里我们使用上采样进行
        self.layers = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=tuple(st)), 
            Conv2d(nin, nout, ks, [1, 1], padding=padding), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = Conv2d(3, 8, [7, 1], [1, 1], padding=[3, 0]) 
        self.layer0 = Conv2d(8, 8, [7, 1], [1, 1], padding=[3, 0]) 
        self.layer1 = Conv2d(8, 16, [7, 1], [4, 1], padding=[3, 0])
        self.layer2 = Conv2d(16, 32, [7, 1], [4, 1], padding=[3, 0])
        self.layer3 = Conv2d(32, 64, [7, 1], [4, 1], padding=[3, 0]) 
        self.layer4 = Conv2d(64, 128, [7, 1], [4, 1], padding=[3, 0]) 
        self.layer5 = Conv2dT(128, 64, [7, 1], [4, 1], padding=[3, 0])
        self.layer6 = Conv2dT(128, 32, [7, 1], [4, 1], padding=[3, 0])
        self.layer7 = Conv2dT(64, 16, [7, 1], [4, 1], padding=[3, 0])
        self.layer8 = Conv2dT(32, 8, [7, 1], [4, 1], padding=[3, 0])
        self.layer9 = nn.Conv2d(16, 3, [7, 1], [1, 1], padding=[3, 0])
    def forward(self, x):
        x = x.unsqueeze(3)
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        x6 = torch.cat([x4, x6], dim=1) # 加入skip connection
        x7 = self.layer6(x6)
        x7 = torch.cat([x3, x7], dim=1) # 加入skip connection
        x8 = self.layer7(x7)
        x8 = torch.cat([x2, x8], dim=1) # 加入skip connection
        x9 = self.layer8(x8)
        x9 = torch.cat([x1, x9], dim=1) # 加入skip connection
        x10 = self.layer9(x9)
        x10 = F.softmax(x10, dim=1)
        x10 = x10.squeeze(dim=3)
        return x10

class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
    def forward(self, x, d):
        loss = - (d * torch.log(x+1e-9)).sum()
        return loss 

if __name__ == "__main__":
    model = UNet() 
    x = torch.randn([10, 3, 6144]) 
    y = model(x) 
    print(y.shape)