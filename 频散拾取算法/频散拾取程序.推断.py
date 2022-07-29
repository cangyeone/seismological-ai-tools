import torch.nn as nn 
import torch 
import torch.nn as nn 

import math 
class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=3, st=2, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, st, padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class Conv2dTT(nn.Module):
    def __init__(self, nin=8, nout=11, ks=3, st=2, padding=1, output_padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, ks, st, padding=padding, output_padding=output_padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class Conv2dT(nn.Module):
    def __init__(self, nin=8, nout=11, ks=3, st=2, padding=1, output_padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=st), 
            Conv2d(nin, nout, ks, [1, 1], padding=padding), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = Conv2d(1, 8, 3, [1, 1], padding=1) 
        self.layer0 = Conv2d(8, 8, 3, [1, 1], padding=1) 
        self.layer1 = Conv2d(8, 16, 3, 2, padding=1)
        self.layer2 = Conv2d(16, 32, 3, 2, padding=1)
        self.layer3 = Conv2d(32, 64, 3, 2, padding=1) 
        self.layer4 = Conv2d(64, 128, 3, 2, padding=1) 
        self.layer5 = Conv2dT(128, 64, 3, 2, padding=1, output_padding=1)
        self.layer6 = Conv2dT(128, 32, 3, 2, padding=1, output_padding=1)
        self.layer7 = Conv2dT(64, 16, 3, 2, padding=1, output_padding=1)
        self.layer8 = Conv2dT(32, 8, 3, 2, padding=1, output_padding=1)
        self.layer9 = nn.Conv2d(16, 2, 3, [1, 1], padding=1)
    def forward(self, x):
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        #Vision Transformer-ViT 
        # PyVista
        #print(x4.shape, x6.shape)
        x6 = torch.cat([x4, x6[:, :, :-1, :-1]], dim=1) 
        x7 = self.layer6(x6)
        #print(x3.shape, x7.shape)
        x7 = torch.cat([x3, x7[:, :, :, :-1]], dim=1) 
        x8 = self.layer7(x7)
        #print(x2.shape, x8.shape)
        x8 = torch.cat([x2, x8[:, :, :-1, :-1]], dim=1) 
        x9 = self.layer8(x8)
        #print(x1.shape, x9.shape)
        x9 = torch.cat([x1, x9[:, :, :-1, :-1]], dim=1) 
        x10 = self.layer9(x9)
        x10 = x10.sigmoid()
        #print(x10.shape)
        #x10 = x10.softmax(dim=1)
        return x10
import numpy as np 

model = UNet() 
model.eval() 
model.load_state_dict(torch.load("ckpt/cnn.disp4")) 
x = np.loadtxt("data/C.S1.S3.dat")
with torch.no_grad():
    xt = torch.tensor(x, dtype=torch.float32)[None, None]
    yt = model(xt)
    yt = yt.squeeze()
    yt = yt.detach().numpy()
    print(yt.shape)


hh, w2 = yt[0].shape
dis = []


for idx in range(w2):
    p = yt[0][:, idx]
    idy = np.argmax(p) 
    prob = p[idy] 
    if prob>=0.1:
        dis.append([idx, idy]) 
dis = np.array(dis)
import matplotlib.pyplot as plt 
h, w = x.shape 
h = np.arange(h) 
w = np.arange(w)
plt.pcolormesh(w, h, x, shading='auto')
plt.scatter(dis[:, 0], dis[:, 1], c="r") 
plt.show()