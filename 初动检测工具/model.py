import torch 
import torch.nn as nn 
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, nin, nout, stride, ks):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(nin, nout, ks, stride=stride, padding=(ks-1)//2), 
            nn.BatchNorm1d(nout), 
            nn.LeakyReLU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class ResBlock(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(nin, nin//2, 1, 1), 
            ConvBNReLU(nin//2, nin, 1, 5), 
        )
    def forward(self, x):
        y = self.layers(x) 
        return y + x 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = 16
        self.layers = nn.Sequential(
            ConvBNReLU(1, base*1, 1, 5), 
            nn.Conv1d(base*1, base*2, 5, 2, padding=2), 
            ResBlock(base*2), 
            nn.Conv1d(base*2, base*3, 5, 2, padding=2), 
            ResBlock(base*3), 
            nn.Conv1d(base*3, base*4, 5, 2, padding=2), 
            ResBlock(base*4), 
            nn.Conv1d(base*4, base*5, 5, 2, padding=2), 
            ResBlock(base*5), 
            nn.Conv1d(base*5, base*6, 5, 2, padding=2), 
            ResBlock(base*6), 
            nn.Conv1d(base*6, base*7, 5, 2, padding=2), 
            ResBlock(base*7), 
            nn.Conv1d(base*7, base*8, 5, 2, padding=2), 
            ResBlock(base*8), 
            nn.Conv1d(base*8, base*9, 5, 2, padding=2), 
            ResBlock(base*9), 
            nn.Conv1d(base*9, base*10, 5, 2, padding=2), 
            ResBlock(base*10), 
            nn.Conv1d(base*10, base*11, 5, 2, padding=2), 
            ResBlock(base*11), 
        )
        self.out1 = nn.Linear(base*11, 2) # 用于初动判定
        self.out2 = nn.Linear(base*11, 3) # 用于初动质量评定
    def forward(self, x):
        T, B, C = x.shape 
        h = self.layers(x) 
        h = h.squeeze()
        y1 = self.out1(h) 
        y2 = self.out2(h)
        return y1, y2 
