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
        self.out1 = nn.Linear(base*11, 2)
        self.out2 = nn.Linear(base*11, 3)
    def forward(self, x):
        T, B, C = x.shape 
        h = self.layers(x) 
        h = h.squeeze()
        y1 = self.out1(h) 
        y2 = self.out2(h)
        return y1, y2   

class CTLoss(nn.Module):
    """损失函数"""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="sum")
    def forward(self, y1, y2, d):
        loss1 = self.ce(y1, d[:, 0]) 
        loss2 = self.ce(y2, d[:, 1]) 
        loss = loss1 + loss2 * 0.1 
        return loss  

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
import utils.utils as utils   
import os 
import numpy as np 
def main(args):
    name = "china"
    #data_tool = data.DataHighSNR(file_name="data/2019gzip.h5", stride=8, n_length=5120, padlen=256, maxdist=300)
    data_tool = utils.Datas()
    device = torch.device("cuda:0")
    model = Model()
    model.train()
    try:
        model.load_state_dict(torch.load("ckpt/polar.pt"))
    except:
        pass 
    model.to(device)
    lossfn = CTLoss() 
    lossfn.to(device)
    acc_time = 0
    outloss = open(f"data/loss.txt", "a")
    optim = torch.optim.Adam(model.parameters(), 1e-4)
    for step in range(200000):
        st = time.perf_counter()
        a1, a2 = data_tool.batch_data(32)
        wave = torch.tensor(a1, dtype=torch.float).to(device) 
        wave = wave.permute(0, 2, 1)
        dc = torch.tensor(a2, dtype=torch.long).to(device) 
        y1, y2 = model(wave)
        loss = lossfn(y1, y2, dc) 
        loss.backward()
        optim.step() 
        optim.zero_grad()
        ls = loss.detach().cpu().numpy()
        ed = time.perf_counter()
        d1, d2 = a2[:, 0], a2[:, 1]
        p1 = y1.argmax(dim=1).cpu().detach().numpy()
        p2 = y2.argmax(dim=1).cpu().detach().numpy()
        outloss.write(f"{step},{ed - st:.1f},{ls:.1f},{np.mean(d1==p1):.3f},{np.mean(d2==p2):.3f}\n")
        outloss.flush()
        acc_time += ed - st

        if step % 100 == 0:
            #print("savemodel")
            torch.save(model.state_dict(), "ckpt/polar.pt")
            print(f"{acc_time:6.1f}, {step:8}, Loss:{ls:6.1f}")
            gs = gridspec.GridSpec(1, 1) 
            d1, d2 = a2[0]
            p1 = y1.argmax(dim=1).cpu().detach().numpy()
            p2 = y2.argmax(dim=1).cpu().detach().numpy()
            fig = plt.figure(1, figsize=(16, 8), dpi=100) 
            p1 = y1.argmax(dim=1).cpu().detach().numpy()[0]
            p2 = y2.argmax(dim=1).cpu().detach().numpy()[0]
            
            w = a1[0, :, -1]
            d = a2[0]
            w /= np.max(w)
            name1 = ["U", "D"] 
            name2 = ["I", "M", "E"]
            #print(p.shape, d.shape)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(w, alpha=0.5, c="k") 
            ax.set_title(f"True:{name1[d1]}-{name2[d2]}, Pred:{name1[p1]}-{name2[p2]}", fontsize=18)
            plt.savefig("data/demo.png") 
            plt.cla() 
            plt.clf()
            acc_time = 0 
    print("done!")
#/home/yuzy/software/anaconda39/bin/python /home/yuzy/machinelearning/lppntorch/lppn.train.py
# 3000310
#nohup /home/yuzy/software/anaconda39/bin/python /home/yuzy/machinelearning/polar/ploar.train.py > data/polar.log 2>&1 &
import argparse
if __name__ == "__main__":

    main([])




