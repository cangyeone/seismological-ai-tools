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
        loss = loss1 + loss2 
        return loss  

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
plt.switch_backend('agg')
import utils.utils as utils   
import os 
import numpy as np 
from obspy.signal.filter import bandpass 

from sklearn.metrics import roc_curve, auc
def main(args):
    name = "china"
    #data_tool = data.DataHighSNR(file_name="data/2019gzip.h5", stride=8, n_length=5120, padlen=256, maxdist=300)
    data_tool = utils.DatasTest()
    device = torch.device("cuda:0")
    model = Model()

    model.load_state_dict(torch.load("ckpt/polar.pt"))
    model.to(device)
    model.eval()
    lossfn = CTLoss() 
    lossfn.to(device)
    pp1 = [] 
    pp2 = [] 
    dd = []
    waves = []
    prob =  []
    for step in range(20):
        a1, a2 = data_tool.batch_data(100)
        wave = torch.tensor(a1, dtype=torch.float).to(device) 
        wave = wave.permute(0, 2, 1)
        dc = torch.tensor(a2, dtype=torch.long).to(device) 
        y1, y2 = model(wave)
        p = torch.softmax(y1, dim=1).cpu().detach().numpy() 
        prob.append(p)
        p1 = y1.argmax(dim=1).cpu().detach().numpy()
        p2 = y2.argmax(dim=1).cpu().detach().numpy()
        pp1.append(p1) 
        pp2.append(p2) 
        dd.append(a2)
        waves.append(a1)
        print(step)
    wave = np.concatenate(waves, axis=0)
    p1 = np.concatenate(pp1, axis=0) 
    p2 = np.concatenate(pp2, axis=0) 
    d = np.concatenate(dd, axis=0) 
    prob = np.concatenate(prob, axis=0)
    ww1 = wave[p1==0] 
    ww2 = wave[p2==1]

    def norm(x):
        x = x.reshape(-1)
        x = x/np.max(np.abs(x))
        x = bandpass(x, 1, 10, 100)
        x = x[256:-256]
        return x 
    for k in range(0):
        gs = gridspec.GridSpec(4, 2)
        fig = plt.figure(1, figsize=(12, 12)) 
        for i in range(4):
            w1 = norm(ww1[i+k*4])
            w2 = norm(ww2[i+k*4])
            t = np.arange(len(w1))*0.01
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(t, w1, c="k")
            if i == 0:
                ax.set_title("Up")
            if i == 3:
                ax.set_xlabel("Time/s")
            ax = fig.add_subplot(gs[i, 1])
            ax.plot(t, w2, c="k")
            if i == 0:
                ax.set_title("Down")
            if i == 3:
                ax.set_xlabel("Time/s")
        plt.savefig(f"figs/fig3.{k}.jpg")
        plt.savefig(f"figs/fig3.{k}.jpg")
        plt.close()

    level = np.linspace(0, 1, 30)
    tpr = [] 
    fpr = []
    d = d[:, 0]
    for e in level:
        p = (prob[:, 0]<=e).astype(np.int) 
        tp = np.sum((p==0) * (d==0))
        fp = np.sum((p==0) * (d==1)) 
        tn = np.sum((p==1) * (d==1))
        fn = np.sum((p==1) * (d==0))
        tpr.append(tp/(tp+fn)) 
        fpr.append(fp/(tp+tn)) 
    #print(e, tp/(tp+fp), tp/(tp+fn))
    tpr[0] = 1 
    fpr[0] = 1
    tpr.pop(-1)
    fpr.pop(-1)
    tpr.append(0)
    fpr.append(0)
    print(tpr, fpr)
    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(1, figsize=(12, 12))   
    ax = fig.add_subplot(gs[0])  
    xx = np.linspace(0, 1, 100)
    ax.plot(xx, xx, c="k", linestyle="--", label="Random")
    ax.plot(fpr, tpr, c="k", label="ROC Curve")
    ax.fill(fpr, tpr, color="k", alpha=0.3)
    ax.set_title(f"AUC={auc(fpr, tpr):.3f}")
    
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("TPR")
    ax.set_ylabel("FPR")
    plt.savefig("ROC.png")
    plt.savefig("ROC.svg")
    plt.close()

#/home/yuzy/software/anaconda39/bin/python /home/yuzy/machinelearning/lppntorch/lppn.train.py
#nohup /home/yuzy/software/anaconda39/bin/python /home/yuzy/machinelearning/polar/ploar.train.py > data/polar.log 2>&1 &
import argparse
if __name__ == "__main__":

    main([])




