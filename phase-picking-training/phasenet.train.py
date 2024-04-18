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
    def __init__(self, nin=8, nout=11, ks=[7, 1], st=[4, 1], padding=[3, 0], output_padding=[3, 0]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, ks, st, padding=padding, output_padding=output_padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU(), 
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
        self.layer5 = Conv2dT(128, 64, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer6 = Conv2dT(128, 32, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer7 = Conv2dT(64, 16, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer8 = Conv2dT(32, 8, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer9 = nn.Conv2d(16, 3, [7, 1], [1, 1], padding=[3, 0])
    def forward(self, x):
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        x6 = torch.cat([x4, x6], dim=1) 
        x7 = self.layer6(x6)
        x7 = torch.cat([x3, x7], dim=1) 
        x8 = self.layer7(x7)
        x8 = torch.cat([x2, x8], dim=1) 
        x9 = self.layer8(x8)
        x9 = torch.cat([x1, x9], dim=1) 
        x10 = self.layer9(x9)
        x10 = F.softmax(x10, dim=1)
        x10 = x10.squeeze()
        return x10
class CTLoss(nn.Module):
    """损失函数"""
    def __init__(self):
        super().__init__()
        pass 
    def forward(self, pclass, dclass):
        loss = -(dclass * torch.log(pclass+1e-6)).sum(dim=1).sum()
        return loss
    

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
import utils 
import os 
import numpy as np 
def main(args):
    stride = 8
    nchannel = 4
    model_name = f"ckpt/phasenet.wave"
    data_tool = utils.DataGauss(batch_size=32, n_thread=1, strides=8, n_length=3072)
    device = torch.device("cuda:0")
    model = UNet()
    try :
        model.load_state_dict(torch.load(model_name))
    except:
        pass
    model.to(device)
    lossfn = CTLoss() 
    lossfn.to(device)
    acc_time = 0
    outloss = open(f"data/phasenet.loss.txt", "a")
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-2)
    count = 0 
    for varkey in model.state_dict():
        var = model.state_dict()[varkey] 
        c = 1
        for i in var.shape:
            c *= i 
        count += c 
    print("可训练参数数量", count)

    for step in range(100000000):
        st = time.perf_counter()
        a1, a2, a3, a4 = data_tool.batch_data()
        wave = torch.tensor(a1, dtype=torch.float).to(device) 
        wave = wave.permute(0, 2, 1)
        wave = wave.unsqueeze(dim=3)
        dc = torch.tensor(a2, dtype=torch.float).to(device) 
        dc = dc.permute(0, 2, 1)
        #print(wave.shape) 
        oc = model(wave)
        loss = lossfn(oc, dc) 
        loss.backward()
        if loss.isnan():
            optim.zero_grad()
            continue 
        optim.step() 
        optim.zero_grad()
        ls = loss.detach().cpu().numpy()
        ed = time.perf_counter()
        #print(ls)
        outloss.write(f"{step},{ed - st},{ls}\n")
        outloss.flush()
        #print("writeloss")
        acc_time += ed - st

        if step % 100 == 0:
            #print("savemodel")
            torch.save(model.state_dict(), model_name)
            print(f"{acc_time:6.1f}, {step:8}, Loss:{ls:6.1f}")
            gs = gridspec.GridSpec(3, 1) 
            fig = plt.figure(1, figsize=(16, 8), dpi=100) 
            p = oc.detach().cpu().numpy()[0]
            w = a1[0, :, 0]
            d = a2[0]
            w /= np.max(w)
            names = [
                "None", 
            "Pg", 
            "Sg", 
            "P", 
            "S", 
            "Pn", 
            "Sn"]
            #print(p.shape, d.shape)
            for i in range(3):
                ax = fig.add_subplot(gs[i, 0])
                ax.plot(w, alpha=0.3, c="k") 
                ax.plot(d[:, i], c="b", alpha=0.5)
                ax.plot(p[i, :], c="r", alpha=0.5) 
                ax.set_ylabel(names[i], fontsize=18)
                ax.set_title(f"{a4[0]}")
            plt.savefig(f"data/phasenet.png") 
            plt.cla() 
            plt.clf()
            acc_time = 0 
    print("done!")

import argparse
if __name__ == "__main__":

    main([])