import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np 
from obspy.geodetics import degrees2kilometers, locations2degrees 
import tqdm 
import scipy.interpolate as interpolate 
from scipy.io import loadmat 
def sline(line):
    return [i for i in line.split(" ") if len(i)>0] # 以空格作为间隔
def readdisp(path):
    data1, data2 = [], []
    with open(path, "r", encoding="utf-8") as f:
        tag = 0 

        for line in f.readlines()[2:]:
            sl = sline(line)
            if float(sl[1])<0.01:continue 
            data1.append([float(sl[0]), float(sl[1])])
    data1, data2 = np.array(data1), np.array(data2)
    func1 = interpolate.interp1d(data1[:, 0], data1[:, 1], fill_value=-1, bounds_error=False)
    func2 = interpolate.interp1d(data1[:, 0], data1[:, 1], fill_value=-1, bounds_error=False)
    return func1, func2
def norm(t, mu=0, std=0.02):
    p = np.exp(-(t-mu)**2/2/std**2)
    p = p/(np.max(p)+1e-6)
    return p 
def readmat(path1, path2, path3, path4):

    mats = np.loadtxt(path2) # 图像599,146
    #mats = loadmat(path2) # 
    #mat1 = mats["disper_map_stack_A2B"]
    #mat2 = mats["disper_map_stack_B2A"]
    #mats = mats["disper_map_stack_SYM"]
    h, w = mats.shape
    #print(mats.shape)
    #print("格式", mats.shape)
    velo = np.linspace(0.4, 4, h) 
    prid = np.linspace(0.2, 5, w)
    
    funcs = readdisp(path1) # 读取的频散
    label = np.zeros([h, w, 2])
    NV = len(velo)
    dots = []
    for nk in range(2):
        for idx, vel in enumerate(prid):
            v = funcs[nk](vel) 
            if v<=0:continue 
            p = norm(velo, v, 0.06)
            dots.append([idx, np.argmax(p)])
            label[:, idx, nk] = p 
    return mats, label, np.array(dots) 
class NCFDataset(Dataset):
    def __init__(self):
        basedir = "data" # 数据文件夹位置
        file_names = os.listdir(basedir) 
        self.paths = []
        for fn in file_names:
            if "GDisp." not in fn:continue 
            path1 = os.path.join(basedir, fn)       # 相速度图
            fn2 = fn.replace("GDisp.", "G.")
            #path2 = path1.replace(".npy", ".npv") # 群速度图
            #path3 = path1.replace(".npy", ".npf") # 群速度图
            #path4 = path1.replace(".npy", ".txt") # 群速度图
            path2 = os.path.join(basedir, fn2)
            mat = np.loadtxt(path2) 
            sp = mat.shape 
            if sp[0]!=181 or sp[1]!=49:continue  
            self.paths.append([path1, path2, path1, path2])
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p1, p2, p3, p4 = self.paths[idx] 
        x, d, c = readmat(p1, p2, p3, p4)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(d, dtype=torch.float32), c

def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds, cs = [], [], []
    for x, d, c in batch:
        xs.append(x) 
        ds.append(d)
        cs.append(c)
    xs = torch.stack(xs, dim=0) 
    ds = torch.stack(ds, dim=0) 
    return xs, ds, cs
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


class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
    def forward(self, x, d):
        #loss = - (d * torch.log(x+1e-9)).sum()
        loss = F.l1_loss(x, d)
        return loss 

import matplotlib.pyplot as plt 
import matplotlib.gridspec as grid 
plt.switch_backend("agg")
def main():
    train_dataset = NCFDataset()     
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch, num_workers=3)

    device = torch.device("cpu")
    name = "ckpt/cnn.disp4"
    model = UNet()
    model.train() 
    model.to(device) 
    try:
        if os.path.exists(name):
            model.load_state_dict(torch.load(name))
    except:
        pass 
    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0e-1)
    lossfn = Loss()
    n_epoch = 1000
    count = 0
    if os.path.exists("ckpt") == False:os.mkdir("ckpt")
    #loss_file = open("ckpt/loss1.txt", "a", encoding="utf-8")
    for b in range(n_epoch):
        for x, d, disp in train_dataloader:
            #print(x.shape, d.shape, m.shape)
            x = x.unsqueeze(1)[:, :, :512, :128]
            d = d.permute(0, 3, 1, 2)[:, :, :512, :128]
            #print(x.shape, d.shape)
            x = x.to(device)
            d = d.to(device)
            y = model(x)
            loss = lossfn(y, d) 
            loss.backward() 
            optim.step() 
            optim.zero_grad()
            count += 1
            if count % 2 ==0:
                print(count, loss)
                x = x.detach().cpu().numpy()[0] 
                y = y.detach().cpu().numpy()[0]
                d = d.cpu().numpy()[0]
                torch.save(model.state_dict(), name)  
                fig = plt.figure(1, figsize=(18, 6))
                gs = grid.GridSpec(2, 3) 
                c, h, w = x.shape 
                h = np.arange(h) 
                w = np.arange(w)
                dis = disp[0]
                ax = fig.add_subplot(gs[0, 0])
                ax.pcolormesh(w, h, x[0], shading='auto')
                ax.scatter(dis[:, 0], dis[:, 1], c="r")
                ax = fig.add_subplot(gs[0, 1])
                ax.pcolormesh(w, h, d[0], shading='auto')
                #ax.scatter(dis[:, 0], dis[:, 1], c="r")
                ax = fig.add_subplot(gs[0, 2])
                ax.pcolormesh(w, h, d[1], shading='auto')
                #  
                ax = fig.add_subplot(gs[1, 0])
                ax.pcolormesh(w, h, x[0], shading='auto')
                hh, w2 = y[0].shape
                dis = []
                for idx in range(w2):
                    p = y[0][:, idx]
                    idy = np.argmax(p) 
                    prob = p[idy] 
                    if prob>0.05:
                        dis.append([idx, idy]) 
                dis = np.array(dis)
                #print(hh, w2, dis)
                if len(dis)>0:
                    ax.scatter(dis[:, 0], dis[:, 1], c="b")
                ax = fig.add_subplot(gs[1, 1])
                ax.pcolormesh(w, h, y[0], shading='auto')
                #ax.scatter(dis[:, 0], dis[:, 1], c="r")
                ax = fig.add_subplot(gs[1, 2])
                ax.pcolormesh(w, h, y[1], shading='auto')
                #ax.scatter(dis[:, 0], dis[:, 1], c="r")                
                plt.savefig(f"logdir/demo.png")
                plt.cla() 
                plt.clf()
#nohup /home/yuzy/software/anaconda39/bin/python /home/yuzy/machinelearning/surfacewave/ncf2disp.train.py > ckpt/ncf2disp.log 2>&1 &
#1802189
if __name__ == "__main__":
    main()