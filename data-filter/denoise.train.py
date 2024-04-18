import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
class AutoEncoder(nn.Module):
    """一维去噪自编码器波形滤波"""
    def __init__(self):
        super().__init__() 
        K = 5        # 卷积核心大小，这里设置为5
        S = 2        # 每次计算步长
        P = (K-1)//2 # 补充0长度
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, K, S, padding=P), 
            nn.BatchNorm1d(16), 
            nn.ReLU(), 
            nn.Conv1d(16, 32, K, S, padding=P), 
            nn.BatchNorm1d(32), 
            nn.ReLU(),            
        )
        # 定义解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, K, S, P, output_padding=S-1), 
            nn.BatchNorm1d(16), 
            nn.ReLU(), 
            nn.ConvTranspose1d(16, 3, K, S, P, output_padding=S-1), 
            nn.Tanh() # 约束到-1~1区间，迭代更加稳定
        )
    def forward(self, x):
        h = self.encoder(x) # 编码器构建特征
        y = self.decoder(h) # 解码器输出滤波波形
        return y 

class CTLoss(nn.Module):
    """损失函数"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss("sum") 
    def forward(self, a, b):
        loss = self.mse(a, b) 
        return loss  
    

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.size'] = 24
import utils 
import os 
import numpy as np 
def main(args):
    ##name = "NSZONE-all-gzip2"
    ##if os.path.exists(f"h5data/{name}")==False:
    ##    os.makedirs(f"h5data/{name}")
    #data_tool = data.DataHighSNR(file_name="data/2019gzip.h5", stride=8, n_length=5120, padlen=256, maxdist=300)
    dist = args.dist 
    model_name = f"ckpt/denoise.pt"
    data_tool = utils.Data(batch_size=32, n_thread=1, strides=8, n_length=3072)
    device = torch.device("cuda:0")
    model = AutoEncoder()
    try :
        model.load_state_dict(torch.load(model_name))
    except:
        pass
    model.to(device)
    lossfn = CTLoss() 
    lossfn.to(device)
    acc_time = 0
    outloss = open(f"data/loss.{dist}km.txt", "a")
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0.0)
    for step in range(100000000):
        st = time.perf_counter()
        a1, a2 = data_tool.batch_data()
        wavenoise = torch.tensor(a1, dtype=torch.float).to(device) 
        wave = torch.tensor(a2, dtype=torch.float).to(device)  
        wavenoise = wavenoise.permute(0, 2, 1)
        #print(wave.shape) 
        clean = model(wavenoise)
        clean = clean.permute(0, 2, 1)
        loss = ((clean - wave) ** 2).sum() 
        loss.backward()
        if loss.isnan():
            print("NAN error")
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
            fig = plt.figure(1, figsize=(16, 16), dpi=100) 
            oc = clean 
            p = oc.detach().cpu().numpy()
            w = a1[0, :, 0]
            d = a2[0, :, 0] 
            c = p[0, :, 0]
            t = np.arange(len(w)) * 0.01
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(t, w, alpha=1, lw=1, c="k", label="噪声+波形")
            ax.set_ylim((-1.2, 1.2))

            ax.legend(loc="upper right")
            ax = fig.add_subplot(gs[1, 0]) 
            ax.plot(t, d, alpha=1, lw=1, c="k", label="原始波形") 
            ax.set_ylim((-1.2, 1.2))
            ax.legend(loc="upper right")
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(t, c, alpha=1, lw=1, c="k", label="滤波后") 
            ax.set_ylim((-1.2, 1.2))
            ax.legend(loc="upper right")

            print(np.sum((d-c)**2))
            plt.savefig(f"logdir/神经网络滤波.jpg") 
            plt.savefig(f"logdir/神经网络滤波.svg") 
            plt.cla() 
            plt.clf()
            acc_time = 0 
    print("done!")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形")       
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名")      
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件lppnmodel")                                                            
    args = parser.parse_args()      
    main(args)




