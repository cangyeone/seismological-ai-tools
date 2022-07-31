import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
class ConvBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=5, stride=1, 
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        padding = (kernel_size - 1) // 2
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(n_in, n_out, [1, kernel_size], 
                      stride, [0, padding], groups=1, 
                      bias=False),
            nn.BatchNorm2d(n_out),
            nn.Tanh()
        )
class ConvTBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=5, stride=1, padding=1, output_padding=1, bias=True, dilation=1,  
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvTBNReLU, self).__init__(
            nn.UpsamplingBilinear2d(scale_factor=tuple(stride)), 
            nn.Conv2d(n_in, n_out, kernel_size, stride=1, padding=padding), 
            #nn.ConvTranspose2d(n_in, n_out, 
            #kernel_size, stride=stride, padding=padding, 
            #output_padding=output_padding, bias=False, dilation=1),
            nn.BatchNorm2d(n_out),
            nn.Tanh()
        )
class InvertedResidual(nn.Module):
    """
    本个模块为MobileNetV2中的可分离卷积层
    中间带有扩张部分，如图10-2所示
    """
    def __init__(self, n_in, n_out, 
                 stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        # 隐藏层需要进行特征拓张，以防止信息损失
        hidden_dim = int(round(n_in * expand_ratio))
        # 当输出和输出维度相同时，使用残差结构
        self.use_res = self.stride == 1 and n_in == n_out
        # 构建多层
        layers = []
        if expand_ratio != 1:
            # 逐点卷积，增加通道数
            layers.append(
                ConvBNReLU(n_in, hidden_dim, kernel_size=1, 
                            norm_layer=norm_layer))
        layers.extend([
            # 逐层卷积，提取特征。当groups=输入通道数时为逐层卷积
            ConvBNReLU(
                hidden_dim, hidden_dim, 
                stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # 逐点卷积，本层不加激活函数
            nn.Conv2d(hidden_dim, n_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(n_out),
        ])
        # 定义多个层
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)
class QInvertedResidual(InvertedResidual):
    """量化模型修改"""
    def __init__(self, *args, **kwargs):
        super(QInvertedResidual, self).__init__(*args, **kwargs)
        # 量化模型应当使用量化计算方法
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, x):
        if self.use_res:
            # 量化加法
            #return self.skip_add.add(x, self.conv(x))
            return x + self.conv(x)
        else:
            return self.conv(x)
    def fuse_model(self):
        # 模型融合
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                # 将本个模块最后的卷积层和BN层融合
                fuse_modules(
                    self.conv, 
                    [str(idx), str(idx + 1)], inplace=True)
class Model(nn.Module):
    def __init__(self, n_stride=8):
        super().__init__()
        self.n_stride = n_stride # 总步长 
        self.encoder = nn.Sequential(
            QInvertedResidual(3, 8, 2, 2), 
            QInvertedResidual(8, 16, 2, 2), 
            QInvertedResidual(16, 32, 2, 2), 
            QInvertedResidual(32, 64, 2, 2), 
            QInvertedResidual(64, 96, 2, 2),           
        )
        self.decoder = nn.Sequential(
            ConvTBNReLU(96, 64, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(64, 32, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(32, 16, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(16, 8, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(8, 8, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
            nn.Conv2d(8, 3, [1, 5], stride=1, padding=[0, 2])
        )
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x1 = self.encoder(x)
        x  = self.decoder(x1) 
        x = x.squeeze()
        return x
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == ConvTBNReLU:
                fuse_modules(m, ['1', '2', '3'], inplace=True)
            if type(m) == QInvertedResidual:
                m.fuse_model()    

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
    model_name = f"ckpt/denoise.wave"
    data_tool = utils.Data(batch_size=32, n_thread=1, strides=8, n_length=3072)
    device = torch.device("cuda:0")
    model = Model()
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
            
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(w, alpha=0.3, c="k", label="噪声+波形")
            ax.set_ylim((-1.2, 1.2))

            ax.legend(loc="upper right")
            ax = fig.add_subplot(gs[1, 0]) 
            ax.plot(d, c="k", alpha=0.5, label="波形") 
            ax.set_ylim((-1.2, 1.2))
            ax.legend(loc="upper right")
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(c, c="k", alpha=0.5, label="滤波后") 
            ax.set_ylim((-1.2, 1.2))
            ax.legend(loc="upper right")

            print(np.sum((d-c)**2))
            plt.savefig(f"logdir/demo.png") 
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




