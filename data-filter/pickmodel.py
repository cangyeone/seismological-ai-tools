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
                 kernel_size=3, stride=1, 
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        padding = (kernel_size - 1) // 2
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(n_in, n_out, [1, kernel_size], 
                      stride, [0, padding], groups=groups, 
                      bias=False),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )
class ConvTBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=3, stride=1, padding=1, output_padding=1, bias=True, dilation=1,  
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvTBNReLU, self).__init__(
            nn.UpsamplingBilinear2d(scale_factor=tuple(stride)), 
            nn.Conv2d(n_in, n_out, kernel_size, stride=1, padding=padding), 
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
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
    def __init__(self, n_stride=128, n_channel=4):
        super().__init__()
        self.n_stride = n_stride # 总步长 
        base = n_channel 
        self.layers = nn.Sequential(
            QInvertedResidual(     3, base*1, 1, 2), 
            QInvertedResidual(base*1, base*2, 1, 2), 
            QInvertedResidual(base*2, base*2, 2, 2), 
            QInvertedResidual(base*2, base*3, 1, 2), 
            QInvertedResidual(base*3, base*3, 2, 2),
            QInvertedResidual(base*3, base*4, 1, 2), 
            QInvertedResidual(base*4, base*5, 2, 2)             
        )
        self.class_encoder = nn.Sequential(
            QInvertedResidual(base*5, base*5, 2, 2), 
            QInvertedResidual(base*5, base*5, 2, 2), 
            QInvertedResidual(base*5, base*5, 2, 2), 
            ConvTBNReLU(base*5, base*5, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1), 
            ConvTBNReLU(base*5, base*5, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),  
            ConvTBNReLU(base*5, base*5, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1), 
        )
        self.cl = nn.Conv2d(base * 5 * 2, 3, 1) 
        self.tm = nn.Conv2d(base * 5 * 2, 1, 1)
    def forward(self, x, device):
        B, C, T = x.shape 
        t = torch.arange(T) * 2 * 3.141592658 / 4 
        p = torch.stack([torch.sin(t), torch.sin(2*t), torch.sin(4*t)], dim=0).to(device) 
        p = torch.unsqueeze(p, 0) 
        x = x + p 
        x = x.unsqueeze(2)
        x1 = self.layers(x)
        x2 = self.class_encoder(x1) 
        x = torch.cat([x1, x2], dim=1)
        out_class = self.cl(x).squeeze()
        out_time = self.tm(x).squeeze()
        out_time = out_time * self.n_stride 
        out_time = out_time.squeeze()
        return out_class, out_time
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
        self.mse = nn.MSELoss(reduction="none") 
        # 加权
        weight = torch.Tensor([1, 1, 1]).float()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="sum")
    def forward(self, pclass, ptime, dclass, dtime):
        loss_class = self.ce(pclass, dclass+1) 
        loss_time_none = self.mse(ptime, dtime) * (dclass+1).clamp(0, 1).float() 
        loss_time = loss_time_none.sum() 
        loss = loss_class + loss_time / 10 
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
    nchannel = 8
    model_name = f"ckpt/{stride}-{nchannel}.wave"
    data_tool = utils.Data(batch_size=32, n_thread=1, strides=8, n_length=3072)
    device = torch.device("cuda:1")
    model = Model(n_stride=stride, n_channel=nchannel)
    try :
        model.load_state_dict(torch.load(model_name))
    except:
        pass
    model.to(device)
    lossfn = CTLoss() 
    lossfn.to(device)
    acc_time = 0
    outloss = open(f"data/{stride}-{nchannel}.loss.txt", "a")
    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0e-3)
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
        dc = torch.tensor(a2[:, :, 0], dtype=torch.long).to(device) 
        dt = torch.tensor(a2[:, :, 1], dtype=torch.float).to(device)
        #print(wave.shape) 
        oc, ot = model(wave, device)
        loss = lossfn(oc, ot, dc, dt) 
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
            oc = F.softmax(oc, 1)
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
                if i == 0:
                    ax.plot(np.repeat(d[:, 0], stride), c="b", alpha=0.5)
                ax.plot(np.repeat(p[i, :], stride), c="r", alpha=0.5) 
                ax.set_ylabel(names[i], fontsize=18)
            plt.savefig(f"data/{stride}-{nchannel}.png") 
            plt.cla() 
            plt.clf()
            acc_time = 0 
    print("done!")

import argparse
if __name__ == "__main__":

    main([])




