from http import client
import tqdm
import matplotlib.gridspec as gridspec
import argparse
from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
import time
import multiprocessing
from multiprocessing import Barrier, Lock, Process
import time
from datetime import datetime
import pickle
from numpy.lib.function_base import percentile
import obspy
import numpy as np
import math
import os
import obspy
import scipy.signal as signal
import datetime
from obspy.geodetics.base import calc_vincenty_inverse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from config.picker import Parameter as global_parameter
import re 

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
                 kernel_size=5, stride=1, padding=1, output_padding=1, bias=True, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvTBNReLU, self).__init__(
            nn.UpsamplingBilinear2d(scale_factor=tuple(stride)),
            nn.Conv2d(n_in, n_out, kernel_size, stride=1, padding=padding),
            # nn.ConvTranspose2d(n_in, n_out,
            #kernel_size, stride=stride, padding=padding,
            # output_padding=output_padding, bias=False, dilation=1),
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
            # return self.skip_add.add(x, self.conv(x))
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
        self.n_stride = n_stride  # 总步长
        self.layers = nn.Sequential(
            QInvertedResidual(3, 8, 1, 2),
            QInvertedResidual(8, 16, 1, 2),
            QInvertedResidual(16, 16, 2, 2),
            QInvertedResidual(16, 32, 1, 2),
            QInvertedResidual(32, 32, 2, 2),
            QInvertedResidual(32, 64, 1, 2),
            QInvertedResidual(64, 96, 2, 2)
        )
        self.class_encoder = nn.Sequential(
            QInvertedResidual(96, 128, 2, 2),
            QInvertedResidual(128, 156, 2, 2),
            QInvertedResidual(156, 200, 2, 2),
            QInvertedResidual(200, 256, 2, 2),
            ConvTBNReLU(256, 200, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(200, 156, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(156, 128, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(128, 96, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
        )
        self.cl = nn.Conv2d(96 * 2, 7, 1)
        self.tm = nn.Conv2d(96 * 2, 1, 1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.qfunc = nn.quantized.FloatFunctional()

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == ConvTBNReLU:
                fuse_modules(m, ['1', '2', '3'], inplace=True)
            if type(m) == QInvertedResidual:
                m.fuse_model()

    def forward(self, x, device):
        with torch.no_grad():
            #print("数据维度", x.shape)
            T, C = x.shape
            seqlen = 5120
            batchstride = 5120 - 256
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(
                0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            idx = idx.clamp(min=0, max=T-1).long()
            x = x.to(device)
            wave = x[idx, :]
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            max1, max1idx = torch.max(torch.abs(wave), dim=2, keepdim=True)
            max2, max2idx = torch.max(max1, dim=2, keepdim=True)
            wave /= (max2 + 1e-6)
            wave = wave.unsqueeze(2)
            #print("数据维度", wave.shape)
            x1 = self.layers(wave)
            x2 = self.class_encoder(x1)
            x = torch.cat([x1, x2], dim=1)
            out_class = self.cl(x)
            out_time = self.tm(x)
            out_time = out_time * self.n_stride

            oc = out_class.squeeze(dim=2)
            ot = out_time.squeeze(dim=2).squeeze(dim=1)
            #print(oc.shape, ot.shape)
            #print(oc.shape, ot.shape)
            B, C, T = oc.shape
            oc = F.softmax(oc, 1)
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(
                0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C)
            ot += tgrid.squeeze()
            ot = ot.reshape(-1)
            output = []
            #print("NN处理完成", oc.shape, ot.shape)
            for itr in range(6):
                pc = oc[:, itr+1]
                time_sel = torch.masked_select(ot, pc > global_parameter.prob)
                score = torch.masked_select(pc, pc > global_parameter.prob)
                _, order = score.sort(0, descending=True)    # 降序排列
                ntime = time_sel[order]
                nprob = score[order]
                #print(batchstride, ntime, nprob)
                select = -torch.ones_like(order)
                selidx = torch.arange(
                    0, order.numel(), 1, dtype=torch.long, device=device)
                count = 0
                while True:
                    if nprob.numel() < 1:
                        break
                    ref = ntime[0]
                    idx = selidx[0]
                    select[idx] = 1
                    count += 1
                    selidx = torch.masked_select(selidx, torch.abs(
                        ref-ntime) > global_parameter.nmslen)
                    nprob = torch.masked_select(nprob, torch.abs(
                        ref-ntime) > global_parameter.nmslen)
                    ntime = torch.masked_select(ntime, torch.abs(
                        ref-ntime) > global_parameter.nmslen)
                p_time = torch.masked_select(time_sel[order], select > 0.0)
                p_prob = torch.masked_select(score[order], select > 0.0)
                p_type = torch.ones_like(p_time) * itr
                y = torch.stack([p_type, p_time, p_prob], dim=1)
                output.append(y)
            y = torch.cat(output, dim=0)
        return y

class Linker(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),  
            nn.Tanh(), 
            nn.Linear(32, 32), 
            nn.BatchNorm1d(32), 
            nn.Tanh(), 
            nn.Linear(32, 5)
        )
    def forward(self, x):
        y = self.dense(x) 
        return y 

def caldist(pars, r=6371):
    lat1 = torch.deg2rad(pars[:, 1:2])
    lat2 = torch.deg2rad(pars[:, 3:4])
    long1 = torch.deg2rad(pars[:, 0:1])
    long2 = torch.deg2rad(pars[:, 2:3])
    long_diff = long2 - long1
    gd = torch.atan2(
            torch.sqrt((
                torch.cos(lat2) * torch.sin(long_diff)) ** 2 +
                (torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) *
                    torch.cos(lat2) * torch.cos(long_diff)) ** 2),
            torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) *
            torch.cos(long_diff)) * r
    disttime = torch.cat([gd, pars[:, 4:5]], dim=1)
    return disttime/100
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
from obspy.geodetics.base import kilometer2degrees
from obspy.geodetics.base import locations2degrees, degrees2kilometers 
from obspy.taup import TauPyModel
from utils.mseed.dbdata import Client 
from obspy import UTCDateTime
class Process():
    def __init__(self, infile="data", outfile="data/out", stationfile="", model="", mseedidx="", mseeddir="", device="cuda:0"):
        #self.base_dir = "/data/workWANGWT/yangbi/rawdata_semirealtime_archive/archive149"
        self.station = {}
        file_ = open(stationfile, "r", encoding="utf-8") 
        for line in file_.readlines():
            sline = [i for i in line.split(" ") if len(i)>0] 
            key = ".".join(sline[:3]) 
            x, y = float(sline[3]), float(sline[4]) 
            self.station[key] = [x, y]
        file_.close() 
        self.base_dir = infile
        self.modeldir = model
        self.device_name = device 
        self.mseedidx = mseedidx 
        self.mseeddir = mseeddir 
        # 这里通道数不太合理，取得比较大
        n_thread = 6   # 16
        dataq = multiprocessing.Queue(maxsize=100)
        feedq = multiprocessing.Queue(maxsize=100)
        batchq = multiprocessing.Queue(maxsize=100)
        npq = multiprocessing.Queue(maxsize=100)
        pickq = multiprocessing.Queue(maxsize=100)
        outq = multiprocessing.Queue(maxsize=100)
        self.outfile = outfile
        self.infile = infile
        self.processed_dict = {}
        if os.path.exists(f"{self.outfile}.log"):
            logfile = open(f"{self.outfile}.log", "r", encoding="utf-8")
            for line in logfile.readlines():
                key = line.strip()
                self.processed_dict[key] = 0
            logfile.close()
        self.data = [[(j, i) for i in range(3)] for j in range(10)]
        for itr in range(n_thread):
            t_data = multiprocessing.Process(
                target=self.process, args=(feedq, outq))
            t_data.start()
        multiprocessing.Process(target=self.feed3, args=(
                feedq, self.processed_dict)).start()

        multiprocessing.Process(target=self.write, args=(outq, )).start()

    def process(self, feedq, dataq):
        seq_len = global_parameter.nsamplesdots
        model = TauPyModel()
        def caltime(d):
            if d<200:
                arrivals = model.get_travel_times(source_depth_in_km=10,
                                            distance_in_degree=kilometer2degrees(d), 
                                            phase_list=["p", "Pg"]) 
            else:
                arrivals = model.get_travel_times(source_depth_in_km=10,
                                            distance_in_degree=kilometer2degrees(d), 
                                            phase_list=["P", "p", "Pg"])         
            arrivp = arrivals[0].time
            if d <200:
                arrivals = model.get_travel_times(source_depth_in_km=10,
                                            distance_in_degree=kilometer2degrees(d), 
                                            phase_list=["s", "Sg"]) 
            else:
                arrivals = model.get_travel_times(source_depth_in_km=10,
                                            distance_in_degree=kilometer2degrees(d), 
                                            phase_list=["S", "s", "Sg"])         
            arrivs = arrivals[0].time
            return arrivp, arrivs
        client = Client(self.mseedidx, datapath_replace=["^", self.mseeddir])
        device = torch.device(self.device_name)
        lppn = Model(8)
        lppn.eval()
        #script = torch.jit.script(lppn)
        #torch.jit.save(script, "lppn2.pt")
        lppn.to(device)
        lppn.load_state_dict(torch.load(self.modeldir))

        lppn.fuse_model()
        linker = Linker() 
        linker.eval() 
        linker.to(device) 
        linker.load_state_dict(torch.load("ckpt/link32.ckpt"))
        while True:
            temp = feedq.get()
            ekey = temp["ekey"]
            etime = temp["etime"]
            outputs = []
            retypes = []
            for skey in self.station:
                net, sta, loc = skey.split(".")
                x1, y1 = self.station[skey] 
                x2, y2 = temp["eloc"]
                dist = degrees2kilometers(locations2degrees(y1, x1, y2, x2))
                pt, st = caltime(dist)
                time11 = etime + datetime.timedelta(seconds=pt-30) # 截取开始前10秒
                time12 = etime + datetime.timedelta(seconds=st+30) # 截取S结束后10秒
                t1 = UTCDateTime(time11.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
                t2 = UTCDateTime(time12.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
                cha = "?H?"
                st1 = client.get_waveforms(net, sta, loc, cha, t1, t2)
                data = [] 
                for tr in st1[:3]:
                    data.append(tr.data) 
                if len(data)!=3:
                    continue 
                ldata = [len(d) for d in data]
                minl = np.min(ldata)
                if minl <10:continue 
                data = [d[:minl] for d in data]
                data = np.vstack(data).T 
                #print("开始拾取")
                with torch.no_grad():
                    nnout = lppn(torch.tensor(data, dtype=torch.float), device)
                    loc = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32) 
                    x = torch.zeros([len(nnout), 5]).to(device) 
                    x[:, :4] = loc 
                    x[:, 4] = nnout[:, 1]/100 + pt - 30 
                    y = linker(caldist(x)) 
                    retype = y.argmax(dim=1)
                    retype = retype.cpu().numpy()
                    nnout = nnout.cpu().numpy()
                #print("拾取完成", ekey, skey)
                outputs.append([time11, nnout, skey, [x2, y2], dist, retype]) 
            outinfo = {"ekey":ekey, "etime":etime, "eline":temp["eline"], "out":outputs, "retype":retype}
            dataq.put(outinfo)

    def feed3(self, feedq, processed_dict):
        if True:
            # 其他文件夹
            # #EVENT SC.202001312355.0001 eq 2020 01 31 031 15 55 35 550000 LOC  104.519   29.474 DEP        4 MAG   ML  0.5 20200131
            file_ = open(self.infile, "r", encoding="utf-8")
            for line in file_.readlines():
                if "#" not in line:continue 
                sline = [i for i in line.strip().split(" ") if len(i)>0]
                ekey = sline[1]
                if ekey in processed_dict:continue 
                etime = datetime.datetime.strptime(f"{sline[3]}-{sline[4]}-{sline[5]} {sline[7]}:{sline[8]}:{sline[9]}.{sline[10]}", "%Y-%m-%d %H:%M:%S.%f")
                info = {"ekey": ekey, "etime": etime, "eloc":[float(sline[12]), float(sline[13])], "eline":line}
                feedq.put(info)
            #print(f"当前文件夹{root}, {len(files)}, {feedq.qsize()}")

    def write(self, outq):
        # 输出文件
        import scipy.signal as signal
        files = open(f"{self.outfile}.txt", "w", encoding="utf-8")
        logfile = open(f"{self.outfile}.log", "w", encoding="utf-8")
        files.write(
            "##数据格式为:\n##数据位置\n##震相,相对时间（秒）,置信度,绝对时间（格式%Y-%m-%d %H:%M:%S.%f）,信噪比,前后200个采样点振幅均值,前95%分位数,后95%分位数,最大值,标准差,峰值\n")
        count = 0
        acc_time = np.ones([100])
        acc_index = 0
        ifplot = global_parameter.ifplot
        fcount = 0
        while True:
            start = time.perf_counter()
            # 文件位置，拾取结果，数据，起始时间
            outinfo = outq.get()
            ekey = outinfo["ekey"]
            eline = outinfo["eline"]
            etime = outinfo["etime"]
            outputs = outinfo["out"] 
            retype = outinfo["retype"]
            files.write(f"{eline.strip()}\n")
            logfile.write(f"{ekey}\n")
            phase_dict = {
                -1:"N", 
                0: "Pg",
                1: "Sg",
                2: "Pn",
                3: "Sn",
                4: "P",
                5: "S"
            }
            
            for out in outputs:
                time11, nnpick, skey, sloc, dist, retype = out 
                for itr in range(6):  # 六个震相
                    phases = nnpick[nnpick[:, 0] == itr]
                    rtype = retype[nnpick[:, 0] == itr]
                    for p, t, rt in zip(phases[:, 2], phases[:, 1], rtype):
                        pidx = int(t)
                        ptime = time11 + datetime.timedelta(seconds=t/100)
                        tdelta = (ptime-etime).total_seconds()
                        pstr = ptime.strftime('%Y-%m-%d %H:%M:%S.%f')
                        files.write(
f"PHASE,{phase_dict[itr]},{phase_dict[rt-1]},{tdelta:.3f},{p:.3f},{pstr},{skey},{dist:.2f},{sloc[0]:.3f},{sloc[1]:.3f}\n")
                        files.flush()
            files.flush()
            logfile.flush()
            end = time.perf_counter()
            acc_time[acc_index % 100] = end-start
            acc_index += 1
            print(
                f"已处理:{acc_index}, 当前完成{ekey}, 平均用时{np.mean(acc_time)}")


# nohup python lppn.py -o result/3days.new -m ckpt/new.200km.wave -d cuda:0 > result/2020.200km.log 2>&1 &
# 873412
if __name__ == "__main__":
    # 处理输出某个时间之后的所有震相
    parser = argparse.ArgumentParser(description="拾取连续波形")
    parser.add_argument('-i', '--input', default="data/data3DAY2020ForTest/phase.rpt.threedays.20200410TO0412.GMTTIME", help="输入地震目录")
    parser.add_argument(
        '-o', '--output', default="odata/segdata", help="输出文件名")
    parser.add_argument(
        '-m', '--model', default="ckpt/new.200km.wave", help="模型文件lppnmodel")
    parser.add_argument('-d', '--device', default="cuda:0",
                        help="模型文件lppnmodel")
    parser.add_argument(
        '-s', '--station', default="data/station.113.FOR.ASSO", help="台站位置")
    parser.add_argument(
        '-q', '--mseedidx', default="/data/waveindexDATA/mseedIndexDB/mseedidxdb.CNNET.BHSH.100HZ_2020.db.sqlite3", help="索引文件")
    parser.add_argument(
        '-r', '--mseeddir', default="/data/CSNDATA/", help="数据位置")    
    args = parser.parse_args()
    infile = args.input
    outfile = args.output
    Process(infile, outfile, 
    stationfile=args.station, 
    model=args.model,
     mseedidx=args.mseedidx, 
     mseeddir=args.mseeddir)
