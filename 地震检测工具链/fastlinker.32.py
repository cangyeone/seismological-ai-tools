import pickle
from matplotlib.pyplot import grid 
import numpy as np 
from obspy.geodetics import calc_vincenty_inverse, gps2dist_azimuth
import matplotlib.pyplot as plt 
import tqdm 
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.loss import CrossEntropyLoss 
from obspy.geodetics import calc_vincenty_inverse, gps2dist_azimuth, locations2degrees
from obspy.geodetics import degrees2kilometers, locations2degrees 
import matplotlib.pyplot as plt 
from config.fastlink import Parameter as global_par 
import scipy.signal as signal 
import datetime 
plt.switch_backend("agg")
class Model(nn.Module):
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
class Link2d():
    def __init__(self):
        self.link2dnet = Model() 
        self.link2dnet.eval() 
        self.link2dnet.cuda() 
        self.link2dnet.load_state_dict(torch.load("ckpt/link32.ckpt"))
    def links(self, data_tool):
        grid_size = 20
        #print(x, np.max(x, axis=0))
        #min_lon, min_lat, _ = np.min(data_tool.data[:, :3], axis=0) 
        #max_lon, max_lat, _ = np.max(data_tool.data[:, :3], axis=0) 

        if len(global_par.lonrange) == 0:
            min_lon, min_lat = data_tool.minloc - global_par.locpad  
            max_lon, max_lat = data_tool.maxloc + global_par.locpad 
        else: 
            min_lat, max_lat = global_par.latrange
            min_lon, max_lon = global_par.lonrange 

        grid_size = global_par.ngrid    
        L = int((data_tool.maxtm-data_tool.mintm) / global_par.win_stirde)
        par = np.pi/180 * 6371.
        dlon = (max_lon-min_lon)/grid_size 
        dlat = (max_lat-min_lat)/grid_size 
        mgrids = []
        mlinks = []
        mphase = []
        mclass = []
        grid1 = (torch.zeros([grid_size, 1], dtype=torch.float32) + torch.arange(min_lon, max_lon, dlon).reshape(1, grid_size)).cuda().reshape(grid_size, grid_size, 1)   
        grid2 = (torch.arange(min_lat, max_lat, dlat).reshape(grid_size, 1) + torch.zeros([1, grid_size], dtype=torch.float32)).cuda().reshape(grid_size, grid_size, 1)
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
            #r /= 100
            return disttime/100
        if os.path.exists(global_par.datadir)==False:
            os.mkdir(global_par.datadir)
        btime = data_tool.basetime + datetime.timedelta(seconds=data_tool.start)
        for t in tqdm.tqdm(range(L)):
            datas = data_tool.getdata() 
            infos = datas[:, 3]
            data = datas[:, :3].astype(np.float32)  
            N, C = data.shape
            #ptype = datas[:, 4].astype(np.int32)
            #ptype = torch.from_numpy(ptype).long() 
            ptype = torch.zeros([grid_size, grid_size, N]).cuda()
            ptype[:, :, :] = torch.from_numpy(datas[:, 4].astype(np.int64)).long().cuda() 
            
            if N < 3:
                mgrids.append([])
                mlinks.append(0)
                #print(i, j)
                mclass.append(0)
                mphase.append([]) 
                continue 
            d = torch.from_numpy(data).float().cuda()
            grid = torch.zeros([grid_size, grid_size, N, 5]).cuda()
            grid[:, :, :, 0] = grid1 
            grid[:, :, :, 1] = grid2 
            grid[:, :, :, 2:] = d.reshape(1, 1, N, 3)
            grid = torch.reshape(grid, [-1, 5])
            disttime = caldist(grid)
            logit = self.link2dnet(disttime.cuda()) 
            
            prob = F.softmax(logit, dim=1)
            num = (prob[:, 0].reshape([grid_size * grid_size, N])<0.5).float().sum(1)

            pids = torch.argmax(logit, dim=1).reshape(grid_size * grid_size, N) 
            rids = torch.reshape(ptype, [grid_size * grid_size, N])
            num = (pids==rids).float().sum(1)
            
            max_num_id = torch.argmax(num)
            max_num_id = max_num_id.detach().cpu().numpy()
            class_ = logit.argmax(dim=1).reshape(grid_size * grid_size, N).detach().cpu().numpy()
            class_ = class_[max_num_id] 
            posgrid = torch.cat([grid1, grid2], dim=2) 
            posgrid = posgrid.reshape(grid_size * grid_size, 2)
            ph = disttime.reshape(grid_size * grid_size, N, 2).detach().cpu().numpy()[max_num_id]
            pos = posgrid.detach().cpu().numpy()[max_num_id]
            nrow = max_num_id // grid_size 
            ncol = max_num_id % grid_size 
            elon = min_lon + ncol * dlon 
            elat = min_lat + nrow * dlat 
            #print(class_.shape)
            mgrids.append([class_, ph * 100, max_num_id, infos, pos])
            num = num.max().detach().cpu().numpy() 
            #print(num)
            mlinks.append(num
            )
            if (t+1) % global_par.saveitv == 0:
                np.savez(f"{global_par.datadir}/link{t}.npz", mlinks=mlinks, mgrids=mgrids, mphase=mphase, mclass=mclass, mtime=btime)
                mgrids = []
                mlinks = []
                mphase = []
                mclass = []
                btime = data_tool.basetime + datetime.timedelta(seconds=data_tool.start)
        np.savez(f"{global_par.datadir}/link{t}.npz", mlinks=mlinks, mgrids=mgrids, mphase=mphase, mclass=mclass, mtime=btime)
import datetime 
import os 

class DataLPPN():
    def __init__(self, infilename, stationfile):
        self.pos_dict = {}
        file_ = open(stationfile, "r", encoding="utf-8") 
        for line in file_.readlines():
            sline = [i for i in line.split(" ") if len(i)>0] 
            key = ".".join(sline[:2]) 
            x, y = float(sline[3]), float(sline[4]) 
            self.pos_dict[key] = [x, y]
        file_.close() 
        basetime = datetime.datetime.strptime(global_par.basetime, "%Y-%m-%d") # 设置数据起始时间
        self.basetime = basetime 
        file_dir = infilename

        p2id = {"Pg":1, "Sg":2, "Pn":3, "Sn":4, "P":5, "S":6}
        file_ = open(file_dir, "r", encoding="utf-8") 
        datas = []
        for line in file_.readlines():
            if "#" in line:continue
            sline = [i for i in line.strip().split(",") if len(i)>0]
            ptime = datetime.datetime.strptime(sline[3], "%Y-%m-%d %H:%M:%S.%f")
            delta = (ptime-basetime).total_seconds()
            key = sline[6]
            p = float(sline[2])
            snr = float(sline[4])
            #if p < 0.5:continue 
            pname = sline[0]
            if key not in self.pos_dict:continue 
            loc = self.pos_dict[key]
            info = {"st":key, "time":ptime, "pname":pname}
            datas.append([loc[0], loc[1], delta, info, p2id[pname]])
        self.datas = np.array(datas)       
        idx = np.argsort(self.datas[:, 2])
        self.datas = self.datas[idx] 
        self.mintm = np.min(self.datas[:, 2])
        self.start = self.mintm 
        self.data = self.datas[:, :3]
        self.maxtm = np.max(self.datas[:, 2])
        self.minloc = np.min(self.datas[:, :2], axis=0) 
        self.maxloc = np.max(self.datas[:, :2], axis=0)
        self.index = torch.from_numpy(self.datas[:, 2].astype(np.float32)).cuda()
        self.tdata = torch.from_numpy(self.data.astype(np.float32)).cuda()
        print("数据读取完成！", len(datas))
    def getdata(self):
        idx = ((self.index>self.start) * (self.index<self.start + global_par.win_length)).cpu().numpy()
        temp = np.copy(self.datas[idx])
        temp[:, 2] -= self.start 
        self.start += global_par.win_stirde 
        return temp
def link(outfilename):
    outfile = open(outfilename, "w", encoding="utf-8")
    filenames = os.listdir(global_par.datadir)
    for fn in filenames:
        if fn.endswith(".npz") == False:continue 
        path = os.path.join(global_par.datadir, fn)
        file_ = np.load(path, allow_pickle=True)
        mlinks = file_["mlinks"] 
        mgrids = file_["mgrids"] 
        peaks, _ = signal.find_peaks(mlinks)
        L = len(peaks)
        count = 0
        reftime = file_["mtime"]


        #outfile.write("#是否在目录中,EVENT,位置编码,时间\n##PHASE,时间,修正类型,台站,原始类型\n")
        id2t = {0:"N", 1:"Pg", 2:"Sg", 3:"Pn", 4:"Sn", 5:"P", 6:"S"}
        linkstats = []
        n = 0
        for i, nlink in enumerate(peaks):
            if len(mgrids[peaks[i]])==0:continue 
            ph = mgrids[peaks[i]][1]
            if type(ph) == list:
                continue 
            linktype, disttime, temp, infos, loc = mgrids[peaks[i]]
            lnum = np.sum(linktype!=0)
            if lnum<global_par.nps:
                continue 
            if np.sum(linktype==1)<global_par.np:
                continue 
            if np.sum(linktype==2)<global_par.ns:
                continue 
            linked = infos[linktype!=0] 
            stdict = {} 
            for li in linked:
                if li["st"] in stdict:
                    stdict[li["st"]].append(li["pname"]) 
                else:
                    stdict[li["st"]] = [li["pname"]]
            ndual = 0 
            for key in stdict:
                if len(stdict[key])>=2:
                    ndual += 1 
            if ndual < global_par.nboth:
                continue 
            n += 1 
            etime = reftime + datetime.timedelta(seconds=float(peaks[i])*global_par.win_stirde)
            estr = etime.strftime('%Y-%m-%d %H:%M:%S.%f')
            outfile.write(f"#EVENT,{temp},{estr},{loc[0]:.4f},{loc[1]:.4f}\n")
            for a, b, c  in zip(linktype, disttime, infos):
                ptime = etime + datetime.timedelta(seconds=float(b[1]))
                pstr = ptime.strftime('%Y-%m-%d %H:%M:%S.%f')
                stkey = c["st"] 
                pname = c["pname"]
                a = int(a)
                outfile.write(f"PHASE,{pstr},{id2t[a]},{stkey},{pname}\n")
        print("峰值个数", len(peaks), "地震个数", n)

import argparse
if __name__ == "__main__":
    # 处理输出某个时间之后的所有震相
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-i', '--input', default="odata/3days.new.s8.txt", help="拾取文件")       
    parser.add_argument('-o', '--output', default="odata/3days.new.s8.link.txt", help="输出文件名")   
    parser.add_argument('-s', '--station', default="data/station.113.FOR.ASSO", help="输出文件名") 
    args = parser.parse_args()   
    data = DataLPPN(args.input, args.station)
    tool = Link2d()
    tool.links(data)
    link(args.output) 

