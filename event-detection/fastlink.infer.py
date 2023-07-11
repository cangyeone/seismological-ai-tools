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
import heapq 
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
        grid_size = 200 
        min_lon, min_lat = 95.987, 31.3039
        max_lon, max_lat = 109.4132, 42.879
        dlon = (max_lon-min_lon)/grid_size 
        dlat = (max_lat-min_lat)/grid_size 
        grid1 = (torch.zeros([grid_size, 1]) + torch.arange(min_lon, max_lon, dlon)[:grid_size].reshape(1, grid_size)).reshape(grid_size, grid_size, 1)   
        grid2 = (torch.arange(min_lat, max_lat, dlat)[:grid_size].reshape(grid_size, 1) + torch.zeros([1, grid_size])).reshape(grid_size, grid_size, 1)
        self.register_buffer("grid1", grid1)
        self.register_buffer("grid2", grid2)
        self.register_buffer("posgrid", torch.cat([self.grid1, self.grid2], dim=2).reshape(grid_size * grid_size, 2))
    def forward(self, x):
        grid_size = 200 
        device = x.device 
        N, C = x.shape 
        dtype = x.dtype
        ptype = torch.zeros([grid_size, grid_size, N], device=device, dtype=dtype)
        ptype[:, :, :] = x[:, 3]
        x = x[:, :3]
        grid = torch.zeros([grid_size, grid_size, N, 5], device=device, dtype=dtype)
        grid[:, :, :, 0] = self.grid1 
        grid[:, :, :, 1] = self.grid2 
        grid[:, :, :, 2:] = x.reshape(1, 1, N, 3)
        grid = torch.reshape(grid, [-1, 5])
        lat1 = torch.deg2rad(grid[:, 1:2])
        lat2 = torch.deg2rad(grid[:, 3:4])
        long1 = torch.deg2rad(grid[:, 0:1])
        long2 = torch.deg2rad(grid[:, 2:3])
        long_diff = long2 - long1
        gd = torch.atan2(
                torch.sqrt((
                    torch.cos(lat2) * torch.sin(long_diff)) ** 2 +
                    (torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) *
                        torch.cos(lat2) * torch.cos(long_diff)) ** 2),
                torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) *
                torch.cos(long_diff)) * 6371
        disttime = torch.cat([gd, grid[:, 4:5]], dim=1) / 100

        logit = self.dense(disttime) 
        
        prob = F.softmax(logit, dim=1)
        num = (prob[:, 0].reshape([grid_size * grid_size, N])<0.5).half().sum(1)

        pids = torch.argmax(logit, dim=1).reshape(grid_size * grid_size, N) 
        rids = torch.reshape(ptype, [grid_size * grid_size, N])
        num = (pids==rids).half().sum(1)
        
        max_num_id = torch.argmax(num)
        class_ = logit.argmax(dim=1).reshape(grid_size * grid_size, N)
        class_ = class_[max_num_id] 
        ph = disttime.reshape(grid_size * grid_size, N, 2)[max_num_id] * 100 
        pos = self.posgrid[max_num_id]
        return class_, ph, num.max(), pos 
class Link2d():
    def __init__(self):
        pass 
    def links(self, data_tool):
        dtype = torch.half
        device = torch.device("cuda")
        model = Model()
        model.eval() 
        grid_size = 200 
        #min_lon, min_lat = 95.987, 31.3039
        #max_lon, max_lat = 109.4132, 42.879
        #dlon = (max_lon-min_lon)/grid_size 
        #dlat = (max_lat-min_lat)/grid_size 

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
        grid1 = (torch.zeros([grid_size, 1], dtype=dtype) + torch.arange(min_lon, max_lon, dlon)[:grid_size].reshape(1, grid_size)).to(device).reshape(grid_size, grid_size, 1)   
        grid2 = (torch.arange(min_lat, max_lat, dlat)[:grid_size].reshape(grid_size, 1) + torch.zeros([1, grid_size], dtype=dtype)).to(device).reshape(grid_size, grid_size, 1)
        #grid1 = (torch.zeros([grid_size, 1]) + torch.arange(min_lon, max_lon, dlon)[:grid_size].reshape(1, grid_size)).reshape(grid_size, grid_size, 1)   
        #grid2 = (torch.arange(min_lat, max_lat, dlat)[:grid_size].reshape(grid_size, 1) + torch.zeros([1, grid_size])).reshape(grid_size, grid_size, 1)
        posgrid = torch.cat([grid1, grid2], dim=2).reshape(grid_size * grid_size, 2)

        edict = torch.load("ckpt/link32.ckpt", map_location="cpu") 
        edict["grid1"] = grid1 
        edict["grid2"] = grid2 
        edict["posgrid"] = posgrid 

        model.load_state_dict(edict)
        model.to(dtype)
        model.to(device)
        
        maxitr = 0 
        if os.path.exists(global_par.datadir)==False:
            os.mkdir(global_par.datadir)
        else:
            tempfiles = os.listdir(global_par.datadir) 
            nums = [] 
            for tf in tempfiles:
                if tf.endswith(".npz")==False:continue 
                nums.append(int(tf.split(".")[0][4:])) 
            if len(nums)==0:
                maxitr = 0
            else:
                maxitr = int(np.max(nums))
        btime = data_tool.basetime + datetime.timedelta(seconds=data_tool.start)
        
        for t in tqdm.tqdm(range(L)):
            if t < maxitr:
                data_tool.skipdata()
                continue 
            datas = data_tool.getdata() 
            if len(datas)==0:
                mgrids.append([])
                mlinks.append(0)
                #print(i, j)
                mclass.append(0)
                mphase.append([]) 
                if (t+1) % global_par.saveitv == 0:
                    np.savez(f"{global_par.datadir}/link{t}.npz", mlinks=mlinks, mgrids=mgrids, mphase=mphase, mclass=mclass, mtime=btime)
                    mgrids = []
                    mlinks = []
                    mphase = []
                    mclass = []
                    btime = data_tool.basetime + datetime.timedelta(seconds=data_tool.start)
                continue                 
            types = datas[:, 4].astype(np.int64)
            N = len(types)
            if N < global_par.nps or np.sum(types==1)<global_par.np or np.sum(types==2)<global_par.ns:
                mgrids.append([])
                mlinks.append(0)
                #print(i, j)
                mclass.append(0)
                mphase.append([]) 
                if (t+1) % global_par.saveitv == 0:
                    np.savez(f"{global_par.datadir}/link{t}.npz", mlinks=mlinks, mgrids=mgrids, mphase=mphase, mclass=mclass, mtime=btime)
                    mgrids = []
                    mlinks = []
                    mphase = []
                    mclass = []
                    btime = data_tool.basetime + datetime.timedelta(seconds=data_tool.start)
                continue 
            infos = datas[:, 3]
            input_data = np.concatenate([datas[:, :3], datas[:, 4:5]], axis=1).astype(np.float32)
            with torch.no_grad():
                data = torch.tensor(input_data, dtype=dtype, device=device)
                class_, ph, max_num_id, pos = model(data)
                class_ = class_.cpu().numpy()
                ph = ph.cpu().numpy()
                pos = pos.cpu().numpy()
                max_num_id = max_num_id.cpu().numpy()
            mgrids.append([class_, ph, max_num_id, infos, pos])
            num = max_num_id
            #print(num)
            mlinks.append(num)
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
import heapq 
class Item(object):
    def __init__(self, idx, val) -> None:
        self.idx = idx 
        self.val = val 
    def __lt__(self, new):
        return self.idx < new.idx 
    def __getitem__(self, a):
        return self.idx  
class DataLPPN():
    def __init__(self, infilename, stationfile):
        self.pos_dict = {}
        file_ = open(stationfile, "r", encoding="utf-8") 
        for line in file_.readlines():
            sline = [i for i in line.split(" ") if len(i)>0] 
            # 台站ID：SC.JHO
            key = ".".join(sline[:2])
            #if "SC" != sline[0] and "YN" != sline[1]:continue 
            # 台站经纬度
            x, y = float(sline[3]), float(sline[4]) 
            self.pos_dict[key] = [x, y]
        print("台站数量", len(self.pos_dict))
        file_.close() 
        basetime = datetime.datetime.strptime(global_par.basetime, "%Y-%m-%d") # 设置数据起始时间
        self.basetime = basetime 
        file_dir = infilename

        p2id = {"Pg":1, "Sg":2, "Pn":3, "Sn":4, "P":5, "S":6}
        file_ = open(file_dir, "r", encoding="utf-8") 
        datas = []
        self.dataheap = []
        print("数据读取中")
        for line in tqdm.tqdm(file_.readlines()[:]):
            if "##" in line:continue 
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
            datas.append([loc[0], loc[1], delta])
            #self.dataheap.append((delta, [loc[0], loc[1], delta, info, p2id[pname]]))
            itm = Item(delta, [loc[0], loc[1], delta, info, p2id[pname]])
            heapq.heappush(self.dataheap, itm)
        #heapq.heapify(self.dataheap)

        self.datas = np.array(datas)       
        self.mintm = np.min(self.datas[:, 2])
        self.start = self.mintm 
        self.maxtm = np.max(self.datas[:, 2])
        self.minloc = np.min(self.datas[:, :2], axis=0) 
        self.maxloc = np.max(self.datas[:, :2], axis=0)
        self.winheap = []
        #while True:
        #    if self.dataheap[0][0] < self.mintm + global_par.win_length:
        #        heapq.heappush(self.winheap, heapq.heappop(self.dataheap))
        #    else:
        #        break 
        print("数据读取完成！", "震相数量", len(datas), "数据天数", datetime.timedelta(seconds=self.maxtm-self.mintm).days)
    def getdata(self):
        while True:
            if len(self.dataheap)==0:
                break 
            if self.dataheap[0][0] < self.start + global_par.win_length:
                heapq.heappush(self.winheap, heapq.heappop(self.dataheap))
            else:
                break 
        while True:
            if len(self.winheap)==0:
                break 
            if self.winheap[0][0] < self.start:
                heapq.heappop(self.winheap) 
            else:
                break 
        if len(self.winheap)==0:
            self.start += global_par.win_stirde 
            return [] 
        temp = []
        for d in self.winheap:
            temp.append(d.val)
        temp = np.array(temp)
        temp[:, 2] -= self.start 
        self.start += global_par.win_stirde 
        return temp
    def skipdata(self):
        #idx = ((self.index>self.start) * (self.index<self.start + global_par.win_length)).cpu().numpy()
        #temp = np.copy(self.datas[idx])
        #temp[:, 2] -= self.start 
        self.start += global_par.win_stirde 
        return 0        
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
                outfile.write(f"PHASE,{pstr},{id2t[a]},{stkey},{pname},{b[0]},{b[1]}\n")
        print("峰值个数", len(peaks), "地震个数", n)

import argparse
if __name__ == "__main__":
    # 处理输出某个时间之后的所有震相
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-i', '--input', default="odata/X2_2015_szy.txt", help="拾取震相文件")       
    parser.add_argument('-o', '--output', default="odata/x2.2015.rnn.txt", help="输出文件名")   
    parser.add_argument('-s', '--station', default="odata/x2.pos", help="台站位置信息") 
    args = parser.parse_args()   
    global_par.datadir = f"{args.output}.tempdata"
    if os.path.exists(global_par.datadir):
        pass
        #os.system(f"rm {global_par.datadir}/*.npz")
    else:
        os.makedirs(global_par.datadir)
    data = DataLPPN(args.input, args.station)
    tool = Link2d()
    tool.links(data)
    link(args.output) 
