
import datetime 
import h5py 
import numpy as np 
import time 
import multiprocessing 

import threading 
import h5py 
import datetime 
import numpy as np 
import matplotlib.pyplot as plt 
plt.switch_backend("agg")
def _label(a=0, b=20, c=40): 
    'Used for triangolar labeling'
    
    z = np.linspace(a, c, num = 2*(b-a)+1)
    y = np.zeros(z.shape)
    y[z <= a] = 0
    y[z >= c] = 0
    first_half = np.logical_and(a < z, z <= b)
    y[first_half] = (z[first_half]-a) / (b-a)
    second_half = np.logical_and(b < z, z < c)
    y[second_half] = (c-z[second_half]) / (c-b)
    return y


class DitingData():
    def __init__(self, file_name="h5data/diting/DiTing.v2.0.h5", n_length=10240, stride=16, padlen=256):
        self.file_name = file_name
        self.length = n_length 
        self.stride = stride  
        self.padlen = padlen 
        self.n_thread = 2
        self.phase_dict = {
            "Pg":0, 
            "Sg":1, 
            "P":0, 
            "S":1,
        } 
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        self.epoch = multiprocessing.Value("d", 0.0)
        self.file_name = file_name 
        self.p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, self.epoch))
        self.p1.start() 
        self.p2s = []
        for _ in range(self.n_thread):
            p = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue))
            p.start() 
            self.p2s.append(p)
    def get_epoch(self):
        return self.epoch.value 
    def kill_all(self):
        self.p1.terminate() 
        for p in self.p2s:
            p.terminate() 
    def feed_data(self, fqueue, epoch):
        while True:
            h5file = h5py.File(self.file_name, "r")
            train = h5file["train"]
            for ekey in train:
                event = train[ekey] 
                for skey in event:
                    station = event[skey] 
                    data = station[:]
                    pt, st = station.attrs["p_pick"], station.attrs["s_pick"]
                    fqueue.put([data, [int(pt), int(st)]])
            h5file.close()
            epoch.value += 1 # 计算epoch 
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, pidx = fqueue.get()
            pdic = {"P":pidx[0], "S":pidx[1]}
            # 做增强的，随机一个位置
            bidx = np.random.choice(pidx) - np.random.randint(self.padlen, self.length-self.padlen)
            eidx = bidx + self.length 
            rdata = np.zeros([self.length, 3])
            len_data = len(data)
            if bidx >= 0 and eidx < len_data:
                rdata = data[bidx:eidx, :] 
            if bidx < 0 and eidx < len_data:
                before = -bidx 
                rdata = np.pad(data[:eidx], ((before, 0), (0, 0)))
            if bidx > 0 and eidx >= len_data:
                after = eidx - len_data 
                rdata = np.pad(data[bidx:], ((0, after), (0, 0))) 
            if bidx < 0 and eidx >= len_data:
                after = eidx - len_data 
                before = -bidx 
                rdata = np.pad(data, ((before, after), (0, 0))) 
            rdata = rdata.astype(np.float32) 
            rdata -= np.mean(rdata, axis=0, keepdims=True) 
            rdata /= (np.max(np.abs(rdata))+1e-6)
            rdata *= np.random.uniform(0.5, 2)
            rdata = rdata.T 
            if rdata.shape[1] != self.length:continue 
            label1 = np.zeros([1, 2, llen]) # LPPN 标签
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)//self.stride
                if idx-1>0:
                    label1[0, :, idx-1:idx+2] = -1
                if idx > 0 and idx < llen:
                    label1[0, 0, idx] = pid + 1
                    label1[0, 1, idx] = (pdic[pkey] - bidx)%self.stride
            
            def tri(t, mu, std=0.1):
                midx = int(mu*100) 
                p = np.zeros_like(t) 
                bidx = np.max([0, midx-20])
                eidx = np.min([self.length, midx+21])
                lent = np.abs(eidx - bidx) 
                p[bidx:eidx] = _label()[:lent]
                return p 
            def norm(t, mu, std=0.1):
                p = np.exp(-(t-mu)**2/std**2/2)
                p /= (np.max(p)+1e-9) 
                return p 
            t = np.arange(self.length) * 0.01 
            label2 = np.zeros([1, 3, self.length]) # 其他类型标签
            label3 = np.zeros([1, 3, self.length]) # 其他类型标签
            phase_intv = {"P":0, "S":0} 
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)
                if idx > 0 and idx < self.length:
                    label2[0, pid+1, :] = tri(t, idx*0.01, 0.1)
                    label3[0, pid, :] = tri(t, idx*0.01, 0.1)
                if pid == 0:
                    if idx < self.length:
                        phase_intv["P"] = np.max([idx, 0]) 
                    else:
                        phase_intv["P"] = self.length 
                if pid == 1:
                    if idx > 0:
                        idx = int(idx + (pdic["S"]-pdic["P"]) * 1.4) 
                        phase_intv["S"] = np.min([idx, self.length])
                    else:
                        phase_intv["S"] = 0 
            label2[0, 0, :] = np.clip(1-label2[0, 1, :]-label2[0, 2, :], 0, 1)
            label3[0, 2, phase_intv["P"]:phase_intv["S"]] = 1    
            dqueue.put([rdata.astype(np.float32), label1, label2, label3])
            count += 1

    def batch_data(self, batch_size=32):
        x1, x2, x3, x4 = [], [], [], []
        for _ in range(batch_size):
            data, label1, label2, label3 = self.dqueue.get() 
            x1.append(data) 
            x2.append(label1) 
            x3.append(label2)
            x4.append(label3)
        x1 = np.stack(x1, axis=0) 
        x2 = np.concatenate(x2, axis=0) 
        x3 = np.concatenate(x3, axis=0) 
        x4 = np.concatenate(x4, axis=0) 
        return x1, x2, x3, x4 

class DitingDataTest():
    def __init__(self, file_name="h5data/diting/DiTing.v2.0.h5", n_length=10240, stride=16, padlen=256):
        self.file_name = file_name
        self.length = n_length 
        self.stride = stride  
        self.padlen = padlen 
        self.n_thread = 2
        self.phase_dict = {
            "Pg":0, 
            "Sg":1, 
            "P":0, 
            "S":1,
        } 
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        self.epoch = multiprocessing.Value("d", 0.0)
        self.file_name = file_name 
        self.p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, self.epoch))
        self.p1.start() 
        self.p2s = []
        for _ in range(self.n_thread):
            p = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue))
            p.start() 
            self.p2s.append(p)
    def get_epoch(self):
        return self.epoch.value 
    def kill_all(self):
        self.p1.terminate() 
        for p in self.p2s:
            p.terminate() 
    def feed_data(self, fqueue, epoch):
        while True:
            h5file = h5py.File(self.file_name, "r")
            train = h5file["valid"]
            for ekey in train:
                event = train[ekey] 
                for skey in event:
                    station = event[skey] 
                    data = station[:]
                    pt, st = station.attrs["p_pick"], station.attrs["s_pick"]
                    fqueue.put([data, [int(pt), int(st)]])
            epoch.value += 1 # 计算epoch 
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, pidx = fqueue.get()
            pdic = {"P":pidx[0], "S":pidx[1]}
            # 做增强的，随机一个位置
            bidx = np.random.choice(pidx) - np.random.randint(self.padlen, self.length-self.padlen)
            eidx = bidx + self.length 
            rdata = np.zeros([self.length, 3])
            len_data = len(data)
            if bidx >= 0 and eidx < len_data:
                rdata = data[bidx:eidx, :] 
            if bidx < 0 and eidx < len_data:
                before = -bidx 
                rdata = np.pad(data[:eidx], ((before, 0), (0, 0)))
            if bidx > 0 and eidx >= len_data:
                after = eidx - len_data 
                rdata = np.pad(data[bidx:], ((0, after), (0, 0))) 
            if bidx < 0 and eidx >= len_data:
                after = eidx - len_data 
                before = -bidx 
                rdata = np.pad(data, ((before, after), (0, 0))) 
            rdata = rdata.astype(np.float32) 
            rdata -= np.mean(rdata, axis=0, keepdims=True) 
            rdata /= (np.max(np.abs(rdata))+1e-6)
            rdata *= np.random.uniform(0.5, 2)
            rdata = rdata.T 
            if rdata.shape[1] != self.length:continue 
            phase_time = {0:-1, 1:-1}
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)
                if idx > 0 and idx < self.length:
                    phase_time[pid] = idx    
            dqueue.put([rdata, [phase_time[0], phase_time[1]]])

            count += 1

    def batch_data(self, batch_size=50):
        x1, x2 = [], []
        for _ in range(batch_size):
            data, label1 = self.dqueue.get() 
            x1.append(data) 
            x2.append(label1) 
            
        x1 = np.stack(x1, axis=0) 
        return x1, x2 

import queue 
class DitingDataThread():
    def __init__(self, file_name="h5data/diting/DiTing.v2.0.h5", n_length=10240, stride=16, padlen=256):
        self.file_name = file_name
        self.length = n_length 
        self.stride = stride  
        self.padlen = padlen 
        self.n_thread = 1
        self.phase_dict = {
            "Pg":0, 
            "Sg":1, 
            "P":0, 
            "S":1,
        } 
        fqueue = queue.Queue(10)
        self.dqueue = queue.Queue(10)
        self.epoch = 0
        self.file_name = file_name 
        self.p1 = threading.Thread(target=self.feed_data, args=(fqueue, self.epoch))
        self.p1.start() 
        self.p2s = []
        for _ in range(self.n_thread):
            p = threading.Thread(target=self.process, args=(fqueue, self.dqueue))
            p.start() 
            self.p2s.append(p)
    def get_epoch(self):
        return self.epoch
    def kill_all(self):
        self.p1.terminate() 
        for p in self.p2s:
            p.terminate() 
    def feed_data(self, fqueue, epoch):
        while True:
            h5file = h5py.File(self.file_name, "r")
            train = h5file["train"]
            for ekey in train:
                event = train[ekey] 
                for skey in event:
                    station = event[skey] 
                    data = station[:]
                    pt, st = station.attrs["p_pick"], station.attrs["s_pick"]
                    fqueue.put([data, [int(pt), int(st)]])
            h5file.close()
            self.epoch += 1 # 计算epoch 
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, pidx = fqueue.get()
            pdic = {"P":pidx[0], "S":pidx[1]}
            # 做增强的，随机一个位置
            bidx = np.random.choice(pidx) - np.random.randint(self.padlen, self.length-self.padlen)
            eidx = bidx + self.length 
            rdata = np.zeros([self.length, 3])
            len_data = len(data)
            if bidx >= 0 and eidx < len_data:
                rdata = data[bidx:eidx, :] 
            if bidx < 0 and eidx < len_data:
                before = -bidx 
                rdata = np.pad(data[:eidx], ((before, 0), (0, 0)))
            if bidx > 0 and eidx >= len_data:
                after = eidx - len_data 
                rdata = np.pad(data[bidx:], ((0, after), (0, 0))) 
            if bidx < 0 and eidx >= len_data:
                after = eidx - len_data 
                before = -bidx 
                rdata = np.pad(data, ((before, after), (0, 0))) 
            rdata = rdata.astype(np.float32) 
            rdata -= np.mean(rdata, axis=0, keepdims=True) 
            rdata /= (np.max(np.abs(rdata))+1e-6)
            rdata *= np.random.uniform(0.5, 2)
            rdata = rdata.T 
            if rdata.shape[1] != self.length:continue 
            label1 = np.zeros([1, 2, llen]) # LPPN 标签
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)//self.stride
                if idx-1>0:
                    label1[0, :, idx-1:idx+2] = -1
                if idx > 0 and idx < llen:
                    label1[0, 0, idx] = pid + 1
                    label1[0, 1, idx] = (pdic[pkey] - bidx)%self.stride
            
            def tri(t, mu, std=0.1):
                midx = int(mu*100) 
                p = np.zeros_like(t) 
                bidx = np.max([0, midx-20])
                eidx = np.min([self.length, midx+21])
                lent = np.abs(eidx - bidx) 
                p[bidx:eidx] = _label()[:lent]
                return p 
            def norm(t, mu, std=0.1):
                p = np.exp(-(t-mu)**2/std**2/2)
                p /= (np.max(p)+1e-9) 
                return p 
            t = np.arange(self.length) * 0.01 
            label2 = np.zeros([1, 3, self.length]) # 其他类型标签
            label3 = np.zeros([1, 3, self.length]) # 其他类型标签
            phase_intv = {"P":0, "S":0} 
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)
                if idx > 0 and idx < self.length:
                    label2[0, pid+1, :] = tri(t, idx*0.01, 0.1)
                    label3[0, pid, :] = tri(t, idx*0.01, 0.1)
                if pid == 0:
                    if idx < self.length:
                        phase_intv["P"] = np.max([idx, 0]) 
                    else:
                        phase_intv["P"] = self.length 
                if pid == 1:
                    if idx > 0:
                        idx = int(idx + (pdic["S"]-pdic["P"]) * 1.4) 
                        phase_intv["S"] = np.min([idx, self.length])
                    else:
                        phase_intv["S"] = 0 
            label2[0, 0, :] = np.clip(1-label2[0, 1, :]-label2[0, 2, :], 0, 1)
            label3[0, 2, phase_intv["P"]:phase_intv["S"]] = 1    
            dqueue.put([rdata.astype(np.float32), label1, label2, label3])
            count += 1
    def batch_data(self, batch_size=32):
        x1, x2, x3, x4 = [], [], [], []
        for _ in range(batch_size):
            data, label1, label2, label3 = self.dqueue.get() 
            x1.append(data) 
            x2.append(label1) 
            x3.append(label2)
            x4.append(label3)
        x1 = np.stack(x1, axis=0) 
        x2 = np.concatenate(x2, axis=0) 
        x3 = np.concatenate(x3, axis=0) 
        x4 = np.concatenate(x4, axis=0) 
        return x1, x2, x3, x4 
        
class DitingDataTestThread():
    def __init__(self, file_name="h5data/diting/DiTing.v2.0.h5", n_length=10240, stride=16, padlen=256):
        self.file_name = file_name
        self.length = n_length 
        self.stride = stride  
        self.padlen = padlen 
        self.n_thread = 1
        self.phase_dict = {
            "Pg":0, 
            "Sg":1, 
            "P":0, 
            "S":1,
        } 
        fqueue = queue.Queue(10)
        self.dqueue = queue.Queue(10)
        self.epoch = 0
        self.file_name = file_name 
        self.p1 = threading.Thread(target=self.feed_data, args=(fqueue, self.epoch))
        self.p1.start() 
        self.p2s = []
        for _ in range(self.n_thread):
            p = threading.Thread(target=self.process, args=(fqueue, self.dqueue))
            p.start() 
            self.p2s.append(p)
    def get_epoch(self):
        return self.epoch.value 
    def kill_all(self):
        for p in self.p2s:
            p.terminate() 
        self.p1.terminate() 
    def feed_data(self, fqueue, epoch):
        h5file = h5py.File(self.file_name, "r")
        train = h5file["test"]
        for ekey in train:
            event = train[ekey] 
            for skey in event:
                station = event[skey] 
                data = station[:]
                pt, st = station.attrs["p_pick"], station.attrs["s_pick"]
                fqueue.put([data, [int(pt), int(st)]])
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, pidx = fqueue.get()
            pdic = {"P":pidx[0], "S":pidx[1]}
            # 做增强的，随机一个位置
            bidx = np.random.choice(pidx) - np.random.randint(self.padlen, self.length-self.padlen)
            eidx = bidx + self.length 
            rdata = np.zeros([self.length, 3])
            len_data = len(data)
            if bidx >= 0 and eidx < len_data:
                rdata = data[bidx:eidx, :] 
            if bidx < 0 and eidx < len_data:
                before = -bidx 
                rdata = np.pad(data[:eidx], ((before, 0), (0, 0)))
            if bidx > 0 and eidx >= len_data:
                after = eidx - len_data 
                rdata = np.pad(data[bidx:], ((0, after), (0, 0))) 
            if bidx < 0 and eidx >= len_data:
                after = eidx - len_data 
                before = -bidx 
                rdata = np.pad(data, ((before, after), (0, 0))) 
            rdata = rdata.astype(np.float32) 
            rdata -= np.mean(rdata, axis=0, keepdims=True) 
            rdata /= (np.max(np.abs(rdata))+1e-6)
            rdata *= np.random.uniform(0.5, 2)
            rdata = rdata.T 
            if rdata.shape[1] != self.length:continue 
            phase_time = {0:-1, 1:-1}
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)
                if idx > 0 and idx < self.length:
                    phase_time[pid] = idx    
            dqueue.put([rdata, [phase_time[0], phase_time[1]]])

            count += 1

    def batch_data(self, batch_size=50):
        x1, x2 = [], []
        for _ in range(batch_size):
            data, label1 = self.dqueue.get() 
            x1.append(data) 
            x2.append(label1) 
            
        x1 = np.stack(x1, axis=0) 
        return x1, x2 



class DitingDataForPlot():
    def __init__(self, file_name="h5data/diting/DiTing.v2.0.h5", n_length=10240, stride=16, padlen=256):
        self.file_name = file_name
        self.length = n_length 
        self.stride = stride  
        self.padlen = padlen 
        self.n_thread = 2
        self.phase_dict = {
            "Pg":0, 
            "Sg":1, 
            "P":0, 
            "S":1,
        } 
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        self.epoch = multiprocessing.Value("d", 0.0)
        self.file_name = file_name 
        self.p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, self.epoch))
        self.p1.start() 
        self.p2s = []
        for _ in range(self.n_thread):
            p = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue))
            p.start() 
            self.p2s.append(p)
    def get_epoch(self):
        return self.epoch.value 
    def kill_all(self):
        self.p1.terminate() 
        for p in self.p2s:
            p.terminate() 
    def feed_data(self, fqueue, epoch):
        while True:
            h5file = h5py.File(self.file_name, "r")
            train = h5file["train"]
            for ekey in train:
                event = train[ekey] 
                for skey in event:
                    station = event[skey] 
                    data = station[:]
                    snr = station.attrs["Z_P_amplitude_snr"]
                    if snr < 10:continue 
                    pt, st = station.attrs["p_pick"], station.attrs["s_pick"]
                    fqueue.put([data, [int(pt), int(st)]])
            h5file.close()
            epoch.value += 1 # 计算epoch 
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, pidx = fqueue.get()
            pdic = {"P":pidx[0], "S":pidx[1]}
            # 做增强的，随机一个位置
            bidx = 0#np.random.choice(pidx) - np.random.randint(self.padlen, self.length-self.padlen)
            eidx = bidx + self.length 
            rdata = np.zeros([self.length, 3])
            len_data = len(data)
            if bidx >= 0 and eidx < len_data:
                rdata = data[bidx:eidx, :] 
            if bidx < 0 and eidx < len_data:
                before = -bidx 
                rdata = np.pad(data[:eidx], ((before, 0), (0, 0)))
            if bidx > 0 and eidx >= len_data:
                after = eidx - len_data 
                rdata = np.pad(data[bidx:], ((0, after), (0, 0))) 
            if bidx < 0 and eidx >= len_data:
                after = eidx - len_data 
                before = -bidx 
                rdata = np.pad(data, ((before, after), (0, 0))) 
            rdata = rdata.astype(np.float32) 
            rdata -= np.mean(rdata, axis=0, keepdims=True) 
            rdata /= (np.max(np.abs(rdata))+1e-6)
            rdata *= np.random.uniform(0.5, 2)
            if rdata.shape[1] != self.length:continue 
            label1 = np.zeros([1, llen, 2]) # LPPN 标签
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)//self.stride # 计算网格索引，下取整
                if idx-1>0:
                    label1[0, idx-1:idx+2] = -1
                if idx > 0 and idx < llen:
                    label1[0, idx, 0] = pid + 1                        # 类别 
                    label1[0, idx, 1] = (pdic[pkey] - bidx)%self.stride# 回归
            
            def tri(t, mu, std=0.1):
                midx = int(mu*100) 
                p = np.zeros_like(t) 
                bidx = np.max([0, midx-20])
                eidx = np.min([self.length, midx+21])
                lent = np.abs(eidx - bidx) 
                p[bidx:eidx] = _label()[:lent]
                return p 
            def norm(t, mu, std=0.1):
                p = np.exp(-(t-mu)**2/std**2/2)
                p /= (np.max(p)+1e-6) 
                return p 
            t = np.arange(self.length) * 0.01 
            label2 = np.zeros([1, self.length, 4]) # 其他类型标签
            phase_intv = {"P":0, "S":0} 
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)
                if idx > 0 and idx < self.length:
                    label2[0, :, pid+1] = tri(t, idx*0.01, 0.1)
                if pid == 0:
                    if idx < self.length:
                        phase_intv["P"] = np.max([idx, 0]) 
                    else:
                        phase_intv["P"] = self.length 
                if pid == 1:
                    if idx > 0:
                        idx = int(idx + (pdic["S"]-pdic["P"]) * 1.4) 
                        phase_intv["S"] = np.min([idx, self.length])
                    else:
                        phase_intv["S"] = 0 
            label2[0, :, 0] = np.clip(1-label2[0, :, 1]-label2[0, :, 2], 0, 1)
            label2[0, phase_intv["P"]:phase_intv["S"], 3] = 1    
            dqueue.put([rdata.astype(np.float32), label1, label2])
            count += 1

    def batch_data(self, batch_size=32):
        x1, x2, x3 = [], [], []
        for _ in range(batch_size):
            data, label1, label2 = self.dqueue.get() 
            x1.append(data) 
            x2.append(label1) 
            x3.append(label2)
        x1 = np.stack(x1, axis=0) 
        x2 = np.concatenate(x2, axis=0) 
        x3 = np.concatenate(x3, axis=0) 
        return x1, x2, x3

class DitingDataTestForPlot():
    def __init__(self, file_name="h5data/diting/DiTing.v2.0.h5", n_length=10240, stride=16, padlen=256):
        self.file_name = file_name
        self.length = n_length 
        self.stride = stride  
        self.padlen = padlen 
        self.n_thread = 2
        self.phase_dict = {
            "Pg":0, 
            "Sg":1, 
            "P":0, 
            "S":1,
        } 
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        self.epoch = multiprocessing.Value("d", 0.0)
        self.file_name = file_name 
        self.p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, self.epoch))
        self.p1.start() 
        self.p2s = []
        for _ in range(self.n_thread):
            p = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue))
            p.start() 
            self.p2s.append(p)
    def get_epoch(self):
        return self.epoch.value 
    def kill_all(self):
        self.p1.terminate() 
        for p in self.p2s:
            p.terminate() 
    def feed_data(self, fqueue, epoch):
        while True:
            h5file = h5py.File(self.file_name, "r")
            train = h5file["valid"]
            count = 0 
            for ekey in train:
                event = train[ekey] 
                for skey in event:
                    station = event[skey] 
                    data = station[:]
                    snr = station.attrs["Z_P_amplitude_snr"]
                    if count !=0:
                        if snr < 10:continue 
                    pt, st = station.attrs["p_pick"], station.attrs["s_pick"]
                    fqueue.put([data, [int(pt), int(st)]])
                    count += 1 
            epoch.value += 1 # 计算epoch 
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, pidx = fqueue.get()
            pdic = {"P":pidx[0], "S":pidx[1]}
            # 做增强的，随机一个位置
            bidx = 0#np.random.choice(pidx) - np.random.randint(self.padlen, self.length-self.padlen)
            eidx = bidx + self.length 
            rdata = np.zeros([self.length, 3])
            len_data = len(data)
            if bidx >= 0 and eidx < len_data:
                rdata = data[bidx:eidx, :] 
            if bidx < 0 and eidx < len_data:
                before = -bidx 
                rdata = np.pad(data[:eidx], ((before, 0), (0, 0)))
            if bidx > 0 and eidx >= len_data:
                after = eidx - len_data 
                rdata = np.pad(data[bidx:], ((0, after), (0, 0))) 
            if bidx < 0 and eidx >= len_data:
                after = eidx - len_data 
                before = -bidx 
                rdata = np.pad(data, ((before, after), (0, 0))) 
            rdata = rdata.astype(np.float32) 
            rdata -= np.mean(rdata, axis=0, keepdims=True) 
            rdata /= (np.max(np.abs(rdata))+1e-6)
            rdata *= np.random.uniform(0.5, 2)
            if rdata.shape[1] != self.length:continue 
            phase_time = {0:-1, 1:-1}
            for pkey in pdic:
                pid = self.phase_dict[pkey] 
                idx = (pdic[pkey] - bidx)
                if idx > 0 and idx < self.length:
                    phase_time[pid] = idx    
            dqueue.put([rdata, [phase_time[0], phase_time[1]]])

            count += 1

    def batch_data(self, batch_size=50):
        x1, x2 = [], []
        for _ in range(batch_size):
            data, label1 = self.dqueue.get() 
            x1.append(data) 
            x2.append(label1) 
            
        x1 = np.stack(x1, axis=0) 
        return x1, x2 



import matplotlib.pyplot as plt 
if __name__ == "__main__":
    data = DitingData(n_length=6144, stride=8)
    
    for e in range(3):
        a1, a2, a3 = data.batch_data() 
        print(a1.shape, a2.shape, a3.shape, data.get_epoch()) 
    w = a1[0, :, 0] 
    #w /= np.max(w)
    plt.plot(w, c="k") 
    plt.plot(np.repeat(a2[0, :, 0], 8), c="r") 
    #plt.plot(a3[0, :, 0], c="r")
    plt.plot(a3[0, :, 1], c="g")
    plt.plot(a3[0, :, 2], c="b")
    #plt.plot(a3[0, :, 3], c="c")
    plt.savefig("temp/demo.jpg")
    data.kill_all()