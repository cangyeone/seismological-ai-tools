import os 
import obspy 
import pickle 
import datetime 
import h5py 
import numpy as np 
import time 
import multiprocessing 


class Datas():
    def __init__(self, file_name="h5data", n_length=1024, stride=16, padlen=128, mindist=0, maxdist=1000):
        self.file_name = file_name
        self.length = n_length  
        self.stride = stride 
        self.padlen = padlen
        self.n_thread = 2 
        self.maxdist = maxdist 
        self.mindist = mindist 
        self.phase_dict = {
            "Pg":0, 
        } 
        self.ploar1 = {
            "C":0, 
            "U":0, 
            "R":1, 
            "D":1, 
        }
        self.ploar2 = {
            "I":0, 
            "M":1, 
            "E":2, 
        }
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        for i in range(11):
            p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, i+2009),daemon=True)
            p1.start() 
        for _ in range(self.n_thread):
            c1 = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue),daemon=True)
            c1.start()
        #multiprocessing.Process(target=self.batch_data, args=(dqueue, )).start()
    def feed_data(self, fqueue, year):
        while True:
            h5file = h5py.File(os.path.join(self.file_name, f"{year}.h5"), "r")
            for ekey in h5file:
                event = h5file[ekey] 
                for skey in event:
                    station = event[skey] 
                    if "POLARITY.Pg" not in station.attrs:continue 
                    if "POLARITY.Pg.UPDOWN" not in station.attrs:continue
                    if "POLARITY.Pg.CLARITY" not in station.attrs:continue 
                    ptype1 = station.attrs["POLARITY.Pg.UPDOWN"]
                    ptype2 = station.attrs["POLARITY.Pg.CLARITY" ]
                    if ptype1 not in self.ploar1 or ptype2 not in self.ploar2:continue 

                    data = []
                    for dkey in station:
                        if "HZ" not in dkey:continue 
                        data.append(station[dkey][:]) 
                        btime = datetime.datetime.strptime(station[dkey].attrs['btime'], "%Y/%m/%d %H:%M:%S.%f") 
                    if len(data)!=1:continue 
                    phases = {}
                    dist = -1
                    for akey in station.attrs:
                        if "dist" in akey:
                            dist = float(station.attrs[akey])
                        if "Pg" in akey:
                            pname = akey.split(".")[-1]
                            if pname in self.phase_dict:
                                phases[pname] = datetime.datetime.strptime(station.attrs[akey], "%Y/%m/%d %H:%M:%S.%f")
                        else:
                            if akey in self.phase_dict:
                                phases[akey] = datetime.datetime.strptime(station.attrs[akey], "%Y/%m/%d %H:%M:%S.%f")
                    if len(phases)==0:continue
                    fqueue.put([data, btime, phases, dist, [self.ploar1[ptype1], self.ploar2[ptype2]]])
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, btime, phases, dist, ptypes = fqueue.get()
            if dist>self.maxdist or dist < self.mindist:
                continue 
            pidx = {}
            plist = []
            for pkey in phases:
                ptime = phases[pkey]
                delta = (ptime-btime).total_seconds()
                delta_idx = int(delta * 100)
                pidx[pkey] = delta_idx
                plist.append(delta_idx) 

            cidx = np.random.choice(plist) - np.random.randint(self.padlen, self.length-self.padlen)
            rdata = [] 
            flen = False
            for d in data:
                w = d[cidx:cidx+self.length] 
                wlen = len(w) 
                if wlen!=self.length:
                    flen = True
                    break 
                w = w - np.mean(w)
                #w = w / (np.max(w)+1e-6)
                rdata.append(w[np.newaxis, :, np.newaxis]) 
            if flen:
                continue
            rdata = np.concatenate(rdata, axis=2) 
            rdata /= (np.std(rdata) + 1e-6) 
            rdata *= np.random.uniform(0.5, 1.5)
            label = np.array(ptypes)[np.newaxis, ...]
            snr = 0
            if snr < -13:
                continue 
            dqueue.put([rdata, label])
            count += 1

    def batch_data(self, batch_size=32):
        x1, x2 = [], []
        for _ in range(batch_size):
            data, label = self.dqueue.get() 
            #print(data.shape, label.shape)
            x1.append(data) 
            x2.append(label) 
        x1 = np.concatenate(x1, axis=0) 
        x2 = np.concatenate(x2, axis=0) 
        return x1, x2 

class Data3d():
    def __init__(self, file_name="h5data", n_length=1024, stride=16, padlen=128, mindist=0, maxdist=1000):
        self.file_name = file_name
        self.length = n_length  
        self.stride = stride 
        self.padlen = padlen
        self.n_thread = 2 
        self.maxdist = maxdist 
        self.mindist = mindist 
        self.phase_dict = {
            "Pg":0, 
        } 
        self.ploar1 = {
            "C":0, 
            "U":0, 
            "R":1, 
            "D":1, 
        }
        self.ploar2 = {
            "I":0, 
            "M":1, 
            "E":2, 
        }
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        for i in range(10):
            p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, i+2009),daemon=True)
            p1.start() 
        for _ in range(self.n_thread):
            c1 = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue),daemon=True)
            c1.start()
        #multiprocessing.Process(target=self.batch_data, args=(dqueue, )).start()
    def feed_data(self, fqueue, year):
        while True:
            h5file = h5py.File(os.path.join(self.file_name, f"{year}.h5"), "r")
            for ekey in h5file:
                event = h5file[ekey] 
                for skey in event:
                    station = event[skey] 
                    if "POLARITY.Pg" not in station.attrs:continue 
                    if "POLARITY.Pg.UPDOWN" not in station.attrs:continue
                    if "POLARITY.Pg.CLARITY" not in station.attrs:continue 
                    ptype1 = station.attrs["POLARITY.Pg.UPDOWN"]
                    ptype2 = station.attrs["POLARITY.Pg.CLARITY" ]
                    if ptype1 not in self.ploar1 or ptype2 not in self.ploar2:continue 

                    data = []
                    for dkey in ["BHE", "BHN", "BHZ"]:
                        if dkey not in station:
                            dkey = dkey.replace("B", "S") 
                        if dkey not in station:
                            continue 
                        data.append(station[dkey][:]) 
                        btime = datetime.datetime.strptime(station[dkey].attrs['btime'], "%Y/%m/%d %H:%M:%S.%f") 
                    if len(data)!=3:continue 
                    phases = {}
                    dist = -1
                    for akey in station.attrs:
                        if "dist" in akey:
                            dist = float(station.attrs[akey])
                        if "Pg" in akey:
                            pname = akey.split(".")[-1]
                            if pname in self.phase_dict:
                                phases[pname] = datetime.datetime.strptime(station.attrs[akey], "%Y/%m/%d %H:%M:%S.%f")
                        else:
                            if akey in self.phase_dict:
                                phases[akey] = datetime.datetime.strptime(station.attrs[akey], "%Y/%m/%d %H:%M:%S.%f")
                    if len(phases)==0:continue
                    fqueue.put([data, btime, phases, dist, [self.ploar1[ptype1], self.ploar2[ptype2]]])
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, btime, phases, dist, ptypes = fqueue.get()
            if dist>self.maxdist or dist < self.mindist:
                continue 
            pidx = {}
            plist = []
            for pkey in phases:
                ptime = phases[pkey]
                delta = (ptime-btime).total_seconds()
                delta_idx = int(delta * 100)
                pidx[pkey] = delta_idx
                plist.append(delta_idx) 

            cidx = np.random.choice(plist) - np.random.randint(self.padlen, self.length-self.padlen)
            rdata = [] 
            flen = False
            for d in data:
                w = d[cidx:cidx+self.length] 
                wlen = len(w) 
                if wlen!=self.length:
                    flen = True
                    break 
                w = w - np.mean(w)
                #w = w / (np.max(w)+1e-6)
                rdata.append(w[np.newaxis, :, np.newaxis]) 
            if flen:
                continue
            rdata = np.concatenate(rdata, axis=2) 
            rdata /= (np.max(np.abs(rdata)) + 1e-6) 
            rdata *= np.random.uniform(0.5, 1.5)
            label = np.array(ptypes)[np.newaxis, ...]
            snr = 0
            if snr < -13:
                continue 
            dqueue.put([rdata, label])
            count += 1

    def batch_data(self, batch_size=32):
        x1, x2 = [], []
        for _ in range(batch_size):
            data, label = self.dqueue.get() 
            #print(data.shape, label.shape)
            x1.append(data) 
            x2.append(label) 
        x1 = np.concatenate(x1, axis=0) 
        x2 = np.concatenate(x2, axis=0) 
        return x1, x2 



class DatasTest():
    def __init__(self, file_name="h5data", n_length=1024, stride=16, padlen=256, mindist=0, maxdist=1000):
        self.file_name = file_name
        self.length = n_length  
        self.stride = stride 
        self.padlen = padlen
        self.n_thread = 2 
        self.maxdist = maxdist 
        self.mindist = mindist 
        self.phase_dict = {
            "Pg":0, 
        } 
        self.ploar1 = {
            "C":0, 
            "U":0, 
            "R":1, 
            "D":1, 
        }
        self.ploar2 = {
            "I":0, 
            "M":1, 
            "E":2, 
        }
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        for i in range(1):
            p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, i+2020),daemon=True)
            p1.start() 
        for _ in range(self.n_thread):
            c1 = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue),daemon=True)
            c1.start()
        #multiprocessing.Process(target=self.batch_data, args=(dqueue, )).start()
    def feed_data(self, fqueue, year):
        
        while True:
            h5file = h5py.File(os.path.join(self.file_name, f"{year}.h5"), "r")
            for ekey in h5file:
                event = h5file[ekey] 
                for skey in event:
                    station = event[skey] 
                    if "POLARITY.Pg" not in station.attrs:continue 
                    if "POLARITY.Pg.UPDOWN" not in station.attrs:continue
                    if "POLARITY.Pg.CLARITY" not in station.attrs:continue 
                    ptype1 = station.attrs["POLARITY.Pg.UPDOWN"]
                    ptype2 = station.attrs["POLARITY.Pg.CLARITY" ]

                    if ptype1 not in self.ploar1 or ptype2 not in self.ploar2:continue 

                    data = []
                    for dkey in station:
                        if "HZ" not in dkey:continue 
                        data.append(station[dkey][:]) 
                        btime = datetime.datetime.strptime(station[dkey].attrs['btime'], "%Y/%m/%d %H:%M:%S.%f") 
                    if len(data)!=1:continue 
                    phases = {}
                    dist = -1
                    for akey in station.attrs:
                        if "dist" in akey:
                            dist = float(station.attrs[akey])
                        if "Pg" in akey:
                            pname = akey.split(".")[-1]
                            if pname in self.phase_dict:
                                phases[pname] = datetime.datetime.strptime(station.attrs[akey], "%Y/%m/%d %H:%M:%S.%f")
                        else:
                            if akey in self.phase_dict:
                                phases[akey] = datetime.datetime.strptime(station.attrs[akey], "%Y/%m/%d %H:%M:%S.%f")
                    if len(phases)==0:continue
                    fqueue.put([data, btime, phases, dist, [self.ploar1[ptype1], self.ploar2[ptype2]]])
    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, btime, phases, dist, ptypes = fqueue.get()
            if dist>self.maxdist or dist < self.mindist:
                continue 
            pidx = {}
            plist = []
            for pkey in phases:
                ptime = phases[pkey]
                delta = (ptime-btime).total_seconds()
                delta_idx = int(delta * 100)
                pidx[pkey] = delta_idx
                plist.append(delta_idx) 

            cidx = np.random.choice(plist) - 512#np.random.randint(self.padlen, self.length-self.padlen)
            rdata = [] 
            flen = False
            for d in data:
                w = d[cidx:cidx+self.length] 
                wlen = len(w) 
                if wlen!=self.length:
                    flen = True
                    break 
                w = w - np.mean(w)
                #w = w / (np.max(w)+1e-6)
                rdata.append(w[np.newaxis, :, np.newaxis]) 
            if flen:
                continue
            rdata = np.concatenate(rdata, axis=2) 
            rdata /= (np.std(rdata) + 1e-6) 
            #rdata *= np.random.uniform(0.5, 1.5)
            label = np.array(ptypes)[np.newaxis, ...]
            snr = 0
            if snr < -13:
                continue 
            dqueue.put([rdata, label])
            count += 1

    def batch_data(self, batch_size=32):
        x1, x2 = [], []
        for _ in range(batch_size):
            data, label = self.dqueue.get() 
            x1.append(data) 
            x2.append(label) 
        x1 = np.concatenate(x1, axis=0) 
        x2 = np.concatenate(x2, axis=0) 
        return x1, x2 

if __name__=="__main__":
    datatool = DatasTest()
    for i in range(3):
        datatool.batch_data()
