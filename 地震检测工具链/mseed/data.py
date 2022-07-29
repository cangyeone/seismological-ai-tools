from obspy.signal import filter  
import numpy as np 
import multiprocessing 
from .dbdata import Client 
import datetime 
from obspy import UTCDateTime
import h5py 
class DataHighSNR():
    def __init__(self, file_name="models/h5test/all-gzip4.h5", n_length=10240, stride=16, padlen=256, mindist=0, maxdist=1000):
        self.file_name = file_name
        self.length = n_length  
        self.stride = stride 
        self.padlen = padlen
        self.n_thread = 2 
        self.maxdist = maxdist 
        self.mindist = mindist 
        self.phase_dict = {
            "P":0, 
            "Pg":0
        } 
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, ),daemon=True)
        p1.start() 
        for _ in range(self.n_thread):
            c1 = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue),daemon=True)
            c1.start()
        #multiprocessing.Process(target=self.batch_data, args=(dqueue, )).start()
    def feed_data(self, fqueue):
        
        while True:
            ##ii=0
            #print("Feed Inited")
            h5file = h5py.File(self.file_name, "r")
            for ekey in h5file:
                #print("edata1")
                event = h5file[ekey] 
                for skey in event:
                    station = event[skey] 
                    data = []
                    for dkey in station:
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
                    fqueue.put([data, btime, phases, dist])
                    ##ii = ii+1
            ##print(f"Feeded {ii} phases!")

    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, btime, phases, dist = fqueue.get()
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
            if np.random.random()<0.0:
                rdata = [] 
                flen = False
                cidx = np.random.randint(0, 20000)
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
                label = -np.ones([1, llen, 2])    
                dqueue.put([rdata, label])          
            else:
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
                label = -np.ones([1, llen, 2])
                filter_data = filter.bandpass(w, freqmin=0.1, freqmax=5, df=100, corners=4)
                snr = 0
                cpass = False 
                for pkey in pidx:
                    if pkey not in self.phase_dict:
                        cpass = True 
                        continue 
                    pid = self.phase_dict[pkey] 
                    idx = (pidx[pkey] - cidx)//self.stride
                    if idx > 0 and idx < llen:
                        if pid == 0:
                            b = np.max([0, idx-1000])
                            pre = filter_data[b:idx] 
                            aft = np.max(np.abs(filter_data[b:idx+1000]))
                            if len(pre)==0:pre=np.zeros([100])
                            snr = aft / (np.std(pre) + 1e-6)
                            break  
                        else:
                            b = np.max([0, idx-1000])
                            pre = filter_data[b:idx] 
                            aft = np.max(np.abs(filter_data[b:idx+1000]))
                            if len(pre)==0:pre=np.zeros([100])
                            snr = aft / (np.std(pre) + 1e-6)
                if snr < 3.0:
                    continue 
                for pkey in pidx:
                    if pkey not in self.phase_dict:
                        cpass = True 
                        continue 
                    pid = self.phase_dict[pkey] 
                    idx = (pidx[pkey] - cidx)//self.stride
                    if idx > 0 and idx < llen:
                        label[0, idx, 0] = pid 
                        label[0, idx, 1] = (pidx[pkey] - cidx)%self.stride
                if cpass == True:
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
class DataHighSNRTest():
    def __init__(self, file_name="models/h5test/all-gzip4.h5", n_length=10240, stride=16, padlen=256, mindist=0, maxdist=1000):
        self.file_name = file_name
        self.length = n_length  
        self.stride = stride 
        self.padlen = padlen
        self.n_thread = 2 
        self.maxdist = maxdist 
        self.mindist = mindist 
        self.phase_dict = {
            "P":0, 
            "Pg":0
        } 
        fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        p1 = multiprocessing.Process(target=self.feed_data, args=(fqueue, ),daemon=True)
        p1.start() 
        for _ in range(self.n_thread):
            c1 = multiprocessing.Process(target=self.process, args=(fqueue, self.dqueue),daemon=True)
            c1.start()
        #multiprocessing.Process(target=self.batch_data, args=(dqueue, )).start()
    def feed_data(self, fqueue):
        
        while True:
            ##ii=0
            #print("Feed Inited")
            h5file = h5py.File(self.file_name, "r")
            for ekey in h5file:
                #print("edata1")
                event = h5file[ekey] 
                for skey in event:
                    station = event[skey] 
                    data = []
                    for dkey in station:
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
                    fqueue.put([data, btime, phases, dist])
                    ##ii = ii+1
            ##print(f"Feeded {ii} phases!")

    def process(self, fqueue, dqueue):
        count = 0
        llen = self.length//self.stride
        while True:
            data, btime, phases, dist = fqueue.get()
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
            if np.random.random()<0.0:
                rdata = [] 
                flen = False
                cidx = np.random.randint(0, 20000)
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
                label = -np.ones([1, llen, 2])    
                dqueue.put([rdata, label])          
            else:
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
                label = -np.ones([1, llen, 2])
                filter_data = filter.bandpass(w, freqmin=0.1, freqmax=5, df=100, corners=4)
                snr = 0
                cpass = False 
                for pkey in pidx:
                    if pkey not in self.phase_dict:
                        cpass = True 
                        continue 
                    pid = self.phase_dict[pkey] 
                    idx = (pidx[pkey] - cidx)//self.stride
                    if idx > 0 and idx < llen:
                        if pid == 0:
                            b = np.max([0, idx-1000])
                            pre = filter_data[b:idx] 
                            aft = np.max(np.abs(filter_data[b:idx+1000]))
                            if len(pre)==0:pre=np.zeros([100])
                            snr = aft / (np.std(pre) + 1e-6)
                            break  
                        else:
                            b = np.max([0, idx-1000])
                            pre = filter_data[b:idx] 
                            aft = np.max(np.abs(filter_data[b:idx+1000]))
                            if len(pre)==0:pre=np.zeros([100])
                            snr = aft / (np.std(pre) + 1e-6)
                #if snr > 3.0:
                #    continue 
                crridx = 0
                for pkey in pidx:
                    if pkey not in self.phase_dict:
                        cpass = True 
                        continue 
                    pid = self.phase_dict[pkey] 
                    idx = (pidx[pkey] - cidx)//self.stride
                    if idx > 0 and idx < llen:
                        label[0, idx, 0] = pid 
                        label[0, idx, 1] = (pidx[pkey] - cidx)%self.stride
                        crridx = pidx[pkey] - cidx
                if cpass == True:
                    continue 
                dqueue.put([rdata, crridx])
            count += 1

    def batch_data(self, batch_size=32):
        x1, x2 = [], []
        for _ in range(batch_size):
            data, label = self.dqueue.get() 
            #print(data.shape, label.shape)
            x1.append(data) 
            x2.append(label) 
        x1 = np.concatenate(x1, axis=0) 
        return x1, x2 