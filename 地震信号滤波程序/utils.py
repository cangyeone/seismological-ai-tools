import pickle 
import matplotlib.pyplot as plt 
import numpy as np 
import obspy 
import os 
import datetime 
import math 
import scipy.signal as signal 
import time 
import multiprocessing 
plt.switch_backend('agg')
import h5py 
def _csv_file():
    if os.path.exists("ckpt/phasedata.pkl")==True:
        outfile = open("ckpt/phasedata.pkl", "rb")
        phase_data = pickle.load(outfile) 
        outfile.close()
        phase_data_clean, phase_data_noise = phase_data 
    else:
        phasefile = open("data/metadata_11_13_19.csv", "r")
        line1 = phasefile.readline()
        phase_data_clean = []
        phase_data_noise = [] 

        for itr in phasefile.readlines():
            line = itr.strip().split(",")
            if len(line)!=35:continue
            #print([a for a in zip(range(len(line)), line)], len(line))
            if True:
                if "noise" == line[33]:
                    name = line[-1].strip()
                    pt = -100
                    st = -100 
                    snr = -100
                    phase_data_noise.append([name, pt, st, snr]) 
                else:
                    name = line[-1].strip()
                    lon = float(line[4])
                    lat = float(line[3]) 
                    if lon < -126 or lon > -118:continue 
                    if lat < 34 or lat > 41:continue  
                    lon = float(line[17])
                    lat = float(line[16]) 
                    if lon < -126 or lon > -118:continue 
                    if lat < 34 or lat > 41:continue  
                    pt = float(line[6])
                    st = float(line[10]) 
                    snr = [float(a) for a in line[30][1:-1].split(" ") if len(a)>3]
                    if np.mean(snr) <20:continue 
                    #snr = [float(a) for a in line[30][1:-1].split(" ") if len(a)>3]
                    phase_data_clean.append([name, pt, st, snr])
            else:
                print([a for a in zip(range(len(line)), line)], len(line))
        outfile = open("ckpt/phasedata.pkl", "wb")
        pickle.dump([phase_data_clean, phase_data_noise], outfile)
        phasefile.close() 
    return phase_data_clean, phase_data_noise
import time 
class Data():
    def __init__(self, batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.n_length = n_length
        self.n_stride = strides
        self.clean, self.noise = _csv_file()
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue_clean = multiprocessing.Queue(maxsize=10) 
        self.in_queue_noise = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        
        for i in range(3):
            multiprocessing.Process(target=self.batch_data_input_clean, args=(self.in_queue_clean, self.clean)).start() 
            multiprocessing.Process(target=self.batch_data_input_noise, args=(self.in_queue_noise, self.noise)).start() 
            multiprocessing.Process(target=self.process_multithread, args=(self.in_queue_clean, self.in_queue_noise, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input_clean(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") # HDF5文件位置
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            np.random.shuffle(train_label)
            for key, pt, st, snr in train_label:
                waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def batch_data_input_noise(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            np.random.shuffle(train_label)
            for key, pt, st, snr in train_label:
                waves = noise[key][:, :]
                snr = [1, 1, 1]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue_clean, in_queue_noise, out_queue):
        """CCC"""
        while True:
            temp = in_queue_clean.get()
            wave, pt, st, snr = temp 
            d_legnth, n_channel = wave.shape
            wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
            
            p_idx = int(pt)
            s_idx = int(st)
            begin = np.random.randint(0, max(p_idx, 2)) 
            
            begin = max(0, begin) 
            wave = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 


            temp = in_queue_noise.get()
            noise, pt, st, snr = temp 
            noise /= (np.max(np.abs(noise), axis=0, keepdims=True)+1e-6)
            begin = np.random.randint(0, d_legnth-self.n_length) 
            noise = np.reshape(noise, [1, -1, 3])[:, begin:begin+self.n_length, :]   


            p_idx = p_idx - begin 
            s_idx = s_idx - begin 
            n_stride = self.n_stride
            n_legnth_s = self.n_length // n_stride
            p_idx_s = p_idx // n_stride 
            s_idx_s = s_idx // n_stride
            logit = -np.ones([1, n_legnth_s, 2]) 
            if p_idx_s > 0:
                logit[0, p_idx_s:p_idx_s+1, 0] = 0 
                logit[0, p_idx_s:p_idx_s+1, 1] = p_idx % n_stride  
            if s_idx_s > 0:
                logit[0, s_idx_s:s_idx_s+1, 0] = 1  
                logit[0, s_idx_s:s_idx_s+1, 1] = s_idx % n_stride  


            snr = np.random.uniform(0.3, 0.6) 
            wave_noise = wave + noise * snr

            out_queue.put([wave_noise, wave]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels = out_queue.get() 
                a1.append(wave)
                a2.append(labels) 

            a1 = np.concatenate(a1, axis=0)
            a2 = np.concatenate(a2, axis=0)
            batch_queue.put([a1, a2])
    def batch_data(self):
        a1, a2 = self.batch_queue.get() 
        return a1, a2


class DataWithPick():
    def __init__(self, batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.n_length = n_length
        self.n_stride = strides
        self.clean, self.noise = _csv_file()
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue_clean = multiprocessing.Queue(maxsize=10) 
        self.in_queue_noise = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        
        for i in range(3):
            multiprocessing.Process(target=self.batch_data_input_clean, args=(self.in_queue_clean, self.clean)).start() 
            multiprocessing.Process(target=self.batch_data_input_noise, args=(self.in_queue_noise, self.noise)).start() 
            multiprocessing.Process(target=self.process_multithread, args=(self.in_queue_clean, self.in_queue_noise, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input_clean(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            np.random.shuffle(train_label)
            for key, pt, st, snr in train_label:
                waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def batch_data_input_noise(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            np.random.shuffle(train_label)
            for key, pt, st, snr in train_label:
                waves = noise[key][:, :]
                snr = [1, 1, 1]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue_clean, in_queue_noise, out_queue):
        """CCC"""
        while True:

            temp = in_queue_noise.get()
            noise, _, _, snr = temp 
            d_legnth, n_channel = noise.shape
            noise /= (np.max(np.abs(noise), axis=0, keepdims=True)+1e-6)
            begin = np.random.randint(0, d_legnth-self.n_length) 
            noise = np.reshape(noise, [1, -1, 3])[:, begin:begin+self.n_length, :]   


            temp = in_queue_clean.get()
            wave, pt, st, snr = temp 
            d_legnth, n_channel = wave.shape
            wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
            
            p_idx = int(pt)
            s_idx = int(st)
            begin = np.random.randint(0, max(p_idx, 2)) 
            
            begin = max(0, begin) 
            wave = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 
            p_idx = int(pt)
            s_idx = int(st)
            p_idx = p_idx - begin 
            s_idx = s_idx - begin 
            n_stride = self.n_stride
            n_legnth_s = self.n_length // n_stride
            p_idx_s = p_idx // n_stride 
            s_idx_s = s_idx // n_stride
            logit = -np.ones([1, n_legnth_s, 2]) 
            if p_idx_s > 0:
                logit[0, p_idx_s:p_idx_s+1, 0] = 0 
                logit[0, p_idx_s:p_idx_s+1, 1] = p_idx % n_stride  
            if s_idx_s > 0:
                logit[0, s_idx_s:s_idx_s+1, 0] = 1  
                logit[0, s_idx_s:s_idx_s+1, 1] = s_idx % n_stride  


            snr = np.random.uniform(0.3, 0.6) 
            wave_noise = wave + noise * snr

            out_queue.put([wave_noise, wave, logit, [p_idx, s_idx]]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels, picks, times = out_queue.get() 
                a1.append(wave)
                a2.append(labels) 
                a3.append(picks)
                a4.append(times)
            a1 = np.concatenate(a1, axis=0)
            a2 = np.concatenate(a2, axis=0)
            a3 = np.concatenate(a3, axis=0)
            batch_queue.put([a1, a2, a3, a4])
    def batch_data(self):
        a1, a2, a3, a4 = self.batch_queue.get() 
        return a1, a2, a3, a4

import time 
class DataTest():
    def __init__(self, batch_size=32, n_thread=1, strides=8, n_length=3000):
        self.batch_size = batch_size 
        self.freq = 100
        self.time_length = 4.0
        self.rage = 0.05
        self.n_length = n_length
        self.n_stride = strides
        self.train_label, self.test_label = _csv_file()
        self.queue = multiprocessing.Queue(maxsize=10) 
        self.in_queue = multiprocessing.Queue(maxsize=10) 
        self.out_queue = multiprocessing.Queue(maxsize=10)
        self.batch_queue = multiprocessing.Queue(maxsize=10) 
        self.data_thread = []
        multiprocessing.Process(target=self.batch_data_input, args=(self.in_queue, self.test_label)).start() 
        for i in range(6):
            multiprocessing.Process(target=self.process_multithread, args=(self.in_queue, self.out_queue)).start()
        multiprocessing.Process(target=self.batch_data_output, args=(self.out_queue, self.batch_queue)).start() 
    def batch_data_input(self, in_queue, train_label):
        h5files = h5py.File("data/waveforms_11_13_19.hdf5", "r") 
        earthquake = h5files["earthquake"]["local"]
        noise = h5files["non_earthquake"]["noise"] 
        while True:
            for key, pt, st, snr in train_label:
                if pt == -100 and st == -100:
                    waves = noise[key][:, :]
                    snr = [1, 1, 1]
                else:
                    waves = earthquake[key][:, :]
                in_queue.put([waves, pt, st, snr])
    def process_multithread(self, in_queue, out_queue):
        """CCC"""
        while True:
            temp = in_queue.get()
            wave, pt, st, snr = temp 
            if pt == -100 and st == -100:
                continue 
                n_legnth, n_channel = wave.shape
                n_stride = self.n_stride 
                n_legnth_s = self.n_length // n_stride
                n_range = 50 
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                #
                p_idx = -500
                s_idx = -500
                logit = -np.ones([1, n_legnth_s, 2])
                wave = np.reshape(wave, [1, -1, 3])[:, :self.n_length, :]
                n_stride = self.n_stride
                
            else:
                d_legnth, n_channel = wave.shape
                wave /= (np.max(np.abs(wave), axis=0, keepdims=True)+1e-6)
                
                p_idx = int(pt)
                s_idx = int(st)
                begin = np.random.randint(0, s_idx)
                begin = 0 
                p_idx = p_idx - begin 
                s_idx = s_idx - begin 

                wave = np.reshape(wave, [1, -1, 3])[:, begin:begin+self.n_length, :] 
            if p_idx < 0:
                isnoise = True 
            else:
                isnoise = False 
            out_queue.put([wave, isnoise, snr, [p_idx, s_idx]]) 
    def batch_data_output(self, out_queue, batch_queue):
        while True:
            a1, a2, a3, a4 = [], [], [], []
            for itr in range(self.batch_size):
                wave, labels, snr, tm = out_queue.get() 
                a1.append(wave)
                a2.append(labels) 
                a3.append(snr) 
                a4.append(tm) 
            
            a1 = np.concatenate(a1, axis=0)
            batch_queue.put([a1, a2, a3, a4])
    def batch_data(self):
        a1, a2, a3, a4 = self.batch_queue.get() 
        return a1, a2, a3, a4


if __name__=="__main__":
    tool = PhaseData() 
    for step in range(20):
        a1, a2, a3, a4, a5, a6 = tool.next_batch() 
        plt.cla()
        plt.clf()
        plt.plot(a1[0, :, 0]) 
        plt.plot(a2[0, :])
        plt.plot(a3[0, :, 0])
        plt.savefig(f"datafig/a-{step}.png")
        print(f"datafig/a-{step}.png")


