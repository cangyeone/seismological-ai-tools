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
from config.picker import Parameter as global_parameter


plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150

def post(prob, time, a=0.1, b=200):
    # ab分别是概率阈值和避免重复的间隔
    # ONNX模型需要后处理
    output = []
    t, c = prob.shape 
    for itr in range(c-1):
        pc = prob[:, itr+1] 
        time_sel = time[pc>a]
        score = pc[pc>a]
        order = np.argsort(score)[::-1]
        ntime = time_sel[order] 
        nprob = score[order]
        #print(batchstride, ntime, nprob)
        select = -np.ones_like(order)
        selidx = np.arange(len(order))
        count = 0
        while True:
            if len(nprob)<1:
                break 
            ref = ntime[0]
            idx = selidx[0]
            select[idx] = 1 
            count += 1 
            #print(ref, nprob[idx])
            kidx = np.abs(ref-ntime)>b
            selidx = selidx[kidx] 
            nprob = nprob[kidx] 
            ntime = ntime[kidx]
        p_time = time_sel[order][select>0] 
        p_prob = score[order][select>0] 
        p_type = np.ones_like(p_time) * itr 
        y = np.stack([p_type, p_time, p_prob], axis=1)
        output.append(y) 
    if len(output) == 0:
        return []
    y = np.concatenate(output, axis=0) 
    return y    
class Process():
    def __init__(self, infile="data", outfile="data/out", model="", device="cpu:0"):
        #self.base_dir = "/data/workWANGWT/yangbi/rawdata_semirealtime_archive/archive149"
        self.base_dir = infile
        self.modeldir = model
        self.device_name = device
        # 这里通道数不太合理，取得比较大
        n_thread = global_parameter.npre   # 16
        dataq = multiprocessing.Queue(maxsize=5)
        feedq = multiprocessing.Queue(maxsize=5)
        batchq = multiprocessing.Queue(maxsize=5)
        npq = multiprocessing.Queue(maxsize=5)
        pickq = multiprocessing.Queue(maxsize=5)
        outq = multiprocessing.Queue(maxsize=5)
        self.outfile = outfile
        self.infile = infile
        self.n_thread = n_thread 
        self.processed_dict = {}
        if os.path.exists(f"{self.outfile}.log"):
            logfile = open(f"{self.outfile}.log", "r", encoding="utf-8")
            for line in logfile.readlines():
                key = line.strip()
                self.processed_dict[key] = 0
            logfile.close()
        self.data = [[(j, i) for i in range(3)] for j in range(10)]
        processt = []
        for itr in range(n_thread):
            t_data = multiprocessing.Process(
                target=self.process, args=(feedq, dataq, itr))
            t_data.start()
            processt.append(t_data)
        if os.path.isdir(infile):  # 检查是否是文件，如果是目录则遍历目录找mseed
            feedt = multiprocessing.Process(target=self.feed2, args=(
                feedq, self.processed_dict))
            feedt.start()
        else:  # 如果是文件则读取文件
            feedt = multiprocessing.Process(target=self.feed3, args=(
                feedq, self.processed_dict))
            feedt.start()
        pickert = []
        for itr in range(global_parameter.npicker):
            if self.modeldir.endswith(".onnx"):
                t = multiprocessing.Process(
                    target=self.infer2, args=(dataq, outq))
                t.start()
                pickert.append(t)
            else:
                t = multiprocessing.Process(
                    target=self.infer, args=(dataq, outq))
                t.start()
                pickert.append(t)
        outt = multiprocessing.Process(target=self.write, args=(outq, ))
        outt.start()

        while True:
            if dataq.empty() and feedq.empty() and outq.empty():
                print("终止feed进程")
                feedt.join()
                print("终止预处理进程")
                for i in range(self.n_thread):
                    feedq.put({}) 
                for t in processt:
                    t.join() 
                print("终止拾取进程")
                for i in range(global_parameter.npicker):
                    dataq.put({})
                for t in pickert:
                    t.join()
                print("终止输出进程")
                outq.put([])
                outt.join()
                print("所有数据处理完毕！")
                break 
            else:
                time.sleep(0.1)

    def process(self, feedq, dataq, fid):
        seq_len = 0#global_parameter.nsamplesdots
        #ofile = open(f"logdir/errordata.{fid}.log", "w", encoding="utf8")
        while True:
            temp = feedq.get()
            if len(temp)==0:
                break 
            base_dir = temp["root"]
            file_dir = temp["dirs"]
            #print("测试", file_dir)
            if len(file_dir) != global_parameter.nchannel:
                print(f"数据不足{global_parameter.nchannel}个", file_dir)
                continue
            # if not (file_dir[0].endswith("mseed") or file_dir[1].endswith("mseed") or file_dir[2].endswith("mseed")):
            #    continue
            # print(file_dir)
            datas = []
            is100Hz = True
            waves = []
            btime = []
            errinfo = {"largegap":[], "dataerr":[]}
            maxlen = 0 
            try:
                for fdir in file_dir:
                    st = obspy.read(os.path.join(base_dir, fdir)
                                    ).merge(fill_value=0)
                    st.resample(global_parameter.samplerate)
                    st.trim(pad=True, nearest_sample=True, fill_value=0)
                    file_data = st[0]
                    # print("起始时间")
                    stime = datetime.datetime.strptime(
                        f"{file_data.stats.starttime}", r"%Y-%m-%dT%H:%M:%S.%fZ")
                    wave = file_data.data
                    if len(wave)>maxlen:
                        maxlen = len(wave)
                    waves.append(wave)
                    btime.append(stime)
            except:
                print("数据错误", os.path.join(base_dir, fdir))
                #ofile.write(f"{os.path.join(base_dir, fdir)}\n")
                #ofile.flush()
                errinfo["dataerr"].append(os.path.join(base_dir, fdir))
                outinfo = {"errinfo":errinfo}
                dataq.put(outinfo)
                continue 
            seq_len = maxlen 
            td1 = (btime[1]-btime[0]).total_seconds()
            td2 = (btime[2]-btime[0]).total_seconds()
            idx = np.argmax([0, td1, td2])
            tmax = np.max([0, td1, td2])
            if tmax > 1:
                errinfo["largegap"].append([tmax, os.path.join(base_dir, fdir)])
                print("数据差距过大", fdir)
            reftime = btime[idx]
            for w, t in zip(waves, btime):
                bidx = (reftime-t).total_seconds()
                bidx = int(bidx*100)
                wave = w[bidx:]
                indata = np.zeros([seq_len])
                n_dots = len(wave)
                if n_dots > seq_len:
                    indata[:] = wave[:seq_len]
                elif n_dots < seq_len:
                    indata[:n_dots] = wave
                else:
                    indata[:] = wave
                datas.append(indata)
            #tr_filt = file_data.copy()
            #tr_filt.filter('bandpass',  freqmin=1, freqmax=10, corners=4, zerophase=True)
            fdata = bandpass(
                wave, global_parameter.bandpass[0], global_parameter.bandpass[1], 100)
            if is100Hz == False:
                print("数据不是100Hz", file_dir)
                continue
            outinfo = {"root": base_dir, "key": file_dir[0], "data": datas,
                       "stime": reftime, "fdata": fdata, "pkey": temp["pkey"], "errinfo":errinfo}
            dataq.put(outinfo)

    def feed2(self, feedq, processed_dict):
        if True:
            # 其他文件夹
            print("遍历目录为", self.infile)
            feedinfos = []
            for root, dirs, files in os.walk(self.infile):
                # if len(dirs)!=0:continue
                if len(files) == 0:
                    continue
                file_dict = {}
                for name in files:
                    if name.endswith(global_parameter.filenametag) == False:
                        continue
                    sname = name.split(".")
                    key = ".".join([sname[i]
                                    for i in global_parameter.namekeyindex])
                    if key in file_dict:
                        file_dict[key].append(name)
                    else:
                        file_dict[key] = [name]
                #print("需要处理文件", file_dict)
                for key in file_dict:
                    # print(key)
                    processed_key = f"{root}/{key}"
                    name = key.split(".")[0]
                    if processed_key in processed_dict:
                        continue
                    threes = [[0 for i in range(global_parameter.nchannel)] for i in range(len(global_parameter.chnames))]
                    #three2 = []
                    for fn in file_dict[key]:
                        # print(fn)
                        # 需要标注BHE或SHE,N,Z
                        sfn = fn.split(".")[global_parameter.channelindex]
                        for cid, chn in enumerate(global_parameter.chnames):
                            sid, = np.where(np.array(chn)==sfn)
                            if len(sid)==0:continue 
                            #print("CNO=", cid, sid)
                            threes[cid][sid[0]] = fn 
                            ##if sfn in chn:
                            ##    threes[cid].append(fn)
                    for three in threes:
                        if 0 not in three:
                            info = {"root": root, "key": key,
                                "dirs": three, "pkey": processed_key}
                            # print(info)
                            feedinfos.append([info, processed_key])
                # print(len(feedinfos))
            print("数据总量", len(feedinfos))
            for info, processed_key in feedinfos:
                processed_dict[processed_key] = 0
                feedq.put(info)
            #print(f"当前文件夹{root}, {len(files)}, {feedq.qsize()}")
            #for i in range(self.n_thread):
            #    feedq.put({})
    def feed3(self, feedq, processed_dict):
        if True:
            # 其他文件夹
            file_ = open(self.infile, "r", encoding="utf-8")
            for line in file_.readlines():
                root, fn1, fn2, fn3 = line.strip().split(",")
                dirs = [os.path.join(root, i) for i in [fn1, fn2, fn3]]
                info = {"root": root, "key": root+fn1, "dirs": dirs}
                feedq.put(info)
            #print(f"当前文件夹{root}, {len(files)}, {feedq.qsize()}")

    def infer(self, dataq, outq):
        with torch.no_grad():
            device = torch.device(self.device_name)
            lppn = torch.jit.load(self.modeldir)
            lppn.eval()
            lppn.to(device)
            #lppn.half()
            print("处理进程加载完成，准备处理数据")
            while True:
                temp = dataq.get()
                if len(temp)==0:break 
                if len(temp)==1:
                    outq.put([temp["errinfo"]])
                    continue 
                #print("已获取", temp["root"])
                root = f"{temp['root']}/{temp['key']}"
                data = temp["data"]
                #print("当前数据中", f"{temp['root']}+++++{temp['key']}" )
                t1 = time.perf_counter()
                #print("处理开始", dataq.qsize())
                data = np.vstack(data).T 
                with torch.no_grad():
                    nnout = lppn(torch.tensor(data, dtype=torch.float, device=device))
                    nnout = nnout.cpu().numpy()
                # print("处理结束")
                t2 = time.perf_counter()
                #print("ROOT", root)
                # $print(f"数据{temp['root']}/{temp['key']}纯处理时间：{t2-t1}")
                outq.put([root, nnout, temp["data"], temp["stime"],
                          temp["fdata"], temp["pkey"], temp["errinfo"]])
    def infer2(self, dataq, outq):
        import onnxruntime as ort
        mname =self.modeldir # 其他onnx均可
        sess = ort.InferenceSession(mname, providers=['CPUExecutionProvider'])#使用pickers中的onnx文件
        if self.device_name == "cuda":
            sess.set_providers(['CUDAExecutionProvider'])
        #lppn.half()
        print("处理进程加载完成，准备处理数据")
        while True:
            temp = dataq.get()
            #print("已获取", temp["root"])
            if len(temp)==0:break 
            if len(temp)==1:
                outq.put([temp["errinfo"]])
                continue 
            root = f"{temp['root']}/{temp['key']}"
            data = temp["data"]
            #print("当前数据中", f"{temp['root']}+++++{temp['key']}" )
            #t1 = time.perf_counter()
            #print("处理开始", dataq.qsize())
            data = np.vstack(data).T 
            prob, time = sess.run(["prob", "time"], {"wave":data.astype(np.float32)})
            nnout = post(prob, time, global_parameter.prob, global_parameter.nmslen)
            # print("处理结束")
            #t2 = time.perf_counter()
            #print("ROOT", root)
            # $print(f"数据{temp['root']}/{temp['key']}纯处理时间：{t2-t1}")
            outq.put([root, nnout, temp["data"], temp["stime"],
                      temp["fdata"], temp["pkey"], temp["errinfo"]])
    def write(self, outq):
        # 输出文件
        import scipy.signal as signal
        if global_parameter.polar:
            import onnxruntime as ort
            sess = ort.InferenceSession("pickers/polar.onnx", providers=['CPUExecutionProvider'])#使用pickers中的onnx文件
        files = open(f"{self.outfile}.txt", "a", encoding="utf-8")
        logfile = open(f"{self.outfile}.log", "a", encoding="utf-8")
        errfile = open(f"{self.outfile}.err", "a", encoding="utf-8")
        files.write(
            "##数据格式为:\n##数据位置\n##震相,相对时间（秒）,置信度,绝对时间（格式%Y-%m-%d %H:%M:%S.%f）,信噪比,前后200个采样点振幅均值,台站,初动,初动概率\n")
        count = 0
        acc_time = np.ones([10])
        acc_index = 0
        ifplot = global_parameter.ifplot
        fcount = 0
        while True:
            start = time.perf_counter()
            # 文件位置，拾取结果，数据，起始时间
            temp = outq.get()
            if len(temp)==0:
                break 
            if len(temp)==1:
                errfile.write(f"{str(temp[0])}\n")
                continue 
            root, nnpick, data, stime, fdata, pkey, errinfo = temp
            errfile.write(f"{str(errinfo)}\n")
            errfile.flush()
            # 校正时间+8
            #stime = stime + datetime.timedelta(hours=8)
            files.write(f"#{root}\n")
            # files.write(f"#starttime:{stime.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
            logfile.write(f"{pkey}\n")
            phase_dict = {
                0: "Pg",
                1: "Sg",
                2: "Pn",
                3: "Sn",
                4: "P",
                5: "S"
            }
            #pres = os.path.split(root)[-1].split(".")[:2]
            #sta = ".".join(pres)
            #sta = os.path.split("\\")[-1].split("/")[-1]
            sta = os.path.split(pkey)[-1]
            for itr in range(6):  # 六个震相
                pname = "N"
                if itr == 0:
                    pname = "P"
                else:
                    pname = "S"
                phases = nnpick[nnpick[:, 0] == itr]
                for p, t in zip(phases[:, 2], phases[:, 1]):
                    # t[0]相对到时
                    # p置信度
                    #print(p, t)
                    pidx = int(t)
                    if "P" in phase_dict[itr]:
                        b = np.max([0, pidx-global_parameter.snritv])
                        pre = fdata[b:pidx]
                        aft = fdata[pidx:pidx+global_parameter.snritv]

                        if len(pre) == 0:
                            pre = np.ones([global_parameter.snritv])
                        if len(aft) == 0:
                            aft = np.ones([global_parameter.snritv])
                        pre -= np.mean(pre)
                        aft -= np.mean(aft)
                        snr = np.std(aft)/(np.std(pre) + 1e-6)
                        b1 = np.percentile(np.abs(pre), 95)
                        e1 = np.percentile(np.abs(aft), 95)
                        b2 = np.max(np.abs(pre))
                        e2 = np.max(np.abs(aft))
                        b3 = np.std(pre)
                        e3 = np.std(aft)
                        b = np.max([0, pidx-global_parameter.snritv])
                        segs = fdata[b:pidx+global_parameter.snritv]
                        segs -= np.mean(segs)
                        peaks, _ = signal.find_peaks(segs)
                        pki = peaks[np.argsort(
                            np.abs(peaks-global_parameter.snritv))]
                        pkv = segs[pki]
                    else:
                        b = np.max([0, pidx-global_parameter.snritv])
                        pre = fdata[b:pidx]
                        aft = fdata[pidx:pidx+global_parameter.snritv]
                        if len(pre) == 0:
                            pre = np.ones([100])
                        if len(aft) == 0:
                            aft = np.ones([100])
                        pre -= np.mean(pre)
                        aft -= np.mean(aft)
                        snr = np.std(aft)/(np.std(pre) + 1e-6)
                        b1 = np.percentile(np.abs(pre), 95)
                        e1 = np.percentile(np.abs(aft), 95)
                        b2 = np.max(np.abs(pre))
                        e2 = np.max(np.abs(aft))
                        b3 = np.std(pre)
                        e3 = np.std(aft)
                        b = np.max([0, pidx-global_parameter.snritv])
                        segs = fdata[b:pidx+global_parameter.snritv]
                        segs -= np.mean(segs)
                        peaks, _ = signal.find_peaks(segs)
                        pki = peaks[np.argsort(
                            np.abs(peaks-global_parameter.snritv))]
                        pkv = segs[pki]
                    b = np.max([0, pidx-global_parameter.snritv])
                    w = fdata[b:pidx+global_parameter.snritv]
                    if ifplot and (snr > 5):
                        b = np.max([0, pidx-1000])  # 截取前500秒
                        w = fdata[b:pidx+3000]  # 截取后10秒
                        w -= np.mean(w)
                        w /= np.max(np.abs(w))
                        plt.cla()
                        plt.clf()
                        plt.plot(w)
                        plt.axvline(1000, c="r")
                        plt.savefig(f"logdir/x1.{fcount}.png")
                        fcount += 1
                    if len(w) > 0:
                        amp = np.max(np.abs(fdata[b:pidx+100]))
                    else:
                        amp = -1
                    ptime = stime + \
                        datetime.timedelta(
                            seconds=t/global_parameter.samplerate)
                    polar_name = "N" 
                    polar_prob = 0.0 
                    if global_parameter.polar and phase_dict[itr]=="Pg":
                        pidx = int(t) 
                        w = data[-1] 
                        if pidx <= 512:
                            pidx = 512 
                        if pidx >= len(w) - 512:
                            pidx = len(w) - 512 
                        pdata = w.astype(np.float32)[pidx-512:pidx+512]
                        if len(pdata)>1024:
                            pdata = pdata[:512]
                        if len(pdata)<1024:
                            pdata = np.pad(pdata, (0, 1024-len(pdata)))
                        #print(pidx-512, pidx+512, pidx, pdata.shape)
                        prob,  = sess.run(["prob"], {"wave":pdata.astype(np.float32)})
                        polarid = np.argmax(prob)
                        if polarid == 0:
                            polar_name = "U" 
                        else:
                            polar_name = "D"
                        polar_prob = np.max(prob)
                    if t/100 < 0:
                        print(t, ptime)
                    if len(pki) >= 2:
                        files.write(
                            f"{phase_dict[itr]},{t/global_parameter.samplerate:.3f},{p:.3f},{ptime.strftime('%Y-%m-%d %H:%M:%S.%f')},{snr:.3f},{amp},{sta},{polar_name},{polar_prob:.3f}\n")
                    elif len(pki) == 1:
                        files.write(
                            f"{phase_dict[itr]},{t/global_parameter.samplerate:.3f},{p:.3f},{ptime.strftime('%Y-%m-%d %H:%M:%S.%f')},{snr:.3f},{amp},{sta},{polar_name},{polar_prob:.3f}\n")
                    elif len(pki) == 0:
                        files.write(
                            f"{phase_dict[itr]},{t/global_parameter.samplerate:.3f},{p:.3f},{ptime.strftime('%Y-%m-%d %H:%M:%S.%f')},{snr:.3f},{amp},{sta},{polar_name},{polar_prob:.3f}\n")
                    if pname != "N" and global_parameter.ifreal:
                        # 制作REAL日期文件夹
                        timedir = ptime.strftime("%Y%m%d")
                        baseptime = datetime.datetime.strptime(
                            timedir, "%Y%m%d")
                        realroot = self.outfile + "realdata"
                        if os.path.exists(realroot) == False:
                            os.mkdir(realroot)
                        basedir = os.path.join(realroot, "REAL"+timedir)
                        if os.path.exists(basedir) == False:
                            os.mkdir(basedir)
                        # 台站名
                        file_name = ".".join(pres + [pname, "txt"])  # 文件名
                        file_ = open(os.path.join(
                            basedir, file_name), "a")  # 关联文件名
                        # 52231.555 0.986 0
                        file_.write(
                            f"{(ptime-baseptime).total_seconds()} {p:.3f} {0}\n")
                        file_.flush()
                        file_.close()
                # if pname != "N":
                    # file_.close()
            files.flush()
            logfile.flush()
            end = time.perf_counter()
            acc_time[acc_index % 10] = end-start
            acc_index += 1
            print(
                f"已处理:{acc_index}, 当前完成{root}, {stime}, 平均用时{np.mean(acc_time)}")


# nohup python lppn.py -o result/3days.new -m ckpt/new.200km.wave -d cuda:0 > result/2020.200km.log 2>&1 &
# 873412
if __name__ == "__main__":
    # 处理输出某个时间之后的所有震相
    parser = argparse.ArgumentParser(description="拾取连续波形")
    parser.add_argument('-i', '--input', default="data", help="输入连续波形")
    parser.add_argument(
        '-o', '--output', default="data/20220619224053N00", help="输出文件名")
    parser.add_argument(
        '-m', '--model', default="pickers/lppnm.onnx", help="模型文件lppnmodel")
    parser.add_argument('-d', '--device', default="cpu",
                        help="模型文件lppnmodel")
    args = parser.parse_args()
    infile = args.input
    outfile = args.output
    Process(infile, outfile, args.model, args.device)
