
import pandas as pd
from datetime import datetime, timedelta
from gamma.mix import BayesianGaussianMixture, GaussianMixture, calc_time
#from gamma.utils import convert_picks_csv, association, from_seconds
import numpy as np
from sklearn.cluster import DBSCAN 
from datetime import datetime, timedelta
import os
import json
import pickle
from pyproj import Proj
import tqdm 
import datetime 
import os 
import heapq 
import multiprocessing 
from config.gamma import Parameter as gpars 

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
        stloc = []
        for line in file_.readlines():
            sline = [i for i in line.split(" ") if len(i)>0] 
            # 台站ID：SC.JHO
            key = ".".join(sline[:2])
            #if "SC" != sline[0] and "YN" != sline[1]:continue 
            # 台站经纬度
            x, y, z = float(sline[3]), float(sline[4]), float(sline[5])
            self.pos_dict[key] = [x, y, z]
            stloc.append([x, y, z])
        print("台站数量", len(self.pos_dict))
        file_.close() 
        basetime = datetime.datetime.strptime(gpars.basetime, "%Y-%m-%d") # 设置数据起始时间
        self.basetime = basetime 
        file_dir = infilename
        cx, cy, cz = np.mean(stloc, axis=0)
        self.proj = Proj(f"+proj=sterea +lon_0={cx} +lat_0={cy} +units=km")
        self.pos_dict_km = {}
        cxkm, cykm = self.proj(cx, cy)
        poskm = []
        for key, val in self.pos_dict.items():
            x, y, z = val 
            xkm, ykm = self.proj(x, y)
            zkm = z / 1000 
            self.pos_dict_km[key] = [xkm, ykm, zkm]
            poskm.append([xkm, ykm, zkm])
        self.loc_range_km = (np.min(poskm, axis=0) - np.mean(poskm, axis=0), np.max(poskm, axis=0) - np.mean(poskm, axis=0))
        p2id = {"Pg":1, "Sg":2, "Pn":3, "Sn":4, "P":5, "S":6}
        p2pp = {"Pg":"p", "Sg":"s", "Pn":"p", "Sn":"s", "P":"p", "S":"s"}
        file_ = open(file_dir, "r", encoding="utf-8") 
        datas = []
        self.dataheap = []
        print("数据读取中")
        for line in tqdm.tqdm(file_.readlines()[:]):
            if "#" in line:continue
            sline = [i for i in line.strip().split(",") if len(i)>0]
            ptime = datetime.datetime.strptime(sline[3], "%Y-%m-%d %H:%M:%S.%f")
            delta = (ptime-basetime).total_seconds()
            key = sline[6]
            p = float(sline[2])
            snr = float(sline[4])
            #if p < 0.5:continue 
            pname = sline[0]
            if key not in self.pos_dict_km:continue 
            loc = self.pos_dict_km[key]
            #print(loc)
            info = {"st":key, "time":ptime, "pname":pname}
            datas.append([loc[0], loc[1], delta])
            #self.dataheap.append((delta, [loc[0], loc[1], delta, info, p2id[pname]]))
            itm = Item(delta, [loc[0], loc[1], loc[2], delta, info, p2id[pname], p2pp[pname], p, key])
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
        #    if self.dataheap[0][0] < self.mintm + gpars.win_length:
        #        heapq.heappush(self.winheap, heapq.heappop(self.dataheap))
        #    else:
        #        break 
        print("数据读取完成！", "震相数量", len(datas), "数据天数", datetime.timedelta(seconds=self.maxtm-self.mintm).days)
    def getdata(self):
        while True:
            if len(self.dataheap)==0:
                break 
            if self.dataheap[0][0] < self.start + gpars.win_length:
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
            self.start += gpars.win_stirde 
            return [] 
        ddict = {"data":[], "loc":[], "phase":[], "prob":[], "station":[]}
        #[loc[0], loc[1], loc[2], delta, info, p2id[pname], pn]
        for d in self.winheap:
            ddict["data"].append([d.val[3]-self.start, 1])
            ddict["phase"].append(d.val[6])
            ddict["prob"].append(d.val[7])
            ddict["loc"].append(d.val[:3])
            ddict["station"].append(d.val[8])
        ddict["data"] = np.array(ddict["data"])
        ddict["phase"] = np.array(ddict["phase"])
        ddict["prob"] = np.array(ddict["prob"])
        ddict["loc"] = np.array(ddict["loc"])
        ddict["station"] = np.array(ddict["station"])
        ddict["num_sta"] = len(self.pos_dict)
        ddict["timestamp"] = (self.basetime+datetime.timedelta(seconds=self.start)).timestamp()
        self.start += gpars.win_stirde 
        return ddict
    def skipdata(self):
        #idx = ((self.index>self.start) * (self.index<self.start + global_par.win_length)).cpu().numpy()
        #temp = np.copy(self.datas[idx])
        #temp[:, 2] -= self.start 
        self.start += gpars.win_stirde 
        return 0       


def associationt_mt(iqueue, oqueue, data_tool):
    while True:
        picks, logt = iqueue.get()
        vel = gpars.vel 
        data = picks["data"] # 到时+振幅
        locs = picks["loc"]
        db = DBSCAN(eps=gpars.dbscan_eps, min_samples=gpars.dbscan_min_samples).fit(
            np.hstack([data[:, 0:1], locs[:, :2] / vel["p"]])
        ) # 基于密度的搜索算法查找候选样本
        labels = db.labels_
        unique_labels = set(labels)

        events = []
        assignment = []  # 找到地震和震相

        pbar = tqdm.tqdm(total=len(data), desc="Association")
        phase_type = picks["phase"]
        phase_weight = picks["prob"]
        pick_idx = np.arange(len(phase_type))
        pick_station_id = picks["station"]
        timestamp0 = picks["timestamp"]
        basetime = picks["basetime"]
        event_idx = 0 
        for k in unique_labels: # 迭代每个候选样本，这里可以改成其他关联算法
            if k == -1:
                data_ = data[labels == k]
                pbar.set_description(f"Skip {len(data_)} picks")
                pbar.update(len(data_))
                continue
            class_mask = labels == k
            data_ = data[class_mask]
            locs_ = locs[class_mask]
            phase_type_ = phase_type[class_mask]
            phase_weight_ = phase_weight[class_mask]
            pick_idx_ = pick_idx[class_mask]
            pick_station_id_ = pick_station_id[class_mask]

            if len(pick_idx_) < gpars.min_picks_per_eq:
                pbar.set_description(f"Skip {len(data_)} picks")
                pbar.update(len(data_))
                continue

            if pbar is not None:
                pbar.set_description(f"Process {len(data_)} picks")
                pbar.update(len(data_))

            time_range = max(data_[:, 0].max() - data_[:, 0].min(), 1)
            
            initial_mode = "one_point"
            if (initial_mode == "one_point") or (len(data_) < len(centers_init)):
                num_event_init = min(
                        max(int(len(data_) / picks["num_sta"] * gpars.oversample_factor), 3),
                        len(data_),
                    )
                if gpars.dims == ["x(km)", "y(km)", "z(km)"]:
                    centers_init = np.vstack(
                        [
                            np.ones(num_event_init) * 0,
                            np.ones(num_event_init) * 0,
                            np.ones(num_event_init) * 0,
                            np.linspace(
                                data_[:, 0].min() - 0.1 * time_range,
                                data_[:, 0].max() + 0.1 * time_range,
                                num_event_init,
                            ),
                        ]
                    ).T
                elif gpars.dims == ["x(km)", "y(km)"]:
                    centers_init = np.vstack(
                        [
                            np.ones(num_event_init) * 0,
                            np.ones(num_event_init) * 0,
                            np.linspace(
                                data_[:, 0].min() - 0.1 * time_range,
                                data_[:, 0].max() + 0.1 * time_range,
                                num_event_init,
                            ),
                        ]
                    ).T
                elif gpars.dims == ["x(km)"]:
                    centers_init = np.vstack(
                        [
                            np.ones(num_event_init) * 0,
                            np.linspace(
                                data_[:, 0].min() - 0.1 * time_range,
                                data_[:, 0].max() + 0.1 * time_range,
                                num_event_init,
                            ),
                        ]
                    ).T
                else:
                    raise (ValueError("Unsupported dims"))

            ## run clustering
            mean_precision_prior = 0.01 / time_range
            if not gpars.useamp:
                covariance_prior = np.array([[1.0]]) * 5
                data_ = data_[:, 0:1]
            else:
                covariance_prior = np.array([[1.0, 0.0], [0.0, 0.5]]) * 5
            method = "BGMM"
            bmin, bmax = data_tool.loc_range_km #搜索范围
            bounds = ((bmin[0], bmax[0]), (bmin[1], bmax[1]), (0, 20), (None, None))

            if method == "BGMM":
                gmm = BayesianGaussianMixture(
                    n_components=len(centers_init),
                    weight_concentration_prior=1 / len(centers_init),
                    mean_precision_prior=mean_precision_prior,
                    covariance_prior=covariance_prior,
                    init_params="centers",
                    centers_init=centers_init.copy(),
                    station_locs=locs_,
                    phase_type=phase_type_,
                    phase_weight=phase_weight_,
                    vel=vel,
                    loss_type="l1",
                    bounds=bounds,
                ).fit(data_)
            elif method == "GMM":
                gmm = GaussianMixture(
                    n_components=len(centers_init) + 1,
                    init_params="centers",
                    centers_init=centers_init.copy(),
                    station_locs=locs_,
                    phase_type=phase_type_,
                    phase_weight=phase_weight_,
                    vel=vel,
                    loss_type="l1",
                    bounds=bounds,
                    # max_covar=20 ** 2,
                    dummy_comp=True,
                    dummy_prob=1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-1 / 2),
                    dummy_quantile=0.1,
                ).fit(data_)
            else:
                raise (f"Unknown method {method}; Should be 'BGMM' or 'GMM'")

            ## run prediction
            pred = gmm.predict(data_)
            prob = np.exp(gmm.score_samples(data_))
            prob_matrix = gmm.predict_proba(data_)
            prob_eq = prob_matrix.sum(axis=0)
            #  prob = prob_matrix[range(len(data_)), pred]
            #  score = gmm.score(data_)
            #  score_sample = gmm.score_samples(data_)

            ## filtering
            for i in range(len(centers_init)):
                tmp_data = data_[pred == i]
                tmp_locs = locs_[pred == i]
                tmp_pick_station_id = pick_station_id_[pred == i]
                tmp_phase_type = phase_type_[pred == i]
                if (len(tmp_data) == 0) or (len(tmp_data) < gpars.min_picks_per_eq):
                    continue

                ## filter by time
                t_ = calc_time(gmm.centers_[i : i + 1, : len(gpars.dims) + 1], tmp_locs, tmp_phase_type, vel=vel)
                diff_t = np.abs(t_ - tmp_data[:, 0:1])
                idx_t = (diff_t < gpars.max_sigma11).squeeze(axis=1)
                idx_filter = idx_t
                if len(tmp_data[idx_filter]) <gpars.min_picks_per_eq:
                    continue

                ## filter multiple picks at the same station
                unique_sta_id = {}
                for j, k in enumerate(tmp_pick_station_id):
                    if (k not in unique_sta_id) or (diff_t[j] < unique_sta_id[k][1]):
                        unique_sta_id[k] = (j, diff_t[j])
                idx_s = np.zeros(len(idx_t)).astype(bool)  ## based on station
                for k in unique_sta_id:
                    idx_s[unique_sta_id[k][0]] = True
                idx_filter = idx_filter & idx_s
                if len(tmp_data[idx_filter]) < gpars.min_picks_per_eq:
                    continue
                gmm.covariances_[i, 0, 0] = np.mean((diff_t[idx_t]) ** 2)

                if len(tmp_data[idx_filter & (tmp_phase_type == "p")]) < gpars.min_p_picks_per_eq:
                    continue
                if len(tmp_data[idx_filter & (tmp_phase_type == "s")]) < gpars.min_s_picks_per_eq:
                    continue

                event = {
                    # "time": from_seconds(gmm.centers_[i, len(config["dims"])]),
                    "time": datetime.datetime.fromtimestamp(gmm.centers_[i, len(gpars.dims)] + timestamp0).isoformat(
                        timespec="milliseconds"
                    ),
                    # "time(s)": gmm.centers_[i, len(config["dims"])],
                    "magnitude": gmm.centers_[i, len(gpars.dims) + 1] if gpars.useamp else 999,
                    "sigma_time": np.sqrt(gmm.covariances_[i, 0, 0]),
                    "sigma_amp": np.sqrt(gmm.covariances_[i, 1, 1]) if gpars.useamp else 0,
                    "cov_time_amp": gmm.covariances_[i, 0, 1] if gpars.useamp else 0,
                    "gamma_score": prob_eq[i],
                    "event_index": event_idx,
                }
                for j, k in enumerate(gpars.dims):  ## add location
                    event[k] = gmm.centers_[i, j]
                phases = []
                for pi, pr in zip(pick_idx_[pred == i][idx_filter], prob):
                    ptime = basetime + datetime.timedelta(seconds=data[pi])
                    sname = pick_station_id[pi]
                    pname = phase_type[pi]
                    phases.append([ptime, sname, pname])
                    assignment.append((pi, event_idx, pr))
                events.append([event, phases])
                event_idx += 1
        oqueue.put([[events, assignment], logt])

def toutput(oqueue, data_tool, outname):
    file_ = open(outname, "w", encoding="utf-8")
    logpath = outname+".log" 
    file_log = open(logpath, "a", encoding="utf-8")
    while True:
        [events, assiment], logt = oqueue.get()
        for eve, phase in events:
            x, y, z = eve["x(km)"], eve["y(km)"], eve["z(km)"]
            lon, lat =  data_tool.proj(x, y, inverse=True)
            tstr = eve["time"].strptime("%Y-%m-%d %H:%M:%S.%f")
            file_.write(f"#EVENT,123456,{tstr},{lon:.4f},{lat:.4f},{z:.3f},{x:3f},{y:.3f}\n")
            for pha in phase:
                ptime, sname, pname = pha 
                tstr = ptime.strptime("%Y-%m-%d %H:%M:%S.%f")
                file_.write(f"{pname},{tstr},{sname}\n")
        file_.flush()
        file_log.write(f"{logt}\n")    
        file_log.flush()    
def main(args):
    data_tool = DataLPPN(args.input, args.station)
    L = int((data_tool.maxtm-data_tool.mintm) / gpars.win_stirde) + 1
    logpath = args.output+".log" 
    maxt = 0 
    if os.path.exists(logpath):
        seqs = [-1]
        with open(logpath, "r") as f:
            for line in f.readlines():
                seqs.append(float(line.strip()))
        maxt = np.max(seqs)
    iqueue = multiprocessing.Queue(maxsize=1)
    oqueue = multiprocessing.Queue(maxsize=100)
    t = multiprocessing.Process(target=toutput, args=(oqueue, data_tool, args.output))
    t.start()
    for i in range(2):
        t = multiprocessing.Process(target=associationt_mt, args=(iqueue, oqueue, data_tool))
        t.start()
    for t in range(L):
        if t < maxt:continue 
        data = data_tool.getdata()
        if len(data)==0:continue 
        print(f"第{t}天")
        iqueue.put([data, t])

        
import argparse
if __name__ == "__main__":
    # 处理输出某个时间之后的所有震相
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-i', '--input', default="odata/X2_2015_szy.txt", help="拾取震相文件")       
    parser.add_argument('-o', '--output', default="odata/x2.gamma.txt", help="输出文件名")   
    parser.add_argument('-s', '--station', default="odata/x2.pos", help="台站位置信息") 
    parser.add_argument('-d', '--device', default="cpu", help="运算设备") 
    args = parser.parse_args()   
    main(args)
