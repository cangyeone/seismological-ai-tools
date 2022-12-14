import os 
import obspy 
import pickle 
import datetime 
import h5py 
import numpy as np 
from obspy.clients.fdsn.header import FDSNNoDataException
#from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import time 
from utils.fetchdata import OBSPYData
from utils.dbdata import Client 
class WriteH5():
    def __init__(self, n_length=6144, area="AH", root="/data/arrayData/dataX1/X1DATA/dbX1MSEEDV2016FINAL/X1", tablebase="tables"):
        self.root = "/data/data4AI_20200223/NFS_CN_DATA/POS.CHINA.STATION.V20200605"  
        #self.phase_file = "/data/data4AI_20200223/NEW_V20200608/phase_ctlgevid_NSZONEV20200708/GMT.MICROSECOND.NSZONE.M2.ALL.TYPE.PHASERPT.YEAR.09TO19.V20200707.txt"
        #self.phase_file = "whatWEneed/GMT.MICROSECOND.NSZONE.M2.ALL.TYPE.PHASERPT.YEAR.09TO19.V20200707.txt"
        #self.phase_file = "/data/manyproject/aiHUABEI/eventCTLGPHA_AIHUABEIV20210114/phase4ai.HUABEI.108.127.33.45.ALLMAG.EQONLY.Y09TO19.V20210114"
        #self.phase_file = "/data/manyproject/aiHUABEI/eventCTLGPHA_AIHUABEIV20210114/phase4ai.HUABEI.108.127.33.45.ALLMAG.EQONLY.Y09TO19.V20210114"
        self.phase_file = "/data/manyprojects/teleseismicPickCNV20210807/teleCN/pickphase.TELESEIS.CN.M5.2KTO1WKM.V20210210"
        self.cn_loc_file = "data/POS.CHINA.STATION.V20200605"
        self.range = 500
        self.get_cn_loc()
        self.process_phase_file()
        self.clint_dict = {}
        self.year = True 
        for itr in range(13):
            self.clint_dict[f"{2007+itr}"] = Client(f"/data/waveindexDATA/mseedIndexDB/mseedidxdb.CNNET.BHSH.100HZ_{2007+itr}.db.sqlite3", datapath_replace=["^", "/data/CSNDATA/"])
        #self.clint_cn = Client("/home/yuzy/makeh5/models/h5test/mseedidxdb.CNNET.BHSH.100HZ_merge.db.sqlite3", datapath_replace=["^", "/data/data4AI_20200223/NFS_CN_DATA/"])
    def get_cn_loc(self):
        self.loc_dict = {}
        with open(self.cn_loc_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sline = [itr for itr in line.split(" ") if len(itr)>0]
                key = ".".join(sline[:3]) 
                self.loc_dict[key] = {"lon":float(sline[3]), "lat":float(sline[4]), "elevation":float(sline[5])}

    def process_phase_file(self): 
        """
        ??????????????????
        ??????????????????
        """
        if False:#os.path.exists("models/sdata.pkl"):
            with open("models/sdata.pkl", "rb") as f:
                self.file_data = pickle.load(f)   
                return          
        with open(self.phase_file, "r", encoding="utf-8") as f:
            file_data = f.read() 
        file_data_s = file_data.split("#") 
        self.file_data = [itr.split("\n") for itr in file_data_s] 
        #with open("models/sdata.pkl", "wb") as f:
        #    pickle.dump(self.file_data, f)
    def get_event(self, data):
        """
        ??????????????????????????????
        data:????????????????????????
        retrun:[??????????????????, ??????????????????]
        """
        head = data[0] # ?????????????????????#??????????????????
        body = data[1:] # ????????????????????????PHASE??????????????????
        def sline(line):
            """????????????????????????????????????re????????????????????????"""
            s = [i for i in line.split(" ") if len(i)>0] 
            return s 
        head = sline(head) 
        body = [sline(itr) for itr in body] 

        if head[15] == "NONE":head[15]='-1'
        event = { # ??????????????????????????????????????????????????????
            "evid": head[1], 
            "time": datetime.datetime.strptime(f"{head[3]}/{head[4]}/{head[5]} {head[7]}:{head[8]}:{head[9]}.{head[10]}", "%Y/%m/%d %H:%M:%S.%f"), 
            "lat": float(head[13]), 
            "lon": float(head[12]), 
            "mag": float(head[18]), 
            "depth": float(head[15]), 
            "type": head[2], 
            "strs":",".join(head)
        }
        phases = [] 
        pgdict = []
        for p in body:
            if len(p)<10:continue 
            if "YYYY" == p[12]:
                ptime = datetime.datetime.strptime(f"{head[3]}/{head[4]}/{head[5]} {head[7]}:{head[8]}:{head[9]}.{head[10]}", "%Y/%m/%d %H:%M:%S.%f")
            else:
                ptime = datetime.datetime.strptime(f"{p[12]}/{p[13]}/{p[14]} {p[16]}:{p[17]}:{p[18]}.{p[19]}", "%Y/%m/%d %H:%M:%S.%f")
            if "NONE" == p[24]:p[24]="-123" 
            if "NONE" == p[25]:p[25]="-123"
            # TRAVTIME RMS WT 
            # AMPLITUD AMP PERIOD 
            # POLARITY UPDOWN CLARITY
            name1 = "name1" 
            name2 = "name2"
            if p[3] == "TRAVTIME":
                name1 = "RMS"
                name2 = "WT" 
            if p[3] == "AMPLITUD":
                name1 = "AMP"
                name2 = "PERIOD" 
            if p[3] == "POLARITY":
                name1 = "UPDOWN"
                name2 = "CLARITY"     
            pname = p[8]
            if p[8] == "Pg":
                pname = f"{p[3]}.{p[8]}"
            #skey = f"{p[4]}.{p[5]}.{p[6]}"
            #if p[8]=="Pg" and skey in pgdict:
            #    pname = f"{p[8]}.{p[3]}"
            #elif p[8]=="Pg" and skey not in pgdict:
            #    pname = 
            #    pgdict[skey]
            #PHASE  SC.200901311837.0001 eq TRAVTIME SC   JJS 00 BHZ     Pg ttime      14.21  RECORDTYPE 2009 01 31 031 10 37 15 940000 MAGSOURCE   MAGTYPE  MAG DISTAZ     79.9  111.5 LOC_RMSWT    RMS      WT
            pdic = { # ?????????????????????????????????????????????
                    "type": pname, 
                    "time": ptime,
                    "dist": float(p[24]),
                    name1:p[27], # ?????????????????????
                    name2:p[28], # ????????????????????????
                    "stid":f"{p[4]}.{p[5]}.{p[6]}", 
                    "phasetype": p[3], 
                    "recordtype": p[11], 
                    "magsource": p[20], 
                    "magtype": p[21], 
                    "mag": p[22], 
                    "distaz": p[24], 
                    "strs": ",".join(p)
                }
            phases.append(pdic) 
        return [event, phases]
    def getdata(self, pinfo, maxdelta=0):  
        """
        ?????????????????????????????????????????????
        maxdelta:??????????????????????????????????????????????????????????????????????????????????????????
        pinfo:?????????????????????????????????
        source:
        ??????:???????????????
        """
        times = pinfo["time"]
        stids = pinfo["stid"].split(".")
        stid = ".".join(stids[:2])
        time1 = times - datetime.timedelta(seconds=self.range) # ??????????????????
        time2 = times + datetime.timedelta(seconds=self.range+maxdelta) # ??????????????????
        selfile = []
        seltime = []
        for itr in range(3):
            # ???????????????????????????????????????????????????
            reftime = time1 + datetime.timedelta(days=itr-2) 
            base_dir = os.path.join(self.root, f"BKC100HZ_{reftime.strftime('%y')}", reftime.strftime("%Y%m%d"), stids[0]) 
            if os.path.exists(base_dir)==False:
                continue 
            file_name = os.listdir(base_dir) 
            for f in file_name:
                if stid in f:
                    selfile.append(os.path.join(base_dir, f)) 
                    tstr = f.split(".")[2]# 2009051160003
                    seltime.append(datetime.datetime.strptime(tstr, "%Y%j%H%M%S")) 
        if len(selfile)==0:return []
        getdata = {}
        deltas = {}
        for f, t in zip(selfile, seltime):
            delta1 = (time1 - t).total_seconds()
            delta2 = (time2 - t).total_seconds()
            if delta1 > 0 and delta1<86400 and delta2>0 and delta2<86400: 
                # ??????????????????????????????????????????????????????????????????????????????
                trace = obspy.read(f)[0] 
                infos = trace.stats 
                #print(str(infos.starttime))
                start = datetime.datetime.strptime(str(infos.starttime), "%Y-%m-%dT%H:%M:%S.%fZ") 
                delta = infos.delta 
                channel = infos.channel 
                datas = trace.data 
                d1 = int((time1 - start).total_seconds() / delta)
                d2 = int((time2 - start).total_seconds() / delta)
                if channel in getdata: 
                    getdata[channel].append(datas[d1:d2])
                else:
                    deltas[channel] = [delta, time1] 
                    getdata[channel] = [datas[d1:d2]]
            elif (delta1 > 0 and delta1<86400) and not (delta2>0 and delta2<86400):
                # ??????????????????????????????????????????????????????????????????????????????
                trace = obspy.read(f)[0] 
                infos = trace.stats 
                start = datetime.datetime.strptime(str(infos.starttime), "%Y-%m-%dT%H:%M:%S.%fZ") 
                delta = infos.delta 
                channel = infos.channel 
                datas = trace.data 
                d1 = int((time1 - start).total_seconds() / delta)
                d2 = int((time2 - start).total_seconds() / delta)
                if channel in getdata: 
                    getdata[channel].append(datas[d1:])
                else:
                    getdata[channel] = [datas[d1:]]
                    deltas[channel] = [delta, time1] 
            elif not (delta1 > 0 and delta1<86400) and (delta2>0 and delta2<86400):   
                # ??????????????????????????????????????????????????????????????????????????????
                trace = obspy.read(f)[0] 
                infos = trace.stats 
                start = datetime.datetime.strptime(str(infos.starttime), "%Y-%m-%dT%H:%M:%S.%fZ") 
                delta = infos.delta 
                channel = infos.channel 
                datas = trace.data 
                d1 = int((time1 - start).total_seconds() / delta)
                d2 = int((time2 - start).total_seconds() / delta)
                if channel in getdata: 
                    getdata[channel].append(datas[:d2])
                else:
                    #deltas[channel] = [delta, time1] 
                    getdata[channel] = [datas[:d2]] 
        for key in getdata:
            getdata[key] = np.concatenate(getdata[key]) 
        infos = {
            "stid":stid, # ????????????
            "data":getdata, 
            "delta":deltas # ???????????????????????????????????????
        }
        return infos 
    def getdata_from_net(self, pinfo, maxdelta):
        times = pinfo["time"]
        stids = pinfo["stid"].split(".")
        stid = ".".join(stids[:2])
        #print(maxdelta)
        maxdelta = np.max(maxdelta, 0)
        time1 = times - datetime.timedelta(seconds=self.range) # ??????????????????
        time2 = times + datetime.timedelta(seconds=self.range+maxdelta) # ??????????????????
        t1 = UTCDateTime(time1.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
        t2 = UTCDateTime(time2.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
        cha = "?H?"
        #query = (stids[0], stids[1], stids[2], cha, t1, t2)
        if self.year:
            #print(time1.strftime["%Y"], type(self.clint_dict))
            #clint = self.clint_dict[time1.strftime["%Y"]]
            #print(clint)
            st = self.clint_dict[time1.strftime("%Y")].get_waveforms(stids[0], stids[1], stids[2], cha, t1, t2)
        else:
            st = self.clint_cn.get_waveforms(stids[0], stids[1], stids[2], cha, t1, t2)
        #st = self.clint_cn.get_waveforms_bulk(query)
        getdata = {}
        stats = {}
        for s in st:
            infos = s.stats
            key = infos.channel
            start = datetime.datetime.strptime(str(infos.starttime), "%Y-%m-%dT%H:%M:%S.%fZ") 
            getdata[key] = s.data 
            stats[key] = [infos.delta, start] 
            #print(s.stats)
        infos = {
            "stid":stid, # ????????????
            "data":getdata, 
            "delta":stats # ???????????????????????????????????????
        }
        return infos          
    def outputh5(self, year="all"):
        h5file = h5py.File(f"data/tele.h5", "w") 
        logfile = open(f"data/tele.log", "w", encoding="utf-8")
        count = 0
        start_time = time.perf_counter()
        for eve in self.file_data[1:]:
            processed_time1 = time.perf_counter()
            evinfo = self.get_event(eve) 
            event, phases = evinfo 
            evtime = event["time"] 
            #if np.abs((evtime-datetime.datetime.strptime("2019/01/01 00:00:00.000000", "%Y/%m/%d %H:%M:%S.%f")).total_seconds()) > 86400 * 10:continue 
            if event["evid"] in h5file:continue 
            group = h5file.create_group(event["evid"])
            group.attrs["time"] = evtime.strftime("%Y/%m/%d %H:%M:%S.%f") 
            group.attrs["lat"] = event["lat"] 
            group.attrs["lon"] = event["lon"] 
            group.attrs["mag"] = event["mag"] 
            group.attrs["depth"] = event["depth"] 
            group.attrs["type"] = event["type"] 
            group.attrs["strs"] = event["strs"]
            for phase in phases:
                stids = phase["stid"].split(".")
                stid = ".".join(stids[:3])                
                if stid in group: # ?????????????????????????????????????????????????????????
                    subgroup = group[stid]
                    subgroup.attrs[phase["type"]] = phase["time"].strftime("%Y/%m/%d %H:%M:%S.%f")  
                    for pkey in phase:
                        if "time" == pkey or "stid" == pkey:continue 
                        #if "Pg" not in phase:
                        #    if "updown" in pkey:
                        #        continue 
                        subgroup.attrs[f"{phase['type']}.{pkey}"] = phase[pkey]
                    continue 
                else:  # ????????????????????????????????????????????????????????????????????????????????????????????????????????????
                    reftime = phase["time"]  #############################????????????
                    relative_time = []
                    alltype = []
                    for subphase in phases:
                        if subphase["stid"] == phase["stid"]:
                            alltype.append(subphase["type"])
                            relative_time.append((subphase["time"]-reftime).total_seconds()) 
                    delta = np.max(relative_time) - np.min(relative_time)
                    delta = np.max([delta, 10]) # ??????10???????????????10???????????????
                    delta = np.min([delta, 1000])
                    subgroup = group.create_group(stid) 
                    subgroup.attrs["types"] = ",".join(alltype)
                    for pkey in phase:
                        if "time" == pkey or "stid" == pkey:continue 
                        #if "Pg" not in phase:
                        #    if "updown" in pkey:
                        #        continue 
                        subgroup.attrs[f"{phase['type']}.{pkey}"] = phase[pkey]
                    if stid in self.loc_dict:
                        loc = self.loc_dict[stid] 
                        for key in loc:
                            subgroup.attrs[key] = loc[key]
                    subgroup.attrs[phase["type"]] = phase["time"].strftime("%Y/%m/%d %H:%M:%S.%f")  
                pinfo = self.getdata_from_net(phase, delta) 
                if len(pinfo)==0:
                    continue 
                data = pinfo["data"]
                for key in data:
                    #continue
                    dataset = subgroup.create_dataset(key, data=data[key].astype(np.int32), dtype="i4", chunks=True, compression="gzip")
                    dataset.attrs["delta"] = pinfo["delta"][key][0]
                    dataset.attrs["btime"] = pinfo["delta"][key][1].strftime("%Y/%m/%d %H:%M:%S.%f") 
                #print(f"{stid} processed!")
            count += 1
            #if count > 5: # ????????????????????????????????????????????????
            #    break
            processed_time2 = time.perf_counter()
            print(f"{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds")
            logfile.write(f"{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds\n")
            logfile.flush()
    
import multiprocessing 
class WriteH5MT():
    def __init__(self, n_length=6144, area="AH", root="/data/arrayData/dataX1/X1DATA/dbX1MSEEDV2016FINAL/X1", tablebase="tables"):
        self.phase_file = "/data/manyprojects/teleseismicPickCNV20210807/teleCN/pickphase.TELESEIS.CN.M5.2KTO1WKM.V20210210"
        self.cn_loc_file = "data/POS.CHINA.STATION.V20200605"
        self.range = 500
        self.get_cn_loc()
        self.process_phase_file()
        #self.clint_cn = Client("http://10.2.28.155:38000")
        self.clint_dict = {}
        self.year = True 
        #feedq = []
        self.feedq = {}
        for itr in range(13):
            self.clint_dict[f"{2007+itr}"] = Client(f"/data/waveindexDATA/mseedIndexDB/mseedidxdb.CNNET.BHSH.100HZ_{2007+itr}.db.sqlite3", datapath_replace=["^", "/data/CSNDATA/"])
            self.feedq[f"{2007+itr}"] = multiprocessing.Queue(100)
        dataq = multiprocessing.Queue(100)
        for key in self.feedq:
            multiprocessing.Process(target=self.feed_thread, args=(key, self.feedq[key])).start()
            multiprocessing.Process(target=self.get_event, args=(self.feedq[key], dataq, self.clint_dict[key])).start()
        multiprocessing.Process(target=self.outputh5, args=(dataq, )).start()
    
    def get_cn_loc(self):
        self.loc_dict = {}
        with open(self.cn_loc_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sline = [itr for itr in line.split(" ") if len(itr)>0]
                key = ".".join(sline[:3]) 
                self.loc_dict[key] = {"lon":float(sline[3]), "lat":float(sline[4]), "elevation":float(sline[5])}

    def process_phase_file(self): 
        """
        ??????????????????
        ??????????????????
        """
        if False:#os.path.exists("models/sdata.pkl"):
            with open("models/sdata.pkl", "rb") as f:
                self.file_data = pickle.load(f)   
                return          
        with open(self.phase_file, "r", encoding="utf-8") as f:
            file_data = f.read() 
        file_data_s = file_data.split("#") 
        self.file_data = [itr.split("\n") for itr in file_data_s] 
        #with open("models/sdata.pkl", "wb") as f:
        #    pickle.dump(self.file_data, f)
    def get_event(self, queue1, queue2, clint):
        """
        ??????????????????????????????
        data:????????????????????????
        retrun:[??????????????????, ??????????????????]
        """
        while True:
            data = queue1.get()
            if data == None:
                break
            head = data[0] # ?????????????????????#??????????????????
            body = data[1:] # ????????????????????????PHASE??????????????????
            def sline(line):
                """????????????????????????????????????re????????????????????????"""
                s = [i for i in line.split(" ") if len(i)>0] 
                return s 
            head = sline(head) 
            body = [sline(itr) for itr in body] 

            if head[15] == "NONE":head[15]='-1'
            event = { # ??????????????????????????????????????????????????????
                "evid": head[1], 
                "time": datetime.datetime.strptime(f"{head[3]}/{head[4]}/{head[5]} {head[7]}:{head[8]}:{head[9]}.{head[10]}", "%Y/%m/%d %H:%M:%S.%f"), 
                "lat": float(head[13]), 
                "lon": float(head[12]), 
                "mag": float(head[18]), 
                "depth": float(head[15]), 
                "type": head[2], 
                "strs":",".join(head)
            }
            phases = [] 
            pgdict = []
            for p in body:
                if len(p)<10:continue 
                if "YYYY" == p[12]:
                    ptime = datetime.datetime.strptime(f"{head[3]}/{head[4]}/{head[5]} {head[7]}:{head[8]}:{head[9]}.{head[10]}", "%Y/%m/%d %H:%M:%S.%f")
                else:
                    ptime = datetime.datetime.strptime(f"{p[12]}/{p[13]}/{p[14]} {p[16]}:{p[17]}:{p[18]}.{p[19]}", "%Y/%m/%d %H:%M:%S.%f")
                if "NONE" == p[24]:p[24]="-123" 
                if "NONE" == p[25]:p[25]="-123"
                # TRAVTIME RMS WT 
                # AMPLITUD AMP PERIOD 
                # POLARITY UPDOWN CLARITY
                name1 = "name1" 
                name2 = "name2"
                if p[3] == "TRAVTIME":
                    name1 = "RMS"
                    name2 = "WT" 
                if p[3] == "AMPLITUD":
                    name1 = "AMP"
                    name2 = "PERIOD" 
                if p[3] == "POLARITY":
                    name1 = "UPDOWN"
                    name2 = "CLARITY"     
                pname = p[8]
                if p[8] == "Pg":
                    pname = f"{p[3]}.{p[8]}"
                pdic = { # ?????????????????????????????????????????????
                        "type": pname, 
                        "time": ptime,
                        "dist": float(p[24]),
                        name1:p[27], # ?????????????????????
                        name2:p[28], # ????????????????????????
                        "stid":f"{p[4]}.{p[5]}.{p[6]}", 
                        "phasetype": p[3], 
                        "recordtype": p[11], 
                        "magsource": p[20], 
                        "magtype": p[21], 
                        "mag": p[22], 
                        "distaz": p[24], 
                        "strs": ",".join(p)
                    }
                phases.append(pdic) 
            phase_delta = {}
            
            for subphase in phases:
                if subphase["stid"] in phase_delta:
                    phase_delta[subphase["stid"]].append((subphase["time"]-event["time"]).total_seconds())
                else:
                    phase_delta[subphase["stid"]] = [(subphase["time"]-event["time"]).total_seconds()] 
            data_info = {}
            for pkey in phase_delta:
                stids = pkey.split(".")
                time1 = event["time"] + datetime.timedelta(seconds=np.min(phase_delta[pkey])-self.range) # ??????????????????
                time2 = event["time"] + datetime.timedelta(seconds=np.max(phase_delta[pkey])+self.range) # ??????????????????
                t1 = UTCDateTime(time1.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
                t2 = UTCDateTime(time2.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
                cha = "?H?"
                st = clint.get_waveforms(stids[0], stids[1], stids[2], cha, t1, t2)
                #st = self.clint_cn.get_waveforms_bulk(query)
                getdata = {}
                stats = {}
                for s in st:
                    infos = s.stats
                    key = infos.channel
                    start = datetime.datetime.strptime(str(infos.starttime), "%Y-%m-%dT%H:%M:%S.%fZ") 
                    getdata[key] = s.data 
                    stats[key] = [infos.delta, start] 
                    #print(s.stats)
                infos = {
                    "stid":pkey, # ????????????
                    "data":getdata, 
                    "delta":stats # ???????????????????????????????????????
                }
                data_info[pkey] = infos
            #print([key for key in data_info])
            queue2.put([event, phases, data_info])
   
    def feed_thread(self, year, feedq):
        def sline(line):
            """????????????????????????????????????re????????????????????????"""
            s = [i for i in line.split(" ") if len(i)>0] 
            return s 
        for eve in self.file_data[1:]:
            head = eve[0] # ?????????????????????#??????????????????
            head = sline(head) 
            if head[3]==year:
                feedq.put(eve)
    def outputh5(self, outq):
        h5file = h5py.File(f"data/tele-mt.h5", "w") 
        logfile = open(f"data/tele-mt.log", "w", encoding="utf-8")
        count = 0
        start_time = time.perf_counter()
        while True:
            processed_time1 = time.perf_counter()
            evinfo = outq.get()
            event, phases, datainfo = evinfo 
            evtime = event["time"] 
            #if np.abs((evtime-datetime.datetime.strptime("2019/01/01 00:00:00.000000", "%Y/%m/%d %H:%M:%S.%f")).total_seconds()) > 86400 * 10:continue 
            if event["evid"] in h5file:continue 
            group = h5file.create_group(event["evid"])
            group.attrs["time"] = evtime.strftime("%Y/%m/%d %H:%M:%S.%f") 
            group.attrs["lat"] = event["lat"] 
            group.attrs["lon"] = event["lon"] 
            group.attrs["mag"] = event["mag"] 
            group.attrs["depth"] = event["depth"] 
            group.attrs["type"] = event["type"] 
            group.attrs["strs"] = event["strs"]
            for phase in phases:
                stids = phase["stid"].split(".")
                stid = phase["stid"]               
                if stid in group: # ?????????????????????????????????????????????????????????
                    subgroup = group[stid]
                    subgroup.attrs[phase["type"]] = phase["time"].strftime("%Y/%m/%d %H:%M:%S.%f")  
                    for pkey in phase:
                        if "time" == pkey or "stid" == pkey:continue 
                        #if "Pg" not in phase:
                        #    if "updown" in pkey:
                        #        continue 
                        subgroup.attrs[f"{phase['type']}.{pkey}"] = phase[pkey]
                    continue 
                else:  # ????????????????????????????????????????????????????????????????????????????????????????????????????????????
                    reftime = phase["time"]  #############################????????????
                    relative_time = []
                    alltype = []
                    for subphase in phases:
                        if subphase["stid"] == phase["stid"]:
                            alltype.append(subphase["type"])
                            relative_time.append((subphase["time"]-reftime).total_seconds()) 
                    delta = np.max(relative_time) - np.min(relative_time)
                    delta = np.max([delta, 10]) # ??????10???????????????10???????????????
                    delta = np.min([delta, 1000])
                    subgroup = group.create_group(stid) 
                    subgroup.attrs["types"] = ",".join(alltype)
                    for pkey in phase:
                        if "time" == pkey or "stid" == pkey:continue 
                        #if "Pg" not in phase:
                        #    if "updown" in pkey:
                        #        continue 
                        subgroup.attrs[f"{phase['type']}.{pkey}"] = phase[pkey]
                    if stid in self.loc_dict:
                        loc = self.loc_dict[stid] 
                        for key in loc:
                            subgroup.attrs[key] = loc[key]
                    subgroup.attrs[phase["type"]] = phase["time"].strftime("%Y/%m/%d %H:%M:%S.%f")  
                #if stid in datainfo:
                #print(datainfo)
                pinfo = datainfo[stid]
                #else:
                #    continue
                data = pinfo["data"]
                for key in data:
                    #continue
                    dataset = subgroup.create_dataset(key, data=data[key].astype(np.int32), dtype="i4", chunks=True, compression="gzip")
                    dataset.attrs["delta"] = pinfo["delta"][key][0]
                    dataset.attrs["btime"] = pinfo["delta"][key][1].strftime("%Y/%m/%d %H:%M:%S.%f") 
                #print(f"{stid} processed!")
            count += 1
            #if count > 5: # ????????????????????????????????????????????????
            #    break
            processed_time2 = time.perf_counter()
            print(f"{event['evid']},{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds")
            logfile.write(f"{event['evid']},{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds\n")
            logfile.flush()

class WriteH5MT2():
    def __init__(self, n_length=6144, area="AH", root="/data/arrayData/dataX1/X1DATA/dbX1MSEEDV2016FINAL/X1", tablebase="tables"):
        self.phase_file = "/data/manyprojects/teleseismicPickCNV20210807/teleCN/pickphase.TELESEIS.CN.M5.2KTO1WKM.V20210210"
        self.cn_loc_file = "data/POS.CHINA.STATION.V20200605"
        self.range = 500
        self.get_cn_loc()
        self.process_phase_file()
        #self.clint_cn = Client("http://10.2.28.155:38000")
        self.clint_dict = {}
        self.year = True 
        #feedq = []
        self.feedq = {}

        dataq = multiprocessing.Queue(100)
        feedq = multiprocessing.Queue(100)
        nthread = 15
        barrier = multiprocessing.Barrier(nthread+1)
        for key in range(nthread):
            multiprocessing.Process(target=self.feed_thread, args=(feedq, nthread, dataq, barrier)).start()
            multiprocessing.Process(target=self.get_event, args=(feedq, dataq, barrier)).start()
        multiprocessing.Process(target=self.outputh5, args=(dataq, )).start()
    
    def get_cn_loc(self):
        self.loc_dict = {}
        with open(self.cn_loc_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sline = [itr for itr in line.split(" ") if len(itr)>0]
                key = ".".join(sline[:3]) 
                self.loc_dict[key] = {"lon":float(sline[3]), "lat":float(sline[4]), "elevation":float(sline[5])}

    def process_phase_file(self): 
        """
        ??????????????????
        ??????????????????
        """
        if False:#os.path.exists("models/sdata.pkl"):
            with open("models/sdata.pkl", "rb") as f:
                self.file_data = pickle.load(f)   
                return          
        with open(self.phase_file, "r", encoding="utf-8") as f:
            file_data = f.read() 
        file_data_s = file_data.split("#") 
        self.file_data = [itr.split("\n") for itr in file_data_s] 
        #with open("models/sdata.pkl", "wb") as f:
        #    pickle.dump(self.file_data, f)
    def get_event(self, queue1, queue2, barrier):
        """
        ??????????????????????????????
        data:????????????????????????
        retrun:[??????????????????, ??????????????????]
        """
        client = {}
        for itr in range(13):
            client[f"{2007+itr}"] = Client(f"/data/waveindexDATA/mseedIndexDB/mseedidxdb.CNNET.BHSH.100HZ_{2007+itr}.db.sqlite3", datapath_replace=["^", "/data/CSNDATA/"])
        while True:
            data = queue1.get()
            if len(data) == 0:
                barrier.wait()
                break
            head = data[0] # ?????????????????????#??????????????????
            body = data[1:] # ????????????????????????PHASE??????????????????
            def sline(line):
                """????????????????????????????????????re????????????????????????"""
                s = [i for i in line.split(" ") if len(i)>0] 
                return s 
            head = sline(head) 
            body = [sline(itr) for itr in body] 

            if head[15] == "NONE":head[15]='-1'
            event = { # ??????????????????????????????????????????????????????
                "evid": head[1], 
                "time": datetime.datetime.strptime(f"{head[3]}/{head[4]}/{head[5]} {head[7]}:{head[8]}:{head[9]}.{head[10]}", "%Y/%m/%d %H:%M:%S.%f"), 
                "lat": float(head[13]), 
                "lon": float(head[12]), 
                "mag": float(head[18]), 
                "depth": float(head[15]), 
                "type": head[2], 
                "strs":",".join(head)
            }
            phases = [] 
            pgdict = []
            for p in body:
                if len(p)<10:continue 
                if "YYYY" == p[12]:
                    ptime = datetime.datetime.strptime(f"{head[3]}/{head[4]}/{head[5]} {head[7]}:{head[8]}:{head[9]}.{head[10]}", "%Y/%m/%d %H:%M:%S.%f")
                else:
                    ptime = datetime.datetime.strptime(f"{p[12]}/{p[13]}/{p[14]} {p[16]}:{p[17]}:{p[18]}.{p[19]}", "%Y/%m/%d %H:%M:%S.%f")
                if "NONE" == p[24]:p[24]="-123" 
                if "NONE" == p[25]:p[25]="-123"
                # TRAVTIME RMS WT 
                # AMPLITUD AMP PERIOD 
                # POLARITY UPDOWN CLARITY
                name1 = "name1" 
                name2 = "name2"
                if p[3] == "TRAVTIME":
                    name1 = "RMS"
                    name2 = "WT" 
                if p[3] == "AMPLITUD":
                    name1 = "AMP"
                    name2 = "PERIOD" 
                if p[3] == "POLARITY":
                    name1 = "UPDOWN"
                    name2 = "CLARITY"     
                pname = p[8]
                if p[8] == "Pg":
                    pname = f"{p[3]}.{p[8]}"
                pdic = { # ?????????????????????????????????????????????
                        "type": pname, 
                        "time": ptime,
                        "dist": float(p[24]),
                        name1:p[27], # ?????????????????????
                        name2:p[28], # ????????????????????????
                        "stid":f"{p[4]}.{p[5]}.{p[6]}", 
                        "phasetype": p[3], 
                        "recordtype": p[11], 
                        "magsource": p[20], 
                        "magtype": p[21], 
                        "mag": p[22], 
                        "distaz": p[24], 
                        "strs": ",".join(p)
                    }
                phases.append(pdic) 
            phase_delta = {}
            
            for subphase in phases:
                if subphase["stid"] in phase_delta:
                    phase_delta[subphase["stid"]].append((subphase["time"]-event["time"]).total_seconds())
                else:
                    phase_delta[subphase["stid"]] = [(subphase["time"]-event["time"]).total_seconds()] 
            data_info = {}
            for pkey in phase_delta:
                stids = pkey.split(".")
                time1 = event["time"] + datetime.timedelta(seconds=np.min(phase_delta[pkey])-self.range) # ??????????????????
                time2 = event["time"] + datetime.timedelta(seconds=np.max(phase_delta[pkey])+self.range) # ??????????????????
                t1 = UTCDateTime(time1.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
                t2 = UTCDateTime(time2.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
                cha = "?H?"
                st = client[event["time"].strftime("%Y")].get_waveforms(stids[0], stids[1], stids[2], cha, t1, t2)
                #st = clint.get_waveforms(stids[0], stids[1], stids[2], cha, t1, t2)
                #st = self.clint_cn.get_waveforms_bulk(query)
                getdata = {}
                stats = {}
                for s in st:
                    infos = s.stats
                    key = infos.channel
                    start = datetime.datetime.strptime(str(infos.starttime), "%Y-%m-%dT%H:%M:%S.%fZ") 
                    getdata[key] = s.data 
                    stats[key] = [infos.delta, start] 
                    #print(s.stats)
                infos = {
                    "stid":pkey, # ????????????
                    "data":getdata, 
                    "delta":stats # ???????????????????????????????????????
                }
                data_info[pkey] = infos
            #print([key for key in data_info])
            queue2.put([event, phases, data_info])
   
    def feed_thread(self, feedq, num, dataq, barrier):
        def sline(line):
            """????????????????????????????????????re????????????????????????"""
            s = [i for i in line.split(" ") if len(i)>0] 
            return s 
        for eve in self.file_data[1:]:
            head = eve[0] # ?????????????????????#??????????????????
            head = sline(head) 
            feedq.put(eve)
        for i in range(num):
            feedq.put([])
        barrier.wait() 
        dataq.put([])
    def outputh5(self, outq):
        h5file = h5py.File(f"data/tele-mt2.h5", "w") 
        logfile = open(f"data/tele-mt.log", "w", encoding="utf-8")
        count = 0
        start_time = time.perf_counter()
        while True:
            processed_time1 = time.perf_counter()
            evinfo = outq.get()
            if len(evinfo)==0:
                h5file.close() 
                break 
            event, phases, datainfo = evinfo 
            evtime = event["time"] 
            #if np.abs((evtime-datetime.datetime.strptime("2019/01/01 00:00:00.000000", "%Y/%m/%d %H:%M:%S.%f")).total_seconds()) > 86400 * 10:continue 
            if event["evid"] in h5file:continue 
            group = h5file.create_group(event["evid"])
            group.attrs["time"] = evtime.strftime("%Y/%m/%d %H:%M:%S.%f") 
            group.attrs["lat"] = event["lat"] 
            group.attrs["lon"] = event["lon"] 
            group.attrs["mag"] = event["mag"] 
            group.attrs["depth"] = event["depth"] 
            group.attrs["type"] = event["type"] 
            group.attrs["strs"] = event["strs"]
            for phase in phases:
                stids = phase["stid"].split(".")
                stid = phase["stid"]               
                if stid in group: # ?????????????????????????????????????????????????????????
                    subgroup = group[stid]
                    subgroup.attrs[phase["type"]] = phase["time"].strftime("%Y/%m/%d %H:%M:%S.%f")  
                    for pkey in phase:
                        if "time" == pkey or "stid" == pkey:continue 
                        #if "Pg" not in phase:
                        #    if "updown" in pkey:
                        #        continue 
                        subgroup.attrs[f"{phase['type']}.{pkey}"] = phase[pkey]
                    continue 
                else:  # ????????????????????????????????????????????????????????????????????????????????????????????????????????????
                    reftime = phase["time"]  #############################????????????
                    relative_time = []
                    alltype = []
                    for subphase in phases:
                        if subphase["stid"] == phase["stid"]:
                            alltype.append(subphase["type"])
                            relative_time.append((subphase["time"]-reftime).total_seconds()) 
                    delta = np.max(relative_time) - np.min(relative_time)
                    delta = np.max([delta, 10]) # ??????10???????????????10???????????????
                    delta = np.min([delta, 1000])
                    subgroup = group.create_group(stid) 
                    subgroup.attrs["types"] = ",".join(alltype)
                    for pkey in phase:
                        if "time" == pkey or "stid" == pkey:continue 
                        #if "Pg" not in phase:
                        #    if "updown" in pkey:
                        #        continue 
                        subgroup.attrs[f"{phase['type']}.{pkey}"] = phase[pkey]
                    if stid in self.loc_dict:
                        loc = self.loc_dict[stid] 
                        for key in loc:
                            subgroup.attrs[key] = loc[key]
                    subgroup.attrs[phase["type"]] = phase["time"].strftime("%Y/%m/%d %H:%M:%S.%f")  
                #if stid in datainfo:
                #print(datainfo)
                pinfo = datainfo[stid]
                #else:
                #    continue
                data = pinfo["data"]
                for key in data:
                    #continue
                    dataset = subgroup.create_dataset(key, data=data[key].astype(np.int32), dtype="i4", chunks=True, compression="gzip")
                    dataset.attrs["delta"] = pinfo["delta"][key][0]
                    dataset.attrs["btime"] = pinfo["delta"][key][1].strftime("%Y/%m/%d %H:%M:%S.%f") 
                #print(f"{stid} processed!")
            count += 1
            #if count > 5: # ????????????????????????????????????????????????
            #    break
            processed_time2 = time.perf_counter()
            print(f"{event['evid']},{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds")
            logfile.write(f"{event['evid']},{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds\n")
            logfile.flush()

#nohup python makeh5.py > data/mh5.log 2>&1 &
#1332677
if __name__=="__main__":
    makeh5 = WriteH5()
    makeh5.outputh5()
    #datatool = ReadH5("2019")
    #datatool.plot()
    #testh5 = TestH5() 
    #testh5.test()
    #PHASE  SC.200901311837.0001 eq TRAVTIME SC   JJS 00 BHZ     Pg ttime      14.21  RECORDTYPE 2009 01 31 031 10 37 15 940000 MAGSOURCE   MAGTYPE  MAG DISTAZ     79.9  111.5 LOC_RMSWT    RMS      WT
    # TRAVTIME RMS WT 
    # AMPLITUD AMP PERIOD 
    # POLARITY UPDOWN CLARITY
    # DISTAZ
    # distnace
    # azimuth
