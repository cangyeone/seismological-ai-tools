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
import multiprocessing 
class WriteH5():
    def __init__(self, args):

        self.root = args.root 
        #self.phase_file = "/data/data4AI_20200223/NEW_V20200608/phase_ctlgevid_NSZONEV20200708/GMT.MICROSECOND.NSZONE.M2.ALL.TYPE.PHASERPT.YEAR.09TO19.V20200707.txt"
        #self.phase_file = "whatWEneed/GMT.MICROSECOND.NSZONE.M2.ALL.TYPE.PHASERPT.YEAR.09TO19.V20200707.txt"
        #self.phase_file = "/data/manyproject/aiHUABEI/eventCTLGPHA_AIHUABEIV20210114/phase4ai.HUABEI.108.127.33.45.ALLMAG.EQONLY.Y09TO19.V20210114"
        self.phase_file = args.ctlg
        self.cn_loc_file = args.station 
        self.range = args.range 
        self.outfile = args.output
        self.get_cn_loc()
        self.process_phase_file()
        #self.clint_cn = Client("http://10.2.28.155:38000")
        self.clint_dict = {}
        self.year = True 
        #feedq = []
        self.feedq = {}
        self.client = Client(args.index)

        self.feedq = multiprocessing.Queue(100)
        dataq = multiprocessing.Queue(100)
        for key in range(args.nthread):
            multiprocessing.Process(target=self.feed_thread, args=(key, self.feedq)).start()
            multiprocessing.Process(target=self.get_event, args=(self.feedq, dataq, self.client)).start()
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
        处理震相文件
        仅做分割处理
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
        获取单个地震事件信息
        data:单个地震事件列表
        retrun:[地震事件信息, 地震震相信息]
        """
        while True:
            data = queue1.get()
            if data == None:
                break
            head = data[0] # 地震信息行，以#开头的那一行
            body = data[1:] # 震相信息列表，以PHASE开头的所有列
            def sline(line):
                """进行字符串分割，应当是用re的，但是太麻烦了"""
                s = [i for i in line.split(" ") if len(i)>0] 
                return s 
            head = sline(head) 
            body = [sline(itr) for itr in body] 

            if head[15] == "NONE":head[15]='-1'
            event = { # 头段信息，包括地震震级、位置等信息。
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
                pdic = { # 地震震相信息，可以添加更多信息
                        "type": pname, 
                        "time": ptime,
                        "dist": float(p[24]),
                        name1:p[27], # 这个是初动信息
                        name2:p[28], # 这个也是初动信息
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
                time1 = event["time"] + datetime.timedelta(seconds=np.min(phase_delta[pkey])-self.range) # 截取开始时间
                time2 = event["time"] + datetime.timedelta(seconds=np.max(phase_delta[pkey])+self.range) # 截取结束时间
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
                    "stid":pkey, # 台站信息
                    "data":getdata, 
                    "delta":stats # 包含数据开始时间，和采样率
                }
                data_info[pkey] = infos
            #print([key for key in data_info])
            queue2.put([event, phases, data_info])
   
    def feed_thread(self, year, feedq):
        def sline(line):
            """进行字符串分割，应当是用re的，但是太麻烦了"""
            s = [i for i in line.split(" ") if len(i)>0] 
            return s 
        for eve in self.file_data[1:]:
            head = eve[0] # 地震信息行，以#开头的那一行
            head = sline(head) 
            if head[3]==year:
                feedq.put(eve)
    def outputh5(self, outq):
        h5file = h5py.File(self.outfile, "w") 
        logfile = open(f"allmt.log", "w", encoding="utf-8")
        count = 0
        start_time = time.perf_counter()
        while True:
            processed_time1 = time.perf_counter()
            evinfo = outq.get()
            event, phases, datainfo = evinfo 
            evtime = event["time"] 
            #if np.abs((evtime-datetime.datetime.strptime("2019/01/01 00:00:00.000000", "%Y/%m/%d %H:%M:%S.%f")).total_seconds()) > 86400 * 10:continue 
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
                if stid in group: # 如果数据已经存在数据则直接附加震相信息
                    subgroup = group[stid]
                    subgroup.attrs[phase["type"]] = phase["time"].strftime("%Y/%m/%d %H:%M:%S.%f")  
                    for pkey in phase:
                        if "time" == pkey or "stid" == pkey:continue 
                        #if "Pg" not in phase:
                        #    if "updown" in pkey:
                        #        continue 
                        subgroup.attrs[f"{phase['type']}.{pkey}"] = phase[pkey]
                    continue 
                else:  # 如果数据不存在则进行读取，这里读取尽量应当包含所有震相，但是没有进行判断
                    reftime = phase["time"]  #############################需要检查
                    relative_time = []
                    alltype = []
                    for subphase in phases:
                        if subphase["stid"] == phase["stid"]:
                            alltype.append(subphase["type"])
                            relative_time.append((subphase["time"]-reftime).total_seconds()) 
                    delta = np.max(relative_time) - np.min(relative_time)
                    delta = np.max([delta, 10]) # 小于10秒都设置为10的震相间隔
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
            #if count > 5: # 测试几个震相，实际操作中需要删除
            #    break
            processed_time2 = time.perf_counter()
            print(f"{event['evid']},{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds")
            logfile.write(f"{event['evid']},{evtime}:{count} processed in {(processed_time2-processed_time1):.2f} seconds. Total {processed_time2-start_time:2f} seconds\n")
            logfile.flush()
import matplotlib.pyplot as plt 
plt.switch_backend("agg")
class ReadH5(): # 纯粹为了输出测试用的
    def __init__(self, year="2019"):
        self.h5file = h5py.File(f"models/h5test/{year}gzip4.h5", "r") 
    def plot(self):
        os.system("rm -rf tfig/*")

        for ekey in self.h5file:
            event = self.h5file[ekey] 
            print(f"EventID:{ekey}")
            #continue 
            for key in event.attrs:
                print(f"|-{key},{event.attrs[key]}")
            for skey in event:
                #plt.cla() 
                #plt.clf()
                station = event[skey] 
                print(f"|-StationID:{skey}")
                btime = False
                for key in station:
                    print(station[key][:].dtype)
                    print(f"  |-Data:{key},{station[key].attrs['btime']},{station[key].attrs['delta']},{len(station[key])}")
                    #if "Z" in key:
                    #    w = station[key][:] 
                    #    w = w.astype(np.float) 
                    #    w /= np.max(w) 
                    #    plt.plot(w, c="k", lw=1) 
                    btime = datetime.datetime.strptime(station[key].attrs['btime'], "%Y/%m/%d %H:%M:%S.%f") 
                
                for key in station.attrs:
                    print(f"  |-AttrName:{key},AttrValue:{station.attrs[key]}")
                    #if btime:
                    #    if "Pg" == key:
                    #        ptime = datetime.datetime.strptime(station.attrs[key], "%Y/%m/%d %H:%M:%S.%f")
                    #        plt.axvline(x=((ptime-btime).total_seconds())*100, c="r", label="P") 
                    #    if "Sg" == key:
                    #        ptime = datetime.datetime.strptime(station.attrs[key], "%Y/%m/%d %H:%M:%S.%f")
                    #        plt.axvline(x=((ptime-btime).total_seconds())*100, c="b", label="S") 
                #plt.legend() 
                #plt.xlim([49000, 51000])
                #plt.savefig(f"tfig/{ekey}.{skey}.png")
class TestH5():
    def __init__(self, phasefile):
        self.phase_file = phasefile 
        self.process_phase_file()
    def get_event(self, data):
        """
        获取单个地震事件信息
        data:单个地震事件列表
        retrun:[地震事件信息, 地震震相信息]
        """
        head = data[0] # 地震信息行，以#开头的那一行
        body = data[1:] # 震相信息列表，以PHASE开头的所有列
        def sline(line):
            """进行字符串分割，应当是用re的，但是太麻烦了"""
            s = [i for i in line.split(" ") if len(i)>0] 
            return s 
        head = sline(head) 
        body = [sline(itr) for itr in body] 

        if head[15] == "NONE":head[15]='-1'
        event = { # 头段信息，包括地震震级、位置等信息。
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
            pdic = { # 地震震相信息，可以添加更多信息
                    "type": pname, 
                    "time": ptime,
                    "dist": float(p[24]),
                    name1:p[27], # 这个是初动信息
                    name2:p[28], # 这个也是初动信息
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
    def process_phase_file(self): 
        """
        处理震相文件
        仅做分割处理
        """
        if False:#os.path.exists("models/sdata.pkl"):
            with open("models/sdata.pkl", "rb") as f:
                self.file_data = pickle.load(f)   
                return          
        with open(self.phase_file, "r", encoding="utf-8") as f:
            file_data = f.read() 
        file_data_s = file_data.split("#") 
        self.file_data = [itr.split("\n") for itr in file_data_s] 
    def test(self, file_name="models/h5test/allmt-gzip4.h5", stfile="out.txt"):
        h5file = h5py.File(file_name, "r")  
        eventN = [0, 0]
        phaseN = [0, 0] 
        dataN = [0, 0, 0, 0]
        N = 0
        length = 0
        file_ = open(stfile, "w")
        for eve in tqdm.tqdm(self.file_data[1:]):
            evinfo = self.get_event(eve) 
            event, phases = evinfo 
            evtime = event["time"] 
            ekey = event["evid"] 
            if ekey in h5file:
                eventN[1] += 1 
            else:
                eventN[0] += 1 
                break 
            group = h5file[ekey]
            stiddict = {}
            for phase in phases:
                stids = phase["stid"].split(".")
                stid = ".".join(stids[:3]) 
                if stid in stiddict:
                    continue 
                else:
                    stiddict[stid] = 0
                if stid in group:
                    phaseN[1] += 1 
                else:
                    phaseN[0] += 1      
                    continue 
                subgroup = group[stid]
                dN = 0 
                for k in subgroup:
                    dN += 1 
                    #length += len(subgroup[k])
                if dN != 3:
                    file_.write(f"地震:{ekey},台站:{stid},分量数:{dN}\n")
                dataN[dN] += 1 
            if N % 100 == 0:
                print(f"地震缺失与保存数量：{eventN}，台站缺失与保存数量：{phaseN}，数据0~3个统计：{dataN}，采样点总数：{length}")
            N += 1
#nohup python maketrainh5.py  > mh5.log 2>&1 &
#15191
