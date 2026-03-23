from obspy import UTCDateTime
import datetime  
from utils.dbdata import Client 
import shutil 
import argparse
import os 
import obspy 
import multiprocessing 
import logging

logger = logging.getLogger('obspy')
logger.setLevel(logging.INFO)

import tqdm 
from obspy.io.sac.sactrace import SACTrace
from obspy.geodetics import calc_vincenty_inverse 
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.io.sac.sactrace import SACTrace
from obspy.geodetics import calc_vincenty_inverse 
class Config():
    filter_by_mag_distance = True 
    station_loc_file = None  #台站位置信息 初始值设为None,稍后从外部获取
    filter_mags = [[-10, 0], [0, 1], [1, 2], [2, 3], [3, 10]] #震级范围
    
    filter_dist = {"20":[5000000, 10000000, 15000000, 20000000, 50000000], #20仪器范围，对应filter_mags的震级
                   "00":[5000000, 10000000, 15000000, 20000000, 50000000], 
                   "40":[5000000, 10000000, 15000000, 20000000, 50000000],
                   "02":[5000000, 10000000, 15000000, 20000000, 50000000],
                   "01":[5000000, 10000000, 15000000, 20000000, 50000000]}

def read_station_loc(station_loc_file):
    station_loc = {} 
    with open(station_loc_file, "r") as file_:
        for line in file_.readlines():
            sline = [i for i in line.split() if len(i)>0]
        skey2 = f"{sline[0]}.{sline[1]}.{sline[2]}"
        station_loc[skey2] = [float(sline[3]), float(sline[4]), float(sline[5])]
    return station_loc 
               
def cal_dist(loc1, loc2):
    return degrees2kilometers(locations2degrees(loc1[1], loc1[0], loc2[1], loc2[0]))
def read_station_loc(root):
    station_loc = {} 
    file_ = open(root, "r") 
    for line in file_.readlines():
        sline = [i for i in line.split() if len(i)>0]
        #skey1 = f"{sline[0]}.{sline[1]}"
        skey2 = f"{sline[0]}.{sline[1]}.{sline[2]}"
        #station_loc[skey1] = [float(sline[3]), float(sline[4]), float(sline[5])]
        station_loc[skey2] = [float(sline[3]), float(sline[4]), float(sline[5])]
    return station_loc 

def read_ctlg_event(path, comma=" "):#读取地震目录
    events = []
    file_ = open(path, "r", encoding="utf8")
    for line in file_.readlines():
        if "#EVENT" not in line:continue 
        #if "NONE" in line:continue 
        head = [i for i in line.split(" ") if len(i)>0]
        if head[15] == "NONE":head[15]='-1'
        if head[13] == "NONE":head[13]="-360"
        if head[12] == "NONE":head[12] = "-360"
        if head[18] == "NONE":head[18] = "-360"
        evid = head[1] 
        #if "NM" not in evid:continue 
        emag = float(head[18])
        etype = head[2]
        #if  emag > 3.6 or emag<1.5:
        #    continue 
        #if etype!="eq":continue 
        prov = evid.split(".")[0] 
        #if prov not in ["SC", "YN"]:continue 
        event = { # 头段信息，包括地震震级、位置等信息。
                    "evid": head[1], 
                    "mag": float(head[18]),
                    "loc": [float(head[12]), float(head[13]), float(head[15])],
                    "time": datetime.datetime.strptime(f"{head[3]}/{head[4]}/{head[5]} {head[7]}:{head[8]}:{head[9]}.{head[10]}", "%Y/%m/%d %H:%M:%S.%f")
                }
        events.append(event)
    file_.close() 
    return events 


def cut_event(event_queue, dbfile, outdir, station_loc_file, base_data_dir):
    #print("数据库开始初始化...")
    config = Config()
    config.station_loc_file = station_loc_file  # 更新配置中的文件路径
    client = Client(dbfile, datapath_replace=["^", base_data_dir])#波形数据库
    #print("数据库初始化完成！")
    stloc = read_station_loc(config.station_loc_file)  # 使用更新后的配置文件路径
    while True:
        event = event_queue.get() 
        if len(event)==0:break 
        try:
            etime = UTCDateTime(event["time"].strftime("%Y/%m/%dT%H:%M:%S.%f"))#obspy时间格式
            emag = event["mag"]
            eloc = event["loc"]
            begin = -30 #减去10秒
            end   = 360 #结束时间
            t1 = etime + begin #截取开始时间
            t2 = etime + end   #截取结束时间
            root = os.path.join(outdir, event["evid"])
            if os.path.exists(root)==False:
                os.makedirs(root)
            tkey = etime.strftime("%Y%m%dT%H%M%S%f")
            filter_id = 0 #设定默认值
            for idx, (r1, r2) in enumerate(config.filter_mags):
                if emag>r1 and emag<r2:
                    filter_id = idx 
            for skey, sloc in stloc.items():
                dist = cal_dist(sloc, eloc)
                net, sta, loc = skey.split(".")
                if loc not in config.filter_dist:
                    loc = "00"
                maxdist = config.filter_dist[loc][filter_id] 
                #print(f"{root}/{net}.{sta}.{loc}.{cha}.mseed")
                if dist >maxdist:
                    continue 
                net, sta, loc = skey.split(".")
                
                dist = cal_dist(stloc[skey], eloc)
                st = client.get_waveforms(net, sta, loc, "*", t1, t2)  # 分量名有三个字符的都算
                if len(st)==0:
                    #print("数据量为0", root)
                    continue 
                for tr in st:
                    stats = tr.stats 
                    net = stats.network 
                    sta = stats.station 
                    loc = stats.location 
                    cha = stats.channel 
                    tr.trim(pad=True, nearest_sample=True, fill_value=0)
                    #tr.write(f"{root}/{net}.{sta}.{loc}.{cha}.{tkey}.mseed")
                    #dist, abaz, baaz = calc_vincenty_inverse(eloc[1], eloc[0], sloc[1], sloc[0])
                    baaz = 0 
                    #EVENT YN.201201312358.0001 eq 2012 01 31 031 15 58 23 770000 LOC  103.498   21.849 DEP        6 MAG   ML  2.3
                    tr2 = SACTrace(
                        b=begin, 
                        nzyear=etime.year, 
                        nzjday=etime.julday, 
                        nzhour=etime.hour, 
                        nzmin=etime.minute, 
                        nzsec=etime.second, 
                        nzmsec=etime.microsecond//1000, 
                        delta=tr.stats.delta, 
                        evla=eloc[1], 
                        evlo=eloc[0], 
                        stla=sloc[1], 
                        stlo=sloc[0], 
                        evdp=eloc[2], 
                        stel=sloc[2], 
                        knetwk=net, 
                        kstnm=sta, 
                        khole=loc, 
                        kcmpnm=cha, 
                        mag=emag, 
                        dist=dist/1000,
                        baz=baaz,  
                        t1=0.0, 
                        data=tr.data)
                    tr2.write(f"{root}/{net}.{sta}.{loc}.{cha}.sac")


            if len(os.listdir(root))==0:
                #print("数据不存在", root)
                shutil.rmtree(root)
        except:
            print("DATA ERROR", root)
            continue 
         
def cut_event_main(args):
    events = [] 
    processed_dict = {}
    nthread = 64  
    if os.path.exists(args.logfile):
        logfile = open(args.logfile, "r") 
        for line in logfile:
            processed_dict[line.strip()] = 0
        logfile.close()
    #for year in ["2021", "2022"]:
    eve = read_ctlg_event(args.ctlgfile)
    events = events + eve
    #stlocs = read_station("data/station.txt")
    event_queue = multiprocessing.Queue(1)
    outdir = args.outdir
    logfile = open(args.logfile, "a")
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)
   
    config = Config()
    config.station_loc_file = args.station_loc  # 设置配置中的台站位置文件路径
    
    for i in range(nthread):
        t = multiprocessing.Process(target=cut_event, args=(event_queue, args.database, args.outdir, config.station_loc_file, args.base_data_dir))
        t.start()

    
    for eve in tqdm.tqdm(events):
        if eve['evid'] in processed_dict:continue 
        #if eve["mag"] < 3.0:continue 
        #print(eve)
        event_queue.put(eve)
        logfile.write(f"{eve['evid']}\n")
        logfile.flush()
    for i in range(nthread):
        event_queue.put([])
#nohup python seedtool/cutdata.filter.py > logdir/cutdata6.log 2>&1 &
#744218

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='截取数据的脚本: python 2.cutdata.py -l odata/cut.log -o odata/cut.miniseed -c data/YN.event.catalog -d odata/miniseed.db.temp -s data/YN.EEW.sta')
    parser.add_argument("-l", "--logfile", help="指定输出路径/*.log",
                        type=str, required=False, default="odata/cut.log")
    #本例子中为 odata/2.cutdata/eg_cutdata.log
    parser.add_argument("-o", "--outdir", help="指定切出sac文件的输出路径",
                        type=str, required=False, default="odata/xyy")
    #odata/2.cutdata/eg_acc_ALQ02
    parser.add_argument("-c", "--ctlgfile", help="地震目录所在路径/*.catalog",
                        type=str, required=False, default="ayrdata/csndata/2012.pha")
    ##EVENT YN.202305010053.0001 eq 2023 04 30 120 16 53 33 800000 LOC 100.081 25.018 DEP 12 MAG ML 1.500 #地震目录格式
    #data/YN.event.catalog
    parser.add_argument("-d", "--database", help="第一步生成的数据库所在目录/*.db",
                        type=str, required=False, default="/data/arrayData/dataX1/X1DATA/dbX1MSEEDV2016FINAL/msindex.X1.100HZ.sqlite3")
    #odata/1.mkindex/eg_acc_ALQ02.db
    parser.add_argument("-s", "--station_loc", help="台站位置文件路径", 
                        type=str, required=False, default="/data/arrayData/dataX1/X1DATA/Info_X1/X1.POS.WITH.LOCID.V2015121")
    parser.add_argument("-b", "--base_data_dir", help="基础数据路径", 
                        type=str, required=False, default="")
    #YN AAN01 00 102.4269  24.9696 1952.00 YN.AAN01.00 #caoying HHZ-HHN-HHE
    #data/YN.EEW.sta.all
    args = parser.parse_args()
    cut_event_main(args)
3
