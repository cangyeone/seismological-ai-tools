from time import strftime
import matplotlib.pyplot as plt 
import matplotlib.gridspec as grid 
import numpy as np 
import datetime 
import os 
from obspy.geodetics import locations2degrees, degrees2kilometers 

def caldist(loc1, loc2):
    return degrees2kilometers(locations2degrees(loc1[1], loc1[0], loc2[1], loc2[0]))
station_dict = {}
with open("odata/china.pos", "r", encoding="utf8") as f:
    for line in f.readlines():
        sline = [i for i in line.split(" ") if len(i)>0]
        skey = ".".join(sline[:2])
        station_dict[skey] = [float(sline[3]), float(sline[4])]


root = "odata/meng.real.txt.temp"
file_names = os.listdir(root)
ofile = open("odata/meng.realt.txt", "w")
for fn in file_names:
    if "phase" not in fn:continue 
    path = os.path.join(root, fn) 
    file_ = open(path, "r", encoding="utf-8")
    basetime = datetime.datetime.strptime(fn.split(".")[0], "%Y%m%d")
    for idx, line in enumerate(file_.readlines()):
        sline = [i for i in line.strip().split(" ") if len(i)>0] 
        if idx == 0:
            if len(sline[2])==3:
                sline[2] = f"0{sline[2]}"
            rbtime = datetime.datetime.strptime(f"{sline[1]}-{sline[2]}", "%Y-%m%d")
            if np.abs((rbtime-basetime).total_seconds())>1:
                break 
        if sline[0].isdigit():
            #print(sline)
            lat, lon, dep = float(sline[7]), float(sline[8]), float(sline[9])
            #print(sline)
            if "-"  in sline[4] or "60" in sline[4]:
                sline[4] = sline[4].replace("-", "")
                sline[4] = sline[4].replace("60", "59")
            #print(sline)
            if len(sline[2])==3:
                sline[2] = f"0{sline[2]}"
            etime = datetime.datetime.strptime(f"{sline[1]}-{sline[2]} {sline[4]}", "%Y-%m%d %H:%M:%S.%f")
            tstr = etime.strftime("%Y-%m-%d %H:%M:%S.%f")
            pn, ns, nps, nboth = [int(sline[12+i]) for i in range(4)]
            eloc = [lon, lat]
            ofile.write(f"#EVENT,{123456},{tstr},{lon:.3f},{lat:.3f},{dep:.3f},{pn},{ns},{nps},{nboth}\n")
        else:
            dt = float(sline[4])
            ptime = etime + datetime.timedelta(seconds=dt)
            tstr = ptime.strftime("%Y-%m-%d %H:%M:%S.%f")
            skey = ".".join(sline[:2])
            ptype = sline[2]
            dist = float(sline[8])
            prob = float(sline[7])
            err = float(sline[6])
            dist = caldist(eloc, station_dict[skey])
            ofile.write(f"{ptype},{tstr},{dt:.3f},{dist:.3f},{prob:.3f},{err:.3f},{skey}\n")
    file_.close()


