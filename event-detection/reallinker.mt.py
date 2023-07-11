import os 
import datetime 
import tqdm 
import pickle 
import time 
import shutil 
import time 
from config.real import Parameter as global_par 
import multiprocessing 
import time 

def mkreal(file_name):
    file_ = open(file_name, "r", encoding="utf-8") 
    outdir = global_par.realdir 
    if os.path.exists(outdir)==False:
        os.mkdir(outdir)
    if True:
        for line in tqdm.tqdm(file_.readlines()):
            if "##" in line:continue 
            if "#" in line:continue 
            sline = [i for i in line.strip().split(",") if len(i)>0] 
            ptype = sline[0] 
            if ptype=="Pg" or ptype=="Pn": 
                pname = "P" 
            elif ptype == "Sg" or ptype=="Sn": 
                pname = "S"
            else:
                continue 
            ptime = datetime.datetime.strptime(sline[3], "%Y-%m-%d %H:%M:%S.%f") 
            daystr = sline[3].split(" ")[0]
            basetime = datetime.datetime.strptime(daystr, "%Y-%m-%d") 
            delta = (ptime-basetime).total_seconds() 
            name = sline[6]
            sname = f"{name[:2]}.{name[2:]}"
            dirname = f"{name}.{pname}.txt" 
            prob = float(sline[2])
            if prob<global_par.prob:
                continue 
            daystr = daystr.replace("-", "")
            path = os.path.join(outdir, daystr) 
            if os.path.exists(path)==False:
                os.mkdir(path)
            with open(os.path.join(path, dirname), "a", encoding="utf-8") as f:
                f.write(f"{delta:.2f} {prob:.3f} {0}\n")


def realmt(realid, fqueue, odata):
    base_dir = global_par.realdir 
    R = global_par.R
    G = global_par.G
    V = global_par.V  
    S = global_par.S
    station="./tt_db/station.dat"
    ttime="./tt_db/tdb.txt"
    cwd = os.getcwd()
    realtooldir = f"realtool/real{realid}"
    while True:
        t1 = time.perf_counter()
        temp = fqueue.get()
        if len(temp)==0:
            break 
        else:
            fn, path, btime = temp 
        D = btime.strftime("%Y/%m%d/25.5")
        t1 = time.perf_counter()
        os.chdir(realtooldir)
        os.system(fr".\real -D{D} -R{R} -G{G} -S{S} -V{V} {station} ../../{path} {ttime}")
        os.chdir(cwd)
        t2 = time.perf_counter() 
        shutil.copy(f"{realtooldir}/catalog_sel.txt", os.path.join(odata, fn+".ctlg.txt"))
        shutil.copy(f"{realtooldir}/phase_sel.txt",  os.path.join(odata, fn+".phase.txt"))
def real(odata):
    base_dir = global_par.realdir 
    file_names = os.listdir(base_dir)
    nthread = global_par.n_thread 
    fqueue = multiprocessing.Queue(1)
    for i in range(nthread):
        t = multiprocessing.Process(target=realmt, args=(i, fqueue, odata))
        t.start()
    for fn in file_names:
        t1 = time.perf_counter()
        btime = datetime.datetime.strptime(fn, "%Y%m%d") 
        path = os.path.join(base_dir, fn)
        fqueue.put([fn, path, btime])
        t2 = time.perf_counter()
        print(f"REAL关联用时：{t2-t1:.3f}")
    for i in range(nthread):
        fqueue.put([])
def mkstation(inname):
    #GX   BSS 00  106.5565   23.8951    179 GX.BSS
    infile = open(inname, "r", encoding="utf-8")  
    outfile = open(os.path.join(global_par.realtooldir, "tt_db", "station.dat"), "w", encoding="utf-8")
    for line in infile.readlines():
        #99.8230 25.8150 GG 53036 BHZ 0
        sline = [i for i in line.split(" ") if len(i)>0] 
        #print(sline)
        #if len(sline)<3:continue 
        lon, lat, dep = float(sline[3]), float(sline[4]), float(sline[5])/1000
        outfile.write(f"{lon:.4f} {lat:.4f} {sline[0]} {sline[1]} BHZ {dep:3f}\n")
from obspy.geodetics import locations2degrees, degrees2kilometers 
import numpy as np 
def caldist(loc1, loc2):
    return degrees2kilometers(locations2degrees(loc1[1], loc1[0], loc2[1], loc2[0]))

def mergedata(ifiledir, ofile, station):
    station_dict = {}
    with open(station, "r", encoding="utf8") as f:
        for line in f.readlines():
            sline = [i for i in line.split(" ") if len(i)>0]
            skey = ".".join(sline[:2])
            station_dict[skey] = [float(sline[3]), float(sline[4])]
    root = ifiledir
    file_names = os.listdir(root)
    ofile = open(ofile, "w")
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
                sloc = station_dict[skey]
                ofile.write(f"{ptype},{tstr},{dt:.3f},{dist:.3f},{prob:.3f},{err:.3f},{skey},{sloc[0]:.4f},{sloc[1]:.4f}\n")
        file_.close()


import argparse
import shutil 
if __name__ == "__main__":
    # 处理输出某个时间之后的所有震相
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-i', '--input', default="odata/sizy.rnn.txt", help="拾取文件")       
    parser.add_argument('-o', '--output', default="odata/meng.real.txt", help="输出文件名")   
    parser.add_argument('-s', '--station', default="null", help="输出文件名")                                                         
    args = parser.parse_args()  
    infile = args.input 
    outfile = args.output 
    if args.station != "null":
        print(type(args.station), args.station)
        mkstation(args.station)
    for i in range(global_par.n_thread):
        if os.path.exists(f"realtool/real{i}"):
            shutil.rmtree(f"realtool/real{i}")
        shutil.copytree(global_par.realtooldir, f"realtool/real{i}")
    tfiledir = f"{outfile}.temp"
    if os.path.exists(tfiledir) == False:
        os.mkdir(tfiledir)
    if os.path.exists(global_par.realdir) == False:
        os.mkdir(global_par.realdir)
    if os.path.exists(tfiledir)==True:
        if len(os.listdir(global_par.realdir))!=0:
            tag = input("已经存在，是否删除原有real数据")
            if tag in ["Y", "y"]:
                shutil.rmtree(global_par.realdir)
                os.mkdir(global_par.realdir)
                mkreal(infile)# 制作REAL关联文件
                real(tfiledir) # 进行real关联
            if tag in ["N", "n"]:
                #mkreal(infile)# 制作REAL关联文件
                real(tfiledir) # 进行real关联
        else:
            mkreal(infile)# 制作REAL关联文件
            real(tfiledir) # 进行real关联        
        mergedata(tfiledir, outfile, args.station)
    else:
        print("已存在临时文件，直接处理临时文件")
        mergedata(tfiledir, outfile, args.station)
