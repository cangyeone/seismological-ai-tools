import os 
import datetime 
import tqdm 
import pickle 
import time 
import shutil 
import time 
from config.real import Parameter as global_par 

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
            if ptype=="Pg": 
                pname = "P" 
            elif ptype == "Sg": 
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
            daystr = daystr.replace("-", "")
            path = os.path.join(outdir, daystr) 
            if os.path.exists(path)==False:
                os.mkdir(path)
            with open(os.path.join(path, dirname), "a", encoding="utf-8") as f:
                f.write(f"{delta:.2f} {prob:.3f} {0}\n")
            
def real(odata):
    R = global_par.R
    G = global_par.G
    V = global_par.V  
    S = global_par.S
    base_dir = global_par.realdir 
    file_names = os.listdir(base_dir)
    station="./tt_db/station.dat"
    ttime="./tt_db/tdb.txt"
    timefile = open("realtime2.txt", "a", encoding="utf-8")
    cwd = os.getcwd()
    for fn in file_names:
        t1 = time.perf_counter()
        btime = datetime.datetime.strptime(fn, "%Y%m%d") 
        path = os.path.join(base_dir, fn)
        D = btime.strftime("%Y/%m%d/25.5")
        t1 = time.perf_counter()
        os.chdir(global_par.realtooldir)
        os.system(fr".\real.exe -D{D} -R{R} -G{G} -S{S} -V{V} {station} ../{path} {ttime}")
        os.chdir(cwd)
        t2 = time.perf_counter() 
        print(f"REAL关联用时：{t2-t1:.3f}")
        t2 = time.perf_counter()
        timefile.write(f"{t2-t1},{t1},{t2}\n")
        timefile.flush()
        shutil.copy(f"{global_par.realtooldir}/catalog_sel.txt", os.path.join(odata, fn+".ctlg.txt"))
        shutil.copy(f"{global_par.realtooldir}/phase_sel.txt",  os.path.join(odata, fn+".phase.txt"))
def readctlg(inname):
    #GX   BSS 00  106.5565   23.8951    179 GX.BSS
    infile = open(inname, "r", encoding="utf-8")  
    outfile = open(os.path.join(global_par.realtooldir, "tt_db", "station.dat"), "w", encoding="utf-8")
    for line in infile.readlines():
        #99.8230 25.8150 GG 53036 BHZ 0
        sline = [i for i in line.split(" ") if len(i)>0] 
        #print(sline)
        #if len(sline)<3:continue 
        lon, lat, dep = float(sline[2]), float(sline[3]), float(sline[4])/1000
        outfile.write(f"{lon:.4f} {lat:.4f} {sline[0]} {sline[1]} BHZ {dep:3f}\n")
import argparse
if __name__ == "__main__":
    # 处理输出某个时间之后的所有震相
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-i', '--input', default="odata/x1.rnn.pnsn.txt", help="拾取文件")       
    parser.add_argument('-o', '--output', default="./odata/reallinkpnsn", help="输出文件名")   
    parser.add_argument('-s', '--station', default="null", help="输出文件名")                                                         
    args = parser.parse_args()  
    infile = args.input 
    outfile = args.output 
    if args.station != "null":
        readctlg(args.station)
    if len(os.listdir(global_par.realdir))!=0:
        tag = input("已经存在，是否删除原有real数据")
        if tag in ["Y", "y"]:
            os.system(f"rm -rf {global_par.realdir}/*")
            mkreal(infile)# 制作REAL关联文件
            real(outfile) # 进行real关联
        if tag in ["N", "n"]:
            #mkreal(infile)# 制作REAL关联文件
            real(outfile) # 进行real关联
    else:
        mkreal(infile)# 制作REAL关联文件
        real(outfile) # 进行real关联        
