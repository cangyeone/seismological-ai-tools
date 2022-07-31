from obspy import UTCDateTime
import datetime  
from utils.dbdata import Client 
from obspy.taup import TauPyModel
from obspy.geodetics import calc_vincenty_inverse, degrees2kilometers, kilometer2degrees, locations2degrees 
from scipy.interpolate import interp1d 
import pickle 
import os 
import tqdm 
import numpy as np 
import torch 

from obspy.imaging.beachball import beach, beachball 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as grid 

from hasha.hashpype import mk_table_add, ran_norm, check_pol, get_tts, get_gap, focalmc,\
    mech_prob, get_misf
import numpy as np
from hasha.scripts.hash_utils import fortran_include, get_sta_coords, test_stereo
def caldist(loc1, loc2):
    return degrees2kilometers(locations2degrees(loc1[1], loc1[0], loc2[1], loc2[0]))
def caldistaz(loc1, loc2):
    dist, amz, _ = calc_vincenty_inverse(loc1[1], loc1[0], loc2[1], loc2[0])
    return dist/1000, amz 
class FetchPolar():
    def __init__(self, findex, basedir, fsta, mdist=120, fpolar="ckpt/polar3d.jit", fphase="ckpt/china.rnn.jit"):
        self.client = Client(findex, datapath_replace=["/data/workWANGWT/yangbi", basedir]) 
        self.station = {}
        self.mdist = mdist 
        self.device = torch.device("cuda")
        self.mploar = torch.jit.load(fpolar) # Polar model
        self.mploar.to(self.device)
        self.mphase = torch.jit.load(fphase) # Phase picking method 
        self.mphase.to(self.device)
        with open(fsta, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sline = [i for i in line.strip().split(" ") if len(i)>0]
                skey = ".".join(sline[:3])
                self.station[skey] = [float(sline[3]), float(sline[4]), float(sline[5])]
        
        if os.path.exists(f"ckpt/disttime{mdist}.pkl"):
            with open(f"ckpt/disttime{mdist}.pkl", "rb") as f:
                self.disttime = pickle.load(f)
        else:
            model = TauPyModel(model="iasp91")
            self.disttime = []
            print("构建走时表")
            for dep in tqdm.tqdm(range(20)):
                dists = [] 
                times = []
                for dist in range(2, mdist+5, 5):
                    arrivals = model.get_travel_times(source_depth_in_km=dep,
                                    distance_in_degree=kilometer2degrees(dist), 
                                    phase_list=["P", "Pg"])
                    if len(arrivals) == 0:
                        #print(dist, dep, arrivals)
                        continue 
                    dists.append(dist) 
                    times.append(arrivals[0].time) 
                func = interp1d(dists, times, fill_value="extrapolate")
                self.disttime.append(func)
            with open(f"ckpt/disttime{mdist}.pkl", "wb") as f:
                pickle.dump(self.disttime, f)
    def get_polar(self, einfo):
        edep = einfo["dep"]
        etime = einfo["time"]
        eloc = einfo["loc"]
        typed = {0:"U", 1:"D"}
        quald = {0:"I", 1:"M", 2:"E"}
        stations = []
        estr = etime.strftime("%Y%m%d%H%M%S%f")
        epath = f"efig/{estr}-{einfo['mag']:.1f}"
        if os.path.exists(epath) == False:
            os.mkdir(epath)
        
        for skey in self.station:# 获取台站
            net, sta, loc = skey.split(".")
            dist = caldist(eloc, self.station[skey])
            if dist > self.mdist:continue 
            if edep > 19:
                func = self.disttime[19] 
                dtime = func(dist)
            else:
                func = self.disttime[int(edep)]
                dtime = func(dist)
            # 截取波形时间
            tb = etime + datetime.timedelta(seconds=dtime-20)
            te = etime + datetime.timedelta(seconds=dtime+50)
            t1 = UTCDateTime(tb.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
            t2 = UTCDateTime(te.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
            datas = []
            for cha in ["?HE", "?HN", "?HZ"]:# 获取三分量波形
                st = self.client.get_waveforms(net, sta, loc, cha, t1, t2)
                if len(st)==0:continue 
                w = st[0].data 
                w = w.astype(np.float32)
                w = w - np.mean(w)
                if len(w)<7000:continue 
                datas.append(w[:6800])
            if len(datas) != 3:continue 
            datas = np.stack(datas, axis=1)
            datas /= (np.max(np.abs(datas)) + 1e-6) 
            with torch.no_grad():
                x = torch.tensor(datas, dtype=torch.float32, device=self.device)
                phas = self.mphase(x) # 震相拾取，拾取P波数据。
                #print(phas)
                if len(phas) == 0:continue 
                pi = int(phas.cpu().numpy()[0, 1])
                if pi < 512:
                    pi = 512 
                if pi + 513 > len(x):
                    pi = len(x)-513
                #print(pi, x.shape)
                x = x[pi-512:pi+512]  # 截取P波部分数据
                x = x.permute(1, 0).unsqueeze(0)
                #print(x.shape)
                y1, y2 = self.mploar(x) # 判断初动方向
                p1 = y1.cpu().numpy() 
                p2 = y2.cpu().numpy()
                i1 = np.argmax(p1, axis=1)[0] 
                i2 = np.argmax(p2, axis=1)[0]
                qul = quald[i2]
                w = datas[:, 2] 
                w -= np.mean(w) 
                wave = datas[pi-512:pi+512]
                pre = w[pi-50:pi] 
                aft = w[pi:pi+50] 
                snr = np.std(aft)/(np.std(pre)+1e-6)
                if p1[0, i1] > 0.6 and snr>0: # 根据阈值判断初动质量。
                    qul = "I" 
                else:
                    qul = "E"
            dist, amz = caldistaz(eloc, self.station[skey]) 
            #gs = grid.GridSpec(1, 1) 
            #fig = plt.figure(1, figsize=(12, 6), dpi=100) 
            #ax = fig.add_subplot(gs[0])
            #t = np.arange(len(w[pi-100:pi+100])) * 0.01 - 1 
            #ax.plot(t, w[pi-100:pi+100], c="k", lw=1)
            #ax.set_xlabel("Time [s]")
            #ax.set_ylabel("Amplitude")
            #ax.set_title(f"ploar={typed[i1]},quality={qul},dist={dist:.1f},amz={amz:.1f}")
            #np.savez(f"data/polar/{estr}-{einfo['mag']:.1f}-{skey}-{typed[i1]}-{qul}.npz", wave=wave, snr=snr, p=p1[0, i1])
            #plt.savefig(f"{epath}/{skey}-{typed[i1]}-{qul}.jpg")
            #plt.savefig(f"{epath}/{skey}-{typed[i1]}-{qul}.svg")
            #plt.close()
            sta = {"polar":typed[i1], "quality":qul, "dist":dist, "amz":amz} 
            sta["loc"] = self.station[skey] 
            sta["name"] = skey    
            stations.append(sta)   
        #print(stations)
        return stations 
def get_events(filename):
    with open(filename, "r", encoding="utf-8") as f:
        einfos = []
        for line in f.readlines():
            line = line.replace("NONE", "0")
            sline = [i for i in line.strip().split(",") if len(i)>0]
            if float(sline[5])<3.5:continue 
            einfo = {
                    "loc":[float(sline[2]), float(sline[3])], 
                    "time":datetime.datetime.strptime(sline[1], "%Y-%m-%d %H:%M:%S.%f"), 
                    "dep":float(sline[4]), 
                    "mag":float(sline[5])
                    }
            einfos.append(einfo)
    return einfos 
class HASH(FetchPolar):
    def __init__(self, findex, basedir, fsta, mdist=160, fpolar="ckpt/polar3d.jit"):
        super().__init__(findex, basedir, fsta, mdist, fpolar)
    def call_hash(self):
        # 模仿HASH中的dirver
        # 每个地震最大样本，最大实验次数，最大可接受参数输出
        npick0, nmc0, nmax0 = fortran_include('hash/param.inc')
        # 网格角度，测试机制数量
        dang0, ncoor        = fortran_include('hash/rot.inc')
        # 输入参数
        sname     = np.empty(npick0, 'a4', 'F') # 台站名
        scomp     = np.empty(npick0, 'a3', 'F')
        snet      = np.empty(npick0, 'a2', 'F')
        pickpol   = np.empty(npick0, 'a1', 'F')
        pickonset = np.empty(npick0, 'a1', 'F')
        p_pol     = np.empty(npick0, int, 'F')
        p_qual    = np.empty(npick0, int, 'F')
        spol      = np.empty(npick0, int, 'F')
        p_azi_mc  = np.empty((npick0,nmc0), float, 'F')
        p_the_mc  = np.empty((npick0,nmc0), float, 'F')
        index     = np.empty(nmc0, int, 'F')
        qdep2     = np.empty(nmc0, float, 'F')

        # 输出数据
        f1norm  = np.empty((3,nmax0), float, 'F')
        f2norm  = np.empty((3,nmax0), float, 'F')
        strike2 = np.empty(nmax0, float, 'F')
        dip2    = np.empty(nmax0, float, 'F')
        rake2   = np.empty(nmax0, float, 'F')
        str_avg = np.empty(5, float, 'F')
        dip_avg = np.empty(5, float, 'F')
        rak_avg = np.empty(5, float, 'F')
        var_est = np.empty((2,5), float, 'F')
        var_avg = np.empty(5, float, 'F')
        mfrac   = np.empty(5, float, 'F')
        stdr    = np.empty(5, float, 'F')
        prob    = np.empty(5, float, 'F')
        qual    = np.empty(5, 'a', 'F')

        degrad = 180. / np.pi
        rad = 1. / degrad
        # 输入示例
        input_file = open("hash/example2.inp").read()
        inp_vars = input_file.split("\n")
        (stfile,plfile,fpfile,outfile1,outfile2,npolmin,max_agap,max_pgap,
        dang,nmc,maxout,badfrac,delmax,cangle,prob_max) = inp_vars[:15]
        num_vel_mods = int(inp_vars[15])
        vmfile = inp_vars[16:16+num_vel_mods]
        for i in range(num_vel_mods):
            ntab = mk_table_add(i+1, vmfile[i])  
        # 台站位置
        npolmin = int(npolmin)
        max_agap = int(max_agap)
        max_pgap = int(max_pgap)
        dang = float(dang)
        dang2 = max(dang0, dang) # don't do finer than dang0
        nmc = int(nmc)
        nmc = 3      # 速度模型数量
        maxout = int(maxout)
        badfrac = float(badfrac)
        delmax = float(delmax)
        cangle = float(cangle)
        prob_max = float(prob_max)
        events = get_events("data/ctlg.txt")
        outfile = open("data/focal.txt", "w")
        for event in events:
            elon, elat = event["loc"]
            edep = event["dep"]
            qdep2[0] = edep 
            index[0] = 1
            sez = None 
            if not sez:
                sez = 1.
            for nm in range(1, nmc):
                val = ran_norm()
                qdep2[nm] = edep# + sez * val # randomly perturbed source depth
                index[nm] = (nm % ntab) + 1  # index used to choose velocity model 
            k = 0 
            str_avg[0] = 0 
            etime = event["time"]
            polars = self.get_polar(event)
            pdatas = []
            for nm, infos in enumerate(polars):
                sname = infos["name"]
                net, sta, _ = sname.split(".")
                snet[k] = net 
                scomp[k] = sta
                pickonset[k] = infos["quality"]
                pickpol[k] = infos["polar"]
                dist = infos["dist"] #/ 100
                qazi = infos["amz"]
                
                if (pickpol[k].decode('UTF-8') in 'Uu+'):
                    p_pol[k] = 1
                elif (pickpol[k].decode('UTF-8') in 'Dd-'):
                    p_pol[k] = -1
                else:
                    continue
                
                if (pickonset[k].decode('UTF-8') in 'Ii'):
                    p_qual[k] = 0
                else:
                    p_qual[k] = 1
                spol = check_pol(
                    plfile, net, etime.year, etime.month, etime.day, etime.hour)  
                p_pol[k] = p_pol[k] * spol
                distss     = np.empty(1, int, 'F')
                distss[0] = dist
                # find azimuth and takeoff angle for each trial
                for nm in range(nmc):
                    p_azi_mc[k, nm] = qazi
                    p_the_mc[k, nm], iflag = get_tts(index[nm], distss[0], qdep2[nm])
                    #print(p_the_mc[k, nm], dist)
                pdatas.append([qazi/180, p_the_mc[k, nm]/180, p_pol[k], p_qual[k]])
                k += 1
                continue        
            pdatas = np.array(pdatas)    
            npol = k# 样本数量
            print(f"样本数量{npol}")
            if (npol < npolmin):
                str_avg[0] = 999
                dip_avg[0] = 99
                rak_avg[0] = 999
                var_est[0,0] = 99
                var_est[1,0] = 99
                mfrac[0] = 0.99
                qual[0] = 'F'
                prob[0] = 0.0
                nout1 = 0
                nout2 = 0
                nmult = 0
                #goto 400   
            if str_avg[0] == 999:
                pass
            else:   
                # determine maximum azimuthal and takeoff gap in polarity observations and stop if either gap is too big
                magap, mpgap = get_gap(p_azi_mc[:npol,0], p_the_mc[:npol, 0], npol)
                #print(magap, mpgap, max_agap, max_pgap)
                if ((magap > max_agap) or (mpgap > max_pgap)):
                    str_avg[0] = 999
                    dip_avg[0] = 99
                    rak_avg[0] = 999
                    var_est[0,0] = 99
                    var_est[1,0] = 99
                    mfrac[0] = 0.99
                    qual[0] = 'E'
                    prob[0] = 0.0
                    nout1 = 0
                    nout2 = 0
                    nmult = 0
            if str_avg[0] == 999:
                pass
            else:
                # 计算可接受的样本数量
                print(f"计算{etime}")
                nmismax = max(int(npol * badfrac),2)        # nint
                nextra  = max(int(npol * badfrac * 0.5),2)  # nint
                # find the set of acceptable focal mechanisms for all trials
                nf2,strike2,dip2,rake2,f1norm,f2norm = focalmc(
                    p_azi_mc, p_the_mc, p_pol[:npol], p_qual[:npol],\
                        nmc, dang2, nmax0, nextra, nmismax, npol)
                nout2 = min(nmax0, nf2)  # number mechs returned from sub
                nout1 = min(maxout, nf2) # number mechs to return
                fig = plt.figure(1, figsize=(12, 12), dpi=100)
                gs = grid.GridSpec(1, 1)
                ax = fig.add_subplot(gs[0])
                for i in range(nf2):
                    sk, dp, rk = strike2[i], dip2[i], rake2[k]
                    fm = (sk, dp, rk)
                    coll = beach(fm, width=2, linewidth=0.5, bgcolor=(1, 1, 1, 0), edgecolor=(0, 0, 0, 1), alpha=0.2, nofill=True)
                    ax.add_collection(coll)
                    if i > 60:break 
                # find the probable mechanism from the set of acceptable solutions
                nmult, str_avg, dip_avg, rak_avg, prob, var_est = mech_prob(f1norm[:,:nout2],f2norm[:,:nout2],cangle,prob_max,nout2) # nout2

                #print(f"{nf2},{nmult}")
                #print(f"FINAL{evid.replace(' ', '')},{fm}")
                for imult in range(nmult):
                    var_avg[imult] = (var_est[0,imult] + var_est[1,imult]) / 2.
                    mfrac[imult],stdr[imult] =  get_misf(p_azi_mc[:npol,0],p_the_mc[:npol,0],p_pol[:npol],p_qual[:npol],str_avg[imult],dip_avg[imult],rak_avg[imult],npol) # npol
                    
                    # solution quality rating  ** YOU MAY WISH TO DEVELOP YOUR OWN QUALITY RATING SYSTEM **
                    if ((prob[imult] > 0.8) and (var_avg[imult] < 25) and (mfrac[imult] <= 0.15) and (stdr[imult] >= 0.5)):
                        qual[imult]='A'
                    elif ((prob[imult] > 0.6) and (var_avg[imult] <= 35) and (mfrac[imult] <= 0.2) and (stdr[imult] >= 0.4)):
                        qual[imult]='B'
                    elif ((prob[imult] > 0.5) and (var_avg[imult] <= 45) and (mfrac[imult] <= 0.3) and (stdr[imult] >= 0.3)):
                        qual[imult]='C'
                    else:
                        qual[imult]='D'
                fm = (str_avg[0], dip_avg[0], rak_avg[0])
                # 输出震源参数
                outfile.write(f"#EVENT,{etime+datetime.timedelta(hours=8)},{str_avg[0]},{dip_avg[0]},{rak_avg[0]},{qual[0].decode('UTF-8')}\n")
                for v, p in zip(pdatas, pdatas[:, 2]):
                    o = v[0]
                    r = v[1]
                    outfile.write(f"Phase,{o:.5f},{r:.5f},{p}\n")
                for i in range(nf2):
                    sk, dp, rk = strike2[i], dip2[i], rake2[k]
                    outfile.write(f"SEL,{sk},{dp},{rk}\n")                
                outfile.flush()
                coll = beach(fm, width=2, linewidth=2, bgcolor=(1, 1, 1, 0), edgecolor=(1, 0, 0, 1), alpha=1, nofill=True)
                ax.add_collection(coll)
                ax.set_title(f"{qual[0]}{etime}-{fm}", fontsize=16)
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
                o = pdatas[:, 0] * 3.1415926 
                r = pdatas[:, 1] #* 3.1415926 
                x1 = r * np.cos(o)
                x2 = r * np.sin(o)
                s = pdatas[:, 3] 
                
                ax.scatter(x1[pdatas[:, 2]== 1], x2[pdatas[:, 2]== 1], c="r", s=(2-s[pdatas[:, 2]== 1])*50)
                ax.scatter(x1[pdatas[:, 2]==-1], x2[pdatas[:, 2]==-1], c="b", s=(2-s[pdatas[:, 2]==-1])*50)
                plt.savefig(f"figs/demo{etime+datetime.timedelta(hours=8)}-{fm}.png")
                plt.savefig(f"figs/demo{etime+datetime.timedelta(hours=8)}-{fm}.svg")
                plt.close()
                print(f"{etime}绘制完成")
def main():
    #polartool = FetchPolar(
    #    "yangbi/mseedindex3.yangbi.0517.to.0528.sqlite3", 
    #    "yangbi", 
    #    "data/YN.GG.loc") 
    #einfo = {
    #    "loc":[99.929, 25.631], 
    #    "time":datetime.datetime.strptime("2021-05-17,21:39:35.88", "%Y-%m-%d,%H:%M:%S.%f"), 
    #    "dep":14
    #}
    #polartool.get_event_wave(einfo)
    hash = HASH( 
        "yangbi/mseedindex3.yangbi.0517.to.0528.sqlite3", # 波形索引数据库
        "yangbi",     
        "data/YN.GG.loc", # 台站位置
        fpolar="ckpt/polar.jit", # 初动方向
        mdist=500
    )
    hash.call_hash()

if __name__== "__main__":
    os.system("rm figs/*")
    os.system("rm efig/* -r")
    main()
