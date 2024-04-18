import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec 
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec 
import scipy.signal as signal 
import os 
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.switch_backend("agg") 

def file_read(file_name="lppn-ctlg.txt", dist=100, gap1=0.1, gap2=0.5):
    files = open(file_name, "r") 
    phase_list = [[aa for aa in itr.strip().split(',') if len(aa)>0] for itr in files.readlines()]  
    files.close()
    phase_head = []
    for itr in phase_list:
        if "#phase" in itr[0] or "#none" in itr[0]:
            phase_head.append(0) 
        else:
            phase_head.append(1)  
    phase_head = np.array(phase_head) 
    phase_list = np.array(phase_list)
    split_idx = np.where(phase_head==0)[0][1:] 
    phase_list = np.split(phase_list, split_idx) 
    #print(phase_list[:3])
    tp = [0 for itr in range(2)] 
    fp = [0 for itr in range(2)] 
    fn = [0 for itr in range(2)] 
    std = [[] for itr in range(2)]

    #print(phase_list[0])
    for phases in phase_list:
        if "#none" not in phases[0][0]:
            mm_time = [float(phases[0][1]), float(phases[0][2])]
            mm_snr = float(phases[0][3])
        allpick = [False, False]
        for idx, stp in enumerate(mm_time):
            if stp < 0:
                allpick[idx] = True 
        for nn_pick in phases[1:]:
            nn_type = int(float(nn_pick[0]))
            nn_time = float(nn_pick[1]) 
            if "#none" not in phases[0][0]:
                ref_time = mm_time[nn_type-1] 
                if ref_time < 0:
                    continue 
                error = np.abs(ref_time-nn_time) / 100
                if error < gap1:
                    tp[nn_type-1] += 1 
                    allpick[nn_type-1] = True 
                else:
                    fp[nn_type-1] += 1 
                if error < gap2:
                #if True:
                    std[nn_type-1].append((ref_time-nn_time)/100) 
        for tps in range(len(tp)):
            if allpick[tps] == False:
                fn[tps] += 1
    name = ["P", "S"]
    P = [0 for itr in range(2)] 
    R = [0 for itr in range(2)]
    for tps in range(2):
        P[tps] = tp[tps]/(tp[tps]+fp[tps]+1e-9)
        R[tps] = tp[tps]/(tp[tps]+fn[tps]+1e-9)
    gs0 = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
    plt.cla() 
    plt.clf()
    fig = plt.figure(1)
    print("样本数量", file_name, len(phase_list))
    for tps in range(2):
        ax = fig.add_subplot(gs0[0, tps])  
        dts = std[tps]
        if len(dts)==0:
            continue
        ax.hist(dts, bins=72, density=True) 
        ax.set_title(name[tps]) 
        ax.set_xlabel("Error [s]")  
        #ax.set_xlim([-gap2, gap2])
        #ax.set_ylim([0, 16])
        ax.text(0.1, 3, f"{gap1},{gap2}\nP={P[tps]:.3f}\nR={R[tps]:.3f}\n$\mu={np.mean(std[tps])*1e3:.3f}$\n$\sigma={np.std(std[tps])*1e3:.3f}$") 
        #ax.set_title(net_name)
    name = os.path.split(file_name)
    plt.savefig(f"fig/{name[-1]}.png")
    #plt.savefig(f"fig/{name[-1]}.svg")
    #print(f"S:P={s_tp/(s_tp+s_fp)}, R={s_tp/(s_tp+s_fn)}")

if __name__ == "__main__":

    #file_read("stdata/tinny16-2-ctlg.txt", 10000, "LPPN")
    file_read("stdata/phasenet2.txt")
    file_read("stdata/fuse.4-8.stat.txt")
    file_read("stdata/fuse.8-8.stat.txt")
    file_read("stdata/fuse.16-8.stat.txt")
    file_read("stdata/fuse.4-16.stat.txt")
    file_read("stdata/fuse.8-16.stat.txt")
    file_read("stdata/fuse.16-16.stat.txt")
    file_read("stdata/fuse.4-32.stat.txt")
    file_read("stdata/fuse.8-32.stat.txt")
    file_read("stdata/fuse.16-32.stat.txt")
    file_read("stdata/nofuse.4-8.stat.txt")
    file_read("stdata/nofuse.8-8.stat.txt")
    file_read("stdata/nofuse.16-8.stat.txt")
    file_read("stdata/nofuse.4-16.stat.txt")
    file_read("stdata/nofuse.8-16.stat.txt")
    file_read("stdata/nofuse.16-16.stat.txt")
    file_read("stdata/nofuse.4-32.stat.txt")
    file_read("stdata/nofuse.8-32.stat.txt")
    file_read("stdata/nofuse.16-32.stat.txt")
    file_read("stdata/4-64.stat.txt")
    file_read("stdata/8-64.stat.txt")
    file_read("stdata/16-64.stat.txt")
    #file_read("stdata/t9-64-3072.txt")
    #file_read("stdata/tinny-ctlg.txt", 10000, "LPPN")
