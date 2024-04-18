import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec 
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec 
import scipy.signal as signal 
import os 
from matplotlib.ticker import FuncFormatter
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.switch_backend("agg") 

def file_read(file_name="lppn-ctlg.txt", dist=100, gap1=0.1, gap2=0.5, axs=[], mname=""):
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
        if len(phases[0])==0:continue 
        if "#none" not in phases[0][0]:
            mm_time = [float(phases[0][1]), float(phases[0][2])]
            mm_snr = float(phases[0][3])
        allpick = [False, False]
        for nn_pick in phases[1:]:
            nn_type = int(float(nn_pick[0]))
            nn_time = float(nn_pick[1]) 
            if "#none" not in phases[0][0]:
                ref_time = mm_time[nn_type-1] 
                if ref_time != -1:
                    error = np.abs(ref_time-nn_time) / 50
                    if error < gap1:
                        tp[nn_type-1] += 1 
                        allpick[nn_type-1] = True 
                    else:
                        fp[nn_type-1] += 1 
                    if error < gap2:
                        std[nn_type-1].append((ref_time-nn_time)/50) 
                else:
                    fp[nn_type-1] += 1 
        for tps in range(len(tp)):
            if allpick[tps] == False:
                fn[tps] += 1
    name = ["P", "S"]
    P = [0 for itr in range(2)] 
    R = [0 for itr in range(2)]
    for tps in range(2):
        P[tps] = tp[tps]/(tp[tps]+fp[tps]+1e-9)
        R[tps] = tp[tps]/(tp[tps]+fn[tps]+1e-9)

    print("样本数量", file_name, len(phase_list))
    nm = os.path.split(file_name)[-1] 
    nm = nm.split(".")[1]
    colors = ["#ff0000", "#0000ff"]
    lims = [[0, 80000], [0, 50000]]
    for tps in range(2):
        ax = axs[tps]
        dts = std[tps]
        if len(dts)==0:
            continue
        def formatnum(x, pos):
            return '$%.1fe{4}$' % (x/10000)
        #formatter = FuncFormatter(formatnum)
        #ax.yaxis.set_major_formatter(formatter)
        ax.hist(dts, bins=36, color=colors[tps], alpha=0.5) 
        #ax.set_title(f"{nm}-{name[tps]}") 
        ax.set_title(f"M:{mname}\nPhase:{name[tps]}\nP={P[tps]:.3f}\nR={R[tps]:.3f}\nF1={P[tps]*R[tps]*2/(P[tps]+R[tps]):.3f}\n$\mu$={np.mean(std[tps])*1e3:.3f}ms\n$\sigma$={np.std(std[tps])*1e3:.3f}ms", fontsize=10, x=0.01, y=0.95, ha="left", va="top")
        ax.set_xlabel("Error [s]")  
        ax.set_xlim([-gap2, gap2])
        ax.set_ylim(lims[tps])
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
        #ax.set_ylim([0, 16])
        #ax.text(0.1, 3, f"{gap1},{gap2}\nP={P[tps]:.3f}\nR={R[tps]:.3f}\n$\mu={np.mean(std[tps])*1e3:.3f}$\n$\sigma={np.std(std[tps])*1e3:.3f}$") 
        #ax.set_title(net_name)
    
    #plt.savefig(f"fig/{name[-1]}.svg")
    #print(f"S:P={s_tp/(s_tp+s_fp)}, R={s_tp/(s_tp+s_fn)}")

if __name__ == "__main__":
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure(1, figsize=(16, 16), dpi=100)
    #file_read("stdata/tinny16-2-ctlg.txt", 10000, "LPPN")


    axs = []
    for i in range(1):
        for j in range(1):
            ax1, ax2 = fig.add_subplot(gs[i, j]), fig.add_subplot(gs[i, j+1]) 
            axs.append([ax1, ax2])
    gap1, gap2 = 0.5, 1.5
    #file_read("stdata/stdt/eqt.pt.txt", gap1=gap1, gap2=gap2, axs=axs[0])

    file_read("odata/diting.rnn.txt",   gap1=gap1, gap2=gap2, axs=axs[0], mname="RNN")

    plt.savefig(f"odata/diting.rnn.jpg")
    plt.savefig(f"odata/diting.rnn.svg")
