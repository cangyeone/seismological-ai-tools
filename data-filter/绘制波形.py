import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.size'] = 24
import utils 
import os 
import numpy as np 
def main(args):
    ##name = "NSZONE-all-gzip2"
    ##if os.path.exists(f"h5data/{name}")==False:
    ##    os.makedirs(f"h5data/{name}")
    #data_tool = data.DataHighSNR(file_name="data/2019gzip.h5", stride=8, n_length=5120, padlen=256, maxdist=300)
    dist = 500
    model_name = f"ckpt/denoise.pt"
    data_tool = utils.Data(batch_size=32, n_thread=1, strides=8, n_length=3072)
    a1, a2 = data_tool.batch_data()
    gs = gridspec.GridSpec(3, 1) 
    fig = plt.figure(1, figsize=(16, 16), dpi=100) 
    w = a1[0, :, 0]
    d = a2[0, :, 0] 
    t = np.arange(len(w)) * 0.01
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, w, alpha=1, lw=1, c="k", label="噪声+波形")
    ax.set_ylim((-1.2, 1.2))
    ax.legend(loc="upper right")
    ax = fig.add_subplot(gs[1, 0])
    m =np.ones_like(d) 
    m[200:400] = 0 
    ax.plot(t, d*m, alpha=1, lw=1, c="k", label="缺失+波形")
    ax.set_ylim((-1.2, 1.2))
    ax.legend(loc="upper right")
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, d, alpha=1, lw=1, c="k", label="完整波形")
    ax.set_ylim((-1.2, 1.2))
    ax.set_xlabel("时间/秒")
    ax.legend(loc="upper right")
    plt.savefig("logdir/噪声波形.jpg")
    plt.savefig("logdir/噪声波形.svg")
    plt.savefig("logdir/噪声波形.pdf")
if __name__ == "__main__":
    main("abc")

