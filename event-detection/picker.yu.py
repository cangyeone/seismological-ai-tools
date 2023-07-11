import numpy as np
import torch 
import obspy # pip install obspy 
import matplotlib.pyplot as plt 
def main():
    mname = "pickers/rnn.jit" # 其他onnx均可
    device = torch.device("cpu") #可设置其他设备
    sess = torch.jit.load(mname)
    sess.eval() # 推断模型
    sess.to(device)

    st = obspy.read("data/SC.202305230113.0001.seed")
    event = {}
    event_ch = {}
    for tr in st:
        stats = tr.stats 
        key = f"{stats.network}.{stats.station}"
        channel = stats.channel
        if key in event:
            event[key].append(tr.data) 
            event_ch[key].append(channel) 
        else:
            event[key] = [tr.data]
            event_ch[key] = [channel] 
    count = 0 
    for key in event:
        station = event[key] 
        chs = event_ch[key] 
        for idx, ch in enumerate(chs):
            if "Z" in ch:
                zidx = idx 
        if len(station)!=3:continue 
        x = np.stack(station, axis=1)
        x = x.astype(np.float32)
        w = x[:, zidx]
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=device) 
            y = sess(x) 
            phase = y.cpu().numpy()# 后处理在文件中
        w -= np.mean(w) 
        w /= np.max(np.abs(w))
        plt.plot(w+count * 2.5, alpha=1, lw=0.5, c="k") 
        for pha in phase:
            if pha[0]==0:
                c = "#ff0000" 
            elif pha[0]==1:
                c = "#0000ff"
            elif pha[0]==2:
                c = "#00ff00"
            elif pha[0]==3:
                c = "#000000"
            else:
                continue 
            plt.plot([pha[1], pha[1]], [count * 2.5+1, count * 2.5-1], c=c)
        count += 1
    plt.scatter([], [], marker="|", color="#ff0000", label="Pg")
    plt.scatter([], [], marker="|", color="#0000ff", label="Sg")
    plt.scatter([], [], marker="|", color="#00ff00", label="Pn")
    plt.scatter([], [], marker="|", color="#000000", label="Sn")
    plt.legend(loc="upper right")
    plt.show()
main()