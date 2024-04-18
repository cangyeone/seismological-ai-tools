import numpy as np
import torch 
import obspy # pip install obspy 

mname = "pickers/rnn.pnsn.jit" # 其他onnx均可
device = torch.device("cpu") #可设置其他设备
sess = torch.jit.load(mname)
sess.eval() # 推断模型
sess.to(device)

# 读取数据
#st1 = obspy.read("data/waveform/X1.53085.01.BHE.D.20122080726235953.sac")
#st2 = obspy.read("data/waveform/X1.53085.01.BHN.D.20122080726235953.sac")
#st3 = obspy.read("data/waveform/X1.53085.01.BHZ.D.20122080726235953.sac")
#data = [st1[0].data, st2[0].data, st3[0].data] 
# 任意长度数据均可
# 数据不需要滤波、预处理、归一化等操作
data = np.random.random([3, 200000])
x = np.stack(data, axis=1).astype(np.float32) #[N, 3]->一天 [8640000]100Hz 
# 需要将数据转换为torch格式
with torch.no_grad():
    x = torch.tensor(x, dtype=torch.float32, device=device) 
    y = sess(x) 
    phase = y.cpu().numpy()# 后处理在文件中
import matplotlib.pyplot as plt 
plt.plot(x[:, 2], alpha=0.5) 
for pha in phase:
    if pha[0]==0:
        c = "r" 
    else:
        c = "b"
    plt.axvline(pha[1], c=c)
plt.show()
