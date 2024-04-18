import numpy as np
import onnxruntime as ort
import obspy # pip install obspy 

def post(prob, time, a=0.1, b=200):
    # ab分别是概率阈值和避免重复的间隔
    # ONNX模型需要后处理
    output = []
    for itr in range(2):
        pc = prob[:, itr+1] 
        time_sel = time[pc>a]
        score = pc[pc>a]
        order = np.argsort(score)[::-1]
        ntime = time_sel[order] 
        nprob = score[order]
        #print(batchstride, ntime, nprob)
        select = -np.ones_like(order)
        selidx = np.arange(len(order))
        count = 0
        while True:
            if len(nprob)<1:
                break 
            ref = ntime[0]
            idx = selidx[0]
            select[idx] = 1 
            count += 1 
            #print(ref, nprob[idx])
            kidx = np.abs(ref-ntime)>b
            selidx = selidx[kidx] 
            nprob = nprob[kidx] 
            ntime = ntime[kidx]
        p_time = time_sel[order][select>0] 
        p_prob = score[order][select>0] 
        p_type = np.ones_like(p_time) * itr 
        y = np.stack([p_type, p_time, p_prob], axis=1)
        output.append(y) 
    if len(output) == 0:
        return []
    y = np.concatenate(output, axis=0) 
    return y    
        
mname = "pickers/lppnm.onnx" # 其他onnx均可
sess = ort.InferenceSession(mname, providers=['CPUExecutionProvider'])#使用pickers中的onnx文件

# 读取数据
st1 = obspy.read("data/waveform/X1.53085.01.BHE.D.20122080726235953.sac")
st2 = obspy.read("data/waveform/X1.53085.01.BHN.D.20122080726235953.sac")
st3 = obspy.read("data/waveform/X1.53085.01.BHZ.D.20122080726235953.sac")
data = [st1[0].data, st2[0].data, st3[0].data] 
# 任意长度数据均可
# 数据不需要滤波、预处理、归一化等操作
x = np.stack(data, axis=1).astype(np.float32)[:7550] #[N, 3]->一天 [8640000]100Hz 
# 直接运行即可
prob, time = sess.run(["prob", "time"], {"wave":x})
phase = post(prob, time, 0.2, 200)
import matplotlib.pyplot as plt 
plt.plot(x[:, 2], alpha=0.5) 
plt.scatter(time, prob[:, 1]*np.max(x[:, 2]), c="r")
plt.scatter(time, prob[:, 2]*np.max(x[:, 2]), c="b")
for pha in phase:
    if pha[0]==0:
        c = "r" 
    else:
        c = "b"
    plt.axvline(pha[1], c=c)
plt.show()
