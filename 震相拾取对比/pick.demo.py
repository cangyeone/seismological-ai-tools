import torch 
import torch.nn as nn 
from models.UNet import UNet 
class Picker(UNet):
    def __init__(self):
        super().__init__()
        self.n_stride = 1 
    def forward(self, x):
        device = x.device
        with torch.no_grad():
            #print("数据维度", x.shape)
            T, C = x.shape 
            seqlen = 6144 
            batchstride = 6144 - 256
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
            idx = idx.clamp(min=0, max=T-1).long()
            x = x.to(device)
            wave = x[idx, :] 
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
            wave /= (max + 1e-6)  
            ### 原始forward函数
            x = wave.unsqueeze(3)
            x = self.inputs(x)
            x1 = self.layer0(x)
            x2 = self.layer1(x1)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4) 
            x6 = self.layer5(x5)
            x6 = torch.cat([x4, x6], dim=1) # 加入skip connection
            x7 = self.layer6(x6)
            x7 = torch.cat([x3, x7], dim=1) # 加入skip connection
            x8 = self.layer7(x7)
            x8 = torch.cat([x2, x8], dim=1) # 加入skip connection
            x9 = self.layer8(x8)
            x9 = torch.cat([x1, x9], dim=1) # 加入skip connection
            x10 = self.layer9(x9)
            x10 = x10.softmax(dim=1)
            oc = x10.squeeze(dim=3)
            # 计算结束
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            ot = tgrid.squeeze()
            ot = ot.reshape(-1)
            output = []
            #print("NN处理完成", oc.shape, ot.shape)
            for itr in range(2):
                pc = oc[:, itr+1] 
                time_sel = torch.masked_select(ot, pc>0.3)
                score = torch.masked_select(pc, pc>0.3)
                _, order = score.sort(0, descending=True)    # 降序排列
                ntime = time_sel[order] 
                nprob = score[order]
                #print(batchstride, ntime, nprob)
                select = -torch.ones_like(order)
                selidx = torch.arange(0, order.numel(), 1, dtype=torch.long, device=device) 
                count = 0
                while True:
                    if nprob.numel()<1:
                        break 
                    ref = ntime[0]
                    idx = selidx[0]
                    select[idx] = 1 
                    count += 1 
                    selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>1000)
                    nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>1000)
                    ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>1000)
                p_time = torch.masked_select(time_sel[order], select>0.0)
                p_prob = torch.masked_select(score[order], select>0.0)
                p_type = torch.ones_like(p_time) * itr 
                y = torch.stack([p_type, p_time, p_prob], dim=1)
                output.append(y) 
            y = torch.cat(output, dim=0)
        return y 

model = Picker() 
torch.jit.save(torch.jit.script(model), "unet.jit.temp")
x = torch.randn([300000, 3])
y = model(x) 
print(y)