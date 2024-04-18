import torch 
import torch.nn as nn 
from models.BRNNPNSN import BRNN 
class Picker(BRNN):
    def __init__(self):
        super().__init__()
        self.n_stride = 1 
    def forward(self, x):
        device = x.device
        with torch.no_grad():
            #print("数据维度", x.shape)
            T, C = x.shape 
            seqlen = 10240 
            batchstride = 10240 - 512
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
            idx = idx.clamp(min=0, max=T-1).long()
            x = x.to(device)
            wave = x[idx, :] 
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            max = torch.std(wave, dim=2, keepdim=True)
            #max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
            wave /= (max + 1e-6)  
            #print(wave.shape)
            x = self.encoder(wave)
            e = self.rnns(x)     # 波形特征
            y = self.decoder(e)  # 输出概率
            oc = y.softmax(dim=1)
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            ot = tgrid.squeeze()
            ot = ot.reshape(-1)
            output = []
            #print("NN处理完成", oc.shape, ot.shape)
            # 接近非极大值抑制（NMS） 
            # .......P........S...... 
            for itr in range(4):
                pc = oc[:, itr+1] 
                time_sel = torch.masked_select(ot, pc>0.1)
                score = torch.masked_select(pc, pc>0.1)
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
                    selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>300)
                    nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>300)
                    ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>300)
                p_time = torch.masked_select(time_sel[order], select>0.0)
                p_prob = torch.masked_select(score[order], select>0.0)
                p_type = torch.ones_like(p_time) * itr 
                y = torch.stack([p_type, p_time, p_prob], dim=1)
                output.append(y) 
            y = torch.cat(output, dim=0)
        return y 

model = Picker() 
model.load_state_dict(torch.load("ckpt/china.rnn.pnsn.pt", map_location="cpu"))
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/rnn.pnsn.01.jit")
x = torch.randn([300000, 3])
y = model(x) 