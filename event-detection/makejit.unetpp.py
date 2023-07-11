import torch 
import torch.nn as nn 
from models.UNetPlusPlus import UNetpp 
class Picker(UNetpp):
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
            #x = wave.unsqueeze(3)
            x0_0 = self.conv0_0(wave)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

            x2_0 = self.conv2_0(self.pool(x1_0))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

            output = self.final(x0_4)
            oc = self.sigmoid(output)
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            ot = tgrid.squeeze()
            ot = ot.reshape(-1)
            output = []
            #print("NN处理完成", oc.shape, ot.shape)
            # 接近非极大值抑制（NMS） 
            # .......P........S...... 
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
model.load_state_dict(torch.load("ckpt/china.unetpp.pt", map_location="cpu"))
model.eval()
torch.jit.save(torch.jit.script(model), "pickers/unetpp.jit")
x = torch.randn([300000, 3])
y = model(x) 