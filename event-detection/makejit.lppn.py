from models.LPPNM import Model 
import torch 

modelname = "lppnl"
if modelname == "lppnt":
    from models.LPPNT import Model
elif modelname == "lppnm":
    from models.LPPNM import Model
elif modelname == "lppnl":
    from models.LPPNL import Model
class Picker(Model):
    def __init__(self, n_stride=8):
        super().__init__()
    def forward(self, x):
        self.n_stride = 8
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
            max1, max1idx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
            max2, max2idx = torch.max(max1, dim=2, keepdim=True) 
            wave /= (max2 + 1e-6)  
            wave = wave.unsqueeze(2)
            x1 = self.layers(wave)
            x2 = self.class_encoder(x1)
            x = torch.cat([x1, x2], dim=1)
            out_class = self.cl(x).squeeze(dim=2)
            out_time = self.tm(x)
            out_time = out_time.sigmoid().squeeze() * self.n_stride

            oc = out_class.squeeze()
            ot = out_time.squeeze()
            #print(oc.shape, ot.shape)
            B, C, T = oc.shape 
            oc = oc.softmax(dim=1)
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            ot += tgrid.squeeze()
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
model.load_state_dict(torch.load(f"ckpt/china.{modelname}.pt", map_location="cpu"))
model.eval()
jitmodel = torch.jit.script(model) 
torch.jit.save(jitmodel, f"pickers/{modelname}.jit") 
x = torch.randn([50000, 3])
y = jitmodel(x)
