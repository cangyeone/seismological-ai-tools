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
            batchstride = seqlen - 512
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
            idx = idx.clamp(min=0, max=T-1).long()
            x = x.to(device)
            wave = x[idx, :] 
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            max = torch.std(wave, dim=2, keepdim=True) 
            wave /= (max + 1e-6)  
            x = self.encoder(wave)
            e = self.rnns(x)     # 波形特征
            y = self.decoder(e)  # 输出概率
            oc = y.softmax(dim=1)
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            ot = tgrid.squeeze()
            ot = ot.reshape(-1) 
        return oc, ot   
model = Picker() 
model.eval()
model.load_state_dict(torch.load("ckpt/china.rnn.pnsn.pt", map_location="cpu"))
input_names = ["wave"]
output_names = ["prob", "time"]
#x = torch.randn([10, 3, 6144, 1])
x = torch.randn([500000, 3])
torch.onnx.export(model, x, 
"pickers/rnn.pnsn.onnx", verbose=True, 
dynamic_axes={"wave":{0:"batch"}, "prob":{0:"batch"}, "time":{0:"batch"}}, 
input_names=input_names, output_names=output_names, opset_version=11)