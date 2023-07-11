from models.LPPNM import Model 
import torch 

modelname = "lppnm"
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
        return oc, ot 

model = Picker() 
model.load_state_dict(torch.load(f"ckpt/china.{modelname}.pt", map_location="cpu"))
model.eval()
input_names = ["wave"]
output_names = ["prob", "time"]
#x = torch.randn([10, 3, 6144, 1])
x = torch.randn([500000, 3])
torch.onnx.export(model, x, 
f"pickers/{modelname}.onnx", verbose=True, 
dynamic_axes={"wave":{0:"batch"}, "prob":{0:"batch"}, "time":{0:"batch"}}, 
input_names=input_names, output_names=output_names, opset_version=11)
