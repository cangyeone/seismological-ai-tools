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
        return oc, ot   
model = Picker() 
model.eval()
model.load_state_dict(torch.load("ckpt/china.unetpp.pt", map_location="cpu"))
input_names = ["wave"]
output_names = ["prob", "time"]
#x = torch.randn([10, 3, 6144, 1])
x = torch.randn([500000, 3])
torch.onnx.export(model, x, 
"pickers/unetpp.onnx", verbose=True, 
dynamic_axes={"wave":{0:"batch"}, "prob":{0:"batch"}, "time":{0:"batch"}}, 
input_names=input_names, output_names=output_names, opset_version=11)