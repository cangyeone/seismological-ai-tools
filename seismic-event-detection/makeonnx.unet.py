from models.UNet import UNet as Model  
import torch 

class Picker(Model):
    def __init__(self):
        super().__init__() 
    def forward2(self, x):
        #x = x.unsqueeze(3)
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
        #x10 = x10.squeeze(dim=3)
        return x10    
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
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * 1 + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            ot = tgrid.squeeze()
            ot = ot.reshape(-1) 
        return oc, ot   
model = Picker() 
model.eval()
model.load_state_dict(torch.load("ckpt/china.unet.pt", map_location="cpu"))
input_names = ["wave"]
output_names = ["prob", "time"]
#x = torch.randn([10, 3, 6144, 1])
x = torch.randn([500000, 3])
torch.onnx.export(model, x, 
"pickers/unet.onnx", verbose=True, 
dynamic_axes={"wave":{0:"batch"}, "prob":{0:"batch"}, "time":{0:"batch"}}, 
input_names=input_names, output_names=output_names, opset_version=11)