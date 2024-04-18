import torch  
import torch.nn as nn 
import torch.nn.functional as F 

class Conv2d(nn.Module):
    def __init__(self, nin=8, nout=11, ks=[7, 1], st=[4, 1], padding=[3, 0]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nin, nout, ks, st, padding=padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class Conv2dT(nn.Module):
    def __init__(self, nin=8, nout=11, ks=[7, 1], st=[4, 1], padding=[3, 0], output_padding=[3, 0]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, ks, st, padding=padding, output_padding=output_padding), 
            nn.BatchNorm2d(nout), 
            nn.ReLU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 
class UNet(nn.Module):
    def __init__(self):
        super().__init__() 
        
        self.inputs = Conv2d(3, 8, [7, 1], [1, 1], padding=[3, 0]) 
        self.layer0 = Conv2d(8, 8, [7, 1], [1, 1], padding=[3, 0]) 
        self.layer1 = Conv2d(8, 16, [7, 1], [4, 1], padding=[3, 0])
        self.layer2 = Conv2d(16, 32, [7, 1], [4, 1], padding=[3, 0])
        self.layer3 = Conv2d(32, 64, [7, 1], [4, 1], padding=[3, 0]) 
        self.layer4 = Conv2d(64, 128, [7, 1], [4, 1], padding=[3, 0]) 
        self.layer5 = Conv2dT(128, 64, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer6 = Conv2dT(128, 32, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer7 = Conv2dT(64, 16, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer8 = Conv2dT(32, 8, [7, 1], [4, 1], padding=[3, 0], output_padding=[3, 0])
        self.layer9 = nn.Conv2d(16, 3, [7, 1], [1, 1], padding=[3, 0])
    def forward(self, x):
        x = self.inputs(x)
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4) 
        x6 = self.layer5(x5)
        x6 = torch.cat([x4, x6], dim=1) 
        x7 = self.layer6(x6)
        x7 = torch.cat([x3, x7], dim=1) 
        x8 = self.layer7(x7)
        x8 = torch.cat([x2, x8], dim=1) 
        x9 = self.layer8(x8)
        x9 = torch.cat([x1, x9], dim=1) 
        x10 = self.layer9(x9)
        x10 = F.softmax(x10, dim=1)
        x10 = x10.squeeze()
        return x10
        
import utils 
import time 
import numpy as np 
import scipy.signal as signal  
#import tensorflow as tf 


def find_phase(pred, height=0.80, dist=1):
    shape = np.shape(pred) 
    all_phase = []
    phase_name = {0:"N", 1:"P", 2:"S"}
    for itr in range(shape[0]):
        phase = []
        for itr_c in [0, 1]:
            p = pred[itr, :, itr_c+1] 
            #p = signal.convolve(p, np.ones([10])/10., mode="same")
            h = height 
            peaks, _ = signal.find_peaks(p, height=h, distance=dist) 
            for itr_p in peaks:
                phase.append(
                    [
                        itr_c+1, #phase_name[itr_c], 
                        itr_p, 
                        pred[itr, itr_p, itr_c], 
                        itr_p
                    ]
                    )
        all_phase.append(phase)
    return all_phase 
def main(args):
    device = torch.device("cuda")
    model = UNet()
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load("ckpt/phasenet.wave"))

    data_tool = utils.DataTest(batch_size=100, n_length=3072)
    outfile = open("stdata/phasenet2.txt", "w")
    datalen = 3072
    acctime = 0 
    for step in range(400):
        a1, a2, a3, a4 = data_tool.batch_data()
        time1 = time.perf_counter()

        with torch.no_grad():
            a1 = torch.tensor(a1, dtype=torch.float32, device=device)
            a1 = a1.permute(0, 2, 1).unsqueeze(dim=3)
            oc = model(a1)
            oc = oc.permute(0, 2, 1)
            oc = oc.cpu().numpy()
            phase = find_phase(oc[:, :, :], height=0.3, dist=500)
        time2 = time.perf_counter()    
        for idx in range(len(a2)):
            is_noise = a2[idx] 
            pt, st = a4[idx] 
            snr = np.mean(a3[idx]) 
            if pt<0 or st<0:
                continue 
            if is_noise:
                outfile.write("#none\n")
            else:
                if st > datalen:
                    outfile.write(f"#phase,{pt},{-100},{snr}\n") 
                else:
                    outfile.write(f"#phase,{pt},{st},{snr}\n") 
            for p in phase[idx]:
                outfile.write(f"{p[0]},{p[1]},{p[2]}\n") 
            outfile.flush()
        if step !=0:
            acctime += time2-time1 
        print(step, f"{time2-time1},{acctime}")

import argparse
if __name__ == "__main__":
    main([])