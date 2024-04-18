
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
from utils.data import DitingData, DitingDataThread 
from models.EQT import EQTransformer as EQT, Loss as EQTLoss
from models.BRNN import BRNN, Loss as RNNLoss 
from models.UNet import UNet, Loss as ULoss 
from models.UNetPlusPlus import UNetpp, Loss as UppLoss 
from models.LPPNT import Model as LPPNT, Loss as TLoss 
from models.LPPNM import Model as LPPNM, Loss as MLoss 
from models.LPPNL import Model as LPPNL, Loss as LLoss 
import os 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150


def main(args):
    model_name = args.model  
    if model_name.lower() == "unet":
        Model = UNet
        lossfn = ULoss() 
        stride = 1 
    elif model_name.lower() == "rnn":
        Model = BRNN
        lossfn = RNNLoss()
        stride = 1 
    elif model_name.lower() == "eqt":
        Model = EQT
        lossfn = EQTLoss()
        stride = 1 
    elif model_name.lower() == "lppnt":
        Model = LPPNT
        lossfn = TLoss()
        stride = 8 
    elif model_name.lower() == "lppnm":
        Model = LPPNM
        lossfn = MLoss()
        stride = 8 
    elif model_name.lower() == "lppnl":
        Model = LPPNL
        lossfn = LLoss()
        stride = 8 
    elif model_name.lower() == "unetpp" or model_name.lower() == "unet++":
        Model = UNetpp
        lossfn = UppLoss()
        stride = 1 
    else:
        print("Model name error")
    model = Model()
    data_tool = DitingDataThread(file_name=args.input, stride=stride, n_length=6144, padlen=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # detect gpu available 
    ckpt_path = args.output 
    pt_path = os.path.join(ckpt_path, f"diting.{model_name.lower()}.pt")
    jit_path = os.path.join(ckpt_path, f"diting.{model_name.lower()}.jit")
    onnx_path = os.path.join(ckpt_path, f"diting.{model_name.lower()}.onnx")
    fig_path = os.path.join(ckpt_path, f"diting.{model_name.lower()}.jpg")
    loss_path = os.path.join(ckpt_path, f"diting.{model_name.lower()}.loss")

    if os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.to(device)
    model.train()
    lossfn.to(device)
    acc_time = 0
    outloss = open(loss_path, "a")
    optim = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0e-3)
    for step in range(500000): # Iteration 500k times. 
        st = time.perf_counter()
        a1, a2, a3, a4 = data_tool.batch_data()
        wave = torch.tensor(a1, dtype=torch.float).to(device) 
        if stride>1:
            d = torch.tensor(a2, dtype=torch.float32).to(device)
        else:
            d = torch.tensor(a3, dtype=torch.float32).to(device)
        oc = model(wave)
        loss = lossfn(oc, d) 
        loss.backward()
        if loss.isnan():
            print("NAN error")
            optim.zero_grad()
            continue 
        optim.step() 
        optim.zero_grad()
        ls = loss.detach().cpu().numpy()
        ed = time.perf_counter()
        outloss.write(f"{step},{ed - st},{ls},{data_tool.get_epoch()}\n")
        outloss.flush()
        acc_time += ed - st
        if step % 100 == 0:
            print(f"{acc_time:6.1f}, {step:8}, Loss:{ls:6.1f}")
            torch.save(model.state_dict(), pt_path)
            print(f"PT saved :{jit_path}")
            cp_model = Model() 
            cp_model.load_state_dict(model.state_dict())
            cp_model.eval()
            jitm = torch.jit.script(cp_model)
            torch.jit.save(jitm, jit_path)
            print(f"JIT saved :{jit_path}")  
            if args.onnx:
                input_names = ["wave"]
                output_names = ["prob"]
                dynamicout = {"wave":{0:"B"}, "prob":{0:"B"}}
                if stride>1: 
                    output_names = ["prob", "time"]
                    dynamicout = {"wave":{0:"B"}, "prob":{0:"B"}, "time":{0:"B"}}
                    cp_model.fuse_model() 
                dummy_input = torch.randn([10, 3, 6144])
                torch.onnx.export(cp_model, dummy_input, 
                onnx_path, 
                verbose=True, input_names=input_names, 
                output_names=output_names, dynamic_axes=dynamicout, opset_version=12)     
                print(f"ONNX saved :{onnx_path}")                  
            if args.plot:
                gs = gridspec.GridSpec(3, 1) 
                fig = plt.figure(1, figsize=(16, 16), dpi=100) 
                if stride>1:
                    p = oc[0].detach().cpu().numpy()[0]
                else:
                    p = oc.detach().cpu().numpy()[0]
                w = a1[0, 0, :]
                d = d.detach().cpu().numpy()[0]
                w /= np.max(w)
                print(p.shape, d.shape)
                for i in range(3):
                    ax = fig.add_subplot(gs[i, 0])
                    #ax.plot(d[i, :], alpha=0.5, c="b") 
                    ax.plot(np.repeat(p[i, :], stride), alpha=0.5, c="r")
                    ax.plot(w, alpha=0.3, c="k")
                plt.savefig(fig_path) 
                plt.cla() 
                plt.clf()
            acc_time = 0 
    data_tool.kill_all()
    print("done!")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with diting")          
    parser.add_argument('-i', '--input', default="data/diting.h5", type=str, help="Path to h5 data")       
    parser.add_argument('-o', '--output', default="ckpt/", type=str, help="output dir")      
    parser.add_argument('-m', '--model', default="unetpp", type=str, 
                choices=["rnn", "eqt", "unet", "unetpp", "lppnt", "lppnm", "lppnl", "RNN", "EQT", "UNet", "UNet++", "LPPNT", "LPPNM", "LPPNL"], help="Train model name")      
    parser.add_argument('--onnx', default=True, type=bool, help="Whether out put onnx ckpt")       
    parser.add_argument('--plot', default=True, type=bool, help="Whether plot training figure")                                                       
    args = parser.parse_args()      
    main(args)
