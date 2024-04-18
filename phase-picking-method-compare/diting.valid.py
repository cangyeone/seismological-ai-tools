import torch 


import time
import numpy as np 
import torch 
import os 
from utils.data import DitingDataTestThread
from utils.post import find_phase_lppn, find_phase_point2point 

def main(args):
    model_name = args.model  
    if "lppn" in model_name.lower():
        stride = 8 
        find_phase = find_phase_lppn
    else:
        stride = 1
        find_phase = find_phase_point2point
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # detect gpu available 
    model = torch.jit.load(model_name)
    model.to(device)
    data_tool = DitingDataTestThread(file_name=args.input, stride=stride, n_length=6144, padlen=512)
    out_path = args.output 
    outfile = open(out_path, "w", encoding="utf-8") 
    acctime = np.ones([50])
    for step in range(200):
        a1, a2 = data_tool.batch_data(batch_size=800)
        a1 = torch.tensor(a1, dtype=torch.float32, device=device)
        with torch.no_grad():
            time1 = time.perf_counter()
            output = model(a1)
            if stride == 1:
                phase = find_phase(output.cpu().numpy(), height=0.3, dist=50)
            else:
                phase = find_phase([out.cpu().numpy() for out in output], height=0.3, dist=50)
            time3 = time.perf_counter()
            for idx in range(len(a2)):
                pt, st = a2[idx] 
                snr = 1.0 
                outfile.write(f"#phase,{pt},{st},{snr}\n") 
                for p in phase[idx]:
                    outfile.write(f"{p[0]},{p[1]},{p[2]}\n") 
                outfile.flush()
            time2 = time.perf_counter()
            acctime[step%50] = time3 - time1
        tstrs = np.mean(acctime) * 1000 
        print(f"Finished:{step}, {tstrs}")
    data_tool.kill_all()
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with diting")          
    parser.add_argument('-i', '--input', default="data/diting.h5", type=str, help="Path to h5 data")       
    parser.add_argument('-o', '--output', default="odata/diting.rnn.txt", type=str, help="output dir")      
    parser.add_argument('-m', '--model', default="ckpt/diting.rnn.jit", type=str, help="Jit model name")                                                            
    args = parser.parse_args()      
    main(args)
