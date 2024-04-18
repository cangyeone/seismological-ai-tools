import h5py 
import re 
import matplotlib.pyplot as plt 
import tqdm 
import os 
def readcsv(filename):
    file_ = open(filename, "r", encoding="utf-8") 
    header = file_.readline()
    shead = header.strip().split(",")
    maginfos = {}
    for line in file_.readlines():
        sline = [i.replace(" ", "") for i in line.strip().split(",")]
        info = {}
        for hidx, head in enumerate(shead):
            if "key" == head:
                pkey = sline[hidx] 
                ekey = pkey.split(".")[0]
            if "snr" in head:
                try:
                    info[head] = float(sline[hidx])
                except:
                    info[head] = -12345
            if "baz" in head:
                try:
                    info[head] = float(sline[hidx])
                except:
                    info[head] = -12345   
            if "residual" in head:
                try:
                    info[head] = float(sline[hidx])
                except:
                    info[head] = -12345  
            if "pick" in head:
                try:
                    info[head] = float(sline[hidx])
                except:
                    info[head] = -12345                  
            if "evmag" in head:
                try:
                    info[head] = float(sline[hidx])
                except:
                    info[head] = -12345    
            if "mag_type" in head:
                info[head] = sline[hidx] 
            if head not in info and len(head)>0:
                info[head] = sline[hidx]
        maginfos[pkey] = info 
    return maginfos 

def main(args):
    outfile = h5py.File(args.output, "w")
    trainf = outfile.create_group("train") 
    validf = outfile.create_group("test")
    datadir = args.input 
    csv_files = []# get csv files 
    h5_files = [] # get h5 files
    for fn in os.listdir(datadir):
        if fn.endswith(".hdf5"):
            h5_files.append(os.path.join(datadir, fn))
        if fn.endswith(".csv") and "_part_" in fn:
            csv_files.append(os.path.join(datadir, fn))
    phase_infos = {} 
    for csv in csv_files:
        infos = readcsv(csv)
        for k, v in infos.items():
            phase_infos[k] = v 
    n_sample = len(phase_infos)
    n_valid = int(n_sample*args.split) # get the number of training samples
    print(f"Number of samples:{n_sample}")
    nitrs = 0 
    for h5 in h5_files:
        file_ = h5py.File(h5, "r")
        datas = file_["earthquake"]
        for i, key in tqdm.tqdm(enumerate(datas), desc=f"File:{h5}"):
            data = datas[key] 
            if key not in phase_infos:continue 
            info = phase_infos[key]
            if nitrs < n_valid:
                group = trainf
            else:
                group = validf
            ekey, skey = key.split(".") 
            if ekey not in group:
                subgroup = group.create_group(ekey)
                if args.compression:
                    dt = subgroup.create_dataset(key, data=data, compression="gzip" )
                else:
                    dt = subgroup.create_dataset(key, data=data)
            else:
                subgroup = group[ekey]
                if args.compression:
                    dt = subgroup.create_dataset(key, data=data, compression="gzip" )
                else:
                    dt = subgroup.create_dataset(key, data=data)
            for key in info:
                dt.attrs[key] = info[key]
            nitrs += 1
        file_.close()
    outfile.close()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make HDF5 file")          
    parser.add_argument('-i', '--input', default="data/", type=str, help="Path to diting")       
    parser.add_argument('-o', '--output', default="data/diting.h5", type=str, help="Output name")     
    parser.add_argument('-s', '--split', default=0.9, type=float, help="Partion of training data")         
    parser.add_argument('-c', '--compression', default=True, type=bool, help="Compression")                                                       
    args = parser.parse_args()      
    main(args)