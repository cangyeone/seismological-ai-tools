import h5py 
import tqdm 
import numpy as np 
outfile = h5py.File("h5data/both.ps.h5", "w")
phase_dict = {
            "Pg":0, 
            "Sg":1, 
            "P":0, 
            "S":1, 
            "Pn":2, 
            "Sn":3
        } 
for year in [2009+i for i in range(11)]:
    h5file = h5py.File(f"h5data/{year}.h5", "r")
    for ekey in tqdm.tqdm(h5file):
        if ekey in outfile:continue 
        event = h5file[ekey] 
        group = outfile.create_group(ekey) 
        group.attrs["time"] = event.attrs["time"]
        group.attrs["lat"] = event.attrs["lat"] 
        group.attrs["lon"] = event.attrs["lon"] 
        group.attrs["mag"] = event.attrs["mag"] 
        group.attrs["depth"] = event.attrs["depth"]
        group.attrs["type"] = event.attrs["type"] 
        group.attrs["strs"] = event.attrs["strs"] 
        for skey in event:
            station = event[skey] 
            lendata = 0
            for dkey in station:
                lendata += 1
            if lendata<3:continue 
            phase_count = {"P":0, "S":0}
            for akey in station.attrs:
                pname = ""
                if "Pg" in akey:
                    pname = akey.split(".")[-1]
                    if pname == "Pg":
                        phase_count["P"] += 1
                else:
                    if akey in phase_dict:
                        pname = akey 
                        if pname == "P" or pname == "Pg":
                            phase_count["P"] += 1
                        if pname == "S" or pname == "Sg":
                            phase_count["S"] += 1 
            if phase_count["P"] !=0 and phase_count["S"]!=0:
                subgroup = group.create_group(skey)
                for akey in station.attrs:
                    subgroup.attrs[akey] = station.attrs[akey] 
                for dkey in station:
                    dataset = subgroup.create_dataset(dkey, data=station[dkey][:].astype(np.int32), dtype="i4", chunks=True, compression="gzip")          

#nohup python utils/merge.py > logdir/merge.log.file 2>&1 &
#1751072
#sftp://wangwt@10.2.202.106/home/wangwt/mkhdf5AI/h5data/2020.h5