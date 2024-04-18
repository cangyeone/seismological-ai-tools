import h5py 

file_ = h5py.File("data/diting.h5", "r")
train = file_["train"]
for key in train:
    print(f"Event:{key}")
    subdata = train[key]
    for skey in subdata:
        data = subdata[skey]
        print(f"  |-Station{skey}-Data:{data.shape}")
        for att in data.attrs:
            print(f"  |  |--{att}:{data.attrs[att]}")
    break 