

class Parameter:
    # GPU计算参数
    win_length = 86400
    win_stirde = 86400 
    ngrid = 200   # 网格大小
    lonrange = [] # 经纬度范围
    latrange = []
    locpad = 0.2 # 若未指定经纬度则在数据范围基础上加pad范围
    datadir = "gamma"
    saveitv = 86400 # 每隔多少步保存一个文件
    vel = {"p": 6.0, "s": 6.0 / 1.73}
    dbscan_eps = 10 
    dbscan_min_samples = 3
    min_picks_per_eq = 5 
    oversample_factor = 0.5 
    dims = ["x(km)", "y(km)", "z(km)"]
    max_sigma11 = 2 
    max_sigma12 = 1 
    max_sigma22 = 1 
    useamp = False 
    min_p_picks_per_eq = 3
    min_s_picks_per_eq = 2
    
    # 模型相关文件
    modeldir = "ckpt/link32.ckpt"
    basetime = "2020-01-01" # 计算相对时间

    # 关联参数
    np = 3 
    ns = 0 
    nps = 3 
    nboth = 0 

