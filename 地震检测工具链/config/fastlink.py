

class Parameter:
    # GPU计算参数
    win_length = 30.0 
    win_stirde = 1 
    ngrid = 200   # 网格大小
    lonrange = [] # 经纬度范围
    latrange = []
    locpad = 0.2 # 若未指定经纬度则在数据范围基础上加pad范围
    datadir = "fastdata"
    saveitv = 86400 # 每隔多少步保存一个文件
    
    # 模型相关文件
    modeldir = "ckpt/link32.ckpt"
    basetime = "2020-01-01" # 计算相对时间

    # 关联参数
    np = 3 
    ns = 0 
    nps = 3 
    nboth = 0 
