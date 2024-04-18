

class Parameter:
    # 数据设置
    #nsamplesdots = 8640000       # 采样点个数用不到了
    nchannel = 3                 # 通道数量
    samplerate = 100             # 采样率

    # 拾取设置，仅对onnx模型有用。.jit版模型是在模型里了。
    prob = 0.3                   # 置信度，仅在onnx模型有用
    nmslen = 1000                # NMS间隔设置，1000代表1000采样点间隔
    npicker = 1                  # 使用多少模型进行拾取单个模型单日数据需要5G，假设有12G内存，可以设置为2
    npre = 2                     # 使用多个进程对数据进行读取等
    # 是否处理的数据
    is_seed = True #如果True则可以读取同一文件中的多道数据多道数据
    filenametag = ".mseed"        # 文件扩展名，在seed文件也会用到
    #SC.A0801.40.EIE.D.20221400520064953.sac
    # 文件名格式NET.STATION.LOC.CHANNEL.OTHERS.mseed
    namekeyindex = [0, 1]        # 文件名NET.KEY索引
    channelindex = 3              # 文件名数据分量标识CHANNEL索引,全国台站是5 
    
    chnames = [["BHE", "BHN", "BHZ"], ["SHE", "SHN", "SHZ"], ["HHE", "HHN", "HHZ"], ["EIE", "EIN", "EIZ"], ["HNN", "HNE", "HNZ"]] # 通道分量名，所有可能的都需要，Z要在最后用于算初动方向
    polar = False #是否输出初动方向
    # 是否输出初动极性
    polar = True 
    # 输出设置
    ifplot = False               # 是否绘制拾取波形
    ifreal = False               # 是否输出REAL关联数据，以天为间隔
    snritv = 100                 # 输出信噪比截取长度
    bandpass = [1, 10]           # 计算信噪比滤波参数


#par = Parameter() 