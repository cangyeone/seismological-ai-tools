

class Parameter:
    # 数据设置
    nsamplesdots = 8640000       # 采样点个数
    nchannel = 3                 # 通道数量
    samplerate = 100             # 采样率

    # 拾取设置，仅对onnx模型有用。.jit版模型是在模型里了。
    prob = 0.3                # 置信度
    nmslen = 1000                # NMS间隔设置，1000代表1000采样点间隔
    # 拾取模型数量
    npicker = 1                  # 使用多少模型进行拾取单个模型单日数据需要5G，假设有12G内存，可以设置为2

    # 文件名格式NET.STATION.LOC.CHANNEL.OTHERS.mseed
    namekeyindex = [0, 1]        # 文件名NET.KEY索引
    channelindex = 3             # 文件名数据分量标识CHANNEL索引,全国台站是5 
    filenametag = "sac"        # 文件结尾标志
    chnames = [["BHE", "BHN", "BHZ"], ["SHE", "SHN", "SHZ"]] # 通道分量名1
    #chname2 = ["E", "N", "Z"] # 通道分量名2

    # 输出设置
    ifplot = False               # 是否绘制拾取波形
    ifreal = False               # 是否输出REAL关联数据，以天为间隔
    snritv = 100                 # 输出信噪比截取长度
    bandpass = [1, 10]           # 计算信噪比滤波参数


#par = Parameter() 
