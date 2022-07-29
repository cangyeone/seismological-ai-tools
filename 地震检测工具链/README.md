### 工具链说明
处理流程：
1. 震相拾取（已完成）
2. 震相关联（已完成）
3. 地震精定位（0%完成）

系统环境
- 操作系统不限
- 基础环境：python3.6 或者以上版本
- 库环境版本不限
  - matplotlib 
  - pytorch 用于拾取和关联
  - obspy 用于数据读取
  - numpy,scipy 用于基础的数据存取
  - tqdm 用于显示进度条

附带预训练模型在ckpt文件夹下
- *.wave 用于震相拾取 
- *.ckpt 用于关联


#### 震相拾取工具 
直接运行
```bash 
python lppnpicker.py -o odata/2020.1107.200km -m ckpt/new.200km.wave -d cuda:0 
```
命令行中的参数分别是
- -o:输出文件名
  - 文件名.log为日志文件，可以随时中断拾取并继续拾取
  - 文件名.txt为拾取结果
- -i:输入文件夹或文件名
  - 当为文件夹时会遍历文件夹
  - 当为文件名时会遍历文件 
- -m:模型文件位置
- -d:执行设备 
  
在config/picker.py需要调整更多拾取参数：
```text
    # 数据设置
    nsamplesdots = 8640000       # 采样点个数
    nchannel = 3                 # 通道数量
    samplerate = 100             # 采样率

    # 拾取设置
    prob = 0.3                   # 置信度
    nmslen = 1000                # NMS间隔设置，1000代表1000采样点间隔
    npicker = 2                  # 使用多少模型进行拾取

    # 文件名格式NET.STATION.LOC.CHANNEL.OTHERS.mseed
    namekeyindex = [0, 1]        # 文件名NET.KEY索引
    channelindex = 5             # 文件名数据分量标识CHANNEL索引,全国台站是5 
    filenametag = "mseed"        # 文件结尾标志
    chname1 = ["BHE", "BHN", "BHZ"] # 通道分量名1
    chname2 = ["SHE", "SHN", "SHZ"] # 通道分量名2

    # 输出设置
    ifplot = False               # 是否绘制拾取波形
    ifreal = False               # 是否输出REAL关联数据，以天为间隔
    snritv = 100                 # 输出信噪比截取长度
    bandpass = [1, 10]           # 计算信噪比滤波参数
```

拾取结果结构：
```text 
#单分量数据位置
震相,相对时间（秒）,置信度,绝对时间（格式%Y-%m-%d %H:%M:%S.%f）,信噪比,前后200个采样点振幅均值,台站名,前95%分位数,后95%分位数,最大值,标准差,峰值
```

#### REAL关联工具
直接运行
```bash 
python lppnpicker.py -o odata/2020.1107.200km -i ckpt/new.200km.wave.txt -s station 
```
运行前需要编译real文件夹下的real.c文件，形成可运行工具

参数依次为
- -o输出文件夹，文件夹应当存在
- -i输入拾取文件，拾取文件为lppnpicker格式
- -s台站文件。如果为null，即不制作台站文件
  
在config/real.py需要调整更多拾取参数：
```
    # REAL参数
    R = "0.4/20/0.02/2/5" 
    G = "2.0/20/0.01/1"
    V = "6.2/3.5"      
    S = "2/4/8/2/0.5/0.1/1.5" 

    # REAL 目录相关
    realdir = "realdata" # real临时文件夹位置
    realtooldir = "real" # real工具位置
```

#### FastLink关联工具
直接运行
```bash 
python fastlinker.py -o odata/2020.1107.200km.txt -i ckpt/new.200km.wave.txt -s station 
```


参数依次为
- -o输出文件夹，文件夹应当存在
- -i输入拾取文件，拾取文件为lppnpicker格式
- -s台站文件。如果为null，即不制作台站文件
  
在config/fastlink.py需要调整更多拾取参数：
```
    # GPU计算参数
    win_length = 25.0   # 单位秒
    win_stirde = 1      # 单位秒
    ngrid = 200 
    lonrange = [] # 经纬度范围
    latrange = []
    locpad = 0.5 # 若未指定经纬度则在数据范围基础上加pad范围，0.5为加0.5度
    datadir = "fastdata"
    saveitv = 86400 # 每隔多少步保存一个文件
    
    # 模型相关文件
    modeldir = "ckpt/link2-mini2d.ckpt"
    basetime = "2020-01-01" # 计算相对时间

    # 关联参数
    np = 2 
    ns = 4 
    nps = 8 
    nboth = 2 
```

关联结果结构：
```text 
#EVENT,位置编码,关联时间,经度,纬度
PHASE,震相时刻,关联类型,台站名,原始类型
```


#### 精定位工具 

### 联系方式
cangye@hotmail.com

### License 
GPLv3 