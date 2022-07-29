# 地震HDF5数据处理与分析

#### 1. 介绍
用于构建MSEED文件索引，可以方便快速的根据台站名、时间等截取任意长度的波形数据。
用于将mseed数据和地震目录整合成独立的h5格式数据，方便后续分析处理工作.

#### 2. 软件架构
软件整合了obspy中的Clint功能和震相数据读取分析功能。软件功能为：
- makeidex.py 产生索引
- makeh5.py 制作数据
- testh5.py 测试数据
- base.py 基础库
- utils 工具文件夹
  - 修改来自于obspy
  - 原始版本存在死锁

#### 3. 安装教程

依赖库包括：obspy,h5py,tqdm 
请编译安装sqlite
#### 4. 使用说明

##### 4.1 创建索引
   1. 使用命令'makeindex.py -r /path/to/data -o index.db' 创建索引
   2. 程序会自动搜索目录下文件
   3. 如果索引文件过大，建议分开存储。比如按年分隔，否则截取时速度较慢
   4. 如果分开存储数据库数据无法截取。
##### 4.2 制作数据
   1. 使用命令'makeh5.py -i index.db -o out.h5 -c /path/to/ctlg -s /path/to/location'
   2. 震相位置需要修改 
   3. 多线程程序需要自行修改
##### 4.3 测试数据
   1. 使用命令'testh5.py -i out.h5 -o stats.txt -c /path/to/ctlg'
   2. 用于测试h5数据完整性

##### 4.4 通过数据库读取mseed数据
```python 
from obspy import UTCDateTime
import datetime  
from utils.dbdata import Client 
clint = Client("path/to/index")
time1 = etime + datetime.timedelta(seconds=-10)# 截取开始时间-10秒
time2 = etime + datetime.timedelta(seconds=20)# 截取结束时间+20秒
t1 = UTCDateTime(time1.strftime("%Y/%m/%dT%H:%M:%S.%f"))  #转换为obspy时间
t2 = UTCDateTime(time2.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
cha = "?HZ"# 获取分量中有HZ的
net = "X1" # 台网名
sta = "00111" # 台站名
loc = "00" # 位置名
st = clint.get_waveforms(net, sta, loc, cha, t1, t2)# 获取波形，obspy.Stream
```

#### 参与贡献

如是：cangye@Hotmail.com


#### LICENSE 
GPL v3
