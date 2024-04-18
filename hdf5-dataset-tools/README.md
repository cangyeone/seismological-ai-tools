# Earthquake HDF5 Data Processing and Analysis
#### 1. Introduction
Introduce the construction of MSEED file index, which can facilitate the quick extraction of waveform data of any length based on station name, time, etc. It is used to integrate mseed data and earthquake catalogs into independent h5 format data for subsequent analysis and processing work.

#### 2. Software Architecture
The software integrates the Clint function in obspy and phase data reading and analysis functions. The software functions include:
- makeidex.py: Generate index
- makeh5.py: Create data
- testh5.py: Test data
- base.py: Basic library
- utils folder: Utility files 
   - Modified from obspy 
   - The original version had deadlocks

#### 3. Installation Tutorial
Dependent libraries include: obspy, h5py, tqdm Please compile and install sqlite.

#### 4. Instructions for Use
##### 4.1 Creating an Index  
1. Use the command 'makeindex.py -r /path/to/data -o index.db' to create an index  
2. The program will automatically search for files in the directory  
3. If the index file is too large, it is recommended to store it separately by year; otherwise, the speed of extraction will be slow  
4.If stored separately, database data cannot be extracted.

##### 4.2 Creating Data   
1.Use the command 'makeh5.py -i index.db -o out.h5 -c /path/to/ctlg -s /path/to/location'   
2.The seismic location needs to be modified    
3.Multi-threaded programs need to be modified manually  

##### 4.3 Testing Data   
1.Use the command 'testh5.py -i out.h5 -o stats.txt -c /path/to/ctlg'   
2.For testing h5 data integrity

##### 4.4 read mseed data 
```python 
from obspy import UTCDateTime
import datetime  
from utils.dbdata import Client 
clint = Client("path/to/index")
time1 = etime + datetime.timedelta(seconds=-10)# strating time -10s
time2 = etime + datetime.timedelta(seconds=20)# end time +20s
t1 = UTCDateTime(time1.strftime("%Y/%m/%dT%H:%M:%S.%f")) 
t2 = UTCDateTime(time2.strftime("%Y/%m/%dT%H:%M:%S.%f"))  
cha = "?HZ"# All channles including HZ
net = "X1" # Net name
sta = "00111" # Station name
loc = "00" # location name
st = clint.get_waveforms(net, sta, loc, cha, t1, t2)# get wave，obspy.Stream
```


#### LICENSE 
GPL v3
