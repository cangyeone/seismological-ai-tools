# Benchmark on the accuracy and efficiency of several neural network based phase pickers using datasets from China Seismic Network
# 基于中国地震台网的震相拾取模型精度和速度对比
[中文版说明](README.cn.md)
更多代码和教程可以加入QQ群：173640919
![QQ](odata/qq.png)

A tutorial on training and validating deep learning based phase pickers 
The deep learning methods have been widely used in seismological studies in recent years. Among all the applications, picking seismic phases is the most popular one. Compared to more complicated neural networks, phase picker is relatively simple for beginners to start the artificial intelligence journey in seismology. 
The recently released DiTing dataset provides good resources to conduct supervised deep learning investigation especially for phase picking (Zhao et al., 2022). This tutorial aims to link the raw dataset and final applicable models by introducing the data organization, model training and validating in a step by step style. Several examples using well-trained models are also provided, so that the graduate students or beginners in this field can take it as one hands-on to begin corresponding research quickly. All materials listed in this tutorial are open-source and can be publicly accessed. 
The tutorial begins from obtaining the original DiTing dataset, all models are implemented using PyTorch framework. 

### 1. Download DiTing dataset. 
The DiTing dataset is shared online through the website of the China Earthquake Data Center [Download data](https://data.earthquake.cn), allowing users to retrieve the metadata information of the dataset online. To obtain the complete dataset, users need to register online, upgrade to an advanced user, and sign the “Data Sharing Agreements of the China Earthquake Data Center” before accessing the data. Please refer to Zhao et al. (2022) for more details. The rest instructions will assume you have obtained the DiTing dataset already. 

### 2. Prepare training and validating datasets	
The DiTing dataset consists of 27 separate HDF5 files holding the waveforms and 27 csv files for the metadata. In order to access the entire dataset in one easier way, we merge all the HDF5 and csv files into one single HDF5 file. The meta information hold in the original csv files are set as the attributes in the single HDF5 file. Additional compression can reduce the file size to about 160 GB instead of original 283 GB, if compressed is set to True. 
One can put the original HDF5 files and csv files into the folder path/to/diting/folder and run the following script to get the single hdf5 file holding all necessary information:

```bash 
python makeh5.py -i path/to/diting/folder -o data/diting.h5 -c True
-i the folder holding the original DiTing files. The folder contains DiTing HDF5 files and CSV metadata. 
-o the output name of h5 file. 
-c whether to compress data. The compressed data will save storage, but will be slower when training. 
```

### 3. Data reading and processing tools 
The single HDF5 file can be read in parallel using the tools imported from utils:

```python
from utils.data import DitingData, DitingTestDataThread
data_tool = DitingData(file_name="data/diting.h5", stride=8, n_length=6144)
train loop:
    wave, label1, label2, label3 = data_tool.batch_data(batch_size)
```

We have designed two classes named as DitingData and DitingTestData, which are used for training and validating purpose. The optional parameters are as follows:
The parameter stride is used to generate the LPPN labels, n_length is the length of training data. The method batch_data () is used to generate a mini-batch training data of batch_size samples. Wave is the normalized waveforms, label1 is used to train LPPN models and label2 is used to train other point-to-point models. label3 is used to train original version of EQTransformer. The format of wave is [B, C, T]. B stands for the number of waveform data B=batch_size; C, that is set to be 3 in practice, stands for the number of channels; T stands for the sampling point T=n_length. The format of label1 is [B, 2, T/stride], the classification label and regression label form two channel. The format of label2 is [B, 3, T], the 3 channels stand for probability of None, P and S. The format of label3 is [B, 3, T], the 3 channels stand for probability of P, S and detection. 
You can load test data set by using the script:

```python
from utils.data import DitingData, DitingTestDataThread
data_tool = DitingTestData(file_name="data/diting.h5", stride=8, n_length=6144)
test loop:
    wave, label = data_tool.batch_data(batch_size)
```

The wave has same preprocessing with train data. The label is arrival time (sampling point) of P, S and SNR of each sample. If there is not P or S in the data, the arrival time will be set to -1. 


### 4. Model training
We have implemented 7 models, UNet, UNet++, EQT, RNN and three LPPN models of Tinny Medium and Large size. These models are organized in the models/ folder. The developers can add new models easier. The original version of EQTransformer is also in the models folder. 
You can write your own training script such as:

```python
from models.EQT import EQTransformer as Model, Loss 
device = torch.device(”cuda”)# recommended device
model = Model() 
lossfn = Loss()
model.to(device)
lossfn.to(device)	
train loop:
wave, label1, label2,label3 = data_tool.batch_data(batch_size)
wave = torch.tensor(wave, dtype=torch.float32)).to(device)
label = torch.tensor(label2, dtype=torch.float32).to(device)
pred = model(wave) 
loss = lossfn(pred, label)
…
```

Or you can easily train the 7 models by using the training script. One can run the following script to finish the model training. Moreover, the scripts can output the binary models in onnx format optionally for fast deployment. 

```bash
python diting.train.py -m modelname -i path/to/diting.h5 -o path/to/save/model/ --onnx true 
```

-m is the name of model. Selected from [RNN, EQT, UNet, UNetpp, LPPNT, LPPNM, LPPNL]
-i is the folder of merged h5 file. 
-o is the name of saved model. 
--onnx is set if you want output onnx format models. 
The optional parameters are as follows:
-m is name of models in our article, that can be: RNN, EQT, UNet, UNet++ and LPPN(T, M, L). -o is the saved path of a model, which will output .pt and .jit format. –onnx is used to output ONNXRuntime script to get a faster infer speed. 


### 5. Model Evaluation 
The evaluation is performed using the DiTing test dataset to measure performance and efficiency of different models. If you want to pick phase from continuous data or event records, please refer to step 6.
Validating the model performance can be made using the valid script as:

```bash
python diting.valid.py -m path/to/jit/model.jit -i path/to/h5file -o path/to/outname.txt
```

-o is the picking result. 
-i is the path to h5 file. The training and test data are in the same folder. 
-m is the saved jit model. 

The result will be listed in path/to/outname.txt and its format is:
#phase,P arrival time (relative sampling point), S arrival time (relative sampling point), SNR
phase type, arrival time (relative sampling point), probability
phase type, arrival time (relative sampling point), probability

To test the efficiency precisely, you need install ONNXRuntime for your system, which is one deployment tool to run the binary models faster than PyTorch. Also you need set –onnx true in step 4 to get models in this specific format. 
Then simply run the following script and specify the running device to get the time cost. 
```bash 
python diting.infertime.py -m path/to/onnx/model -d cpu
```

-m the input onnx model generated by the 4th step. 
-d running devices. 

The shape of input data is [BatchSize, N Channel, N Sampling point], which is [1440, 3, 6144] as default. The script will run 15 cycles and get the latest 10 cycles to calculate the infer time in ms.


### 6. Application
#### 6.1 Performance of the models trained using 100 Hz dataset
How the sample rate difference between training data and test data affects the model performance is still not clear and may vary with the dataset (Woollam et al.,2021). Perform the dataset using models with the same sample rate is preferred. In order to provide more choices for the end-users, we train another set of 7 models with one alternative large-scale dataset with 100 Hz sample rate. In general, it is similar with DiTing dataset for the size and scale, including events occurred in China with magnitude over 0 from year 2009 to 2019. In order to train the models with sufficient data point, longer windows around the picks are used to ensure 6144 points are available without zero padding. The maximum event-station distance is also extended to 800 km to include phases at far stations. The maximum distance is also set to 800 km for evaluation. The performance metrics calculated using 10,000 test samples in year 2020 are summarized in Figure S1. Compared to the 50 Hz models, some metrics are higher and some are lower, but they are very close. The differences may be caused by including large distance waveforms for both training and validating. Generally, the performances are almost same with those trained with 50 Hz dataset and just another model sets for the 100 Hz sampled records.
Before picking phase, one need modify the configuration in config/picker.py 
```python 
class Parameter:
    # configure of picking model 
    samplerate = 100             # sampling rate of picker 

    # picking parameter
    prob = 0.2                   # threshold of prob. 
    nmslen = 1000               # minimal interval between same phase. 
    npicker = 1                  # number of pickers 

    # name format:NET.STATION.LOC.CHANNEL.OTHERS.mseed, separate by “.” 
    namekeyindex = [0, 1]         # index of NET of STATION 
    channelindex = 3             # index of CHANNEL 
    filenametag = "mseed"        # tail of file name
    chname1 = ["BHE", "BHN", "BHZ"] # channel name 
    chname2 = ["SHE", "SHN", "SHZ"] # channel name 

    # output config 
    ifplot = False               # if plot picking figs
    ifreal = False               # if output real data for association 
```

Then run
```bash 
python picker.py -i path/to/data -o outname -m path/to/jit/model -d device  
```

-i is input folder of data. The picker.py will walk through the directory, and find all the files ends with file name tag, i.e., suffix, in configuration file. t
-m is saved model. We recommend that .jit model, which has better compatibility. Also, .onnx models are also available. 

The output file outname.txt contains the phase arrival with the format:
```text
#path/to/file
phase name,relative time(s),confident,aboulute time(%Y-%m-%d %H:%M:%S.%f),SNR,AMP,station name,other information 
```

The SNR is a reference for the waveform quality around the picks, the time window and band pass frequency limits to calculate SNR is controlled by the users. 
The users can interrupt and continue training at any time, and the process will be logged in outname.log.
 


#### 6.2 Picking phases from continuous waveforms
There are two common cases the models will be used. One of them is to pick the P/S phases from continuous data records. We select one day continuous waveforms of one mobile station near the Binchuan Airgun active source to demonstrate the usage. The input files are in mseed format and have been used to analyze the aftershocks of Yangbi Ms 6.4 earthquake (Su et al., 2021; Wang and Yu, 2022).

#### 6.3 Picking phases for specific event
Another case is to pick the phases using the records for a given earthquake. Although this is similar with processing the continuous data, small differences do exist. Given the location of the event and stations, the theoretical travel time can be estimated, giving a reference where the desired phases will appear. This is the main difference compared to the continuous case. Specific treatment can be taken for selecting desired window and discarding the wrong phase types. We use the records of one ML 3.3 event occurred in 2012/07/27 to demonstrate the usage. This event is well recorded by the ChinArray stations deployed across Yunnan province, as shown in Figure S3a. The waveforms at these stations are saved in sac format with the event longitude, latitude and depth properly set in the headers. For the known event, the theoretical arrival times for P and S phase can be estimated for each station, so that we can select proper window within which the waveforms will be processed (Figure S3b). Moreover, the theoretical arrival times also provide references for the phase type. For example, for the waveforms near the P theoretical arrival time, the picks with high S probability will be false picks and will be discarded. This can reduce some post-process procedures. Figure S3c show the automatic picks at one station 180 km away from the event. 
In practice, one need put the sac files with the suffix “.sac” into one single folder, then modify the config/picker.py to change filenametag = “sac” and run the picker.py script descripted in 6.1. We recommend that the records should be longer than 61.44 seconds and there are at least 2.56 seconds before P arrival for a better model performance. 
The eventpick.py can process a list of events in one shot, given the event catalog and station profiles.
You can run:
```bash 
python eventpick.py 0 -m ./ckpt/new.400km.wave -c ./data/test.catalog -s ./data/X1.sta.cyn -i ./data –o ./Results/out
```
-m is the used model, here the 100 Hz model.
-s is the station file. All the stations in this file will be scan through, and their location information will be used to estimate theoretical traveltimes.
-c is the event catalog used for phase picking picking. The second column is event id, which is the directory name holding sac files for each event. The proper window will be set internally. 
```text
#EVENT YN.202001312039.0002 eq 2020 01 31 031 12 39 09 110000 LOC   98.035   24.514 DEP     NONE MAG   ML  1.0
#EVENT YN.202001312011.0001 eq 2020 01 31 031 12 11 42 770000 LOC   98.126   24.503 DEP     NONE MAG   ML  1.4
#EVENT YN.202001311950.0001 eq 2020 01 31 031 11 50 06 820000 LOC   98.002   24.472 DEP     NONE MAG   ML  1.6
…
```

-i is input folder of event data.
-o is the output file name with same output format mentioned in 6.1.

type 
```bash 
Python eventpick.y -h
```
to check the help messages and you can also revise the script to meet your purpose.
 
Figure S3: Picking the arrival times using the records from known earthquakes. a) Location of the earthquake (yellow star) and stations (Blue triangles) b) The vertical component waveforms of the event recorded at the 139 stations. The solid lines indicate the theoretical arrival times of P and S phases and the dashed lines show the time window within which the corresponding arrival are picked. c) Example of the P/S picks at a station ~ 180 km away from the event. The top panel shows the vertical component waveform and the P and S picks are indicated by yellow and red vertical lines, respectively. The lower panels show the waveforms details around P and S picks.


#### 6.4 Detected events form picked phases
Phase association is used to find the locations and number of events from picked phases. Here we supply two methods for association:
1. REAL methods [reallinker.py]
2. LPPN methods [fastlinker.32.py]

The two script takes picker.py output as input. And output the information of events. The format of output:
```
##EVENT,TIME,LAT,LON,DEP
##PHASE,TIME,LAT,LON,TYPE,PROB,STATION,DIST,DELTA,ERROR
#EVENT,2022-04-09 02:28:38.021000,100.6492,25.3660,0.0000
PHASE,2022-04-09 02:28:40.249700,100.5690,25.2658,P,0.958,MD.01311,13.753,2.229,-0.139
PHASE,2022-04-09 02:28:41.929700,100.5690,25.2658,S,0.621,MD.01311,13.753,3.909,-0.238
```

#### 7. List of models and pertrained model
**The pretrained model is trained with GPU, you need to load as "torch.load(pt_path, map_location="cpu")"**
We provide two sets of models for the end-users who may not care about model creation details but are interested in applying them to the seismic records. Each set contains 7 models as mentioned in current paper. The first set of models are trained with DiTing data with 50 Hz sampling rate, and the second set of modes are trained using one dataset at the similar size but with 100 Hz sampling rate. So the users can choose one of them based on the sampling rate of their records. 
For each model, there are three model files in different formats: “.pt” model only contains trainable parameters; “.jit” model contains parameters and the computation graph, which allows the user to run without other scripts; “.onnx” model can be used in ONNXRuntime, under which a high speed is expected.
The details of these model files are listed in Table S1:

Table S1. Models and their statics. 
Model name	Model Size (MB)	Pretrained 50Hz model
(Trained on DiTing)	Pretrained 100Hz model
(Trained on CSN data)
|Model Name|Size|Diting Model format|CSN Model fomat|
|:-:|:-:|:-:|:-:|
|RNN	| 1.75| [DownLoad]()| [DownLoad]()|
|EQT	| 1.72|[DownLoad]()| [DownLoad]()|
|UNet	| 0.70|[DownLoad]()| [DownLoad]()|
|UNet++	| 11.70|[DownLoad]()| [DownLoad]()|
|LPPN(T)|0.09|[DownLoad]()| [DownLoad]()|
|LPPN(M)|0.23|[DownLoad]()| [DownLoad]()|
|LPPN(L)|2.53|[DownLoad]()| [DownLoad]()|

Below are some scripts to call these models:
 
The UNet, UNet++, EQT and RNN can be called as:
“.pt” model:
```
from models.EQT import EQTransformer as Model 
model = Model()
model.eval()
model.load_state_dict(“ckpt/china.eqt.pt”)
x = … #NumPy data in [N, C, T] format 
with torch.no_grad():
prob = model(torch.tensor(x, dtype=torch.float32))
prob = prob.cpu().numpy()
phases = find_phase_point2point(prob, height=0.3, dist=10)
```

“.jit” model:
```
import torch 
model = torch.jit.load(“ckpt/china.eqt.jit”)
x = … #NumPy data in [N, C, T] format 
with torch.no_grad():
prob = model(torch.tensor(x, dtype=torch.float32))
prob = prob.cpu().numpy()
phases = find_phase_point2point(prob, height=0.3, dist=10)
```

“.onnx” model 
```
import onnxruntime as rt 
model = rt.InferenceSession(“ckpt/china.eqt.onnx”, providers=[“CPUExecutionProvider”])
x = … #NumPy data in [N, C, T] format 
prob = model.run([“prob”], {“wave”:x.astype(np.float32)})
phases = find_phase_point2point(prob, height=0.3, dist=10)
```

The LPPN series can be called as:
“.pt” model:
```
from models.LPPNT import Model
model = Model()
model.eval()
model.load_state_dict(“ckpt/china.lppnt.pt”)
x = … #NumPy data in [N, C, T] format 
with torch.no_grad():
prob, time = model(torch.tensor(x, dtype=torch.float32))
prob = prob.cpu().numpy()
time = time.cpu().numpy()
phases = find_phase_lppn([prob, time], height=0.3, dist=10)
```

“.onnx” model 
```
import onnxruntime as rt 
model = rt.InferenceSession(“ckpt/china.eqt.onnx”, providers=[“CPUExecutionProvider”])
x = … #NumPy data in [N, C, T] format 
prob, time = model.run([“prob”, “time”], {“wave”:x.astype(np.float32)})
phases = find_phase_lppn([prob, time], height=0.3, dist=10)
```

The picked phases can be extracted using the find_phase_modelname function, which will output the probability and relative sampling point of each pick. The parameter height of this function denotes the probability threshold to declare a pick. The parameter dist is the minimal interval (number of sampling point) between two picks. If there are two picks within this interval, only the pick with higher probability will be kept. The output phases are with format:
 [Type, Relative time (sampling point), Probability]. 
then, one can take the additional processing to output the arrival times of corresponding picks for further processing. 

