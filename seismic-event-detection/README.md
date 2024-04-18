### 1. Instructions for using the national 100Hz model
1. Training data, all models are based on the 2009-2019 national seismic network training of the 100Hz model, which can be directly used for continuous data picking.
2. The model training data is based on stations within a distance of 800km from the epicenter and includes PS wave data.
3. Currently, it has been tested based on three phases of ChinArray data, and the recall rate of RNN model manually labeled data is not less than 80%.
4. Different models' accuracy and speed are shown in the figure.
![](pickers/speed.jpg)

#### 目前已开源模型
|Model|Size(MB)|P-F1Score|Instrument|Sampling rate|Channel|Max distance|Range|Output phases|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|BRNN|1.9|0.857|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|EQTrasformer|3.1|0.852|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|PhaseNet(UNet)|0.8|0.815|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|LPPN(Large)|2.7|0.813|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|LPPN(Medium)|0.4|0.808|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|LPPN(Tinny)|0.3|0.757|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|UNet++|12|0.798|Broad band|100Hz|EHZ|300km|Global|Pg、Sg|
|BRNN(PnSn)|1.9|0.781|Broad band, MEMS, |100Hz|EHZ|2000km|Global|Pg、Sg、Pn、Sn|
|tele|1.9|0.800|Broad band|20Hz|EHZ|>3000km|Global|P|
|BRNN|1.9|0.807|Broad band|100Hz|Any|300km|Global|Pg、Sg|

#### 1.1 Recommended models:
1. If accuracy is more important, RNN can be used. We have tested it on mobile networks, dense networks, and fixed networks at the global level.
2. If memory is limited and speed is more important, LPPNM can be used.
3. If recall rate is low, we recommend using a threshold of 0.1 (pickers/rnn.01.jit), or using the PnSn model. Although the F1 score was low in testing, this was due to testing with manually labeled data within 2000km.
4. For some tasks that require confidence scores for each sampling point, an onnx model can be used.

#### 1.2 Pn and Sn phase picking model
1. In order to make the model more universal, we trained a new model using 2000km of manually labeled data.
2. The model is called rnn.pnsn.jit.
3. Based on the RNN model, it can simultaneously pick P, S, Pn, and Sn phases.
4. For other code, please visit the Gitee project address.
5. Due to the imbalanced nature of the data and some missing phase labels leading to low confidence, the confidence threshold is currently set at 0.1.
6. The model can be called by pickers.py for automatic picking by traversing directories directly as described in section 4.1.
7. The data needs to be sampled at 100Hz.
8.The accuracy has not been fully tested yet; only 10,000 waveforms of 102.4 seconds within 2000km from year 2020 were used for testing with results shown in the figure.
9.We found that after high-pass filtering (differentiation), the picking effect for large earthquakes was better; therefore we created a model for picking original + differentiated data as an example: makejit.pnsn.diff.py.Output models are: rnn.origdiff.pnsn.jit

![](pickers/china.pnsn.jpg)


Call in python interface
```python 
import torch 
sess = torch.jit.load("rnn.pnsn.jit")
x = ... # [Any length, 3] 
with torch.no_grad():
    x = torch.tensor(x, dtype=torch.float32, device=device) 
    y = sess(x) 
    phase = y.cpu().numpy()# [Number of phases, 1P, 2S, 3Pn, 4Sn]
```

#### 1.3 Distant Earthquake Picking Model
We have added a new model tele.rnn.jit for distant earthquake picking, which is used for picking the PS phase of distant earthquakes.

### 2. Model Usage Instructions
We provide three types of model files:
1. .pt files in the ckpt folder, which can be used for transfer learning and easily transferred to local data. It is recommended to fix some trainable parameters during transfer training.
2. Models for picking any length are located in the pickers folder.
   - .jit for direct use with PyTorch, which can directly output phase relative arrival time and phase type information.
   - .onnx for use with onnxruntime library, which is lighter than PyTorch and suitable for picking on edge devices. Due to the simple API provided, post-processing needs to be done externally.
- The output format of .jit files is: [number of phases, phase type + relative arrival time + confidence], all jit files are like this. Phase types: 1:P, 2:S.
- The .onnx output has two parts: a probability prob and a time time; for example prob[i] represents the probability of different phase types at point i, it is a vector of length 3; time[i] represents the relative moment at point i. Time and prob need to be used together in order to perform picking.
- Example usage of .jit can be found in picker.jit.py
- Example usage of .onnx can be found in picker.onnx.py
  
#### 2.1 Using C Language Version Onnx Model
Due to the complexity of writing programs in C language, we have merged the time and prob outputs from onnx into a .merge.onnx version model where the vector format becomes: 
[    [time length, number of categories,-,-],
     [number of categories, noise probability,P-wave probability,S-wave probability],
     [sample points, noise probability,P-wave probability,S-wave probability],
     .....]
For examples using C language version programs please contact cangye@hotmail.com.

### 3. make onnx and jit files
See the example programs makeonnx.xxx.jit and makejit.xxx.jit. In the .jit file:
```python
time_sel = torch.masked_select(ot, pc>0.3)
score = torch.masked_select(pc, pc>0.3)
```
Here, 0.3 is the minimum confidence level, which seems reasonable at present. If you want to pick up more phases (and consequently more errors), you can lower this value.
```python
selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>1000)
nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>1000)
ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>1000)
```
Here, 1000 represents 1000 sampling points and signifies that only the phase with the highest probability within a window of length 1000 is picked up for the same type of phase. If it is believed that there may be multiple phases within a 10-second window, this value can be lowered.
**The onnx model can use config/picker.py for post-processing as it is outside of the model itself**


### 4. Directly picking up continuous data
#### 4.1 Phase picking
Phase picking provides a more convenient way to directly traverse the directory and pick up all phases.
```bash 
python picker.py -i path/to/data -o outputname -m pickers/rnn.jit -d device
```

1. output file name.txt containing all picked phases 
2. output file name.log containing processed data information
3. output file name.err containing problematic data information 

The format of the output file is:
```text
#path/to/file
phase name,relative time(s),confident,aboulute time(%Y-%m-%d %H:%M:%S.%f),SNR,AMP,station name,other information 
```


#### 4.2 Seimic assosication
The goal of seismic association is to determine the number, location, and timing information of earthquakes from the phase picking results. Currently, there are 3 association algorithms provided: 
1. REAL methods [reallinker.py] 
2. LPPN methods [fastlinker.py] 
3. GaMMA methods [gammalinker.py] 
Both models take the picking results as input.

```bash
python fastlinker.py -i phase_picking_results.txt -o output_file_name.txt -s station_directory
```

The format of the station file is:
```text
network station LOC longitude latitude elevation(m)
```

For example:
```
SC AXX 00 110.00 38.00 1000.00
```


The structure of the output association file is:
```text
##EVENT,TIME,LAT,LON,DEP##
PHASE,TIME,LAT,LON,TYPE,PROB,STATION,DIST,DELTA,ERROR#
EVENT,2022-04-09 02:28:38.021000,100.6492,25.3660,PICKED_PHASE_TIME_LAT_LON_TYPE_PROB_STATION_DIST_DELTA_ERROR#
PHASE_PICKED_TIME_LAT_LON_TYPE_PROB_STATION_DIST_DELTA_ERROR#
```


### Please refer to the code for the format of the station directory.
When citing this work in a paper or publication please use:
1.LPPN: A Lightweight Network for Fast Phase Picking,
https://doi.org/10.1785/02202103092
2.Yu ZY,Wang WT and Chen YN (2022). Benchmark on accuracy and efficiency of several neural network based phase pickers using datasets from China seismic network.
Earthq Sci35,
doi:10 .1016/j.eqs .2022 .10 .001

### Open Source License
GPLv3