### The denoising program
The denoising program for seismic waveforms uses the STEAD dataset to train the denoising network. 

The training program is relatively simple and can be directly read in the code. 

It includes two training codes: 
1. train.py, which uses an encoder-decoder structure for waveform denoising; 
2. trainwithpick.py, which combines the phase picking model training to ensure that the picked PS phase can still be guaranteed after filtering. 
 
The data files are placed in the data folder, including two from the STEAD dataset: 
- data/waveforms_11_13_19.hdf5 
- data/metadata_11_13_19.csv