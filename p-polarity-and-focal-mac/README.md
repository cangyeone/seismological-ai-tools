###focal mechanisms
Earthquake source parameter calculation (through initial motion) is based on the hashpy program, which is a Python version program compiled from HASH1.2.
### Training and Testing
#### Training
The training program is relatively simple, just run ploar.train.py directly. The network input is the Z-component waveform of 1024 samples, and the output is the initial motion direction and quality. However, the quality of the initial motion is not accurate. Training data are prepared using [H5 file production](hdf5-dataset-tools).

#### Testing
Testing data statistics parameters include information such as recall rate and precision rate. Its structure is similar to training.

#### About three-component data for initial motion detection
Single-component and three-component waveform data can be used for initial motion detection, but in testing it was found that there was little difference in accuracy between them, so it is recommended to use a single Z-component waveform.

### Calculation of source parameters based on P-wave initiation
Source parameter calculation through P-wave initiation includes three programs:
1. Phase picking program: Uses the currently most accurate RNN model to pick up earthquake P-waves. It has been made into a jit model.
2. Initial motion judgment program: Uses trained initial motion calculation programs to determine the direction of P-wave initiation. Use make.jit.py to create jit models.
3. HASH calculation for initial motion: Uses binary libraries included in hashpy programs for calculations.Before calculating, processing includes:
   1.Waveform data needs to use mseed format data with three components.
   2.Data needs to be indexed into an mseed database for easy retrieval from the database.
   3.The focal-mechanisms.py program is used for calculating source parameters; there are comments in the code that can be modified as needed.
Convenience during processing includes:
   1.No need for manual labeling of P-waves; only need to provide earthquake location and time as well as station location to automatically pick up P-waves.
   2.No need for labeling of initial motions; they are automatically determined by cutting off at step one.
   3.Automatic determination of initial motion quality based on confidence level outputs.