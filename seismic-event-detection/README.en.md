# seismic-event-detection Guide

This directory contains scripts for seismic phase picking and event association, along with various pretrained models.

## Contents
- `pickers/`: Pretrained `.jit` and `.onnx` models for inference on data of any length.
- `ckpt/`: PyTorch `.pt` weights for transfer learning or retraining.
- `picker.py`: Automatically pick phases from 100 Hz three-component waveforms in a directory.
- `fastlinker.py`, `gammalinker.py`, `reallinker.py`: Implementations of three association algorithms.
- `config/`: Example configurations for picking and association.

## Quick Start
1. Install dependencies such as `obspy`, `torch`, and `onnxruntime`.
2. Prepare three-component 100 Hz waveform files, for example `NET.STA.LOC.CHN.XXXX.mseed`.
3. Run the following command for batch picking:
   ```bash
   python picker.py -i /path/to/waves -o result -m pickers/rnn.jit -d cuda:0
   ```
   This generates `result.txt` (picks), `result.log` (processing log) and `result.err` (errors).

### Output Format
```
#path/to/file
phase,relative_time,confidence,absolute_time,SNR,AMP,station,extra
```

## Event Association
Use the picked phases to associate events with one of these scripts:
- `fastlinker.py`: Fast association based on LPPN.
- `reallinker.py`: REAL algorithm implementation.
- `gammalinker.py`: GaMMA method implementation.

Example command:
```bash
python fastlinker.py -i result.txt -o events.txt -s station.csv
```
Where `station.csv` looks like:
```
network station LOC longitude latitude elevation
SC AXX 00 110.00 38.00 1000.00
```

## Additional Notes
- `makejit.*.py` and `makeonnx.*.py` show how to export models and can be used to create new jit/onnx files.
- See this directory's `README.md` for more model comparisons and parameter details.

License: GPLv3
