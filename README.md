## Seismology AI Tool

### Introduction
This is an open-source tool for seismological data processing and analysis, including phase picking, polarization, and dispersion extraction.

- We have open-sourced the 100Hz model for the China region, with some models trained on the CSNCD dataset. 
- In the "seismic-event-detection" directory of this project, if the PS pick-up rate within 800 kilometers on your dataset is less than 75%, or if the PS pick-up error is greater than 300ms or 350ms respectively, please submit the BUG here or contact cangye@hotmail.com. 
- Among the currently open-sourced models, the picking accuracy of PgSgPnSn four-phase seismic waves is highest.
- Currently, picker.py defaults to outputting initial movements, including their quality and requires using the onnxruntime library; it was also trained using a nationwide fixed network.

My telegram:
![telegram](qr.jpg)

### Software Architecture
The software is entirely built on Python. Each project is independent and relies on:
- obspy: for data reading
- PyTorch: for deep learning
- OpenCV: for image processing
  
### Installation Tutorial
1. It is recommended to install the latest version of Anaconda.
2. Other libraries can be installed using pip.
3. For deep learning libraries, it is suggested to use conda for basic environment installation process.
For installation guidance, please refer to the article [Python Environment and Usage Issues - Such Articles - Zhihu](https://zhuanlan.zhihu.com/p/414300182).

### Instructions for Use
Please refer to the content in each respective folder.

#### Contribution Participation
Contributions can be made by contacting cangye@Hotmail.com.

#### License 
GPLv3
