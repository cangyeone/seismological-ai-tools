import matplotlib.pyplot as plt 
import obspy 

st = obspy.read("data/waveform/X1.53085.01.BHE.D.20122080726235953.sac")
dt = st[0].data 
plt.plot(dt)
plt.axvline(9630)
plt.axvline(12247)
plt.show()