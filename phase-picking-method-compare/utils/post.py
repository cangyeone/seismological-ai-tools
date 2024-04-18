import scipy.signal as signal  
import numpy as np 
def find_phase_lppn(out, delta=8.0, height=0.80, dist=1):
    prob, regr = out 
    shape = np.shape(prob) 
    all_phase = []
    phase_name = {0:"N", 1:"P", 2:"S"}
    for itr in range(shape[0]):
        phase = []
        for itr_c in [0, 1]:
            p = prob[itr, itr_c+1, :] 
            #p = signal.convolve(p, np.ones([10])/10., mode="same")
            h = height 
            peaks, _ = signal.find_peaks(p, height=h, distance=dist) 
            for itr_p in peaks:
                phase.append(
                    [
                        itr_c+1, #phase_name[itr_c], 
                        itr_p*delta+regr[itr, itr_p], 
                        prob[itr, itr_c+1, itr_p], 
                        itr_p*delta
                    ]
                    )
        all_phase.append(phase)
    return all_phase 

def find_phase_point2point(pred, delta=1.0, height=0.80, dist=1):
    shape = np.shape(pred) 
    all_phase = []
    phase_name = {0:"N", 1:"P", 2:"S"}
    for itr in range(shape[0]):
        phase = []
        for itr_c in [0, 1]:
            p = pred[itr, itr_c+1, :] 
            #p = signal.convolve(p, np.ones([10])/10., mode="same")
            h = height 
            peaks, _ = signal.find_peaks(p, height=h, distance=dist) 
            for itr_p in peaks:
                phase.append(
                    [
                        itr_c+1, #phase_name[itr_c], 
                        itr_p, 
                        pred[itr, itr_c+1, itr_p], 
                        itr_p
                    ]
                    )
        all_phase.append(phase)
    return all_phase 
