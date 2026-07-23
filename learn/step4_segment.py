import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

waveforms = np.load("data/EchoNext_val_waveforms.npy")
record = waveforms[0, 0, :, :]
lead_II = record[:, 1]

cleaned = nk.ecg_clean(lead_II, sampling_rate=250)
_, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=250)
rpeaks = rpeaks_info["ECG_R_Peaks"]

# delineate: find P, QRS, T wave boundaries relative to each R-peak
_, waves_info = nk.ecg_delineate(cleaned, rpeaks_info, sampling_rate=250, method="dwt")

print(waves_info.keys())