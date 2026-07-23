import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

waveforms = np.load("data/EchoNext_val_waveforms.npy")
record = waveforms[0, 0, :, :]
lead_II = record[:, 1]

cleaned = nk.ecg_clean(lead_II, sampling_rate=250)

# detect R-peaks
_, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=250)
rpeaks = rpeaks_info["ECG_R_Peaks"]  # array of sample indices where R-peaks occur

print("number of R-peaks detected:", len(rpeaks))
print("R-peak sample indices:", rpeaks)

plt.plot(cleaned, label="cleaned signal")
plt.scatter(rpeaks, cleaned[rpeaks], color="red", label="R-peaks", zorder=5)
plt.title("R-peak detection — Lead II, patient 0")
plt.legend()
plt.show()