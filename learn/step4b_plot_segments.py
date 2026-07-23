import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

waveforms = np.load("data/EchoNext_val_waveforms.npy")
record = waveforms[0, 0, :, :]
lead_II = record[:, 1]

cleaned = nk.ecg_clean(lead_II, sampling_rate=250)
_, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=250)
rpeaks = rpeaks_info["ECG_R_Peaks"]
_, waves_info = nk.ecg_delineate(cleaned, rpeaks_info, sampling_rate=250, method="dwt")

# zoom into just ONE beat so the boundaries are actually visible (2nd beat, index 1)
beat_idx = 1
r = rpeaks[beat_idx]
window = slice(r - 150, r + 150)  # ~600ms window centered on this R-peak

plt.figure(figsize=(10, 5))
plt.plot(range(r-150, r+150), cleaned[window], color="black", label="signal")

def mark(key, color, label):
    val = waves_info[key][beat_idx]
    if val is not None and not np.isnan(val):
        val = int(val)
        plt.axvline(val, color=color, linestyle="--", alpha=0.7, label=label)

mark("ECG_P_Onsets", "green", "P onset")
mark("ECG_P_Offsets", "green", "P offset")
mark("ECG_R_Onsets", "blue", "QRS onset")
mark("ECG_R_Offsets", "blue", "QRS offset")
mark("ECG_T_Onsets", "orange", "T onset")
mark("ECG_T_Offsets", "orange", "T offset")

plt.axvline(r, color="red", linestyle="-", label="R peak")
plt.title("One heartbeat, fully segmented")
plt.legend(loc="upper right", fontsize=8)
plt.show()