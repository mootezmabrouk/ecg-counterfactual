import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

waveforms = np.load("data/EchoNext_val_waveforms.npy")
record = waveforms[0, 0, :, :]
lead_II = record[:, 1]

# neurokit2's cleaning function — designed specifically for ECG signals
cleaned = nk.ecg_clean(lead_II, sampling_rate=250)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(lead_II)
axes[0].set_title("Raw (PhysioNet pre-processed)")
axes[1].plot(cleaned)
axes[1].set_title("After nk.ecg_clean()")
plt.tight_layout()
plt.show()