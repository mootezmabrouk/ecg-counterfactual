import numpy as np
import matplotlib.pyplot as plt

waveforms = np.load("data/EchoNext_val_waveforms.npy")
print("shape:", waveforms.shape)

# grab one patient, one lead, to start simple
record = waveforms[0, 0, :, :]   # drop the channel dim -> (2500, 12)
lead_II = record[:, 1]           # lead index 1 = lead II, a common one to eyeball

plt.plot(lead_II)
plt.title("Raw ECG — Lead II, patient 0")
plt.xlabel("Sample (250 Hz)")
plt.ylabel("Amplitude (standardized)")
plt.show()