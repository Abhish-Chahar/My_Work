import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from scipy import fftpack
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# applying Fourier transform on the EEG data
fft_data = np.fft.fft(eegdat, axis=1)

# Calculating the frequency bins
sampling_rate = 600  # Hz
n_samples = eegdat.shape[1]
freq_bins = np.fft.fftfreq(n_samples, d=1/sampling_rate)

# Plotting the magnitude spectrum for a specific electrode and grasp task
electrode_idx = 0
task_idx = 0

magnitude_spectrum = np.abs(fft_data[electrode_idx, :, task_idx])

# Plotting only the positive side of the frequency spectrum
positive_freq_bins = freq_bins[1:int(n_samples/2)+1]
positive_magnitude_spectrum = magnitude_spectrum[1:int(n_samples/2)+1]

plt.plot(positive_freq_bins, positive_magnitude_spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude Spectrum')
plt.title(f'Magnitude Spectrum for Electrode {electrode_idx}, Task {task_idx}')
plt.grid(True)
plt.show()






