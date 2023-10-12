import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Sampling rate for PSD calculation
sampling_rate = 600  # Sampling rate in Hz

# Initializing an array to store the PSD features
psd_features = np.zeros((eegdat.shape[0], eegdat.shape[2], eegdat.shape[3], eegdat.shape[1]//2 + 1))

# Iterating over each electrode, grasp task, and repetition
for electrode in range(eegdat.shape[0]):
    for grasp_task in range(eegdat.shape[2]):
        for repetition in range(eegdat.shape[3]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=256, noverlap=128)

            # Extracting the power spectral density
            psd = np.mean(np.abs(stft_data)**2, axis=1)

            # Taking the mean of the band power to ensure a scalar value
            mean_band_power = np.mean(psd)

            # Storing the PSD feature
            psd_features[electrode, grasp_task, repetition, :] = mean_band_power

# Plotting the PSD features
mean_psd = np.mean(psd_features, axis=(2, 3)).T
plt.imshow(mean_psd, aspect='auto', origin='lower', cmap='jet')
plt.xlabel('Electrode')
plt.ylabel('Grasp task')
plt.title('Mean Power Spectral Density')
plt.colorbar(label='Power')
plt.show()
