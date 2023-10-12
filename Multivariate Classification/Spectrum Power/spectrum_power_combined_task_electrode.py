import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Defining the sampling rate for spectrum power calculation
sampling_rate = 600  # Sampling rate in Hz

# Initializing an array to store the spectrum power features
spectrum_power_features = np.zeros((eegdat.shape[0], eegdat.shape[2], eegdat.shape[1], eegdat.shape[3]))

# Iterating over each electrode, grasp task, and repetition
for electrode in range(eegdat.shape[0]):
    for grasp_task in range(eegdat.shape[2]):
        for repetition in range(eegdat.shape[3]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=512, noverlap=256)

            # Calculating the spectrum power by summing the squared magnitudes
            spectrum_power = np.sum(np.abs(stft_data)**2, axis=0)

            # Taking the mean of the spectrum power to ensure a scalar value
            expanded_spectrum_power = np.repeat(spectrum_power, [200] * 6)

            # Storing the spectrum power feature
            spectrum_power_features[electrode, grasp_task, :, repetition] = expanded_spectrum_power
        average_spectrogram = np.mean(spectrum_power_features, axis=-1)
    average_spectrogram = np.mean(average_spectrogram, axis=2)

# Plotting the spectrum power features
plt.imshow((average_spectrogram).T, aspect='auto', origin='lower', cmap='jet')
plt.xlabel('Electrode')
plt.ylabel('Grasp task')
plt.title('Mean Spectrum Power')
plt.colorbar(label='Power')
plt.show()