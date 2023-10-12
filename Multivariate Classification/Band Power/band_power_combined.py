import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Defining the frequency range and sampling rate for band power calculation
frequency_range = (8, 30)  # Frequency range in Hz
sampling_rate = 600  # Sampling rate in Hz

# Calculating the frequency bin indices corresponding to the frequency range
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)
start_freq, end_freq = frequency_range
start_idx = np.argmax(frequency_bins >= start_freq)
end_idx = np.argmax(frequency_bins >= end_freq) + 1  # Add 1 to include the end frequency bin

# Initializing an array to store the band power features
band_power_features = np.zeros((eegdat.shape[0], eegdat.shape[2], eegdat.shape[3]))

# Iterating over each electrode, grasp task, and repetition
for electrode in range(eegdat.shape[0]):
    for grasp_task in range(eegdat.shape[2]):
        for repetition in range(eegdat.shape[3]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=256, noverlap=128)

            # Calculating the band power for the desired frequency range
            band_power = np.sum(np.abs(stft_data[start_idx:end_idx])**2, axis=0)

            # Taking the mean of the band power to ensure a scalar value
            mean_band_power = np.mean(band_power)

            # Storing the band power feature
            band_power_features[electrode, grasp_task, repetition] = mean_band_power

# Plotting the band power features
plt.imshow(np.mean(band_power_features, axis=2).T, aspect='auto', origin='lower', cmap='jet')
plt.xlabel('Electrode')
plt.ylabel('Grasp task')
plt.title('Mean Band Power')
plt.colorbar(label='Power')
plt.show()
