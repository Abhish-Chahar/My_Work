import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Defining the frequency range for STFT
frequency_range = (8, 30)  # Frequency range in Hz

# Calculating the frequency bin indices corresponding to the frequency range
sampling_rate = 600  # Sampling rate in Hz
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)

# Finding the indices corresponding to the desired frequency range
start_freq, end_freq = frequency_range
start_idx = np.argmax(frequency_bins >= start_freq)
end_idx = np.argmax(frequency_bins >= end_freq)
frequency_values = frequency_bins[start_idx:end_idx]

# Initializing an array to store the STFT features
stft_features = np.zeros((eegdat.shape[0], eegdat.shape[2], eegdat.shape[3], end_idx - start_idx))

# Iterating over each electrode, grasp task, and repetition
for electrode in range(eegdat.shape[0]):
    for grasp_task in range(eegdat.shape[2]):
        for repetition in range(eegdat.shape[3]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=256, noverlap=128)

            # Extracting the desired frequency range from STFT data
            stft_range = stft_data[start_idx:end_idx]

            # Averaging the STFT data across the time axis
            stft_avg = np.mean(stft_range, axis=1)

            # Storing the STFT features
            stft_features[electrode, grasp_task, repetition] = stft_avg

# Plotting the STFT features for a specific electrode, grasp task, and repetition
electrode_idx = 0
grasp_task_idx = 0
repetition_idx = 0

plt.plot(frequency_values, np.abs(stft_features[electrode_idx, grasp_task_idx, repetition_idx]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('STFT Features')
plt.show()