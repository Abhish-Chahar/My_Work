import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio

# Load the EEG data from the MATLAB file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Define the frequency range for STFT
frequency_range = (8, 30)  # Frequency range in Hz

# Calculate the frequency bin indices corresponding to the frequency range
sampling_rate = 600  # Sampling rate in Hz
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)

# Find the indices corresponding to the desired frequency range
start_freq, end_freq = frequency_range
start_idx = np.argmax(frequency_bins >= start_freq)
end_idx = np.argmax(frequency_bins >= end_freq)
frequency_values = frequency_bins[start_idx:end_idx]

# Initialize an array to store the averaged STFT features
averaged_stft_features = np.zeros((eegdat.shape[0], eegdat.shape[2], end_idx - start_idx))

# Iterate over each electrode and hand task
for electrode in range(eegdat.shape[0]):
    for hand_task in range(eegdat.shape[2]):
        # Initialize an array to store the STFT features for each repetition
        stft_features = []

        # Iterate over each repetition
        for repetition in range(eegdat.shape[3]):
            # Extract the EEG data for the current combination
            eeg_data = eegdat[electrode, :, hand_task, repetition]

            # Apply STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=256, noverlap=128)

            # Extract the desired frequency range from STFT data
            stft_range = stft_data[start_idx:end_idx]

            # Average the STFT data across the time axis
            stft_avg = np.mean(stft_range, axis=1)

            # Append the averaged STFT data to the list of features
            stft_features.append(stft_avg)

        # Calculate the average STFT features across repetitions
        averaged_stft_features[electrode, hand_task] = np.mean(stft_features, axis=0)

# Plot the spectrogram for a specific hand task (e.g., grasp_task_idx = 0)
hand_task_idx = 0

# Iterate over each electrode
for hand_task in range(eegdat.shape[2]):
    # Get the averaged STFT features for the current electrode and hand task
    stft_features = np.abs(averaged_stft_features[hand, hand_task_idx])

    # Plot the spectrogram
    plt.figure()
    plt.imshow([stft_features], cmap='jet', aspect='auto')
    plt.colorbar()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Electrode')
    plt.title('Spectrogram for Electrode {}'.format(electrode_idx))
    plt.show()
