import numpy as np
import scipy.io as sio
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Define the parameters
electrodes = 45
time_samples = 1200
tasks = 8
repetitions = 28
sampling_rate = 600  # Hz

# Define the frequency range for the STFT (8-30 Hz)
frequency_range = [8, 30]
window_length = 800  # You can try smaller values, e.g., 64
overlap = window_length / 2  # Set overlap to half the window size
frequencies = np.linspace(0, sampling_rate / 2, window_length // 2 + 1)
freq_indices = np.where((frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1]))[0]

for task in range(1, tasks + 1):
    # Initialize an empty list to store the STFT data for each electrode and repetition
    stft_data_list = []

    for repetition in range(1, repetitions + 1):
        for electrode in range(1, electrodes + 1):
            # Extract the EEG data for the current electrode, task, and repetition
            eeg_data = eegdat[electrode - 1, :, task - 1, repetition - 1]

            # Compute the STFT with the specified window length and overlap
            f, t, stft_data = spectrogram(eeg_data, window='hamming', nperseg=window_length, noverlap=overlap, fs=sampling_rate)

            # Append the STFT data for the current electrode and repetition to the list
            stft_data_list.append(stft_data[freq_indices, :])

    # Combine the STFT data for all electrodes and repetitions
            stft_data_combined = np.vstack(stft_data_list)

    # Take the mean along time samples (columns) and repetitions (pages)
    # mean_stft_data = np.mean(stft_data_combined, axis=1)

    # Reshape mean_stft_data to a 2-dimensional array
    # mean_stft_data = np.expand_dims(mean_stft_data, axis=1)

    # Plot the average time-frequency spectrum for the current hand task
    t = np.arange(0, time_samples / sampling_rate, time_samples / (sampling_rate * 2))
    plt.figure()
    plt.imshow(10 * np.log10(np.abs(stft_data_combined)), aspect='auto', extent=[t[0], t[-1], f[freq_indices[0]], f[freq_indices[-1]]], cmap='jet')
    plt.colorbar()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Average Time vs. Frequency Spectrum - Hand Task ' + str(task))
    plt.show()
print(stft_data_combined.shape)
print(f.shape)
print(t.shape)