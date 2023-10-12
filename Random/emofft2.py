import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

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

# Define the window length and overlap parameters for STFT
window_length = 256  # Try an even smaller value
overlap = window_length // 2  # Increase the overlap slightly

# Compute the frequencies corresponding to each STFT bin
frequencies = np.linspace(0, sampling_rate / 2, window_length // 2 + 1)

# Compute the time-frequency spectrum for each electrode, task, and repetition
for task in range(tasks):
    for repetition in range(repetitions):
        for electrode in range(electrodes):
            # Extract the EEG data for the current electrode, task, and repetition
            eeg_data = eegdat[electrode, :, task, repetition]

            # Compute the STFT with the specified window length and overlap
            f, t, stft_data = spectrogram(eeg_data, fs=sampling_rate, nperseg=window_length, noverlap=overlap)

            # Plot the STFT for the current task
            print(stft_data.shape)
            plt.figure()
            plt.imshow(np.abs(stft_data), aspect='auto', origin='lower', cmap='jet', extent=[0, time_samples / sampling_rate, frequency_range[0], frequency_range[1]])
            plt.colorbar(label='Power (dB)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Time vs. Frequency Spectrum - Task {task + 1}')
            plt.show()