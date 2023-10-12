import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

sampling_rate = 600
num_electrodes, time_samples, num_tasks, num_repetitions = eegdat.shape
# List of task names (replace with actual task names)
task_names = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7', 'Task 8']

# Perform temporal frequency domain analysis using FFT
# eeg_data = eegdat[electrode, :, grasp_task, repetition]
# num_electrodes, time_samples, num_tasks, num_repetitions = eeg_data.shape
# frequencies, time, magnitude_spectra = stft(eeg_data, fs=sampling_rate, nperseg=128, noverlap=64)
# mean_magnitude_spectra = np.mean(magnitude_spectra, axis=0)

for grasp_task in range(eegdat.shape[2]):
    # Initializing an array to store the spectrum power spectrogram for the current task
    temporal_frequency = np.zeros((eegdat.shape[0], eegdat.shape[1], eegdat.shape[2], eegdat.shape[3]))

    # Iterating over each electrode and repetition
    for electrode in range(eegdat.shape[0]):
        for repetition in range(eegdat.shape[3]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            frequencies, time, magnitude_spectra = stft(eeg_data, fs=sampling_rate, nperseg=128, noverlap=64)

# Plot the spectrogram for each task
num_tasks = len(task_names)
plt.figure(figsize=(12, 6))

for task_index in range(num_tasks):
    # plt.subplot(num_tasks, 1, task_index + 1)
    plt.imshow(magnitude_spectra, origin='lower', aspect='auto', extent=[0, time_samples / sampling_rate, 0, sampling_rate / 2])
    plt.colorbar(label='Log Magnitude')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Temporal Frequency Analysis - Task: {task_names[task_index]}')

    plt.tight_layout()
    plt.show()