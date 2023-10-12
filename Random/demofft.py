import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eeg_data = mat_data['eegdat']

# Define the sampling rate (Hz) for the EEG data
sampling_rate = 600

# List of task names (replace with actual task names)
task_names = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7', 'Task 8']

# Perform temporal frequency domain analysis using FFT
num_electrodes, time_samples, num_tasks, num_repetitions = eeg_data.shape
frequencies = np.fft.fftfreq(time_samples, d=1/sampling_rate)
magnitude_spectra = np.abs(np.fft.fft(eeg_data, axis=1))
mean_magnitude_spectra = np.mean(magnitude_spectra, axis=0)
mean_magnitude_spectra = np.mean(mean_magnitude_spectra, axis=-1)
# Plot the magnitude spectra for each task
num_tasks = len(task_names)
plt.figure(figsize=(12, 6))

for task_index in range(num_tasks):
    plt.subplot(num_tasks, 1, task_index + 1)
    plt.plot(frequencies, mean_magnitude_spectra[:, task_index])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Temporal Frequency Analysis - Task: {task_names[task_index]}')

plt.tight_layout()
plt.show()
print(mean_magnitude_spectra.shape)