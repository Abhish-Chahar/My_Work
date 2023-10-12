import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import stft

# Load EEG data from the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Define the sampling rate for spectrum power calculation
sampling_rate = 600  # Sampling rate in Hz

# Define the baseline and task-related epoch parameters
baseline_start = 0  # Start time of the baseline period in seconds
baseline_end = 1  # End time of the baseline period in seconds
task_start = 1.5  # Start time of the task-related epoch in seconds
task_end = 2.5  # End time of the task-related epoch in seconds

# Initialize an array to store the ERD/ERS values
erd_ers = np.zeros((eegdat.shape[0], eegdat.shape[2], eegdat.shape[3]))

# Iterate over each electrode, grasp task, and repetition
for electrode in range(eegdat.shape[0]):
    for grasp_task in range(eegdat.shape[2]):
        for repetition in range(eegdat.shape[3]):
            # Extract the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Calculate the baseline and task-related indices
            baseline_start_index = int(baseline_start * sampling_rate)
            baseline_end_index = int(baseline_end * sampling_rate)
            task_start_index = int(task_start * sampling_rate)
            task_end_index = int(task_end * sampling_rate)

            # Apply STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=256, noverlap=128)

            # Calculate the baseline and task-related power
            baseline_power = np.mean(np.abs(stft_data[:, baseline_start_index:baseline_end_index])**2, axis=1)
            task_power = np.mean(np.abs(stft_data[:, task_start_index:task_end_index])**2, axis=1)

            # Calculate the ERD/ERS values for each frequency bin
            erd_ers[electrode, grasp_task, repetition] = (task_power - baseline_power) / baseline_power

# Average the ERD/ERS values across repetitions
avg_erd_ers = np.mean(erd_ers, axis=2)

# Plot the ERD/ERS results
plt.imshow(avg_erd_ers.T, aspect='auto', origin='lower', cmap='jet')
plt.xlabel('Grasp Task')
plt.ylabel('Electrode')
plt.title('ERD/ERS')
plt.colorbar(label='ERD/ERS')
plt.show()
