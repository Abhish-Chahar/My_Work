import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']
# Load the EEG data
# Assuming you have the data stored in a variable called 'eegdat'

# Define wavelet transform parameters
wavelet_name = 'morl'  # Name of the wavelet
scales = np.arange(1, 50)  # Range of scales for wavelet transform

# Initialize an array to store the wavelet scalograms
scalograms = []

# Apply wavelet transform to each electrode's data
for electrode_data in eegdat:
    electrode_scalograms = []
    for task_data in electrode_data:
        # Concatenate all repetitions for the current task
        task_repetitions = np.concatenate(task_data, axis=0)

        # Perform continuous wavelet transform
        coefficients, frequencies = pywt.cwt(task_repetitions, scales, wavelet_name, sampling_period=1/600)

        # Calculate the scalogram by taking the squared magnitude of the coefficients
        scalogram = np.abs(coefficients) ** 2

        # Store the scalogram
        electrode_scalograms.append(scalogram)

    scalograms.append(electrode_scalograms)

# Plot the wavelet scalograms
electrode_names = ['Electrode 1', 'Electrode 2', ..., 'Electrode 45']
task_names = ['Task 1', 'Task 2', ..., 'Task 8']

for electrode_idx, electrode_scalogram in enumerate(scalograms):
    for task_idx, task_scalogram in enumerate(electrode_scalogram):
        plt.figure(figsize=(10, 6))
        plt.imshow(task_scalogram, origin='lower', aspect='auto', cmap='jet',
                   extent=[0, task_scalogram.shape[1], scales[0], scales[-1]])
        plt.colorbar(label='Magnitude')
        plt.title(f'Wavelet Scalogram for {electrode_names[electrode_idx]}, {task_names[task_idx]}')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.show()
