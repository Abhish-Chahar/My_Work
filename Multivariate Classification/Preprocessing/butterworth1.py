import numpy as np
import scipy.signal
import scipy.io as sio
import matplotlib.pyplot as plt

# Loading the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGdata.mat")

# Access the EEG data from the loaded .mat file
eegdat = mat_data['data']

# Defining the sampling frequency and desired frequency bands
sampling_freq = 600  # Hz
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
}

# Computing the Nyquist frequency
nyquist_freq = 0.5 * sampling_freq

# Iterating over the frequency bands and apply filtering
filtered_data = {}
for band_name, (low_freq, high_freq) in frequency_bands.items():
    # Computing the normalized frequencies
    low_freq_normalized = low_freq / nyquist_freq
    high_freq_normalized = high_freq / nyquist_freq

    # Designing the Butterworth filter
    b, a = scipy.signal.butter(4, [low_freq_normalized, high_freq_normalized], btype='band')

    # Applying the filter to each repetition and electrode
    filtered_data[band_name] = np.zeros_like(eegdat)
    for subject in range(eegdat.shape[4]):
        for repetition in range(eegdat.shape[2]):
            for electrode in range(eegdat.shape[0]):
                filtered_data[band_name][electrode, :, repetition, :] = scipy.signal.filtfilt(b, a, eegdat[electrode, :, repetition, :], axis=0)

# Create a new dictionary to store filtered data in the same format as original data
filtered_data_same_format = {}

# Iterate over the frequency bands
for band_name, _ in frequency_bands.items():
    filtered_data_same_format[band_name] = np.zeros_like(eegdat)  # Initialize an array with the same shape as the original data
    
    # Iterate over subjects, repetitions, electrodes, and tasks to populate the filtered data
    for subject in range(eegdat.shape[4]):
        for repetition in range(eegdat.shape[2]):
            for electrode in range(eegdat.shape[0]):
                for task in range(eegdat.shape[3]):
                    # Replace the corresponding portion of the filtered_data_same_format with filtered values
                    filtered_data_same_format[band_name][electrode, :, repetition, task, subject] = filtered_data[band_name][electrode, :, repetition, task, subject]

# Accessing filtered data for a specific frequency band
filtered_delta_data = filtered_data['delta']
filtered_theta_data = filtered_data['theta']
filtered_alpha_data = filtered_data['alpha']
filtered_beta_data = filtered_data['beta']

# Select a subject index to plot
subject_index = 0

# Select a sample electrode index to plot
electrode_index = 0

# Select a sample task index to plot
hand_task_index = 0

# Select a sample repetition index to plot
repetition_index = 0

# Get the original EEG data and filtered data for the selected electrode and repetition
original_data = eegdat[electrode_index, :, repetition_index, hand_task_index, subject_index]
specific_filtered_delta_data = filtered_delta_data[electrode_index, :, repetition_index, hand_task_index, subject_index]
specific_filtered_theta_data = filtered_theta_data[electrode_index, :, repetition_index, hand_task_index, subject_index]
specific_filtered_alpha_data = filtered_alpha_data[electrode_index, :, repetition_index, hand_task_index, subject_index]
specific_filtered_beta_data = filtered_beta_data[electrode_index, :, repetition_index, hand_task_index, subject_index]

# Creating a time vector based on the sampling frequency
time = np.arange(original_data.shape[1]) / sampling_freq

# Plot the original EEG and filtered EEG
plt.figure(figsize=(12, 6))
plt.plot(time, original_data, label='Original EEG')
plt.plot(time, specific_filtered_delta_data, label='Filtered EEG (delta band)')
plt.plot(time, specific_filtered_theta_data, label='Filtered EEG (theta band)')
plt.plot(time, specific_filtered_alpha_data, label='Filtered EEG (alpha band)')
plt.plot(time, specific_filtered_beta_data, label='Filtered EEG (beta band)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Electrode_0 Task_0 Repetition_0')
plt.legend()
plt.grid(True)
plt.show()
