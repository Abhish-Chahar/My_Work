import numpy as np
import scipy.signal
import scipy.io as sio
import matplotlib.pyplot as plt

# Loading the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")

# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']

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
    for repetition in range(eegdat.shape[3]):
        for electrode in range(eegdat.shape[0]):
            filtered_data[band_name][electrode, :, :, repetition] = scipy.signal.filtfilt(b, a, eegdat[electrode, :, :, repetition], axis=0)

# Accessing the filtered data for all EEG frequency band
filtered_delta_data = filtered_data['delta']
filtered_theta_data = filtered_data['theta']
filtered_alpha_data = filtered_data['alpha']
filtered_beta_data = filtered_data['beta']
# print(filtered_beta_data)


# Select a sample electrode index to plot
electrode_index = 0

# Select a sample electrode index to plot
hand_task_index = 0

# Select a sample repetition index to plot
repetition_index = 0

# Get the original EEG data and filtered data for the selected electrode and repetition
original_data = eegdat[electrode_index, :, hand_task_index, repetition_index]
# filtered_delta__data = filtered_data['delta'][electrode_index, :, hand_task_index, repetition_index]
filtered_theta__data = filtered_data['theta'][electrode_index, :, hand_task_index, repetition_index]
filtered_alpha__data = filtered_data['alpha'][electrode_index, :, hand_task_index, repetition_index]
filtered_beta__data = filtered_data['beta'][electrode_index, :, hand_task_index, repetition_index]

# Creating a time vector based on the sampling frequency
sampling_freq = 600  # Hz
time = np.arange(original_data.shape[0]) / sampling_freq

# Plot the original EEG and filtered EEG
plt.figure(figsize=(12, 6))
plt.plot(time, original_data, label='Original EEG')
# plt.plot(time, filtered_delta__data, label='Filtered EEG (' + 'delta' + ' band)')
plt.plot(time, filtered_theta__data, label='Filtered EEG (' + 'theta' + ' band)')
plt.plot(time, filtered_alpha__data, label='Filtered EEG (' + 'alpha' + ' band)')
plt.plot(time, filtered_beta__data, label='Filtered EEG (' + 'beta' + ' band)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Electrode_0 Task_0 Repetition_0')
plt.legend()
plt.grid(True)
plt.show()
