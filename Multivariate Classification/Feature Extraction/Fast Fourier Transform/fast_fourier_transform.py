import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat=mat_data['eegdat']
# Defining the frequency range for FFT
frequency_range = (8, 30)  # Frequency range in Hz

# Calculate=ing the frequency bin indices corresponding to the frequency range
sampling_rate = 600  # Sampling rate in Hz
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)

# Finding the indices corresponding to the desired frequency range
start_freq, end_freq = frequency_range
start_idx = np.argmax(frequency_bins >= start_freq)
end_idx = np.argmax(frequency_bins >= end_freq)

# Initializing an array to store the FFT features
fft_features = np.zeros((eegdat.shape[0], eegdat.shape[2], eegdat.shape[3], end_idx - start_idx))

# Iterating over each electrode, grasp task, and repetition
for electrode in range(eegdat.shape[0]):
    for grasp_task in range(eegdat.shape[2]):
        for repetition in range(eegdat.shape[3]):
            # Extract the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]
            
            # Apply FFT to the EEG data
            fft = np.abs(np.fft.rfft(eeg_data))
            
            # Extract the desired frequency range
            fft_range = fft[start_idx:end_idx]
            
            # Store the FFT features
            fft_features[electrode, grasp_task, repetition] = fft_range

# Plotting the FFT features for a specific electrode, grasp task, and repetition
electrode_idx = 0
grasp_task_idx = 0
repetition_idx = 0

plt.plot(frequency_bins[start_idx:end_idx], fft_features[electrode_idx, grasp_task_idx, repetition_idx])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Features')
plt.show()
