import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io as sio
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Defining wavelet transform parameters
wavelet_name = 'morl'  # Name of the wavelet
scales = np.arange(1, 50)  # Range of scales for wavelet transform

electrode_idx = 0
grasp_task_idx = 0
repetition_idx = 0

# Extracting the EEG data for the selected electrode, grasp task, and repetition
eeg_data = eegdat[electrode_idx, :, grasp_task_idx, repetition_idx]

# Performing continuous wavelet transform
coefficients, frequencies = pywt.cwt(eeg_data, scales, wavelet_name, sampling_period=1/600)

# Plotting the wavelet coefficients
plt.figure(figsize=(10, 6))
plt.plot(coefficients.T)
plt.title(f'Wavelet Coefficients for Electrode {electrode_idx+1}, Task {grasp_task_idx+1}, Repetition {repetition_idx+1}')
plt.xlabel('Time')
plt.ylabel('Coefficient')
plt.show()