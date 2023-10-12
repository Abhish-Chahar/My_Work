import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']
# Define the wavelet function to be used
wavelet = 'morl'

# Define the frequency bands of interest
freq_bands = [(8, 12), (13, 17), (18, 22), (23, 30)]

# Perform wavelet transform on each electrode's data
wavelet_coeffs = []
for electrode_data in eegdat:
    electrode_coeffs = []
    for grasp_task_data in electrode_data:
        task_coeffs = []
        for repetition_data in grasp_task_data:
            coeffs, _ = pywt.cwt(repetition_data, np.arange(1, 129), wavelet)
            task_coeffs.append(coeffs)
        electrode_coeffs.append(task_coeffs)
    wavelet_coeffs.append(electrode_coeffs)
wavelet_coeffs = np.array(wavelet_coeffs)

# Calculate the average coefficients across repetitions for each task
avg_coeffs = np.mean(wavelet_coeffs, axis=3)

# Plot the wavelet coefficients for each frequency band
for i, freq_band in enumerate(freq_bands):
    band_coeffs = avg_coeffs[:, :, :, freq_band[0]:freq_band[1]]
    mean_band_coeffs = np.mean(band_coeffs, axis=(0, 1, 3))

    # Plotting the average coefficients for each task
    plt.figure()
    for task in range(8):
        plt.plot(mean_band_coeffs[task], label=f'Task {task+1}')
    plt.title(f'Wavelet Coefficients - Frequency Band {freq_band}')
    plt.xlabel('Time Sample')
    plt.ylabel('Average Coefficient')
    plt.legend()
    plt.show()
