# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import scipy.io as sio
# # Load the .mat file
# mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# # Access the EEG data from the loaded .mat file
# eegdat = mat_data['eegdat']

# # Define STFT parameters
# window_size = 256  # Size of the window for each STFT segment
# overlap = int(window_size * 0.75)  # Amount of overlap between adjacent segments

# # Initialize an array to store the spectrograms
# spectrograms = []

# # Apply STFT to each electrode's data
# for electrode_data in eegdat:
#     electrode_spectrograms = []
#     for task_data in electrode_data:
#         # Concatenate all repetitions for the current task
#         task_repetitions = np.concatenate(task_data, axis=0)

#         # Compute STFT
#         freqs, times, Sxx = signal.stft(task_repetitions, fs=600, nperseg=window_size, noverlap=overlap)

#         # Store the spectrogram
#         electrode_spectrograms.append(np.abs(Sxx))

#     spectrograms.append(electrode_spectrograms)

# # Plot the spectrograms
# electrode_names = ['Electrode 1', 'Electrode 2', ..., 'Electrode 45']
# task_names = ['Task 1', 'Task 2', ..., 'Task 8']

# for electrode_idx, electrode_spectrogram in enumerate(spectrograms):
#     for task_idx, task_spectrogram in enumerate(electrode_spectrogram):
#         # Check if the spectrogram has the correct shape
#         if task_spectrogram.ndim == 2:
#             mean_spectrogram = np.mean(task_spectrogram, axis=0)
#         else:
#             mean_spectrogram = task_spectrogram

#         # Compute the time axis based on window size and overlap
#         time_axis = np.arange(mean_spectrogram.shape[0]) * overlap / 600

#         plt.figure(figsize=(10, 6))
#         plt.imshow(np.log10(mean_spectrogram), origin='lower', aspect='auto', cmap='jet',
#                    extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]])
#         plt.colorbar(label='Log Magnitude')
#         plt.title(f'Spectrogram for {electrode_names[electrode_idx]}, {task_names[task_idx]}')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Frequency (Hz)')
#         plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']
# Assuming you have the EEG data stored in a variable called 'eegdat'
# with shape (45, 1200, 8, 28)

# Select a specific electrode for analysis (e.g., electrode index 0)
electrode_idx = 0
eeg_data = eegdat[electrode_idx]

# Parameters for STFT
window_size = 256
overlap = 128

# Compute the STFT
frequencies, times, stft_data = spectrogram(
    eeg_data, fs=600, window='hann', nperseg=window_size, noverlap=overlap, axis=-3
)
stft_data = stft_data.T
# Plot the STFT
plt.pcolormesh(times, frequencies, 10 * np.log10(stft_data))
plt.title('STFT of EEG Data')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power Spectral Density (dB)')
plt.show()
