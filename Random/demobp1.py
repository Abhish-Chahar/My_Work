import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio

# Load the EEG data
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Define the frequency range for STFT
frequency_range = (8, 30)  # Frequency range in Hz

# Define the sampling rate
sampling_rate = 600  # Sampling rate in Hz

# Calculate the frequency bin indices corresponding to the frequency range
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)
start_freq, end_freq = frequency_range
start_idx = np.argmax(frequency_bins >= start_freq)
end_idx = np.argmax(frequency_bins >= end_freq) + 1  # Add 1 to include the end frequency bin

# Initialize an array to store the spectrogram for each feature
band_power_spectrogram = np.zeros((eegdat.shape[2], eegdat.shape[3], end_idx - start_idx))
# spectrum_power_spectrogram = np.zeros((eegdat.shape[2], eegdat.shape[3], len(frequency_bins)))
# psd_spectrogram = np.zeros((eegdat.shape[2], eegdat.shape[3], end_idx - start_idx))

# Iterate over each grasp task and repetition
for grasp_task in range(eegdat.shape[2]):
    for repetition in range(eegdat.shape[3]):
        # Extract the EEG data for the current grasp task and repetition
        eeg_data = eegdat[:, :, grasp_task, repetition]

        # Apply STFT to the EEG data
        f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=256, noverlap=128)

        # Calculate the band power spectrogram
        band_power_spectrogram[grasp_task, repetition] = np.sum(np.abs(stft_data[start_idx:end_idx])**2, axis=0)

        # # Calculate the spectrum power spectrogram
        # spectrum_power_spectrogram[grasp_task, repetition] = np.sum(np.abs(stft_data)**2, axis=0)

        # # Calculate the PSD spectrogram
        # psd = np.abs(stft_data)**2 / (sampling_rate * np.mean(np.hanning(256)**2))
        # psd_spectrogram[grasp_task, repetition] = psd[start_idx:end_idx]

# Plot the band power spectrogram
plt.figure()
plt.imshow(np.mean(band_power_spectrogram, axis=(0, 1)).T, aspect='auto', origin='lower', cmap='jet')
plt.xlabel('Grasp Task')
plt.ylabel('Time')
plt.title('Band Power Spectrogram')
plt.colorbar(label='Power')
plt.show()

# # Plot the spectrum power spectrogram
# plt.figure()
# plt.imshow(np.mean(spectrum_power_spectrogram, axis=(0, 1)).T, aspect='auto', origin='lower', cmap='jet')
# plt.xlabel('Grasp Task')
# plt.ylabel('Frequency Bin')
# plt.title('Spectrum Power Spectrogram')
# plt.colorbar(label='Power')
# plt.show()

# # Plot the PSD spectrogram
# plt.figure()
# plt.imshow(np.mean(psd_spectrogram, axis=(0, 1)).T, aspect='auto', origin='lower', cmap='jet')
# plt.xlabel('Grasp Task')
# plt.ylabel('Frequency (Hz)')
# plt.title('Power Spectral Density (PSD) Spectrogram')
# plt.colorbar(label='Power')
# plt.show()