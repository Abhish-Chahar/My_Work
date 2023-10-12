import numpy as np

eegdat = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
array2 = np.array([])

for electrode in range(eegdat.shape[0]):
    for repetition in range(eegdat.shape[1]):
        # Extracting the EEG data for the current combination
        eeg_data = eegdat[electrode, :, repetition]

        # Applying STFT to the EEG data
        f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=256, noverlap=128)

        # Calculating the band power spectrogram
        band_power = np.sum(np.abs(stft_data[start_idx:end_idx])**2, axis=0)
        # mean_band_power = np.mean(band_power)
        # band_power_spectrogram[electrode, :, repetition] = band_power

        expanded_band_power = np.repeat(band_power, [109] * 10 + [110])
        band_power_spectrogram[electrode, :, repetition] = expanded_band_power[:n_samples]


# Calculating the average band power spectrogram across repetitions
average_spectrogram = np.sum(band_power_spectrogram, axis=-1)