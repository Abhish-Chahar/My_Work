import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Defining the frequency range and sampling rate for band power calculation
frequency_range = (8, 30)  # Frequency range in Hz
sampling_rate = 600  # Sampling rate in Hz

# Calculating the frequency bin indices corresponding to the frequency range
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)
start_freq, end_freq = frequency_range
start_idx = np.argmax(frequency_bins >= start_freq)
end_idx = np.argmax(frequency_bins >= end_freq) + 1  # Add 1 to include the end frequency bin

# Iterating over each grasp task
for grasp_task in range(eegdat.shape[2]):
    # Initializing an array to store the band power spectrogram for the current task
    band_power_spectrogram = np.zeros((eegdat.shape[0], eegdat.shape[1], end_idx - start_idx))

    # Iterating over each electrode and repetition
    for electrode in range(eegdat.shape[0]):
        for repetition in range(eegdat.shape[3]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=512, noverlap=256)

            # Calculating the band power spectrogram
            band_power = np.sum(np.abs(stft_data[start_idx:end_idx])**2, axis=0)
            expanded_band_power = np.repeat(band_power, [200] * 6)
            band_power_spectrogram[electrode, :, repetition] = expanded_band_power

    # Calculating the average band power spectrogram across repetitions
    average_spectrogram = np.mean(band_power_spectrogram, axis=-1)

    # Plotting the spectrograms
    plt.imshow(average_spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
    plt.xlabel('Electrodes')
    plt.ylabel('Time samples')
    plt.title(f'Task {grasp_task+1} Band Power Spectrogram')
    plt.colorbar(label='Power')
    plt.show()