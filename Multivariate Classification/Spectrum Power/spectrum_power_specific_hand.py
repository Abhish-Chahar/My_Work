import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Defining the sampling rate for spectrum power calculation
sampling_rate = 600  # Sampling rate in Hz

# Calculating the frequency bin indices corresponding to the entire frequency range
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)

# Iterating over each grasp task
for grasp_task in range(eegdat.shape[2]):
    # Initializing an array to store the spectrum power spectrogram for the current task
    spectrum_power_spectrogram = np.zeros((eegdat.shape[0], eegdat.shape[1], eegdat.shape[-1]))

    # Iterating over each electrode and repetition
    for repetition in range(eegdat.shape[3]):
        for electrode in range(eegdat.shape[0]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=512, noverlap=256)

            # Calculating the spectrum power spectrogram
            spectrum_power = np.sum(np.abs(stft_data)**2, axis=0)
            expanded_spectrum_power = np.repeat(spectrum_power, [200] * 6)
            spectrum_power_spectrogram[electrode, :, repetition] = expanded_spectrum_power

        # Calculating the average spectrum power spectrogram across repetitions
        average_spectrogram = np.mean(spectrum_power_spectrogram, axis=-1)

    # Plotting the spectrograms
    # plt.imshow(average_spectrogram.T, aspect='auto', origin='lower', cmap='jet')
    # plt.xlabel('Electrodes')
    # plt.ylabel('Time samples')
    # plt.title(f'Task {grasp_task+1} Spectrum Power Spectrogram')
    # plt.colorbar(label='Power')
    # plt.show()
print(spectrum_power.shape)