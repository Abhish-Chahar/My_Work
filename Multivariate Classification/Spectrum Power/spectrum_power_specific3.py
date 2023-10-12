import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio

# Load the EEG data from the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']

# Defining the sampling rate for spectrum power calculation
sampling_rate = 600  # Sampling rate in Hz

# Calculate the frequency bin indices corresponding to the entire frequency range
n_samples = eegdat.shape[1]
frequency_bins = np.fft.rfftfreq(n_samples, d=1/sampling_rate)

# Number of grasp tasks
num_grasp_tasks = eegdat.shape[2]

# Iterate over each grasp task
for grasp_task in range(num_grasp_tasks):
    # Initializing an array to store the spectrum power spectrogram for the current task
    

    # Iterate over each electrode and repetition
    for electrode in range(eegdat.shape[0]):
        for repetition in range(eegdat.shape[3]):
            # Extracting the EEG data for the current combination
            eeg_data = eegdat[electrode, :, grasp_task, repetition]

            # Applying STFT to the EEG data
            f, t, stft_data = stft(eeg_data, fs=sampling_rate, nperseg=512, noverlap=256)
            spectrum_power_spectrogram = np.zeros((eegdat.shape[0], len(f), eegdat.shape[1], eegdat.shape[3]))
            # Calculating the spectrum power spectrogram
            spectrum_power = np.abs(stft_data)**2
            expanded_spectrum_power = np.repeat(spectrum_power, [200] * 6, axis=-1)
            spectrum_power_spectrogram[electrode, :, :, repetition] = expanded_spectrum_power

    # Calculate the average spectrum power spectrogram across repetitions
    average_spectrogram = np.mean(spectrum_power_spectrogram, axis=-1)

    # Average across electrodes
    average_spectrogram = np.mean(average_spectrogram, axis=0)

    # Plotting the spectrograms
    plt.imshow(average_spectrogram, aspect='auto', origin='lower', cmap='jet', extent=[0, eegdat.shape[1]/sampling_rate, frequency_bins[0], frequency_bins[-1]])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Task {grasp_task+1} Spectrum Power Spectrogram')
    plt.colorbar(label='Power')
    plt.show()
