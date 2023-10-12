import numpy as np
from mne.decoding import CSP
from scipy.signal import spectrogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']


def extract_features(eegdat):
    num_channels, num_samples, num_tasks, num_repetitions = eegdat.shape

    # Reshape the data to match the input format for IRCmvMFE
    reshaped_data = np.reshape(eegdat, (num_channels, -1, num_tasks * num_repetitions))

    # Calculate the spectrogram for each channel
    spectrograms = []
    for channel_data in reshaped_data:
        _, _, Sxx = spectrogram(channel_data, fs=600, nperseg=128, noverlap=64)
        spectrograms.append(Sxx)

    # Apply IRCmvMFE to obtain features
    features = []
    for spectrogram_data in spectrograms:
        feature_vector = np.mean(spectrogram_data, axis=1)
        features.append(feature_vector)

    return np.array(features)


# Example usage
eegdat = np.random.rand(45, 1200, 8, 28)  # Replace with your actual EEG data

# Perform feature extraction
features = extract_features(eegdat)
print(features)  # Print the shape of the extracted features
