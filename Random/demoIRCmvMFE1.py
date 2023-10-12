import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']



# Step 2: Extract features using IRCmvMFE
features = []
for task_idx in range(8):
    task_data = eegdat[:, :, task_idx, :]
    covariances = []
    tangentspace_projections = []
    # for repetition_idx in range(28):
    #     repetition_data = task_data[repetition_idx]
    #     cov_estimator = Covariances()
    #     covariance = cov_estimator.transform(repetition_data)
    #     covariances.append(covariance)

    # mean_covariance = np.mean(covariances, axis=0)
    # for repetition_idx in range(28):
    #     tangent_space = TangentSpace(metric='riemann')
    #     tangent_projection = tangent_space.fit_transform(covariances[repetition_idx], mean_covariance)
    #     tangentspace_projections.append(tangent_projection)

    # features.extend(tangentspace_projections)
features = np.array(features)

# Step 3: Plot the spectrograms
selected_task_idx = 0  # Replace with the desired task index (0-7)
sampling_rate = 600
time_samples = np.arange(0, 2, 1 / sampling_rate)
frequency_range = (8, 30)

task_features = features[selected_task_idx * 28: (selected_task_idx + 1) * 28]
averaged_features = np.mean(task_features, axis=0)
frequencies, times, spectrogram_values = spectrogram(averaged_features.T, fs=sampling_rate, nperseg=256, noverlap=128)

# Step 4: Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram_values, aspect='auto', cmap='hot_r', origin='lower',extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
plt.colorbar(label='Magnitude')
plt.title(f'Spectrogram for Hand Task {selected_task_idx + 1}')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(frequency_range)

plt.show()
