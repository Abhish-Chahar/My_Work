import numpy as np
from pyentrp import entropy
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eeg_data = mat_data['eegdat']
num_electrodes=45
num_grasp_tasks=8
num_repetitions=28

# Define parameters for sample entropy feature extraction
m = 2  # Embedding dimension
r = 0.2 * np.std(eeg_data)  # Tolerance threshold

# Initialize an array to store the sample entropy features
sample_entropy_features = np.zeros((num_electrodes))

# Iterate over each electrode
for electrode in range(num_electrodes):
    # Extract the EEG data for the current electrode
    eeg_segment = eeg_data[electrode].reshape(-1, num_grasp_tasks * num_repetitions)

    # Calculate the sample entropy feature
    sample_entropy = entropy.sample_entropy(eeg_segment, m, r)
    
    # Store the sample entropy feature
    sample_entropy_features[electrode] = sample_entropy

# Perform further analysis or visualization with the sample entropy features
