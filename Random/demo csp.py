import numpy as np
from mne.decoding import CSP
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']

# Reshape the data to be compatible with CSP algorithm
n_channels, n_samples, n_trials = eegdat.shape[:3]
eegdata = np.transpose(eegdat, (2, 3, 1, 0))
eegdata = eegdata.reshape((n_trials * n_channels, n_samples, n_channels))

# Define the CSP parameters
n_components = 4  # Number of CSP components to extract
csp = CSP(n_components=n_components, reg=None, log=True, transform_into='csp_space')

# Apply the CSP algorithm to the EEG data
X_csp = csp.fit_transform(eegdata)

# Reshape the CSP features back to the original trial format
X_csp = X_csp.reshape((n_trials, n_channels, n_components, n_samples))
X_csp = np.transpose(X_csp, (0, 3, 1, 2))

# Average the CSP features across time to obtain a single feature vector per trial
X_csp_mean = np.mean(X_csp, axis=1)  # Shape: (8, 45, 4)
