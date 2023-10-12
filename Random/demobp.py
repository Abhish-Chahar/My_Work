import numpy as np
from mne import Epochs, pick_types, events_from_annotations, concatenate_epochs
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP
from mne.time_frequency import psd
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']
# Define the events from the annotations
events, event_id = events_from_annotations(eegdat)

# Create the epoch object
epochs = Epochs(eegdat, events=events, event_id=event_id, tmin=0, tmax=2, baseline=None, preload=True)


# Assume "eegdat" is a 4D numpy array with shape (45, 1200, 8, 28)
# Define the frequency bands of interest
freq_bands = {'Theta': [4, 8],
              'Alpha': [8, 12], 
              'Beta': [12, 30]}

# Compute the power spectral density (PSD) for each trial in each frequency band
epochs = Epochs(eegdat, events=None, tmin=0, tmax=2, baseline=None, preload=True)
psds, freqs = psd(epochs, fmin=4, fmax=30, picks='eeg', n_jobs=-1)

# Compute the band power for each trial in each frequency band
for band, (fmin, fmax) in freq_bands.items():
    # Find the indices of the frequency band in the PSD frequency vector
    freq_ix = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    # Sum the PSD values within the frequency band for each trial
    band_power = psds[:, freq_ix, :].sum(axis=1)
    # Add the band power as a feature to the data
    eegdat = np.concatenate([eegdat, band_power[:, :, np.newaxis, np.newaxis]], axis=2)

# The resulting "eegdat" array will have shape (45, 1200, 11, 28) 
# (8 original tasks + 3 frequency bands) for each repetition
