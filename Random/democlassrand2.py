import numpy as np
import pandas as pd
import os
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

csv_folder = os.path.expanduser(r"C:\Users\user\Downloads\Telegram Desktop\EEG_CSV_Output\EEG_CSV_Output")

# Choose the task repetition and electrode for plotting
# selected_task = 1  # Choose the task number
selected_repetition = 1  # Choose the repetition number
selected_electrode = 0
for selected_task in range (1, 10):
    # Load EEG data for the selected task repetition and electrode
    filename = os.path.join(csv_folder, f"Task_{selected_task}_Rep_{selected_repetition}.csv")
    df = pd.read_csv(filename)
    selected_electrode_data = df.iloc[:, selected_electrode]

# Define STFT parameters
fs = 600  # Sampling frequency (Hz)
nperseg = 256  # Length of each segment
noverlap = 128  # Overlap between segments

# Calculate the STFT for the selected electrode data
f, t, Sxx = spectrogram(selected_electrode_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
Szz = np.abs(Sxx)

# Plot the spectrogram using imshow
plt.figure(figsize=(10, 6))
plt.imshow(Szz, aspect='auto', cmap='jet', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()])
plt.colorbar(label='Magnitude')
plt.title(f'Spectrogram for Task {selected_task}, Repetition {selected_repetition}, Electrode {selected_electrode}')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

# Load EEG data for the selected task repetition and electrode
filename = os.path.join(csv_folder, f"Task_{selected_task}_Rep_{selected_repetition}.csv")
df = pd.read_csv(filename)
selected_electrode_data = df.iloc[:, selected_electrode]

# Define STFT parameters
fs = 600  # Sampling frequency (Hz)
nperseg = 256  # Length of each segment
noverlap = 128  # Overlap between segments

# Calculate the STFT for the selected electrode data
f, t, Sxx = spectrogram(selected_electrode_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
Szz = np.abs(Sxx)

# Plot the spectrogram using imshow
plt.figure(figsize=(10, 6))
plt.imshow(Szz, aspect='auto', cmap='jet', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()])
plt.colorbar(label='Magnitude')
plt.title(f'Spectrogram for Task {selected_task}, Repetition {selected_repetition}, Electrode {selected_electrode}')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()