import array
import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eeg_data = mat_data['eegdat']
# Define the frequency band and time window of interest
freq_band = [8, 30]  # Hz
time_window = [0.5, 1.5]  # seconds

# Extract the time samples within the time window of interest
time_start = int(time_window[0] * 600)  # Convert time to samples
time_end = int(time_window[1] * 600)
time_samples = eeg_data[:, time_start:time_end, :, :]

# Calculate the baseline activity
baseline_start = int(0.0 * 600)  # Use first second as baseline
baseline_end = int(1.0 * 600)
baseline = np.mean(eeg_data[:, baseline_start:baseline_end, :, :], axis=1, keepdims=True)

# Calculate the event-related desynchronization (ERD)
ERD = (baseline - time_samples) / baseline * 100  # Calculate as a percentage

# Average across repetitions and concatenate across tasks
ERD = np.mean(ERD, axis=3)  # Average across repetitions
ERD = np.concatenate([ERD[:, :, i] for i in range(8)], axis=1) # Concatenate across tasks

# Reshape the data into a 2D matrix for machine learning
X = ERD.reshape(-1, ERD.shape[-1])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM model
clf = svm.SVC(kernel='linear')

# Fit the model to the training data
clf.fit(X_train, y_train)

# Predict the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy:", accuracy)
 