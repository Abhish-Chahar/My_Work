import numpy as np
import scipy.io as sio
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load EEG data
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']
electrodes, time_samples, grasp_tasks, repetitions = 45, 1200, 8, 28
save_directory = r"C:\Users\user\Downloads\split data"

# Prepare data for classification
X = []
y = []

# Loop through each grasp task
for task in range(grasp_tasks):
    for repetition in range(repetitions):
        # Extract data for the current task and repetition
        task_repetition_data = eegdat[:, :, task, repetition]

        # Apply STFT to the data
        f, t, Zxx = stft(task_repetition_data, nperseg=64)  # Adjust nperseg as needed
        
        # Reshape STFT data for classification
        stft_features = Zxx.reshape(-1, Zxx.shape[-1])
        
        X.append(np.abs(stft_features))
        y.append(task)

X = np.array(X)
y = np.array(y)

# Flatten the 3D STFT features into a 2D array
X_flat = X.reshape(X.shape[0], -1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Initialize and train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
