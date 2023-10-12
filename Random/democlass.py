import numpy as np
import scipy.io as sio
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import stft
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
eegdat = mat_data['eegdat']
electrodes, time_samples, grasp_tasks, repetitions = 45, 1200, 8, 28

# Set up STFT parameters
fs = 600  # Sampling rate
nperseg = 512  # Number of data points in each STFT segment
noverlap = nperseg / 2  # Overlap between segments

# Prepare data for classification
X = []
y = []

for task in range(grasp_tasks):
    for repetition in range(repetitions):
        task_repetition_data = eegdat[:, :, task, repetition]
        for electrode in range(electrodes):
            _, _, Zxx = stft(task_repetition_data[electrode, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
            X.append(np.abs(Zxx))  # Use the magnitude of STFT coefficients
            y.append(task)

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Initialize and train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()