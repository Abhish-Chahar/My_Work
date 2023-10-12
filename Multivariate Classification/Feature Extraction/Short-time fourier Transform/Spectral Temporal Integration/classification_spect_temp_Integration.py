import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.signal import spectrogram
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

csv_folder = os.path.expanduser(r"C:\Users\user\Downloads\Telegram Desktop\EEG_CSV_Output\EEG_CSV_Output")

num_grasp_tasks = 8
num_repetitions = 28

# Defining STFT parameters
fs = 600  # Sampling frequency (Hz)
nperseg = 600  # Length of each segment
noverlap = 300  # Overlap between segments


# Initializing a list to store STFT features for each task
stft_features = []

# Loading EEG data for each task and calculate STFT features
for task in range(1, num_grasp_tasks + 1):
    task_stft = []
    for repetition in range(1, num_repetitions + 1):
        filename = os.path.join(csv_folder, f"Task_{task}_Rep_{repetition}.csv")
        df = pd.read_csv(filename)
        for electrode in range(df.shape[1]):
            f, t, Sxx = spectrogram(df.iloc[:, electrode], fs=fs, nperseg=nperseg, noverlap=noverlap)
            task_stft.append(np.abs(Sxx))
    stft_features.append(task_stft)
# Converting the list to a NumPy array
stft_features = np.array(stft_features)
print(stft_features.shape)
print(df.shape)

# Reshaping the STFT features array to have a single dimension for each task
tasks, repetitions, freq_bins, time_stamps = stft_features.shape
X = stft_features.reshape(tasks * repetitions, freq_bins * time_stamps)
y = np.repeat(np.arange(1, tasks + 1), repetitions)
# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Splittig the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Creating an SVM Classifier model
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Classifier Accuracy:", svm_accuracy)


# Initializing and training the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("RF Classifier Accuracy:", accuracy)


# Creating a base classifier (you can choose Random Forest or SVM)
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Creating a Bagging Classifier model
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)
bagging_predictions = bagging_classifier.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
print("Bagging Classifier Accuracy:", bagging_accuracy)