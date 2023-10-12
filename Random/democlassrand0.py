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

# Define STFT parameters
fs = 600  # Sampling frequency (Hz)
nperseg = 600  # Length of each segment
noverlap = 300  # Overlap between segments


# Initialize a list to store feature vectors for each task
feature_vectors = []

# Load EEG data for each task and calculate features
for task in range(1, num_grasp_tasks + 1):
    task_features = []
    for repetition in range(1, num_repetitions + 1):
        filename = os.path.join(csv_folder, f"Task_{task}_Rep_{repetition}.csv")
        df = pd.read_csv(filename)
        for electrode in range(df.shape[1]):
            f, t, Sxx = spectrogram(df.iloc[:, electrode], fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            # Extract relevant features from the STFT matrix
            mean_power = np.mean(np.abs(Sxx), axis=1)  # Mean power in each frequency bin
            max_power = np.max(np.abs(Sxx), axis=1)    # Max power in each frequency bin
            
            electrode_features = np.concatenate((mean_power, max_power))
            task_features.append(electrode_features)
    feature_vectors.append(task_features)
# Convert the list to a NumPy array
feature_vectors = np.array(feature_vectors)
print(feature_vectors.shape)

# Reshape the features for training
tasks, repetitions, rel_features = feature_vectors.shape
X = feature_vectors.reshape(tasks * repetitions, rel_features)
y = np.repeat(np.arange(1, tasks + 1), repetitions)
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM Classifier model
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# Train the model
svm_classifier.fit(X_train, y_train)
# Make predictions
svm_predictions = svm_classifier.predict(X_test)
# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Classifier Accuracy:", svm_accuracy)


# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("RF Classifier Accuracy:", accuracy)


# Create a base classifier (you can choose Random Forest or SVM)
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Create a Bagging Classifier model
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
# Train the model
bagging_classifier.fit(X_train, y_train)
# Make predictions
bagging_predictions = bagging_classifier.predict(X_test)
# Calculate accuracy
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
print("Bagging Classifier Accuracy:", bagging_accuracy)