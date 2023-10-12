import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

csv_folder = os.path.expanduser(r"C:\Users\user\Downloads\new split data")

num_subjects = 7
num_grasp_tasks = 6
num_repetitions = 30

# Defining STFT parameters
fs = 600  # Sampling frequency (Hz)
nperseg = 600  # Length of each segment
noverlap = 300 # Overlap between segments

# Loading EEG data for each task and calculate features
for subject in range(1, num_subjects + 1):
    # Initializing a list to store feature vectors for each task
    feature_vectors = []
    subject_folder = os.path.join(csv_folder, f"subject_{subject}")
    print(f"classifiaction for subject {subject}")
    # Loading EEG data for each task and calculate STFT features
    for task in range(1, num_grasp_tasks + 1):
        task_features = []
        for repetition in range(1, num_repetitions + 1):
            filename = os.path.join(subject_folder, f"task_{task}rep{repetition}_spreadsheet.csv")
            df = pd.read_csv(filename, header = None)
            for electrode in range(df.shape[0]):
                f, t, Sxx = spectrogram(df.iloc[electrode, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
                
                # Extract relevant features from the STFT matrix
                mean_power = np.mean(np.abs(Sxx), axis=1)  # Mean power in each frequency bin
                max_power = np.max(np.abs(Sxx), axis=1)    # Max power in each frequency bin
                
                electrode_features = np.concatenate((mean_power, max_power))
                task_features.append(electrode_features)
        feature_vectors.append(task_features)
    # Converting the list to a NumPy array
    feature_vectors = np.array(feature_vectors)
    print(feature_vectors.shape)

    # Reshaping the STFT features array to have a single dimension for each task
    reshaped_stft_features = feature_vectors.reshape(num_grasp_tasks, num_repetitions, df.shape[0], feature_vectors.shape[2])
    tasks, repetitions, electrodes, rel_features = reshaped_stft_features.shape
    X = reshaped_stft_features.reshape(tasks * repetitions, electrodes, rel_features)
    y = np.repeat(np.arange(1, tasks + 1), repetitions)
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, electrodes * rel_features))
    # Splittig the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, electrodes, rel_features)
    X_train = X_train.reshape(-1, rel_features)
    X_test = X_test.reshape(-1, electrodes, rel_features)
    X_test = X_test.reshape(-1, rel_features)
    y_train = np.repeat(y_train, electrodes)
    y_test = np.repeat(y_test, electrodes)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

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