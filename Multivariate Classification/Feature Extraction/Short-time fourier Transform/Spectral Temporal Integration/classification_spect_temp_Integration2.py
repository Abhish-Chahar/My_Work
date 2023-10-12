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
nperseg = 60  # Length of each segment
noverlap = 40  # Overlap between segments

# Initializing a list to store feature vectors for each task


# Loading EEG data for each task and calculate features
for subject in range(1, num_subjects + 1):
    stft_features = []
    subject_folder = os.path.join(csv_folder, f"subject_{subject}")
    print(f"classification for subject {subject}")
    
    # Loading EEG data for each task and calculate STFT features
    for task in range(1, num_grasp_tasks + 1):
        task_repetition_features = []
        for repetition in range(1, num_repetitions + 1):
            # task_stft = []
            filename = os.path.join(subject_folder, f"task_{task}rep{repetition}_spreadsheet.csv")
            df = pd.read_csv(filename, header=None)
            
            for electrode in range(df.shape[0]):
                f, t, Sxx = spectrogram(electrode.iloc[electrode, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
                task_repetition_features.append(np.abs(Sxx))
            # task_stft = np.array(task_stft)
            # task_stft_mean = np.mean(task_stft, axis=0)
            # task_repetition_features.append(task_stft_mean)
        # stft_features = np.array(task_repetition_features)
        stft_features.append(task_repetition_features)
    
    # Converting the list to a NumPy array
    st_stft_features = np.array(stft_features)
    print(st_stft_features.shape)


    # Reshaping the STFT features array to have a single dimension for each task
    reshaped_stft_features = st_stft_features.reshape(num_grasp_tasks, num_repetitions, 9, st_stft_features.shape[2], st_stft_features.shape[3])
    tasks, repetitions, electrodes, freq_bins, time_stamps = reshaped_stft_features.shape
    X = st_stft_features.reshape(tasks * repetitions, 9, freq_bins * time_stamps)
    y = np.repeat(np.arange(1, tasks + 1), repetitions)
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 9 * freq_bins * time_stamps))
    # Splittig the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(144, 9, 1798)
    X_train = X_train.reshape(144*9, 1798)
    X_test = X_test.reshape(36, 9, 1798)
    X_test = X_test.reshape(36*9, 1798)
    y_train = np.repeat(y_train, 9)
    y_test = np.repeat(y_test, 9)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

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