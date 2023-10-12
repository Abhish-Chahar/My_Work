import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and preprocess the data
data_directory = r"C:\Users\user\Downloads\split data"
grasp_tasks, repetitions = 8, 28
electrodes, time_samples = 45, 1200
num_samples = grasp_tasks * repetitions

# Initialize arrays to hold data and labels
data = np.empty((num_samples, electrodes * time_samples))
labels = np.empty(num_samples)

# Loop through each grasp task
for task in range(grasp_tasks):
    for repetition in range(repetitions):
        file_path = os.path.join(data_directory, f'task_{task + 1}_repetition_{repetition + 1}_spreadsheet.csv')
        task_repetition_data = pd.read_csv(file_path, header=None).values
        data[task * repetitions + repetition, :] = task_repetition_data.flatten()
        labels[task * repetitions + repetition] = task

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', C=1)
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
