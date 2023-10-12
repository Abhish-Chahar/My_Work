import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']
# Assuming you have preprocessed EEG data 'eegdat' with dimensions [45, 1200, 8, 28]
# representing [n_channels, n_samples, n_tasks, n_repetitions]

# Reshape EEG data for feature extraction
n_channels, n_samples, n_tasks, n_repetitions = eegdat.shape
X = np.transpose(eegdat, (2, 3, 0, 1))  # Reshape to [n_tasks, n_repetitions, n_channels, n_samples]

# Define the number of CSP components
n_components = 4

# Initialize the CSP matrices
csp_matrices = np.zeros((n_tasks, n_components, n_channels))

# Loop over tasks
for task in range(n_tasks):
    # Concatenate all repetitions for the current task
    X_task = np.concatenate(X[task], axis=1)
    
    # Compute the class-wise covariance matrices
    class_covariances = []
    for repetition in range(n_repetitions):
        X_rep = X[task][repetition]
        class_covariances.append(np.cov(X_rep))
    
    # Compute the averaged covariance matrix
    avg_covariance = np.mean(class_covariances, axis=0)
    
    # Perform eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(avg_covariance)
    
    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the CSP components
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    # Compute the spatial filter matrix
    spatial_filter = np.linalg.pinv(selected_eigenvectors.T)
    
    # Normalize the spatial filter matrix
    spatial_filter /= np.linalg.norm(spatial_filter, axis=0)
    
    # Store the CSP matrix for the current task
    csp_matrices[task] = spatial_filter.T

# Apply the CSP transformation to the data
X_csp = np.zeros((n_tasks, n_repetitions, n_components, n_samples))
for task in range(n_tasks):
    for repetition in range(n_repetitions):
        X_rep = X[task][repetition]
        X_csp[task, repetition] = np.dot(csp_matrices[task], X_rep)

# Reshape the CSP features for classification
X_csp = X_csp.reshape(-1, n_components * n_samples)

# Assuming you have the corresponding task labels 'y' with shape [n_tasks * n_repetitions]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_csp, y, test_size=0.2, random_state=42)

# Train a classifier on the CSP features
classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, y_train)

# Predict the task labels for the test set
y_pred = classifier.predict(X_test)

# Compute the classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)
