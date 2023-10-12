import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']

def csp_feature_extraction(eegdat, num_filters):
    # Separate data and labels
    X = eegdat[:, :, :, :-1]  # EEG data without the last column (labels)
    y = eegdat[0, 0, 0, -1]   # Labels (assuming they are the same for all data points)

    # Reshape the data for CSP
    num_channels = X.shape[0]
    num_time_samples = X.shape[1]
    num_trials = X.shape[2]
    X = np.reshape(X, (num_channels, -1))

    # Calculate the spatial covariance matrices for each class
    cov_class1 = np.zeros((num_channels, num_channels))
    cov_class2 = np.zeros((num_channels, num_channels))

    for trial in range(num_trials):
        if y[trial] == 1:
            cov_class1 += np.dot(X[:, trial*num_time_samples:(trial+1)*num_time_samples], X[:, trial*num_time_samples:(trial+1)*num_time_samples].T)
        else:
            cov_class2 += np.dot(X[:, trial*num_time_samples:(trial+1)*num_time_samples], X[:, trial*num_time_samples:(trial+1)*num_time_samples].T)

    cov_class1 /= np.trace(cov_class1)
    cov_class2 /= np.trace(cov_class2)

    # Calculate CSP filters
    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(cov_class1 + cov_class2), cov_class1))
    sort_indices = np.argsort(np.abs(eigvals))
    W = eigvecs[:, sort_indices[:num_filters]]

    # Apply CSP filters to EEG data
    X_csp = np.dot(W.T, X)

    # Calculate logarithmic variances as features
    X_features = np.log(np.var(X_csp, axis=1))

    return X_features

# Example usage
# Assuming eegdat is the 4-dimensional EEG data array
# num_filters = 2  # Number of CSP filters to extract
# features = csp_feature_extraction(eegdat, num_filters)
