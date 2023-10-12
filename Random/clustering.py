import numpy as np
from sklearn.cluster import KMeans
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']

# Assuming you have preprocessed EEG data 'eegdat' with dimensions [45, 1200, 8, 28]

# Reshape EEG data for clustering
num_channels, num_samples, num_tasks, num_repetitions = eegdat.shape
X = np.reshape(eegdat, (num_channels*num_samples, num_tasks*num_repetitions)).T

# Perform K-means clustering
num_clusters = 8  # Set the number of clusters
n_init = 10  # Set the value of n_init explicitly to suppress the warning
kmeans = KMeans(n_clusters=num_clusters, n_init=n_init)
kmeans.fit(X)

# Get the cluster labels assigned to each data point
cluster_labels = kmeans.labels_

# Perform further analysis on the clusters
# For example, you can analyze the distribution of data points in each cluster,
# visualize the cluster centroids, or extract meaningful information from the clusters.
#print(kmeans.cluster_centers_.size)
print(X.ndim)
