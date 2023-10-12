import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Create a four-dimensional array with shape (2, 3, 5, 4)
# data = np.zeros((2, 6, 5, 4))

# # Fill in the data according to your instructions
# for i in range(2):
#     for j in range(6):
#         for k in range(5):
#             data[i, j, k, :] = [0.01, 0.02, 0.03, 0.04]
#             data[i, j, k, :] += 0.1 * (k + 1)
#             data[i, j, k, :] += 1 * (j + 1)
#             data[i, j, k, :] += 10 * (i + 1)
# print(data)
# reshaped_data = data.reshape(2, 2, 3, 5, 4)
# tasks, repetitions, electrodes, freq_bins, time_stamps = reshaped_data.shape
# X = data.reshape(tasks * repetitions, electrodes, freq_bins * time_stamps)
# y = np.repeat(np.arange(1, tasks + 1), repetitions)
# print(X)
# # Standardizing the features
# scaler = StandardScaler()
# X_scaled = X.reshape(-1, electrodes * freq_bins * time_stamps)
# print(X_scaled)
# # Splitting the data by repetitions while preserving electrode grouping
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# X_train = X_train.reshape(3, 3, 20)
# X_train = X_train.reshape(3*3, 20)
# X_test = X_test.reshape(1, 3, 20)
# X_test = X_test.reshape(1*3, 20)
# y_train = np.repeat(y_train, 3)
# y_test = np.repeat(y_test, 3)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y.shape)
# print(X_train)
# print(y_train)
# print(reshaped_data)

# # Create a four-dimensional array with shape (2, 3, 5, 4)
# data = np.zeros((2, 5, 4))

# # Fill in the data according to your instructions
# for i in range(2):
#     for k in range(5):
#         data[i, k, :] = [0.01, 0.02, 0.03, 0.04]
#         data[i, k, :] += 0.1 * (k + 1)
#         data[i, k, :] += 1 * (i + 1)
# print(data)
# tasks, freq_bins, time_stamps = data.shape
# X = data.reshape(tasks, freq_bins * time_stamps)
# print(X)

# tasks = 3
# repetitions = 4
# electrodes = 5
# y = np.repeat(np.arange(1, tasks + 1), repetitions*electrodes)
# print(y)
# print(y.shape)

# y = [0.5, 0.02, 23.03, 0.04]
# expanded_spectrum_power = np.repeat(y, 5)
# print(expanded_spectrum_power)

# # Create a four-dimensional array with shape (2, 3, 5, 4)
data = np.zeros((2, 20))
# Fill in the data according to your instructions
for i in range(2):
    data[i, :] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    data[i, :] += 1 * (i + 1)
data = np.mean(data, axis = 0)
print(data)
# tasks, freq_bins, time_stamps = data.shape
# X = data.reshape(tasks, freq_bins * time_stamps)
# print(X)