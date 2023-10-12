import numpy as np
import scipy
import scipy.io as sio

# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")

# Accessing the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']

task_names = ["task_{}".format(i) for i in range(1, 9)]
task_dict = {}
for i, task_name in enumerate(task_names):
    task_dict[task_name] = eegdat[:, :, i, :]
tasks = np.array([task_dict[task_name] for task_name in task_names])
for task_name, task_data in zip(task_names, tasks):
    print(task_name, "shape:", task_data.shape)
print(tasks.shape)

repetitions_names = ["Rep_{}".format(j) for j in range(1, 29)]
repetitions_dict = {}

for j, rep_name in enumerate(repetitions_names):
    repetitions_dict[rep_name] = tasks[:, :, :, j]
repetitions = np.array([repetitions_dict[rep_name] for rep_name in repetitions_names])
for rep_name, repetitions_data in zip(repetitions_names, repetitions):
    print(rep_name, "shape:", repetitions_data.shape)
print(repetitions.shape)

neweeg= repetitions

# Creating a dictionary to hold the array
data = {'data': neweeg}

file_path = r'c:\Users\user\Downloads\EEGset\EEGset'
file_name = 'neweeg_labeled_2D.mat'
save_path = file_path + '\\' + file_name

# Saving the data to the .mat file
scipy.io.savemat(save_path, data)