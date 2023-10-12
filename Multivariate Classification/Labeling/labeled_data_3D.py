import numpy as np
import scipy
import scipy.io as sio

# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")

# Accessing the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']

repetitions_names = ["Rep_{}".format(i) for i in range(1, 29)]
repetitions_dict = {}
for i, repetition_name in enumerate(repetitions_names):
    repetitions_dict[repetition_name] = eegdat[:, :, :, i]
repetitions = np.array([repetitions_dict[repetition_name] for repetition_name in repetitions_names])

# Here, repetions_dict is a dictionary that stores the variables of all repetitions.
# But, repetitions is an array therefore it cannot store variables so we have to use indices.
print(repetitions.shape)

tasks_names = ["Task_{}".format(j) for j in range(1, 9)]
tasks_dict = {}
for j, task_name in enumerate(tasks_names):
    for i, repetition_name in enumerate(repetitions_names): 
        tasks_dict[task_name, repetition_name] = repetitions_dict[repetition_name][:,:,j]

tasks = np.array([[tasks_dict[task_name, repetition_name] for repetition_name in repetitions_names] for task_name in tasks_names])

for task_name, tasks_data in zip(tasks_names, tasks):
    for repetition_name, repetition_data in zip(repetitions_names, tasks_data):
        print(task_name, repetition_name, "shape:", repetition_data.shape)
print(tasks.shape)

variable_names = list(tasks_dict.keys()) 

# Printing all variable names
for name in variable_names:
    print(name)

neweeg= tasks

# Creating a dictionary to hold the array
data = {'data': neweeg}

file_path = r'c:\Users\user\Downloads\EEGset\EEGset'
file_name = 'neweeg_labeled_3D.mat'
save_path = file_path + '\\' + file_name

# Saving the data to the .mat file
scipy.io.savemat(save_path, data)

