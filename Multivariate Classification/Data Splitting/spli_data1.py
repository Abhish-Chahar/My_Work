import numpy as np
import scipy.io as sio
import os
import csv

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGdata.mat")
eegdat = mat_data['eegdat']
save_directory = r"C:\Users\user\Downloads\split data"
# electrodes, time_samples, grasp_tasks, repetitions = 45, 1200, 8, 28

# # Loop through each grasp task
# for task in range(grasp_tasks):
#     for repetition in range(repetitions):
#         # Extract data for the current task and repetition
#         task_repetition_data = eegdat[:, :, task, repetition]
        
#         file_path = os.path.join(save_directory, f'task_{task + 1}_repetition_{repetition + 1}_spreadsheet.csv')

#         with open(file_path, 'w', newline='') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             for row in task_repetition_data:
#                 csvwriter.writerow(row)
print(eegdat.shape)