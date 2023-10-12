import numpy as np
import scipy.io as sio
import os
import csv

mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGdata.mat")
eegdat = mat_data['data']
save_directory = r"C:\Users\user\Downloads\new split data"
electrodes, time_samples, repetitions, grasp_tasks, subjects = eegdat.shape

# Loop through each subject
for subject in range(subjects):
    subject_folder = os.path.join(save_directory, f'subject_{subject + 1}')
    
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)
    
    # Loop through each grasp task
    for task in range(grasp_tasks):
        # Loop through each repetition
        for repetition in range(repetitions):
            # Extract data for the current subject, task, and repetition
            subject_task_repetition_data = eegdat[:, :, repetition, task, subject]
            
            file_name = f'task_{task + 1}rep{repetition + 1}_spreadsheet.csv'
            file_path = os.path.join(subject_folder, file_name)

            with open(file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for row in subject_task_repetition_data:
                    csvwriter.writerow(row)