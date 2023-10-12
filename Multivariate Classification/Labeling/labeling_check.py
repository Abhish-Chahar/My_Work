import scipy.io as sio
import numpy as np
# Loading the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")

variable_names = list(mat_data.keys()) 
# Printing all variable names
for name in variable_names:
    print(name)
