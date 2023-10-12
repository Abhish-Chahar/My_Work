import numpy as np
import scipy.io as sio
# Load the .mat file
mat_data = sio.loadmat(r"C:\Users\user\Downloads\EEGset\EEGset\EEG_MI_data.mat")
# Access the EEG data from the loaded .mat file
eegdat = mat_data['eegdat']
demo = np.array([[[[11, 12],[13, 14]],[[15, 16],[17, 18]]],[[[19, 20],[21, 22]],[[23, 24],[25, 26]]]])
print(demo.size)
print(demo.ndim)
print(demo.shape)