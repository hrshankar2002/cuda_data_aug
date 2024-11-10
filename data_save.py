#%%
import os

import numpy as np
import scipy.io as scio

#%%
base_path = './'
dataset_path =  'dataset/Dataset_1' # Training data

classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
           'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
ClassesNum = len(classes)

X = list()
y = list()

#%%
for root, dirs, files in os.walk(dataset_path, topdown=False):
    for name in files:
    
        data_train = scio.loadmat(os.path.join(root, name))
        
        # arr -> list
        data_arr = data_train.get('val')
        
        data_list = data_arr.tolist()
       
        X.append(data_list[0]) # [[……]] -> [ ]
        y.append(int(os.path.basename(root)[0:2]) - 1)  # name -> num('02' -> 1)
    
X=np.array(X) # (1000, 3600)
y=np.array(y) # (1000, )

X = X.reshape((1000,3600))
y = y.reshape((1000))

#%%
np.save('X.npy', X)
np.save('y.npy', y)