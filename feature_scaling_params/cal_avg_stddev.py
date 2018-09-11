# -*-coding:utf-8-*-

import numpy as np 
import os
from sklearn.decomposition import PCA

x_abs_max = -1000.0
y_abs_max = -1000.0
z_abs_max = -1000.0
i_abs_max = -1000.0

path = "/media/shao/TOSHIBA EXT/data_object_velodyne/Daten_txt_CNN/train"
filelist = os.listdir(path)

for f in filelist:

    data_with_label = np.loadtxt(path + "/" + f)
    i = np.abs(data_with_label[:, 3])
    pca = PCA(n_components=2, copy=False)
    data_with_label[:, 0:2] = pca.fit_transform(data_with_label[:, 0:2])
    data_with_label = data_with_label - np.mean(data_with_label, axis=0)
    x = np.abs(data_with_label[:, 0])
    y = np.abs(data_with_label[:, 1])
    z = np.abs(data_with_label[:, 2])
    
    
    if np.max(x) > x_abs_max:
        x_abs_max = np.max(x)
    if np.max(y) > y_abs_max:
        y_abs_max = np.max(y)
    if np.max(z) > z_abs_max:
        z_abs_max = np.max(z)
    if np.max(i) > i_abs_max:
        i_abs_max = np.max(i)  

print('x_abs_max', x_abs_max)
print('y_abs_max', y_abs_max)
print('z_abs_max', z_abs_max)
print('i_abs_max', i_abs_max)
    
