#####################################
### generate data for robust test ###
#####################################

import numpy as np 

def getOccludedCloud (data_with_label, occlusion_percentage):
    assert(occlusion_percentage >=0.0 and occlusion_percentage <= 100.0)

    occluded_cloud = []
    mp = 0.0
    length = 0.0
    half_occluded_length = 0.0

    # if car, van, truck
    if data_with_label[0, -1] in [0, 1, 3]:
        minpx = data_with_label[0, 0]
        maxpx = data_with_label[0, 0]
        minpy = data_with_label[0, 1]
        maxpy = data_with_label[0, 1]
        
        for i in range(len(data_with_label)):
            if data_with_label[i, 0] > maxpx:
                maxpx = data_with_label[i, 0]
            elif data_with_label[i, 0] < minpx:
                minpx = data_with_label[i, 0]
            if data_with_label[i, 1] > maxpy:
                maxpy = data_with_label[i, 1]
            elif data_with_label[i, 1] < minpy:
                minpy = data_with_label[i, 1]
        
        if maxpx-minpx > maxpy-minpy:
            length = maxpx - minpx
            mp = (maxpx + minpx) / 2.0
            half_occluded_length = length * occlusion_percentage / 200.0
            
            for i in range(len(data_with_label)):
                if  data_with_label[i, 0] <= mp+half_occluded_length and \
                    data_with_label[i, 0] >= mp-half_occluded_length:
                    continue
                else:
                    occluded_cloud.append(data_with_label[i, :])
        else:
            length = maxpy-minpy
            mp = (maxpy + minpy) / 2.0
            half_occluded_length = length * occlusion_percentage / 200.0
            
            for i in range(len(data_with_label)):
                if  data_with_label[i, 1] <= mp+half_occluded_length and \
                    data_with_label[i, 1] >= mp-half_occluded_length:
                    continue
                else:
                    occluded_cloud.append(data_with_label[i])
        occluded_cloud = np.array(occluded_cloud)
        return occluded_cloud

    # if pedestrain, cyclist
    elif data_with_label[0, -1] in [2, 4]:
        maxpz = data_with_label[0, 2]
        minpz = data_with_label[0, 2]

        for i in range(len(data_with_label)):
            if data_with_label[i, 2] > maxpz:
                maxpz = data_with_label[i, 2]
            elif data_with_label[i, 2] < minpz:
                minpz = data_with_label[i, 2]
        
        length = maxpz - minpz
        mp = (maxpz + minpz) / 2.0
        half_occluded_length = length * occlusion_percentage / 200.0

        for i in range(len(data_with_label)):
            if  data_with_label[i, 2] <= mp+half_occluded_length and \
                data_with_label[i, 2] >= mp-half_occluded_length:
                continue
            else:
                occluded_cloud.append(data_with_label[i])
        occluded_cloud = np.array(occluded_cloud)
        return occluded_cloud

    else:
        print('getOccludedCloud(): invalid label name!')
        occluded_cloud = np.array(occluded_cloud)
        return occluded_cloud


def getSparseCloud (data_with_label, sparse_percentage):
    assert(sparse_percentage >= 0.0 and sparse_percentage <= 100.0)

    num_points_after_downsampled = int(sparse_percentage * len(data_with_label) / 100)
    np.random.shuffle(data_with_label)
    sparse_cloud = data_with_label[0:num_points_after_downsampled]
    return sparse_cloud

# def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
#     """ Randomly jitter points. jittering is per point.
#         Input:
#           BxNx3 array, original batch of point clouds
#         Return:
#           BxNx3 array, jittered batch of point clouds
#     """
#     B, N, C = batch_data.shape
#     assert(clip > 0)
#     # randn(): standard norm distribution
#     # clip(): limit the data within min and max
#     jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
#     jittered_data += batch_data
#     return jittered_data

def addNoise (data_with_label, std_dev, clip = 0.12):
    assert(std_dev >= 0.0)

    N = len(data_with_label[:, 0:3])
    C = 3
    jittered_data = np.clip(std_dev * np.random.randn(N, C), -1.0*clip, clip) 
    data_with_label[:, 0:3] += jittered_data
    return data_with_label 