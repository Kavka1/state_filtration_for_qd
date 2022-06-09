import numpy as np

prepare_one_hot_z_expand   = []
for z in range(10):
    temp_matrix = np.zeros((10, 10))
    for z_small in range(10):
        temp_matrix[z_small,:]  =  (np.arange(10) == z_small).astype(np.integer)
    temp_matrix = np.delete(temp_matrix, z, 0)
    prepare_one_hot_z_expand.append(temp_matrix)

prepare_one_hot_z_expand = np.stack(prepare_one_hot_z_expand, 0)


print(prepare_one_hot_z_expand)            