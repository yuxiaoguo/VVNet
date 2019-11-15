import numpy as np


def semantic_down_sample_voxel(full_voxel, scaled_vox_size, label_length=13):
    vox_size = np.array(full_voxel.shape)
    resize_voxel = np.expand_dims(full_voxel, axis=0)
    for axis in range(3):
        resize_voxel = np.split(resize_voxel, scaled_vox_size[axis], axis=3 - axis)
        resize_voxel = np.concatenate(tuple(resize_voxel), axis=0)
    resize_voxel = np.reshape(resize_voxel, [-1, np.prod(vox_size / scaled_vox_size).astype(np.int32)])
    stat_voxel = []
    for elem in resize_voxel:
        stat_voxel.append(np.bincount(elem, minlength=label_length)[1:])
    down_samples = np.stack(stat_voxel, axis=0)
    down_samples_indices = np.argmax(down_samples, axis=-1)
    final_voxel = np.reshape(np.where(down_samples.max(axis=-1) >= 3, down_samples_indices + 1,
                                      np.zeros(down_samples_indices.shape, down_samples_indices.dtype)),
                             scaled_vox_size)
    return final_voxel
