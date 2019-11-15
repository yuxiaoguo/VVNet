import os
import random
import cv2
import struct
import numpy as np
import tensorflow as tf

from .utilize import semantic_down_sample_voxel

np.seterr(divide='ignore', invalid='ignore')

DATA_DIR = os.path.join(os.environ['HOME'], 'datasets', 'SUNCG')
RECORD_DIR = os.path.join(os.environ['HOME'], 'datasets', 'SUNCG-TF-60')


def details_and_fov(img_height, img_width, img_scale, vox_scale):
    vox_details = np.array([0.02 * vox_scale, 0.24], np.float32)
    camera_fov = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                           0., 518.8579 / img_scale, img_height / (2 * img_scale),
                           0., 0., 1.], dtype=np.float32)
    return vox_details, camera_fov


def _diff_vec(img, axis=0):
    img_diff = np.diff(img, 1, axis)
    img_diff_l = img_diff[1:, :] if axis == 0 else img_diff[:, 1:]
    img_diff_h = img_diff[:-1, :] if axis == 0 else img_diff[:, :-1]
    img_diff = img_diff_l + img_diff_h
    pad_tuple = ((1, 1), (0, 0), (0, 0)) if axis == 0 else ((0, 0), (1, 1), (0, 0))
    padded = np.lib.pad(img_diff, pad_tuple, 'edge')
    return padded


def _gen_normal(depth_path, file_path='tmp.png'):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    lower_depth = depth >> 3
    higher_depth = (depth % 8) << 13
    real_depth = (lower_depth | higher_depth).astype(np.float32) / 1000
    _, fov = details_and_fov(*real_depth.shape, 1, 1)

    img_x = np.repeat(np.expand_dims(np.arange(real_depth.shape[0]), axis=1), real_depth.shape[1], axis=1)
    img_y = np.repeat(np.expand_dims(np.arange(real_depth.shape[1]), axis=0), real_depth.shape[0], axis=0)
    point_cam_x = (img_x - fov[2]) * real_depth / fov[0]
    point_cam_y = (img_y - fov[5]) * real_depth / fov[4]
    points = np.stack([point_cam_x, point_cam_y, real_depth], axis=2)

    diff_y = _diff_vec(points, axis=0)
    diff_x = _diff_vec(points, axis=1)
    normal = np.cross(diff_x, diff_y)
    normal_factor = np.expand_dims(np.linalg.norm(normal, axis=2), axis=-1)
    normal = np.where((normal_factor == 0.) | np.isnan(normal_factor), (0, 0, 0), normal / normal_factor)
    normal = (np.clip((normal + 1) / 2, 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(file_path, normal)

    # cooked_normal = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return open(file_path, 'rb').read()


def _gen_zip_voxel(meta_path, vox_size=None, scaled_vox_size=None):
    seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10,
                     10, 11, 8, 10, 11, 9, 11, 11, 11, 12]

    if scaled_vox_size is None:
        scaled_vox_size = np.array([60, 36, 60])
    if vox_size is None:
        vox_size = np.array([240, 144, 240])
    meta = open(meta_path, 'rb')
    header_bytes = meta.read(76)
    vox_meta_info = meta.read()
    vox_info = struct.unpack('%di' % int(len(vox_meta_info) / 4), vox_meta_info)
    labels, vox_nums = [np.squeeze(x) for x in np.split(np.array(vox_info).reshape([-1, 2]), 2, axis=1)]
    full_voxel = np.full(vox_size, 37, np.uint8).reshape([-1])
    offset = 0
    for label, vox_num in zip(labels, vox_nums):
        if label != 255:
            full_voxel[offset:offset+vox_num] = label
        offset += vox_num
    full_voxel = np.take(seg_class_map, full_voxel)
    full_voxel = np.reshape(full_voxel, vox_size)
    final_voxel = semantic_down_sample_voxel(full_voxel, scaled_vox_size)
    final_voxel = np.expand_dims(np.where(final_voxel == 12, np.full(final_voxel.shape, 255, dtype=final_voxel.dtype),
                                          final_voxel), axis=-1)
    meta_bytes = np.reshape(final_voxel, [-1]).astype(np.uint8).tobytes()
    return header_bytes + meta_bytes


def prepare_data(target_path, shuffle=False, normal=False, zip_voxel=False):
    if not os.path.exists(RECORD_DIR):
        os.mkdir(RECORD_DIR)
    print('write samples from %s' % target_path)
    dir_name = os.path.dirname(target_path)
    target_folders = [folder for folder in os.listdir(dir_name) if folder.startswith(os.path.basename(target_path))]
    samples_path = []

    # Get the total samples list
    for target_folder in target_folders:
        folder_path = os.path.join(dir_name, target_folder)
        if not os.path.isdir(folder_path):
            continue
        sub_samples = sorted([os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('bin')])
        samples_path.extend([os.path.join(folder_path, sub_sample) for sub_sample in sub_samples])
    if shuffle:
        random.seed(0)
        random.shuffle(samples_path)

    option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(os.path.join(RECORD_DIR, os.path.split(target_path)[-1] + '.tfrecord'),
                                         options=None)
    current_index = 0
    for sample in samples_path:
        print('--%07d write %s in TFRECORDS' % (current_index, sample))
        current_index += 1
        depth_path = sample + '.png'
        bin_path = sample + '.bin'
        if not os.path.exists(depth_path) or not os.path.exists(bin_path):
            continue

        features_dict = dict()
        features_dict['img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[open(depth_path, 'rb').read()]))

        bin_meta = open(bin_path, 'rb').read() if not zip_voxel else _gen_zip_voxel(bin_path,
                                                                                    scaled_vox_size=[60, 36, 60])
        features_dict['bin'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bin_meta]))

        alias = os.path.split(sample)[-1].encode('utf-8')
        features_dict['alias'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[alias]))

        if normal:
            normal_img = _gen_normal(depth_path)
            features_dict['normal'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[normal_img]))
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    prepare_data(os.path.join(DATA_DIR, 'SUNCGtrain'), shuffle=True, normal=True,zip_voxel=True)
    prepare_data(os.path.join(DATA_DIR, 'SUNCGtest'), shuffle=False, normal=True, zip_voxel=True)
