import os
import cv2
import struct
import tensorflow as tf
import numpy as np

import libs as g_libs
from tools.utilize import semantic_down_sample_voxel
from models.network import Network


def process_mask_via_custom_ops(data_dir, vox_size=None, img_size=None):
    vox_size = [240, 144, 240] if vox_size is None else vox_size
    img_size = [480, 640] if img_size is None else img_size
    all_files = sorted([os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.png')])
    # all_files = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.png')]
    vol_detail_t, cam_int_t = g_libs.scene_base_info()
    label_t = tf.placeholder(tf.uint8, shape=[1, *vox_size])
    depth_t = tf.placeholder(tf.float32, shape=[1, *img_size, 1])
    cam_info_t = tf.placeholder(tf.float32, shape=[1, 30])

    vox_scale = 240 // vox_size[0]  # down-sample or up-sample vox size needs to refresh the camera intrinsic
    _, mask_t = Network.optional_label(label_t, depth_t, cam_info_t, label_vox_size=vox_size, vox_factor=vox_scale)
    with tf.Session() as sess:
        for a_f in all_files:
            print('process %s' % a_f)
            # read from file
            depth = cv2.imread(os.path.join(data_dir, a_f + '.png'), cv2.IMREAD_UNCHANGED)
            lower_depth = depth >> 3
            higher_depth = (depth % 8) << 13
            real_depth = (lower_depth | higher_depth).astype(np.float32) / 1000
            depth = real_depth[None, ..., None]
            meta = open(os.path.join(data_dir, a_f + '.bin'), 'rb')
            header_bytes = meta.read(76)
            origin = struct.unpack('3f', header_bytes[:12])
            cam_pose = struct.unpack('16f', header_bytes[12:])
            data_bytes = meta.read()
            data_raw = np.reshape(struct.unpack('%di' % (len(data_bytes) // 4), data_bytes), [-1, 2])
            data_bg = np.reshape(np.ones([240, 144, 240], dtype=np.uint8), [-1]) * 255
            offset = 0
            for d_r in data_raw:
                if d_r[0] != 255:
                    data_bg[offset: offset + d_r[1]] = d_r[0]
                offset += d_r[1]
            label = semantic_down_sample_voxel(np.reshape(data_bg, [240, 144, 240]), vox_size, 256)[None, ...]
            meta.close()

            # read from internal
            vol_detail, cam_int = sess.run([vol_detail_t, cam_int_t])
            cam_info = np.concatenate([origin, cam_pose, vol_detail, cam_int], axis=0)[None, ...]
            mask = sess.run(mask_t, feed_dict={label_t: label, depth_t: depth, cam_info_t: cam_info})
            mask_dense = np.reshape(np.zeros(vox_size, dtype=np.int32), [-1])
            mask_dense[mask[..., 0]] = 1
            mask_prefix = np.asarray(vox_size, dtype=np.int32)

            binary_raw = np.concatenate([mask_prefix, mask_dense], axis=0).tobytes()
            dataset_name = os.path.split(data_dir)[-1]
            mask_dir = os.path.join('mask_dir', dataset_name)
            os.makedirs(mask_dir, exist_ok=True)
            with open(os.path.join(mask_dir, a_f + '.mask'), 'wb') as fp:
                fp.write(binary_raw)
            fp.close()


if __name__ == '__main__':
    # Please download and unzip the necessary samples from SSCNet first
    process_mask_via_custom_ops(os.path.join('data', 'depthbin', 'NYUtrain'), vox_size=[60, 36, 60])
    process_mask_via_custom_ops(os.path.join('data', 'depthbin', 'NYUtest'), vox_size=[60, 36, 60])
