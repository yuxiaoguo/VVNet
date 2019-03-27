import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from abc import abstractmethod
import libs


class Network(object):
    def __init__(self):
        return

    @staticmethod
    def cook_raw_inputs(reader):
        raw_inputs = reader.get_inputs(scale=1.)
        with tf.name_scope('cook_input') as _:
            depth, normal, scene_info, label = raw_inputs
            return [depth, normal, scene_info], label

    @staticmethod
    def optional_label(label, depth, scene_info, label_vox_size=None):
        if label_vox_size is None:
            label_vox_size = [60, 36, 60]
        with tf.name_scope('cook_label') as _:
            scene_info = Network.modify_scene_info(scene_info, 1., 4.)
            tsdf = libs.tsdf_projection(depth, scene_info, vox_size=label_vox_size, flipped_tsdf=True)
            # _, tsdf, _, _ = libs.feat_fusion(depth, depth, scene_info, vox_size=label_vox_size)
            label = tf.reshape(tf.cast(label, dtype=tf.int32), [-1, ], name='reshape_label')
            tsdf = tf.reshape(tsdf, [-1, ], name='reshape_tsdf')
            tsdf_cond = tf.logical_and(tsdf > -1, tsdf < 0)
            label_cond = tf.logical_and(label > 0, label < 255)
            cond = tf.logical_or(tsdf_cond, label_cond)
            # cond = tf.logical_or(tsdf < -0.5, label_cond)
            cond = tf.logical_and(cond, label < 255)
            valid = tf.where(cond, name='valid_indices')
            return label, valid

    @staticmethod
    def modify_scene_info(old_scene_info, img_factor=1., vox_factor=1.):
        scaled_vox_detail, scaled_fov = libs.scene_base_info(img_factor, vox_factor, old_scene_info.shape[0])
        replace_scene_info, _ = tf.split(old_scene_info, [19, 11], axis=1)
        scene_info = tf.concat([replace_scene_info, scaled_vox_detail, scaled_fov], axis=1)
        return scene_info

    @staticmethod
    def apply_l2_regularization(l2_factor):
        l2_reg = layers.l2_regularizer(l2_factor)
        conv_reg_dict = {'weights_regularizer': l2_reg, 'biases_regularizer': l2_reg}
        bn_reg_dict = None
        return conv_reg_dict, bn_reg_dict

    @abstractmethod
    def instance(self, inputs): pass
