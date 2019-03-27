from abc import abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from models import vvnet
import libs


class VVNetAE(vvnet.VVNet):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE, self).__init__(reuse, is_training, wd)

    def _2d_dnn_level_80(self, inputs):
        return self._resnet_2d(inputs, 64, scope='ResNet80')

    def _3d_dnn_level_60(self, inputs):
        with tf.variable_scope('Volume60', reuse=self._reuse):
            conv = layers.conv3d(inputs, 64, 3, scope='conv1', **self._reg)
            conv = layers.conv3d(conv, 64, 3, activation_fn=None, scope='conv2', **self._reg)
            shortcut = layers.conv3d(inputs, 64, 1, activation_fn=None, scope='shortcut', **self._reg)
            res = tf.nn.relu(conv + shortcut)
            return layers.max_pool3d(res, 2, scope='pool')

    def _3d_dnn_level_30(self, inputs):
        with tf.variable_scope('Volume30', reuse=self._reuse):
            conv = layers.conv3d(inputs, 128, 3, scope='conv1', **self._reg)
            conv = layers.conv3d(conv, 128, 3, activation_fn=None, scope='conv2', **self._reg)
            shortcut = layers.conv3d(inputs, 128, 1, activation_fn=None, scope='shortcut', **self._reg)
            res1 = tf.nn.relu(conv + shortcut)

            conv = layers.conv3d(res1, 128, 3, rate=2, scope='conv3', **self._reg)
            conv = layers.conv3d(conv, 128, 3, activation_fn=None, rate=2, scope='conv4', **self._reg)
            res2 = tf.nn.relu(conv + res1)

            conv = layers.conv3d(res2, 128, 3, rate=2, scope='conv5', **self._reg)
            conv = layers.conv3d(conv, 128, 3, activation_fn=None, rate=2, scope='conv6', **self._reg)
            res3 = tf.nn.relu(conv + res2)

            concat = tf.concat([res1, res2, res3], axis=-1, name='concat')
            return tf.nn.relu(concat)

    def _3d_rdnn_level_60(self, inputs):
        with tf.variable_scope('Volume30Up', reuse=self._reuse):
            return layers.conv3d_transpose(inputs, 256, 3, 2, scope='deconv', **self._reg)

    @abstractmethod
    def _2d_dnn(self, inputs): pass

    @abstractmethod
    def _3d_dnn(self, inputs): pass


class VVNetAE120(VVNetAE):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE120, self).__init__(reuse, is_training, wd)
        self._volume_scale = 2
        self._image_scale = 2

    def _2d_dnn(self, inputs):
        feat_2d = self._2d_dnn_level_640(inputs)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        return self._2d_dnn_level_320(feat_2d)

    def _3d_dnn(self, inputs):
        feat_3d = self._3d_dnn_level_120(inputs)
        feat_3d = self._3d_dnn_level_60(feat_3d)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE60(VVNetAE):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE60, self).__init__(reuse, is_training, wd)
        self._volume_scale = 4
        self._image_scale = 4

    def _2d_dnn(self, inputs):
        feat_2d = self._2d_dnn_level_640(inputs)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        feat_2d = self._2d_dnn_level_320(feat_2d)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        return self._2d_dnn_level_160(feat_2d)

    def _3d_dnn(self, inputs):
        feat_3d = self._3d_dnn_level_60(inputs)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE30(VVNetAE):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30, self).__init__(reuse, is_training, wd)
        self._volume_scale = 8
        self._image_scale = 8

    def _2d_dnn(self, inputs):
        feat_2d = self._2d_dnn_level_640(inputs)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        feat_2d = self._2d_dnn_level_320(feat_2d)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        feat_2d = self._2d_dnn_level_160(feat_2d)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        return self._2d_dnn_level_80(feat_2d)

    def _3d_dnn(self, inputs):
        feat_3d = self._3d_dnn_level_30(inputs)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE60AC(VVNetAE60):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE60AC, self).__init__(reuse, is_training, wd)

    def _2d_dnn_level_160(self, inputs):
        feat = self._resnet_2d(inputs, 32, scope='ResNet160')
        return self._resnet_2d(feat, 32, scope='ResNet160E')


class VVNetAE30UR(VVNetAE30):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30UR, self).__init__(reuse, is_training, wd)
        self._volume_scale = 4
        self._image_scale = 8

    def _3d_dnn(self, inputs):
        with tf.variable_scope('DownSample3D', reuse=self._reuse):
            feat_3d = layers.conv3d(inputs, 64, 3, stride=2, activation_fn=None, **self._reg)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE30AP(VVNetAE30):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30AP, self).__init__(reuse, is_training, wd)
        self._volume_scale = 4
        self._image_scale = 8

    def _3d_dnn(self, inputs):
        with tf.variable_scope('DownSample3D', reuse=self._reuse):
            feat_3d = layers.max_pool3d(inputs, 2, 2)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE30BP(VVNetAE30):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30BP, self).__init__(reuse, is_training, wd)

    def _up_sampling_feature(self, feat, axis_range):
        with tf.variable_scope('UpSampleFeat', self._reuse):
            feat_size = [int(axis) for axis in (np.array(feat.shape[1:3]) * self._image_scale).tolist()]
            return libs.interpolation_2d(feat, feat_size)


class VVNetAE60BP(VVNetAE60):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE60BP, self).__init__(reuse, is_training, wd)

    def _up_sampling_feature(self, feat, axis_range):
        with tf.variable_scope('UpSampleFeat', self._reuse):
            feat_size = [int(axis) for axis in (np.array(feat.shape[1:3]) * self._image_scale).tolist()]
            return libs.interpolation_2d(feat, feat_size)


class VVNetAE30AC(VVNetAE30):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30AC, self).__init__(reuse, is_training, wd)

    def _3d_dnn(self, inputs):
        with tf.variable_scope('ExtraReception', reuse=self._reuse):
            feat_3d = layers.conv3d(inputs, 128, 3, **self._reg)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE60DC(VVNetAE60):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE60DC, self).__init__(reuse, is_training, wd)

    def _3d_dnn_level_60(self, inputs):
        with tf.variable_scope('Volume60', reuse=self._reuse):
            conv = layers.conv3d(inputs, 64, 3, scope='conv1', **self._reg)
            conv = layers.conv3d(conv, 64, 3, activation_fn=None, scope='conv2', **self._reg)
            shortcut = layers.conv3d(inputs, 64, 1, activation_fn=None, scope='shortcut', **self._reg)
            res = tf.nn.relu(conv + shortcut)
            return layers.conv3d(res, 64, 3, stride=2, activation_fn=None, scope='pool', **self._reg)


class VVNetAE60UR(VVNetAE60):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE60UR, self).__init__(reuse, is_training, wd)
        self._volume_scale = 2
        self._image_scale = 4

    def _3d_dnn(self, inputs):
        with tf.variable_scope('DownSample3D', reuse=self._reuse):
            feat_3d = layers.conv3d(inputs, 64, 3, stride=2, activation_fn=None, **self._reg)
        feat_3d = self._3d_dnn_level_60(feat_3d)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE30URS(VVNetAE30UR):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30URS, self).__init__(reuse, is_training, wd)

    def _3d_dnn(self, inputs):
        with tf.variable_scope('DownSample3D', reuse=self._reuse):
            feat_3d = layers.conv3d(inputs, 64, 2, stride=2, activation_fn=None, **self._reg)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE30URAC(VVNetAE30URS):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30URAC, self).__init__(reuse, is_training, wd)

    def _3d_dnn(self, inputs):
        with tf.variable_scope('DownSample3D', reuse=self._reuse):
            feat_3d = layers.conv3d(inputs, 64, 2, stride=2, activation_fn=None, **self._reg)
        with tf.variable_scope('ExtraReception', reuse=self._reuse):
            feat_3d = layers.conv3d(feat_3d, 128, 3, **self._reg)
        feat_3d = self._3d_dnn_level_30(feat_3d)
        return self._3d_rdnn_level_60(feat_3d)


class VVNetAE30EncodePosition(VVNetAE30):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetAE30EncodePosition, self).__init__(reuse, is_training, wd)
        self._hi_res = 2

    def _view_fuse(self, depth, feat_2d, scene_info):
        feat_3d = super(VVNetAE30EncodePosition, self)._view_fuse(depth, feat_2d, scene_info)
        with tf.variable_scope('PosCoding', reuse=self._reuse):
            hi_res_scene_info = self.modify_scene_info(scene_info, 1., self._volume_scale / self._hi_res)
            hi_res_volume = (self._original_volume / self._volume_scale * self._hi_res).tolist()
            _, _, vox_occupied = libs.img2vol_forward_projection(depth, hi_res_scene_info, hi_res_volume)
            pos_coding = tf.cast(vox_occupied, tf.float32)
            for i in range(3):
                new_shape = [int(dim_size) for dim_size in pos_coding.shape]
                new_shape[-2 - i] = int(new_shape[-2 - i] / self._hi_res)
                new_shape.insert(-1 - i, 2)
                pos_coding = tf.reshape(pos_coding, shape=new_shape)
                pos_coding = tf.split(pos_coding, self._hi_res, axis=-2 - i)
                pos_coding = tf.concat(pos_coding, axis=-1)
                new_shape[-1] *= self._hi_res
                del new_shape[-2 - i]
                pos_coding = tf.reshape(pos_coding, shape=new_shape)
            coding_sum = tf.reduce_sum(pos_coding, axis=-1, keep_dims=True)
            coding_sum = tf.where(coding_sum <= 0, tf.ones(coding_sum.shape, tf.float32), coding_sum)
            pos_coding = pos_coding / coding_sum
        pos_sen_feat_3d = tf.concat([feat_3d, pos_coding], axis=-1)
        return pos_sen_feat_3d
