from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib import layers

from models import vvnetae
import libs


class VVNetRP(vvnetae.VVNetAE):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetRP, self).__init__(reuse, is_training, wd)

    def _before_view_fuse(self, feat_2d):
        with tf.variable_scope('BeforeViewFuse', reuse=self._reuse):
            return feat_2d

    def _view_fuse(self, depth, feat_2d, scene_info):
        with tf.variable_scope('ViewFuse', reuse=self._reuse):
            scaled_volume = (self._original_volume / self._volume_scale).tolist()
            return libs.view_volume_reverse_projection(depth, feat_2d, scene_info, scaled_volume)

    def _after_view_fuse(self, feat_3d):
        with tf.variable_scope('AfterViewFuse', reuse=self._reuse):
            return feat_3d

    @abstractmethod
    def _2d_dnn(self, inputs): pass

    @abstractmethod
    def _3d_dnn(self, inputs): pass


class VVNetRP30(VVNetRP):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNetRP30, self).__init__(reuse, is_training, wd)
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
