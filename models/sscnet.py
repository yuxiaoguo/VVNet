import tensorflow as tf
import tensorflow.contrib.layers as layers

from models import vvnet
import libs


class SSCNet(vvnet.VVNet):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(SSCNet, self).__init__(reuse, is_training, wd)

    def _3d_dnn_level_240(self, inputs):
        with tf.variable_scope('Volume240', reuse=self._reuse):
            return layers.conv3d(inputs, 16, 7, stride=2, scope='conv', **self._reg)

    def _view_fuse(self, depth, feat_2d, scene_info):
        with tf.variable_scope('ViewFuse', reuse=self._reuse):
            return libs.tsdf_projection(depth, scene_info, self._original_volume.tolist())

    def _2d_dnn(self, inputs):
        return inputs

    def _3d_dnn(self, inputs):
        feat_3d = self._3d_dnn_level_240(inputs)
        feat_3d = self._3d_dnn_level_120(feat_3d)
        return self._3d_dnn_level_60(feat_3d)

    def instance(self, inputs, scope='SSCNet'):
        return super(SSCNet, self).instance(inputs, scope)
