from abc import abstractmethod
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

from models import network
import libs


class VVNet(network.Network):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNet, self).__init__()
        self._original_volume = np.array([240, 144, 240], np.int32)
        self._volume_scale = 1
        self._image_scale = 1
        self._fuse_methods = libs.FusionType.AVERAGE_POOLING
        self._reuse = reuse
        self._is_training = is_training
        self._reg, _ = self.apply_l2_regularization(wd)

    def _conv_bn_2d(self, inputs, out_channel, kernel_size, fn=None, scope=''):
        conv = layers.conv2d(inputs, out_channel, kernel_size, activation_fn=None,
                             scope='_'.join((scope, 'conv')), **self._reg)
        bn = layers.batch_norm(conv, scale=True, activation_fn=fn, is_training=self._is_training,
                               scope='_'.join((scope, 'bn')))
        return bn

    def _resnet_2d(self, inputs, out_channel, scope='', no_short=False):
        with tf.variable_scope(scope, reuse=self._reuse):
            res_1 = self._conv_bn_2d(inputs, out_channel, 3, tf.nn.relu, scope='res_1')
            if no_short:
                return self._conv_bn_2d(res_1, out_channel, 3, tf.nn.relu, scope='res_2')
            else:
                res_2 = self._conv_bn_2d(res_1, out_channel, 3, None, scope='res_2')
                shortcut = self._conv_bn_2d(inputs, out_channel, 1, None, scope='shortcut')
                return tf.nn.relu(res_2 + shortcut)

    def _up_sampling_feature(self, feat, axis_range):
        with tf.variable_scope('UpSampleFeat', self._reuse):
            def _axis_expand(source_feat, target_axis):
                expand_feat = tf.stack([source_feat] * self._image_scale, axis=target_axis + 1)
                new_shape = [int(dim_size) for dim_size in source_feat.shape]
                new_shape[axis] *= self._image_scale
                expand_feat = tf.reshape(expand_feat, new_shape)
                return expand_feat
            feat_reshape = feat
            for axis in axis_range:
                feat_reshape = _axis_expand(feat_reshape, axis)
            return feat_reshape

    def _inputs(self, inputs):
        with tf.variable_scope('BlendInput', reuse=self._reuse):
            depth, normal, scene_info = inputs
            scene_info = self.modify_scene_info(scene_info, 1., self._volume_scale)
            feat_input = tf.concat([depth, normal], axis=-1)
            return feat_input, depth, scene_info

    def _before_view_fuse(self, feat_2d):
        with tf.variable_scope('BeforeViewFuse', reuse=self._reuse):
            feat_2d = self._up_sampling_feature(feat_2d, [1, 2])
            return feat_2d

    def _view_fuse(self, depth, feat_2d, scene_info):
        with tf.variable_scope('ViewFuse', reuse=self._reuse):
            scaled_volume = (self._original_volume / self._volume_scale).tolist()
            return libs.feat_projection(depth, feat_2d, scene_info, scaled_volume, self._fuse_methods)

    def _after_view_fuse(self, feat_3d):
        with tf.variable_scope('AfterViewFuse', reuse=self._reuse):
            return feat_3d

    def _2d_dnn_level_640(self, inputs):
        return self._resnet_2d(inputs, 8, scope='ResNet640', no_short=True)

    def _2d_dnn_level_320(self, inputs):
        return self._resnet_2d(inputs, 16, scope='ResNet320')

    def _2d_dnn_level_160(self, inputs):
        return self._resnet_2d(inputs, 32, scope='ResNet160')

    def _3d_dnn_level_120(self, inputs):
        with tf.variable_scope('Volume120', reuse=self._reuse):
            conv = layers.conv3d(inputs, 32, 3, scope='conv1', **self._reg)
            conv = layers.conv3d(conv, 32, 3, activation_fn=None, scope='conv2', **self._reg)
            shortcut = layers.conv3d(inputs, 32, kernel_size=1, activation_fn=None, scope='shortcut', **self._reg)
            res = tf.nn.relu(conv + shortcut)
            pool = layers.max_pool3d(res, 2, scope='pool')
            return pool

    def _3d_dnn_level_60(self, inputs):
        with tf.variable_scope('Volume60', reuse=self._reuse):
            conv = layers.conv3d(inputs, 64, 3, scope='conv1', **self._reg)
            conv = layers.conv3d(conv, 64, 3, activation_fn=None, scope='conv2', **self._reg)
            shortcut = layers.conv3d(inputs, 64, 1, activation_fn=None, scope='shortcut1', **self._reg)
            res1 = tf.nn.relu(conv + shortcut)

            conv = layers.conv3d(res1, 64, 3, scope='conv3', **self._reg)
            conv = layers.conv3d(conv, 64, 3, activation_fn=None, scope='conv4', **self._reg)
            res2 = tf.nn.relu(conv + res1)

            conv = layers.conv3d(res2, 64, 3, rate=2, scope='conv5', **self._reg)
            conv = layers.conv3d(conv, 64, 3, activation_fn=None, rate=2, scope='conv6', **self._reg)
            res3 = tf.nn.relu(conv + res2)

            conv = layers.conv3d(res3, 64, 3, rate=2, scope='conv7', **self._reg)
            conv = layers.conv3d(conv, 64, 3, activation_fn=None, rate=2, scope='conv8', **self._reg)
            res4 = tf.nn.relu(conv + res3)

            concat = tf.concat([res1, res2, res3, res4], axis=-1, name='concat')
            return tf.nn.relu(concat)

    @abstractmethod
    def _2d_dnn(self, inputs): pass

    @abstractmethod
    def _3d_dnn(self, inputs): pass

    def _prediction_reduce(self, inputs, conv_reg_dict):
        with tf.variable_scope('PredictReduction', reuse=self._reuse):
            conv = layers.conv3d(inputs, 128, 1, scope='conv4_1', **conv_reg_dict)
            conv = layers.conv3d(conv, 128, 1, scope='conv4_2', **conv_reg_dict)
            fc12 = layers.conv3d(conv, 12, 1, activation_fn=None, scope='fc12', **conv_reg_dict)
            return fc12

    def instance(self, inputs, scope='VVNet'):
        with tf.variable_scope(scope, reuse=self._reuse):
            # acquire the inputs
            feat_input, depth, scene_info = self._inputs(inputs)
            feat_2d = self._2d_dnn(feat_input)
            feat_before_fuse = self._before_view_fuse(feat_2d)
            feat_fuse = self._view_fuse(depth, feat_before_fuse, scene_info)
            feat_after_fuse = self._after_view_fuse(feat_fuse)
            feat_3d = self._3d_dnn(feat_after_fuse)
            predict = self._prediction_reduce(feat_3d, self._reg)
            return predict


class VVNet120(VVNet):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNet120, self).__init__(reuse, is_training, wd)
        self._volume_scale = 2
        self._image_scale = 2

    def _2d_dnn(self, inputs):
        feat_2d = self._2d_dnn_level_640(inputs)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        return self._2d_dnn_level_320(feat_2d)

    def _3d_dnn(self, inputs):
        feat_3d = self._3d_dnn_level_120(inputs)
        return self._3d_dnn_level_60(feat_3d)


class VVNet60(VVNet):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNet60, self).__init__(reuse, is_training, wd)
        self._volume_scale = 4
        self._image_scale = 4

    def _2d_dnn(self, inputs):
        feat_2d = self._2d_dnn_level_640(inputs)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        feat_2d = self._2d_dnn_level_320(feat_2d)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        return self._2d_dnn_level_160(feat_2d)

    def _3d_dnn(self, inputs):
        return self._3d_dnn_level_60(inputs)


class VVNet120Img320(VVNet120):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNet120Img320, self).__init__(reuse, is_training, wd)
        self._image_scale = 1

    def _inputs(self, inputs):
        with tf.variable_scope('BlendLowResInput', reuse=self._reuse):
            depth, normal, scene_info = inputs
            scene_info = self.modify_scene_info(scene_info, 2., self._volume_scale)
            depth = tf.image.resize_images(depth, [240, 320])
            normal = tf.image.resize_images(normal, [240, 320])
            feat_input = tf.concat([depth, normal], axis=-1)
            return feat_input, depth, scene_info

    def _2d_dnn(self, inputs):
        return self._resnet_2d(inputs, 16, scope='ResNet320', no_short=True)


class VVNet60Img320(VVNet60):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNet60Img320, self).__init__(reuse, is_training, wd)
        self._image_scale = 2

    def _inputs(self, inputs):
        with tf.variable_scope('BlendLowResInput', reuse=self._reuse):
            depth, normal, scene_info = inputs
            scene_info = self.modify_scene_info(scene_info, 2., self._volume_scale)
            depth = tf.image.resize_images(depth, [240, 320])
            normal = tf.image.resize_images(normal, [240, 320])
            feat_input = tf.concat([depth, normal], axis=-1)
            return feat_input, depth, scene_info

    def _2d_dnn(self, inputs):
        feat_2d = self._resnet_2d(inputs, 16, scope='ResNet320', no_short=True)
        feat_2d = layers.max_pool2d(feat_2d, 2, 2)
        return self._2d_dnn_level_160(feat_2d)


class VVNet120MaxFuse(VVNet120):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNet120MaxFuse, self).__init__(reuse, is_training, wd)
        self._fuse_methods = libs.FusionType.MAX_POOLING


class VVNet60MaxFuse(VVNet60):
    def __init__(self, reuse, is_training, wd=0.0005):
        super(VVNet60MaxFuse, self).__init__(reuse, is_training, wd)
        self._fuse_methods = libs.FusionType.MAX_POOLING


# class VVNet120Sep(VVNet):
#     def __init__(self):
#         super(VVNet120Sep, self).__init__()
#
#     @staticmethod
#     def separate_feature_along_tsdf(depth, feat_2d, scene_info, vox_size):
#         with tf.name_scope('separate_features'):
#             batch_size = int(depth.shape[0])
#             vox_map, img_map = libs.img_vox_projection(depth, scene_info, vox_size)
#             surface_feat = libs.feat_fusion_vox(feat_2d, img_map, vox_map, libs.FusionType.AVERAGE_POOLING)
#             tsdf, cast_map = libs.tsdf_via_vox(depth, scene_info, vox_map, True)
#             sub_sep_feats = []
#             for sub_tensor in zip(tf.split(cast_map, batch_size, axis=0), tf.split(surface_feat, batch_size, axis=0)):
#                 sub_cast_map, sub_surface_feat = sub_tensor
#                 reshape_feat = tf.reshape(sub_surface_feat, [np.prod(vox_size), -1])
#                 reshape_cast_map = tf.reshape(sub_cast_map, [-1])
#                 invalid_indices = reshape_cast_map < 0
#                 reshape_cast_map = tf.where(invalid_indices, tf.zeros(reshape_cast_map.shape, dtype=tf.int32),
#                                             reshape_cast_map)
#                 sub_feat = tf.gather(reshape_feat, reshape_cast_map)
#                 sub_feat = tf.where(invalid_indices, tf.zeros(sub_feat.shape, dtype=tf.float32), sub_feat)
#                 sub_sep_feats.append(tf.reshape(sub_feat, shape=vox_size + [-1]))
#             signal_feat = tf.stack(sub_sep_feats, axis=0)
#             signal_feat = signal_feat * tsdf
#             return signal_feat
#
#     def instance(self, inputs, is_training=True, reuse=False, l2_factor=0.0005, scope='VVNet120'):
#         conv_reg_dict, _ = SSCNet.apply_l2_regularization(l2_factor)
#         with tf.variable_scope(scope, reuse=reuse):
#             # acquire the inputs
#             depth, normal, label_scene_info = inputs
#             scene_info = Network.modify_scene_info(label_scene_info, 1., 2.)
#             input_2d = tf.concat([depth, normal], axis=-1)
#             stage_2d_640 = self._2d_dnn_level_640(input_2d, is_training, conv_reg_dict)
#             pool_2d_640 = layers.max_pool2d(stage_2d_640, 2, 2)
#             stage_2d_320 = self._2d_dnn_level_320(pool_2d_640, is_training, conv_reg_dict)
#             feat_2d = self._up_sampling_feature(stage_2d_320, [1, 2], 2)
#             feat_3d = VVNet120Sep.separate_feature_along_tsdf(depth, feat_2d, scene_info, [120, 72, 120])
#             stage_3d_120 = self._dnn_level_120(feat_3d, conv_reg_dict)
#             stage_3d_60 = self._dnn_level_60(stage_3d_120, conv_reg_dict)
#             predict = self._prediction_reduce(stage_3d_60, conv_reg_dict)
#             return predict
#
#
# class VVNet60Sep(VVNet120Sep):
#     def __init__(self):
#         super(VVNet60Sep, self).__init__()
#
#     def instance(self, inputs, is_training=True, reuse=False, l2_factor=0.0005, scope='VVNet60Sep'):
#         conv_reg_dict, _ = SSCNet.apply_l2_regularization(l2_factor)
#         with tf.variable_scope(scope, reuse=reuse):
#             # acquire the inputs
#             depth, normal, label_scene_info = inputs
#             scene_info = Network.modify_scene_info(label_scene_info, 1., 4.)
#             input_2d = tf.concat([depth, normal], axis=-1)
#             stage_2d_640 = self._2d_dnn_level_640(input_2d, is_training, conv_reg_dict)
#             pool_2d_640 = layers.max_pool2d(stage_2d_640, 2, 2)
#             stage_2d_320 = self._2d_dnn_level_320(pool_2d_640, is_training, conv_reg_dict)
#             pool_2d_320 = layers.max_pool2d(stage_2d_320, 2, 2)
#             stage_2d_160 = self._2d_dnn_level_160(pool_2d_320, is_training, conv_reg_dict)
#             feat_2d = self._up_sampling_feature(stage_2d_160, [1, 2], 4)
#             feat_3d = VVNet120Sep.separate_feature_along_tsdf(depth, feat_2d, scene_info, [60, 36, 60])
#             stage_3d_60 = self._dnn_level_60(feat_3d, conv_reg_dict)
#             predict = self._prediction_reduce(stage_3d_60, conv_reg_dict)
#             return predict
#
#
# class VVNet60SepDF(VVNet60Sep):
#     def __init__(self):
#         super(VVNet60SepDF, self).__init__()
#
#     @staticmethod
#     def separate_feature_along_tsdf(depth, feat_2d, scene_info, vox_size):
#         with tf.name_scope('separate_features'):
#             batch_size = int(depth.shape[0])
#             vox_map, img_map = libs.img_vox_projection(depth, scene_info, vox_size)
#             surface_feat = libs.feat_fusion_vox(feat_2d, img_map, vox_map, libs.FusionType.AVERAGE_POOLING)
#             tsdf, cast_map = libs.tsdf_via_vox(depth, scene_info, vox_map, True)
#             cond_tsdf = tf.logical_and(tsdf < 1, tsdf > 0)
#             tsdf = tf.where(cond_tsdf, tf.zeros(tsdf.shape, tf.float32), tsdf)
#             tsdf = tf.abs(tsdf)
#             sub_sep_feats = []
#             for sub_tensor in zip(tf.split(cast_map, batch_size, axis=0), tf.split(surface_feat, batch_size, axis=0)):
#                 sub_cast_map, sub_surface_feat = sub_tensor
#                 reshape_feat = tf.reshape(sub_surface_feat, [np.prod(vox_size), -1])
#                 reshape_cast_map = tf.reshape(sub_cast_map, [-1])
#                 invalid_indices = reshape_cast_map < 0
#                 reshape_cast_map = tf.where(invalid_indices, tf.zeros(reshape_cast_map.shape, dtype=tf.int32),
#                                             reshape_cast_map)
#                 sub_feat = tf.gather(reshape_feat, reshape_cast_map)
#                 sub_feat = tf.where(invalid_indices, tf.zeros(sub_feat.shape, dtype=tf.float32), sub_feat)
#                 sub_sep_feats.append(tf.reshape(sub_feat, shape=vox_size + [-1]))
#             signal_feat = tf.stack(sub_sep_feats, axis=0)
#             signal_feat = signal_feat * tsdf
#             return signal_feat


# class VVNet60AddConv(VVNet60):
#     def __init__(self):
#         super(VVNet60AddConv, self).__init__()
#
#     def _after_view_fuse(self, feat_3d, reuse, conv_reg_dict):
#         super(VVNet60AddConv, self)._after_view_fuse(feat_3d, reuse, conv_reg_dict)
#         with tf.variable_scope('AfterViewFuse_AC', reuse=reuse):
#             extra_feat_3d = layers.conv3d(feat_3d, 32, 3, scope='add_conv_0', **conv_reg_dict)
#             self._variable_list.append(extra_feat_3d)
#             return extra_feat_3d
#
#
# class VVNet60AddRes(VVNet60):
#     def __init__(self):
#         super(VVNet60AddRes, self).__init__()
#
#     def _after_view_fuse(self, feat_3d, reuse, conv_reg_dict):
#         super(VVNet60AddRes, self)._after_view_fuse(feat_3d, reuse, conv_reg_dict)
#         with tf.variable_scope('AfterViewFuse_AR', reuse=reuse):
#             extra_res_0 = layers.conv3d(feat_3d, 32, 3, scope='add_res_0', **conv_reg_dict)
#             extra_res_1 = layers.conv3d(extra_res_0, 32, 3, activation_fn=None, scope='add_res_1', **conv_reg_dict)
#             with tf.name_scope('add_res_plus'):
#                 extra_res = tf.nn.relu(extra_res_1 + feat_3d)
#             self._variable_list.extend([extra_res_0, extra_res_1, extra_res])
#             return extra_res
#
#
# class VVNet60AddConvRes(VVNet60):
#     def __init__(self):
#         super(VVNet60AddConvRes, self).__init__()
#
#     def _after_view_fuse(self, feat_3d, reuse, conv_reg_dict):
#         super(VVNet60AddConvRes, self)._after_view_fuse(feat_3d, reuse, conv_reg_dict)
#         with tf.variable_scope('AfterViewFuse_ACR', reuse=reuse):
#             extra_feat_3d = layers.conv3d(feat_3d, 32, 3, scope='add_conv_0', **conv_reg_dict)
#             with tf.name_scope('add_res_plus'):
#                 extra_res = tf.nn.relu(extra_feat_3d + feat_3d)
#             self._variable_list.extend([extra_feat_3d, extra_res])
#             return extra_res
