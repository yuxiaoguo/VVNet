import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib import layers

from enum import Enum

_bin_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bin')
LD_LIB = os.environ['LD_LIBRARY_PATH']
if str(LD_LIB).find(_bin_path) == -1:
    os.environ['LD_LIBRARY_PATH'] += os.pathsep + _bin_path
    os.execv(sys.executable, [sys.executable] + sys.argv)
_tf_graphics_module = tf.load_op_library(os.path.join(_bin_path, 'libtfgraphics.so'))

_TsdfViaVox = _tf_graphics_module.tsdf_via_vox

_FeatFusionVox = _tf_graphics_module.feat_fusion_vox
_FeatFusionVoxGrad = _tf_graphics_module.feat_fusion_vox_grad

_BilinearInterpolation = _tf_graphics_module.bilinear_interpolation
_BilinearInterpolationGrad = _tf_graphics_module.bilinear_interpolation_grad

_ForwardProjection = _tf_graphics_module.forward_projection
_ReverseProjection = _tf_graphics_module.reverse_projection
_InterpolationWeights = _tf_graphics_module.interpolation_weights
_CameraToImage = _tf_graphics_module.camera_to_image


class FusionType(Enum):
    MAX_POOLING = 1
    AVERAGE_POOLING = 2


def tsdf_via_vox(depth, scene_info, vox_map, is_flipped):
    """
    Convert a vox into tsdf format
    :param depth: the depth info
    :param scene_info: the scene info, including voxel resolution and image viewpoint
    :param vox_map: the non-empty voxel
    :param is_flipped: whether to use the flipped signal
    :type depth: tf.Tensor
    :type scene_info: tf.Tensor
    :type vox_map: tf.Tensor
    :type is_flipped: bool
    :return: tf.Tensor
    """
    if vox_map.dtype is not tf.float32:
        vox_map = tf.cast(vox_map, tf.float32)
    return _TsdfViaVox(depth, scene_info, vox_map, is_flipped)


@ops.RegisterGradient('FeatFusionVox')
def feat_fusion_vox_grad(op, vox_grad):
    """
    Gradient operation of feat_fusion_vox
    :param op: info of forward ops
    :param vox_grad: the vox features
    :type op: tf.Operation
    :type vox_grad: tf.Tensor
    :return: tf.Tensor
    """
    img_feat, img_map, img_weights, vox_weights = op.inputs
    vox_feat = op.outputs[0]
    img_grad, _ = _FeatFusionVoxGrad(img_feat, vox_grad, vox_feat, img_map, img_weights, vox_weights,
                                     op.get_attr('fusion_type'))
    return [img_grad, None, None, None]


def feat_fusion_vox(img_feat, img_map, img_weights, vox_weights, fusion_type=FusionType.MAX_POOLING):
    """
    Fusion the image features into voxel
    :param img_feat: the image features
    :param img_map: the cast relationship between voxel and image
    :param img_weights: the weights blended for volume fusion
    :param vox_weights: the volume weights for normalization
    :param fusion_type: the fusion type to deal with multiple corresponding
    :type img_feat: tf.Tensor
    :type img_map: tf.Tensor
    :type img_weights: tf.Tensor
    :type vox_weights: tf.Tensor
    :type fusion_type: FusionType
    :return: tf.Tensor
    """
    return _FeatFusionVox(img_feat, img_map, img_weights, vox_weights,
                          0 if fusion_type == FusionType.MAX_POOLING else 1)


def feat_projection(depth, img_feat, scene_info, vox_size, fusion_type=FusionType.AVERAGE_POOLING):
    img_vox_map, _, vox_occupied = img2vol_forward_projection(depth, scene_info, vox_size)
    weights = tf.constant(1, dtype=tf.float32, shape=depth.shape)
    vox_occupied = tf.cast(vox_occupied, tf.float32)
    vox_feat = feat_fusion_vox(img_feat, img_vox_map, weights, vox_occupied, fusion_type)
    return vox_feat


def tsdf_projection(depth, scene_info, vox_size, flipped_tsdf=True):
    _, _, vox_weights = img2vol_forward_projection(depth, scene_info, vox_size)
    tsdf, _ = tsdf_via_vox(depth, scene_info, vox_weights, flipped_tsdf)
    return tsdf


def feat_fusion(depth, img_feat, scene_info, vox_size, fusion_type=FusionType.MAX_POOLING, flipped_tsdf=True):
    """
    The packaged fusion operation, including projection, fusion, tsdf
    :param depth: the depth info
    :param img_feat: the image features
    :param scene_info: the scene info, including voxel details and viewpoint
    :param vox_size: the vox size
    :param fusion_type: the fusion type, max or average
    :param flipped_tsdf: whether to flip the signal or not
    :type depth: tf.Tensor
    :type img_feat: tf.Tensor or tf.Variable
    :type scene_info: tf.Tensor
    :type vox_size: list(int)
    :type fusion_type: FusionType
    :type flipped_tsdf: bool
    :return: list(tf.Tensor)
    """
    img_map, _, vox_occupied = img2vol_forward_projection(depth, scene_info, vox_size)
    tsdf_feat, _ = tsdf_via_vox(depth, scene_info, vox_occupied, flipped_tsdf)
    weights = tf.constant(1, dtype=tf.float32, shape=depth.shape)
    vox_occupied = tf.cast(vox_occupied, tf.float32)
    vox_feat = feat_fusion_vox(img_feat, img_map, weights, vox_occupied, fusion_type)
    return vox_feat, tsdf_feat, vox_occupied, img_map


def details_and_fov(img_height, img_width, img_scale, vox_scale):
    vox_details = np.array([0.02 * vox_scale, 0.24], np.float32)
    camera_fov = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                           0., 518.8579 / img_scale, img_height / (2 * img_scale),
                           0., 0., 1.], dtype=np.float32)
    return vox_details, camera_fov


def scene_base_info(img_scale=1., vox_scale=0., batch_size=0):
    """
    Get the basic info of scene
    :param img_scale: the image re-scale factor, base size is [480, 640]
    :param batch_size: the batch size of scene, if batch_size=0, will return the non-batch result
    :param vox_scale: the vox scale factor, comparing with 240/144/240
    :return:
    """
    if vox_scale == 0.:
        vox_scale = img_scale
    img_height, img_width = [480, 640]
    vox_details_np, camera_fov_np = details_and_fov(img_height, img_width, img_scale, vox_scale)
    vox_details = tf.constant(vox_details_np, tf.float32, [2, ], name='vox_details')
    camera_fov = tf.constant(camera_fov_np, tf.float32, [9, ], name='camera_fov')

    if batch_size != 0:
        vox_details = tf.stack([vox_details] * batch_size, axis=0)
        camera_fov = tf.stack([camera_fov] * batch_size, axis=0)

    return vox_details, camera_fov


def bilinear_interpolation(sampled_feat, interp_points, interp_weights):
    """
    Bilinear interpolation op
    :param sampled_feat:
    :param interp_points:
    :param interp_weights:
    :return:
    """
    return _BilinearInterpolation(sampled_feat, interp_points, interp_weights)


@ops.RegisterGradient('BilinearInterpolation')
def bilinear_interpolation_grad(op, interp_grad):
    """
    Bilinear interpolation gradient op
    :param op: the operation of bilinear interpolation
    :param interp_grad: the gradient from
    :type op: tf.Operation
    :type interp_grad: tf.Tensor or tf.Variable
    :return:
    """
    sampled_feat, interp_points, interp_weights = op.inputs
    sampled_size = [int(axis_size) for axis_size in sampled_feat.shape[1:-1]]
    return [_BilinearInterpolationGrad(interp_grad, interp_points, interp_weights, sampled_size), None, None]


def interpolation_flooring_2d(sampled_feat, interp_size):
    """
    Using bilinear interpolation to sample an image into higher resolution, with flooring sampling scheme
    :param sampled_feat: the feature map to be sampled
    :param interp_size: the interpolated feature map size
    :type sampled_feat: tf.Tensor or tf.Variable
    :type interp_size: list(int)
    :return: tf.Tensor
    """
    sampled_size = [int(axis_size) for axis_size in sampled_feat.shape[1:-1]]
    sampling_ratio = np.array(interp_size) / np.array(sampled_size)
    interp_map = np.mgrid[0:sampling_ratio[0]:interp_size[0], 0:sampling_ratio[1]:interp_size[1]]
    indices_map = np.stack([interp_map] * int(sampled_feat.shape[0]), axis=0)
    interp_points = tf.Constant(indices_map, dtype=tf.float32)
    interp_weights = tf.Constant(np.floor(indices_map), dtype=tf.float32)
    return _BilinearInterpolation(sampled_feat, interp_points, interp_weights)


def interpolation_average_2d(sampled_feat, interp_size):
    """
    Using bilinear interpolation to sample an image into higher resolution, with average sampling scheme
    :param sampled_feat: the feature map to be sampled
    :param interp_size: the interpolated feature map size
    :type sampled_feat: tf.Tensor or tf.Variable
    :type interp_size: list(int)
    :return: tf.Tensor
    """
    sampled_size = [int(axis_size) for axis_size in sampled_feat.shape[1:-1]]
    sampling_ratio = np.array(interp_size) / np.array(sampled_size)
    interp_map = np.mgrid[0:sampling_ratio[0]:interp_size[0], 0:sampling_ratio[1]:interp_size[1]]
    indices_map = np.stack([interp_map] * int(sampled_feat.shape[0]), axis=0)
    interp_points = tf.Constant(indices_map, dtype=tf.float32)
    interp_weights = tf.Constant(np.full(indices_map.shape, 0.25), dtype=tf.float32)
    return _BilinearInterpolation(sampled_feat, interp_points, interp_weights)


def interpolation_bilinear_2d(sampled_feat, interp_size, reuse):
    """
    Using standard bilinear interpolation to sample an image into higher resolution, width bilinear sampling scheme
    :param sampled_feat: the feature map to be sampled
    :param interp_size: the interpolated feature map size
    :param reuse: whether to reuse the weight within the operation
    :type sampled_feat: tf.Tensor or tf.Variable
    :type interp_size: list(int)
    :type reuse: bool
    :return: tf.Tensor
    """
    with tf.variable_scope('BilinearInterpolation2D', reuse=reuse):
        batch_size = int(sampled_feat.shape[0])
        channels = int(sampled_feat.shape[-1])
        sampled_size = [int(axis_size) for axis_size in sampled_feat.shape[1:-1]]

        interp_x, interp_y = np.meshgrid(np.arange(0, interp_size[1]), np.arange(0, interp_size[0]))
        interp_x = interp_x * (sampled_size[1] - 1) / (interp_size[1] - 1)
        interp_y = interp_y * (sampled_size[0] - 1) / (interp_size[0] - 1)
        interp_points = np.stack([interp_y, interp_x], axis=-1)
        interp_points = np.stack([interp_points] * batch_size, axis=0)
        interp_points = np.reshape(interp_points, [batch_size, -1, 2])
        weights_tl = (np.floor(interp_x + 1) - interp_x) * (np.floor(interp_y + 1) - interp_y)
        weights_tr = (interp_x - np.floor(interp_x)) * (np.floor(interp_y + 1) - interp_y)
        weights_bl = (np.floor(interp_x + 1) - interp_x) * (interp_y - np.floor(interp_y))
        weights_br = (interp_x - np.floor(interp_x)) * (interp_y - np.floor(interp_y))
        interp_weights = np.stack([weights_tl, weights_tr, weights_bl, weights_br], axis=-1)
        interp_weights = np.stack([interp_weights] * batch_size, axis=0)
        interp_weights = np.reshape(interp_weights, [batch_size, -1, 4])

        interp_feat = _BilinearInterpolation(sampled_feat, interp_points, interp_weights)
        return tf.reshape(interp_feat, [batch_size, ] + interp_size + [channels, ])


def img_to_cam_space(depth, cam_k):
    """
    Generate the corresponding xyz position of camera coordinate via depth map and camera focus info
    :param depth: the depth map
    :param cam_k: the camera focus information
    :type depth: tf.Tensor
    :type cam_k: tf.Tensor
    :return: tf.Tensor
    """
    depth_size = [int(item) for item in depth.shape]
    img_size = depth_size[1:-1]
    y_axis = np.arange(0, img_size[0])
    x_axis = np.arange(0, img_size[1])
    x, y = np.meshgrid(x_axis, y_axis)

    img_pos_x = tf.constant(np.stack([x] * depth_size[0], axis=0), tf.float32, depth_size[:-1] + [1, ])
    img_pos_y = tf.constant(np.stack([y] * depth_size[0], axis=0), tf.float32, depth_size[:-1] + [1, ])

    cam_k0 = tf.reshape(tf.gather(cam_k, [0, ], axis=-1), [-1, 1, 1, 1])
    cam_k2 = tf.reshape(tf.gather(cam_k, [2, ], axis=-1), [-1, 1, 1, 1])
    cam_k4 = tf.reshape(tf.gather(cam_k, [4, ], axis=-1), [-1, 1, 1, 1])
    cam_k5 = tf.reshape(tf.gather(cam_k, [5, ], axis=-1), [-1, 1, 1, 1])

    cam_pos_x = (img_pos_x - cam_k2) * depth / cam_k0
    cam_pos_y = (img_pos_y - cam_k5) * depth / cam_k4

    return tf.concat([cam_pos_x, cam_pos_y, depth], axis=-1)


def forward_projection(proj_points, camera_pose, vox_unit, vox_origin, vox_size):
    """
    Forward projection sampling scheme, cast image samples into volume space
    :param proj_points: the points set needed to project into volume space
    :param camera_pose: the transform matrix from camera space into volume space
    :param vox_unit: the vox stride for each volume grid
    :param vox_origin: the origin coordinate of the volume
    :param vox_size: the vox size
    :type proj_points: tf.Tensor or tf.Variable
    :type camera_pose: tf.Tensor or tf.Variable
    :type vox_unit: tf.Tensor or tf.Variable
    :type vox_origin: tf.Tensor or tf.Variable
    :type vox_size: list(int)
    :return: list(tf.Tensor)
    """
    return _ForwardProjection(proj_points, camera_pose, vox_unit, vox_origin, vox_size)


def img2vol_forward_projection(depth, scene_info, vox_size):
    vox_origin, cam_pose, vox_unit, cam_k = tf.split(scene_info, [3, 16, 2, 9], axis=-1)
    img_proj_pos = img_to_cam_space(depth, cam_k)
    proj_points = tf.reshape(img_proj_pos, [int(depth.shape[0]), -1, 3])
    point_vox_map, vox_occupied = forward_projection(proj_points, cam_pose, vox_unit, vox_origin, vox_size)
    point_vox_map = tf.reshape(point_vox_map, [int(depth.shape[0]), int(depth.shape[1]), int(depth.shape[2]), -1])
    return point_vox_map, img_proj_pos, vox_occupied


def reverse_projection(cam_pose, vox_unit, vox_origin, vox_size):
    """
    Reverse projection sampling scheme, cast volume samples into image space
    :param cam_pose: the transform matrix from camera space into volume space
    :param vox_unit: the vox stride for each volume grid
    :param vox_origin: the origin coordinate of the volume
    :param vox_size: the vox size
    :type cam_pose: tf.Tensor or tf.Variable
    :type vox_unit: tf.Tensor or tf.Variable
    :type vox_origin: tf.Tensor or tf.Variable
    :type vox_size: list(int)
    :return: list(tf.Tensor)
    """
    cam_pose = tf.reshape(cam_pose, shape=[-1, 4, 4])
    inv_camera_pose = tf.matrix_inverse(cam_pose)
    return _ReverseProjection(inv_camera_pose, vox_unit, vox_origin, vox_size)


def interpolation_weights(interp_points, vox_proj_pos, img_proj_pos, max_distance=0.02):
    """
    Using spatial distance to blend weights
    :param interp_points: the interpolated positions of image space
    :param vox_proj_pos: the spatial position of interpolated volumes
    :param img_proj_pos: the spatial position of sampled images
    :param max_distance: max distance for the valid points
    :type interp_points: tf.Tensor or tf.Variable
    :type img_proj_pos: tf.Tensor or tf.Variable
    :type max_distance: float
    :return: tf.Tensor
    """
    return _InterpolationWeights(interp_points, vox_proj_pos, img_proj_pos, max_distance)


def camera_to_image(vox_proj_pos, scene_info, scale):
    """
    Convert camera space x,y into image space pixel idx
    :param vox_proj_pos: the camera space position
    :param scene_info: the scene information
    :param scale: the scale factor, to scale up the feature maps
    :type vox_proj_pos: tf.Tensor or tf.Variable
    :type scene_info: tf.Tensor or tf.Variable
    :type scale: float
    :return: tf.Tensor
    """
    return _CameraToImage(vox_proj_pos, scene_info, scale)


def view_volume_reverse_projection(depth, img_feat, scene_info, vox_size, multi_scale=False, test_mode=False):
    """
    View volume projection -- using reverse projection
    :param depth: the depth map
    :param img_feat: the 2d features map
    :param scene_info: the scene information, including volume and transform matrix
    :param vox_size: the projected volume size
    :param multi_scale: using multiple scale features maps, based on the volume and image relationship
    :param test_mode: if in test mode, the function will return more intermediary results.
    :type depth: tf.Tensor or tf.Variable
    :type img_feat: tf.Tensor or tf.Variable
    :type scene_info: tf.Tensor or tf.Variable
    :type vox_size: list(int)
    :type multi_scale: bool
    :type test_mode: bool
    :return: tf.Tensor
    """
    if multi_scale:
        raise NotImplementedError
    vox_size = [int(size) for size in vox_size]
    batch_size = int(depth.shape[0])
    img_scale = int(int(depth.shape[-2]) / int(img_feat.shape[-2]))
    depth_batches, img_feat_batches, scene_info_batches = [tf.split(item, batch_size, axis=0)
                                                           for item in [depth, img_feat, scene_info]]
    feat_3d_batches = []
    intermediate = [[], [], [], []]
    for depth_batch, img_feat_batch, scene_info_batch in zip(depth_batches, img_feat_batches, scene_info_batches):
        _, img_proj_pos, vox_occupied = img2vol_forward_projection(depth_batch, scene_info_batch, vox_size)
        img_proj_pos = layers.avg_pool2d(img_proj_pos, img_scale, img_scale)

        vox_origin, cam_pose, vox_unit, _ = tf.split(scene_info_batch, [3, 16, 2, 9], axis=-1)
        vox_proj_pos = reverse_projection(cam_pose, vox_unit, vox_origin, vox_size)

        vox_occupied_indices = tf.where(tf.reshape(vox_occupied, shape=vox_occupied.shape[:-1]) > 0)
        vox_proj_pos = tf.expand_dims(tf.gather_nd(vox_proj_pos, vox_occupied_indices), axis=0)
        intermediate[0].append(vox_proj_pos)

        interp_points = camera_to_image(vox_proj_pos, scene_info_batch, img_scale)
        intermediate[1].append(interp_points)
        interp_weights, interp_distance = interpolation_weights(interp_points, vox_proj_pos, img_proj_pos,
                                                                max_distance=0.04 * img_scale)
        intermediate[2].append(interp_weights)
        intermediate[3].append(interp_distance)
        interp_feat = bilinear_interpolation(img_feat_batch, interp_points, interp_weights)
        interp_feat = tf.reshape(interp_feat, shape=[-1, int(interp_feat.shape[-1])])

        feat_3d_shape = tf.constant([1, ] + vox_size + [int(img_feat.shape[-1]), ], dtype=tf.int64)
        feat_3d_batch = tf.scatter_nd(vox_occupied_indices, interp_feat, feat_3d_shape)
        feat_3d_batches.append(feat_3d_batch)
    feat_3d = tf.concat(feat_3d_batches, axis=0)
    if test_mode:
        return feat_3d, intermediate
    else:
        return feat_3d
