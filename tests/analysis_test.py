import os
import h5py

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op

import matplotlib.pyplot as plt
from matplotlib import ticker

from tests.utils_test import GlobalConfiguration
import libs

ANALYSIS_VIS_DIR = os.path.join(GlobalConfiguration.Visualization_Dir, 'analysis')
if not os.path.exists(ANALYSIS_VIS_DIR):
    os.mkdir(ANALYSIS_VIS_DIR)


def _norm_to_cdf(raw_dist):
    return np.cumsum(raw_dist / np.sum(raw_dist))


def _norm(raw_dist):
    return raw_dist / np.sum(raw_dist)


class ProjectionAnalysisTest(tf.test.TestCase):
    """
    This test case contains the analysis terms relating to projection ops
    """
    def _Reverse_Projection_Distance(self, batch_size, image_shape=None, vox_shape=None, feat_scale=1):
        if vox_shape is None:
            vox_shape = [240, 144, 240]
        if image_shape is None:
            image_shape = [480, 640]
        batch_feat_image_shape = [batch_size, ] + list((np.array(image_shape) / feat_scale).astype(np.int32).tolist())
        batch_image_shape = [batch_size, ] + image_shape
        with self.test_session() as sess:
            depth = constant_op.constant(0, shape=batch_image_shape + [1, ], dtype=tf.float32)
            scene_info = constant_op.constant(0, shape=[batch_size, 30], dtype=tf.float32)
            feat = constant_op.constant(0, shape=batch_feat_image_shape + [1, ], dtype=tf.float32)
            outputs = libs.view_volume_reverse_projection(depth, feat, scene_info, vox_shape, test_mode=True)
            return [depth, scene_info, feat], outputs, sess

    @staticmethod
    def histogram_curves(distributions, bins, cond, x_axis, y_axis, bins_axis, name):
        filtered = distributions[cond(distributions)]
        filtered = np.reshape(filtered, [-1, ])
        hist, _ = np.histogram(filtered, bins=bins)

        std_axis = np.arange(len(bins_axis))
        plt.xlabel(x_axis)
        plt.ylim([0, 1])
        plt.ylabel(y_axis)

        cdf, = plt.plot((std_axis[:-1] + std_axis[1:]) / 2,  _norm_to_cdf(hist), label='cdf')
        pdf, = plt.plot((std_axis[:-1] + std_axis[1:]) / 2, _norm(hist), label='pdf')

        plt.xticks(std_axis, bins_axis)

        plt.legend(handles=[cdf, pdf])
        plt.savefig(name + '.png')

    def testReverseProjectionDistance(self):
        if not GlobalConfiguration.UNDERSTANDING_TEST or GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Analysis Test: reverse projection distance distribution')
        feat_scale = 8
        vox_scale = 8
        batch_size = 1
        base_slide_size = 0.02
        image_shape = [480, 640]
        vox_shape = [30, 18, 30]
        feat_shape = list((np.array(image_shape) / feat_scale).astype(np.int32).tolist())
        depth, _, scene_info, _ = GlobalConfiguration.read_test_data(batch_size, 8)
        inputs, outputs, sess = self._Reverse_Projection_Distance(1, image_shape, vox_shape, feat_scale=feat_scale)
        _, intermediate = outputs

        location_results = []
        weights_results = []
        distance_results = []
        for feeds in zip(depth, scene_info):
            d, s = [np.expand_dims(feed, axis=0) for feed in feeds]
            feed_dict = {inputs[0]: d, inputs[1]: s, inputs[2]: np.random.rand(1, *feat_shape, 1)}
            _, location, weights, distance = sess.run(intermediate, feed_dict=feed_dict)
            location_results.append(location[0])
            weights_results.append(weights[0])
            distance_results.append(distance[0])
        distance_results = np.concatenate(distance_results, axis=1)

        if os.path.exists(ANALYSIS_VIS_DIR):
            # projection distance visualization
            vox_slide_size = base_slide_size * vox_scale
            bins_scale = np.arange(2, 10, 1)
            left_bins = np.flip(vox_slide_size / bins_scale, axis=0)
            right_bins = vox_slide_size * bins_scale
            bins = np.concatenate((left_bins, [vox_slide_size], right_bins), axis=0)

            bins_axis = []
            bins_axis.extend(['1:%d' % int(item) for item in np.flip(np.arange(2, 10, 1), axis=0)])
            bins_axis.extend(['1:1', ])
            bins_axis.extend(['%d:1' % int(item) for item in np.arange(2, 10, 1)])

            reverse_proj_distance_path = os.path.join(ANALYSIS_VIS_DIR, 'reverse_proj_distance_distribution')
            self.histogram_curves(distance_results, bins, lambda cond: np.all(cond > 0, axis=-1),
                                  'distance', 'prob', bins_axis, reverse_proj_distance_path)

            # reverse projection locations map
            loc_index = 0
            for location_result, weight_result in zip(location_results, weights_results):
                y_axis_size, x_axis_size = [int(item) for item in np.array(image_shape) / feat_scale]
                plt.figure(num=None, figsize=(x_axis_size, y_axis_size), dpi=400, facecolor='w', edgecolor='k')
                plt.xlim([0, x_axis_size])
                plt.ylim([0, y_axis_size])
                for i in range(x_axis_size):
                    plt.axvline(i, color='k')
                for i in range(y_axis_size):
                    plt.axhline(i, color='k')
                ys, xs = [np.squeeze(axis) for axis in np.split(location_result, 2, axis=-1)]
                weight_result = np.squeeze(weight_result)
                plt.plot(xs, ys, 'ro')
                for y, x, weight in zip(ys, xs, weight_result):
                    plt.text(x, y, '(%0.2f, %0.2f)\n (%0.2f, %0.2f, %0.2f, %0.2f)' % (x, y, *list(weight.tolist())))
                plt.savefig(os.path.join(ANALYSIS_VIS_DIR, 'reverse_cast_image_%d.png' % loc_index))
                loc_index += 1

    def _forward_projection(self, batch_size, vox_shape=None, image_shape=None):
        if vox_shape is None:
            vox_shape = [240, 144, 240]
        if image_shape is None:
            image_shape = [480, 640]
        batch_image_shape = [batch_size, ] + image_shape
        with self.test_session() as sess:
            depth = constant_op.constant(0, shape=batch_image_shape + [1, ], dtype=tf.float32)
            scene_info = constant_op.constant(0, shape=[batch_size, 30], dtype=tf.float32)
            outputs = libs.img2vol_forward_projection(depth, scene_info, vox_shape)
            return [depth, scene_info], outputs, sess

    def testDumpOccupiedVoxel(self):
        if not GlobalConfiguration.UNDERSTANDING_TEST or GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Analysis Test: reverse projection distance distribution')
        batch_size = -1
        depth, _, scene_info, _ = GlobalConfiguration.read_test_data(batch_size, 4)

        inputs, outputs, sess = self._forward_projection(1, vox_shape=[60, 36, 60])

        occupied_tensor = []
        for feed in zip(depth, scene_info):
            feed_dict = {inputs[0]: np.expand_dims(feed[0], axis=0), inputs[1]: np.expand_dims(feed[1], axis=0)}
            _, _, occupied = sess.run(outputs, feed_dict=feed_dict)
            occupied_tensor.append(np.squeeze(occupied))

        occupied_tensor = np.array(occupied_tensor)
        if not os.path.exists(ANALYSIS_VIS_DIR):
            os.mkdir(ANALYSIS_VIS_DIR)
        file_name = os.path.join(ANALYSIS_VIS_DIR, 'occupied.hdf5')
        fp = h5py.File(file_name,  'w')
        result = fp.create_dataset('result', occupied_tensor.shape, dtype='f')
        result[...] = occupied_tensor
        fp.close()
        return
