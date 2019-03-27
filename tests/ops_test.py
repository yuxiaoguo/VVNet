import os

import tensorflow as tf
from scipy.interpolate import interp2d
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
import numpy as np

import libs
from tests.utils_test import GlobalConfiguration
from utils import visualize


class SamplingOpsTest(tf.test.TestCase):
    """
    Sampling 2D tests contain variable ops, relating to 2D sampling
    """
    def _bilinearInterpolation(self, device, x_shape, interp_size):
        with self.test_session(force_gpu=device, use_gpu=device) as sess:
            x = constant_op.constant(0, shape=x_shape, dtype=tf.float32)
            x_ops = tf.identity(x)
            y = libs.interpolation_bilinear_2d(x_ops, interp_size, False)
            return x, y, sess

    @staticmethod
    def _reference_bilinear(sampled_object, interp_size):
        batch_res = []
        sampled_size = sampled_object.shape[1:-1]
        interp_y = np.arange(0, interp_size[0]) / (interp_size[0] - 1)
        interp_x = np.arange(0, interp_size[1]) / (interp_size[1] - 1)
        sampled_x = np.arange(0, sampled_size[1]) / (sampled_size[1] - 1)
        sampled_y = np.arange(0, sampled_size[0]) / (sampled_size[0] - 1)
        for batch in np.split(sampled_object, sampled_object.shape[0], axis=0):
            channel_res = []
            for channel in np.split(batch, batch.shape[-1], axis=-1):
                channel_sampled = np.reshape(channel, channel.shape[1:-1])
                interpolated_func = interp2d(sampled_x, sampled_y, channel_sampled)
                interpolated = interpolated_func(interp_x, interp_y)
                interpolated = np.reshape(interpolated, interp_size)
                channel_res.append(interpolated)
            batch_res.append(np.stack(channel_res, axis=-1))
        return np.stack(batch_res, axis=0)

    @staticmethod
    def _reference_flooring(sampled_feat, interp_size):
        pass

    @staticmethod
    def _reference_average(sampled_feat, interp_size):
        pass

    def testInterpolation2DBilinear(self):
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Bilinear Interpolation Test...')
        test_case = np.random.rand(2, 60, 80, 3).astype(np.float32)
        interpolated_size = [480, 640]
        reference = self._reference_bilinear(test_case, interpolated_size)
        x, y, sess = self._bilinearInterpolation(True, test_case.shape, interpolated_size)
        device_results = y.eval(session=sess, feed_dict={x: test_case})
        self.assertAllClose(reference, device_results, atol=1e-5)

    def testInterpolation2DBilinearGrad(self):
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Bilinear Interpolation Test...')
        sampled_size = [2, 2, 3, 2]
        interpolated_size = [2, 4, 6, 2]
        x, y, sess = self._bilinearInterpolation(True, sampled_size, interpolated_size[1:-1])
        with sess:
            error = gradient_checker.compute_gradient(x, sampled_size, y, interpolated_size)
        self.assertAllClose(error[0], error[1], atol=1e-4)

    def testInterpolation2DFlooring(self):
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Interpolation 2D --> Flooring Test')
        # TODO: Add tests here
        return

    def testInterpolation2DFlooringGrad(self):
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Interpolation 2D Gradient --> Flooring Test')
        # TODO: Add tests here
        return

    def testInterpolation2DAverage(self):
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Interpolation 2D --> Average Test')
        # TODO: Add tests here
        return

    def testInterpolation2DAverageGrad(self):
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Interpolation 2D Gradient --> Average Test')
        # TODO: Add tests here
        return


class ProjectionOpsTest(tf.test.TestCase):
    """
    Projection ops test, including forward and reverse porjection
    """
    def _ForwardProjection(self, batch_size, image_shape=None, vox_shape=None):
        if image_shape is None:
            image_shape = [480, 640]
        if vox_shape is None:
            vox_shape = [240, 144, 240]
        batch_image_shape = [batch_size, ] + image_shape
        with self.test_session() as sess:
            depth = constant_op.constant(0, dtype=tf.float32, shape=batch_image_shape + [1, ])
            scene_info = constant_op.constant(0, dtype=tf.float32, shape=[batch_size, 30])
            img_vox_map, img_proj_pos, vox_occupied = libs.img2vol_forward_projection(depth, scene_info, vox_shape)
            return [depth, scene_info], [img_vox_map, img_proj_pos, vox_occupied], sess

    def _ForwardProjectionPointsSet(self, points, vox_shape=None):
        if vox_shape is None:
            vox_shape = [240, 144, 240]
        with self.test_session() as sess:
            points = constant_op.constant(0, dtype=tf.float32, shape=[1, points, 3])
            scene_info = constant_op.constant(0, dtype=tf.float32, shape=[1, 30])
            vox_origin, camera_pose, vox_unit, _ = tf.split(scene_info, [3, 16, 2, 9], axis=-1)
            pnt_proj_map, vox_occupied = libs.forward_projection(points, camera_pose, vox_unit, vox_origin, vox_shape)
            return [points, scene_info], [pnt_proj_map, vox_occupied], sess

    def _ViewVolumeReverseProjection(self, batch_size, image_shape, feat_shape, vox_shape):
        if image_shape is None:
            image_shape = [480, 640]
        if vox_shape is None:
            vox_shape = [240, 144, 240]
        if feat_shape is None:
            feat_shape = image_shape
        with self.test_session() as sess:
            depth = constant_op.constant(0, dtype=tf.float32, shape=[batch_size, ] + image_shape + [1, ])
            feat = constant_op.constant(0, dtype=tf.float32, shape=[batch_size, ] + feat_shape + [1, ])
            scene_info = constant_op.constant(0, dtype=tf.float32, shape=[batch_size, 30])
            outputs = libs.view_volume_reverse_projection(depth, feat, scene_info, vox_shape, test_mode=True)
            return [depth, scene_info, feat], outputs, sess

    def testForwardProjection(self):
        """
        Forward projection test -- casting image into volume
        :return:
        """
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Projection Op -- Forward Projection Test')
        image_size = [480, 640]
        vox_size = [30, 18, 30]
        batch_size = 2
        samples = GlobalConfiguration.read_test_data(batch_size, 8)
        inputs, outputs, sess = self._ForwardProjection(batch_size, image_size, vox_size)
        feed_dict = {inputs[0]: samples[0], inputs[1]: samples[2]}
        outputs = sess.run(outputs, feed_dict=feed_dict)
        cur_index = 0
        for img_vox_map, img_proj_pos, vox_occupied in zip(*outputs):
            # img_vox_map & vox_occupied cross validation
            vox_vox_occupied = np.reshape(vox_occupied, newshape=[-1])
            img_vox_indices, img_vox_counts = np.unique(img_vox_map, return_counts=True)
            img_vox_counts = np.delete(img_vox_counts, np.where(img_vox_indices == -1), axis=0)
            img_vox_indices = np.delete(img_vox_indices, np.where(img_vox_indices == -1), axis=0)
            img_vox_indices = img_vox_indices.astype(np.uint32)
            vox_indices = np.argwhere(vox_vox_occupied > 0)
            self.assertTrue(np.array_equal(img_vox_indices, np.reshape(vox_indices, newshape=[-1])))
            self.assertTrue(np.array_equal(img_vox_counts, np.reshape(vox_vox_occupied[vox_indices], newshape=[-1])))
            # img_proj_pos visual validation
            img_proj_pos = img_proj_pos[img_proj_pos[:, :, 2] > 0]
            img_proj_pos = img_proj_pos / 0.02
            img_proj_pos = img_proj_pos - np.min(img_proj_pos, axis=0)
            bins = np.ceil(np.max(img_proj_pos, axis=0))
            hist, _ = np.histogramdd(img_proj_pos, bins=bins)
            if os.path.exists(GlobalConfiguration.Visualization_Dir):
                saving_dir = os.path.join(GlobalConfiguration.Visualization_Dir, 'forward_projection')
                if not os.path.exists(saving_dir):
                    os.mkdir(saving_dir)
                vis_model_path = os.path.join(saving_dir, 'camera_pose_view_%d' % cur_index)
                hist = np.expand_dims(hist, axis=-1)
                sparse_indices, _ = visualize.cond_sparse_represent(hist, lambda x: x > 0, color_norm=False)
                visualize.sparse_vox2ply(sparse_indices, hist.shape[:-1], name=vis_model_path)
                vis_model_path = os.path.join(saving_dir, 'volume_pose_view_%d' % cur_index)
                visualize.sparse_vox2ply(np.expand_dims(img_vox_indices, axis=-1), vox_size,
                                         1, np.expand_dims(img_vox_counts, axis=-1), face=True, name=vis_model_path)
            cur_index += 1

    def testViewVolumeReverseProjection(self):
        """
        View Volume Reverse Projection Test -- cast 2d feats into 3d volume
        :return:
        """
        if GlobalConfiguration.INDIVIDUAL_TEST:
            self.skipTest('Skip Projection Op -- Forward Projection Test')
        image_size = [480, 640]
        feat_size = [120, 160]
        vox_size = [240, 144, 240]
        batch_size = 2
        samples = GlobalConfiguration.read_test_data(batch_size, 1)
        inputs, outputs, sess = self._ViewVolumeReverseProjection(batch_size, image_size, feat_size, vox_size)
        feed_dict = {inputs[0]: samples[0], inputs[1]: samples[2], inputs[2]: np.random.rand(batch_size, *feat_size, 1)}
        _, intermediate = outputs
        results = sess.run(intermediate, feed_dict=feed_dict)
        cur_index = 0
        for result in zip(*results, samples[2]):
            vox_proj_pos, _, _, _, scene_info = result
            scene_info = np.expand_dims(scene_info, 0)
            inputs_fp, outputs_fp, sess_fp = self._ForwardProjectionPointsSet(int(vox_proj_pos.shape[1]), vox_size)
            feed_dict = {inputs_fp[0]: vox_proj_pos, inputs_fp[1]: scene_info}
            pnt_proj_map, vox_occupied = sess_fp.run(outputs_fp, feed_dict=feed_dict)
            pnt_proj_map = pnt_proj_map[pnt_proj_map >= 0]
            if os.path.exists(GlobalConfiguration.Visualization_Dir):
                saving_dir = os.path.join(GlobalConfiguration.Visualization_Dir, 'view_volume_reverse_projection')
                if not os.path.exists(saving_dir):
                    os.mkdir(saving_dir)
                vis_model_path = os.path.join(saving_dir, 'identity_projection_%d' % cur_index)
                visualize.sparse_vox2ply(np.expand_dims(pnt_proj_map, axis=-1), vox_size, name=vis_model_path)
            cur_index += 1
