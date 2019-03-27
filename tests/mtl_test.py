# This test is used for generating basic scene information of article materials
import os

import tensorflow as tf
from tensorflow.python.framework import constant_op
import numpy as np
import cv2

from tests.utils_test import GlobalConfiguration
from models.network import Network

from utils import visualize
import libs


class ExportMaterialTest(tf.test.TestCase):
    def testExportMaterial(self):
        # if GlobalConfiguration.INDIVIDUAL_TEST:
        #     self.skipTest('Skip Bilinear Interpolation Test...')
        export_dir = os.path.join(GlobalConfiguration.Visualization_Dir, 'export_material')
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
        raw_data = GlobalConfiguration.read_test_data(-1, 1)
        count = 0

        depth_input = constant_op.constant(0, dtype=tf.float32, shape=[1, 480, 640, 1])
        scene_info_input = constant_op.constant(0, dtype=tf.float32, shape=[1, 30])
        _, _, volume = libs.img2vol_forward_projection(depth_input, scene_info_input, [240, 144, 240])

        with self.test_session() as sess:
            for raw in zip(*raw_data):
                if count not in [10, 20, 50, 110, 150, 170, 400]:
                    count += 1
                    continue
                depth, normal, scene_info_240, label = [np.expand_dims(item, axis=0) for item in raw]
                export_normal = ((normal[0, :, :, :] + 1) / 2 * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(export_dir, 'normal_%d.png' % count), export_normal)
                export_depth = (depth[0, :, :, :] * 1000).astype(np.uint16)
                export_depth = ((export_depth / 8192) + (export_depth * 8)).astype(np.uint16)
                cv2.imwrite(os.path.join(export_dir, 'depth_%d.png' % count), export_depth)
                proj_result = sess.run(volume, feed_dict={depth_input: depth, scene_info_input: scene_info_240})
                sp_indices, _ = visualize.cond_sparse_represent(proj_result, lambda x: x > 0)
                visualize.sparse_vox2ply(sp_indices, [240, 144, 240], face=False,
                                         name=os.path.join(export_dir, 'surface_%d' % count))
                count += 1
            return
