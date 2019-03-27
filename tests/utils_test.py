# import os
# import sys
#
# import tensorflow as tf
# import numpy as np
#
# from scripts import dataset
#
# VIS_OUT_DIR = os.path.join('/home', 'ig', 'Shared', 'yuxgu', 'visual')
# # TEST_DATA_ROOT = os.path.abspath(os.path.dirname(__file__))
# TEST_DATA_ROOT = os.path.join(os.environ['HOME'], 'datasets', 'SUNCG-TF-Full')
# NEEDS_VIS = True
# UPDATE_BASELINE = False
#
# os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

import os

import tensorflow as tf
import numpy as np

from scripts import dataset


class GlobalConfiguration(object):
    Visualization_Dir = os.getenv('VIS_DIR') if os.getenv('VIS_DIR') is not None else \
        os.path.join('/mnt', 'yuxgu', 'visual', 'unittest')
    Real_Data_Dir = os.path.join(os.environ['HOME'], 'datasets', 'SUNCG-TF')
    GENERAL_BATCH_SIZE = 2      # default batch size
    INDIVIDUAL_TEST = True      # ignore all tests but the target test
    UNDERSTANDING_TEST = True   # ignore all non-analysis test
    TIME_LIMIT = False          # ignore the time consuming tests
    MEMORY_LIMIT = False        # ignore the memory consuming tests

    @staticmethod
    def read_test_data(size, vox_scale, with_sort=True):
        """
        Read samples from real test cases
        :param size: the total samples to read, -1 means read all samples in given records
        :param vox_scale: the vox scale, comparing with original [240, 144, 240]
        :param with_sort: whether to follow the sort, or will enable multiple threads reading
        :type size: int
        :type vox_scale: list(int)
        :type with_sort: bool
        :return:
        """
        target_records = [os.path.join(GlobalConfiguration.Real_Data_Dir, record) for record
                          in os.listdir(GlobalConfiguration.Real_Data_Dir) if record.endswith('test.tfrecord')]
        num_samples = 0
        if size == -1:
            for target_record in target_records:
                num_samples += sum(1 for _ in tf.python_io.tf_record_iterator(target_record))
        else:
            num_samples = size
        reader = dataset.SceneReader(target_records, batch_size=1, num_threads=10 if not with_sort else 1)
        inputs = reader.get_inputs(vox_scale)
        samples = None
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(num_samples):
                sample = sess.run(inputs)
                if samples is None:
                    samples = [[np.reshape(item, newshape=item.shape[1:])] for item in sample]
                else:
                    for items, item in zip(samples, sample):
                        items.append(np.reshape(item, newshape=item.shape[1:]))
            coord.request_stop()
            coord.join(threads)
        return samples

