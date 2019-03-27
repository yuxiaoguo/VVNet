# MSRA Internal Graphics
#

"""
Examples:
"""
import os

import h5py
import numpy as np
import tensorflow as tf

import models
from scripts import dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def eval_phase_network(net, reader):
    data, _ = net.cook_raw_inputs(reader)

    instance_network = net(False, False)
    logit = instance_network.instance(data)

    softmax_logit = tf.nn.softmax(logit, name='softmax')
    _, predict = tf.nn.top_k(softmax_logit, name='seg_label')

    return predict, softmax_logit


def eval_network(args):
    # network choice
    net = models.NETWORK[args.input_network]

    # data reader
    data_dir = args.input_training_data_path
    data_records = [item for item in os.listdir(data_dir) if item.endswith('.tfrecord')]
    test_records = [os.path.join(data_dir, item) for item in data_records if item.find('test') != -1]
    num_samples = sum(1 for _ in tf.python_io.tf_record_iterator(test_records[0]))
    reader = dataset.SceneReader(test_records)

    outputs = eval_phase_network(net, reader)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        if os.path.isdir(args.output_model_path):
            model_choice = tf.train.latest_checkpoint(args.output_model_path)
        else:
            model_choice = args.output_model_path
        saver.restore(sess, model_choice)
        print('start evaluate model: %s' % model_choice)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        predict_tensor = []
        for sample in range(num_samples):
            [predict, logit] = [np.squeeze(item) for item in sess.run(outputs)]
            # print('run iteration: %03d' % sample)
            if args.eval_platform == 'fusion':
                predict_tensor.append(np.array(predict))
            else:
                predict_tensor.append(np.array(logit))
        # finished training
        coord.request_stop()
        coord.join(threads)

    predict_tensor = np.stack(predict_tensor, axis=0)
    print(predict_tensor.shape)
    if not args.eval_platform == 'fusion':
        predict_tensor = np.swapaxes(predict_tensor, -1, 1)
        predict_tensor = np.swapaxes(predict_tensor, -1, 2)
        predict_tensor = np.swapaxes(predict_tensor, -1, 3)

    if not os.path.exists(args.eval_results):
        os.mkdir(args.eval_results)
    file_name = '.'.join((os.path.split(model_choice)[1].split('.')[0], 'hdf5'))
    fp = h5py.File(os.path.join(args.eval_results, file_name), 'w')
    result = fp.create_dataset('result', predict_tensor.shape, dtype='f')
    result[...] = predict_tensor
    fp.close()
    return
