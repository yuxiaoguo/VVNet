import os

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.client import timeline

import models
import time
from scripts import dataset
from scripts import train


def measured_network(net_alias):
    normal = constant_op.constant(0, shape=(1, 480, 640, 3), dtype=tf.float32)
    depth  = constant_op.constant(0, shape=(1, 480, 640, 1), dtype=tf.float32)
    scene_info = constant_op.constant(0, shape=(1, 30), dtype=tf.float32)
    label = constant_op.constant(0, shape=(1, 129600), dtype=tf.uint8)
    inputs = [depth, normal, scene_info]

    train_net = models.NETWORK[net_alias](False, True)

    train_logit = train_net.instance(inputs)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    starter_lr = 0.01
    lr_scheme = tf.train.exponential_decay(starter_lr, global_step, 100000, 0.1, staircase=True)
    opt = tf.train.MomentumOptimizer(lr_scheme, 0.9)
    train_loss = train.loss(train_logit, label, depth, scene_info, scope='loss')
    gradients = opt.compute_gradients(train_loss)
    apply_gradient_op = opt.apply_gradients(gradients)
    train_op = tf.group(apply_gradient_op)

    test_net = models.NETWORK[net_alias](True, False)

    test_logit = test_net.instance(inputs)
    softmax_logit = tf.nn.softmax(test_logit, name='softmax')
    _, predict = tf.nn.top_k(softmax_logit, name='seg_label')

    return test_logit, train_op, inputs, label


def measure_time(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    predict, train_op, inputs, label = measured_network(args.network)

    data_records = [item for item in os.listdir(args.data_dir) if item.endswith('.tfrecord')]
    test_records = [os.path.join(args.data_dir, item) for item in data_records if item.find('test') != -1]
    reader = dataset.SceneReader(test_records, batch_size=1, num_threads=10)
    inputs_feed = reader.get_inputs()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        samples = []
        for i in range(10):
            samples.append(sess.run(inputs_feed))
        coord.request_stop()
        coord.join(threads)

        test_time_periods = []
        count = 0
        for sample in samples:
            start_time = time.time()
            sess.run(predict, feed_dict={inputs[0]: sample[0], inputs[1]: sample[1], inputs[2]: sample[2]},
                     options=run_options, run_metadata=run_metadata)
            test_time_periods.append(time.time() - start_time)
            tl = timeline.Timeline(run_metadata.step_stats)
            with open('test_step_%d.json' % count, 'w') as trace_file:
                trace_file.write(tl.generate_chrome_trace_format(show_memory=True))
            count += 1
        print(np.mean(test_time_periods[1:]))

        train_time_periods = []
        count = 0
        for sample in samples:
            start_time = time.time()
            sess.run(train_op, feed_dict={inputs[0]: sample[0], inputs[1]: sample[1], inputs[2]: sample[2],
                                          label: sample[3]}, options=run_options, run_metadata=run_metadata)
            train_time_periods.append(time.time() - start_time)
            tl = timeline.Timeline(run_metadata.step_stats)
            with open('train_step_%d.json' % count, 'w') as trace_file:
                trace_file.write(tl.generate_chrome_trace_format(show_memory=True))
            count += 1
        print(np.mean(train_time_periods[1:]))
    return
