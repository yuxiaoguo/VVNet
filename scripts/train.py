# MSRA Internal Graphics
#

"""
Examples:
"""
import os
import logging

import tensorflow as tf

import models
from scripts import dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def loss(logit, label, depth, scene_info, categories=12, scope='loss'):
    with tf.name_scope(scope) as _:
        label, indices = models.network.Network.optional_label(label, depth, scene_info)
        if indices is not None:
            logit = tf.reshape(logit, [-1, 12])
            logit = tf.gather_nd(logit, indices)
            label = tf.cast(label, tf.float32)
            label = tf.gather_nd(label, indices)
            label = tf.cast(label, tf.int32)
        label_oh = tf.one_hot(label, categories)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label_oh, logits=logit, name='ce_vox')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='ce_mean')
        return cross_entropy_mean


def average_gradient(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_losses(tower_losses):
    losses = []
    for tower_loss in tower_losses:
        expand_loss = tf.expand_dims(tower_loss, 0)
        losses.append(expand_loss)
    average_loss = tf.concat(losses, axis=0)
    average_loss = tf.reduce_mean(average_loss, 0)
    return average_loss


def train_phase_network(net, args):
    logging.info('=====train network definition=====')
    # data
    training_data_dir = args.input_training_data_path
    data_records = [item for item in os.listdir(training_data_dir) if item.endswith('.tfrecord')]
    train_records = [os.path.join(training_data_dir, item) for item in data_records if item.find('train') != -1]
    logging.info('available training resource: %s', train_records)

    batch_size = args.batch_per_device * args.input_gpu_nums
    reader = dataset.SceneReader(train_records, shuffle=True, batch_size=batch_size, num_threads=20)
    logging.info('batch size for reader: %s', batch_size)

    # optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')
    starter_lr = 0.01
    lr_scheme = tf.train.exponential_decay(starter_lr, global_step, 100000, 0.1, staircase=True)
    opt = tf.train.MomentumOptimizer(lr_scheme, 0.9)

    data, label = net.cook_raw_inputs(reader)
    net_instance = net(False, True)
    with tf.name_scope('multiple_gpus'):
        gpu_num = args.input_gpu_nums
        gpu_depth = tf.split(data[0], gpu_num, axis=0)
        gpu_normal = tf.split(data[1], gpu_num, axis=0)
        gpu_scene_info = tf.split(data[2], gpu_num, axis=0)
        gpu_label = tf.split(label, gpu_num, axis=0)
        gpu_data = [item for item in zip(gpu_depth, gpu_normal, gpu_scene_info)]

    tower_grads = []
    tower_losses = []
    with tf.variable_scope(tf.get_variable_scope()):
        gpu_id = 0
        for data, label in zip(gpu_data, gpu_label):
            with tf.device('/gpu:%s' % gpu_id):
                with tf.name_scope('tower_%s' % gpu_id) as _:
                    logit = net_instance.instance(data)
                    train_loss = loss(logit, label, data[0], data[2], scope='loss')

                    tf.get_variable_scope().reuse_variables()

                    tower_grads.append(opt.compute_gradients(train_loss))
                    tower_losses.append(train_loss)
                    gpu_id += 1

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads = average_gradient(tower_grads)
        train_loss = average_losses(tower_losses)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op)

        summary_lr_scheme = tf.summary.scalar('learning_rate', lr_scheme)

        average_train_loss = tf.placeholder(tf.float32, name='average_train_loss')
        summary_train_loss = tf.summary.scalar('train_ce', average_train_loss)
        train_merged = tf.summary.merge([summary_train_loss, summary_lr_scheme])

        return train_merged, average_train_loss, train_loss, train_op


def test_phase_network(net, args):
    logging.info('=====test network definition=====')
    # data
    validation_data_dir = args.input_validation_data_path
    data_records = [item for item in os.listdir(validation_data_dir) if item.endswith('.tfrecord')]
    test_records = [os.path.join(validation_data_dir, item) for item in data_records if item.find('test') != -1]
    logging.info('available training resource: %s', test_records)

    batch_size = args.batch_per_device
    num_samples = sum(1 for _ in tf.python_io.tf_record_iterator(test_records[0]))
    test_iters = int(num_samples / batch_size) + 1
    reader = dataset.SceneReader(test_records, shuffle=False, batch_size=batch_size, num_threads=10)
    logging.info('batch size for reader: %s', batch_size)

    net_instance = net(True, False)

    data, label = net.cook_raw_inputs(reader)

    logit = net_instance.instance(data)

    test_loss = loss(logit, label, data[0], data[2], scope='test_loss')

    mean_test_loss = tf.placeholder(tf.float32, name='average_test_loss')
    summary_test_loss = tf.summary.scalar('test_ce', mean_test_loss)
    test_merged = tf.summary.merge([summary_test_loss])
    return test_merged, mean_test_loss, test_loss, test_iters


def train_network(args):
    # network choice
    net = models.NETWORK[args.input_network]
    ckpt = tf.train.latest_checkpoint(args.output_model_path)
    start_iters = 0 if not ckpt else int(ckpt[ckpt.find('iter') + 4:-5])

    # train & test targets
    train_summary, average_train_loss, train_loss, train_op = train_phase_network(net, args)
    test_summary, average_test_loss, test_loss, test_iters = test_phase_network(net, args)

    # saver
    tf_saver = tf.train.Saver(max_to_keep=600)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    logging.info('=====start training=====')

    with tf.Session(config=config) as sess:
        # tf summary
        train_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'train'), sess.graph)

        # initialize
        init = tf.global_variables_initializer()
        sess.run(init)

        if ckpt:
            tf_saver.restore(sess, ckpt)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # start training
        avg_train_loss = 0.
        for i in range(start_iters, args.max_iters):

            if (i + 1) % args.record_iters == 0:
                avg_loss = 0.
                for test_iter in range(test_iters):
                    iter_test_loss = sess.run(test_loss)
                    avg_loss += iter_test_loss
                avg_loss /= test_iters
                summary = sess.run(test_summary, feed_dict={average_test_loss: avg_loss})
                train_writer.add_summary(summary, i)
                logging.info('test iters %d: %f', i, avg_loss)

            if (i + 1) % 100 == 0 and i != start_iters:
                avg_train_loss /= 100
                summary = sess.run(train_summary, feed_dict={average_train_loss: avg_train_loss})
                train_writer.add_summary(summary, i)
                logging.info('train iters %d: %f', i, avg_train_loss)
                avg_train_loss = 0

            if (i + 1) % args.record_iters == 0:
                tf_saver.save(sess, os.path.join(args.output_model_path, 'model_iter%06d.ckpt' % i))

            iter_train_loss, _ = sess.run([train_loss, train_op])
            avg_train_loss += iter_train_loss

        # finished training
        coord.request_stop()
        coord.join(threads)
