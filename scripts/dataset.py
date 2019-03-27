import tensorflow as tf

import libs


class SceneReader(object):
    def __init__(self, tfrecord_list, shuffle=False, num_threads=1, batch_size=1):
        self.reader = None
        self.queue = None
        self.shuffle = shuffle
        self.num_threads = num_threads
        self.tfrecord_list = tfrecord_list
        self.batch_size = batch_size

    @staticmethod
    def _process_depth_image(depth_raw):
        img = tf.image.decode_png(depth_raw, channels=1, dtype=tf.uint16)
        img = tf.cast(img, tf.int32)
        img_upper = tf.mod(img, 8) * 8192
        img_lower = tf.div(img, 8)
        img = tf.bitwise.bitwise_or(img_upper, img_lower)
        img = tf.cast(img, tf.float32) / 1000
        return img

    @staticmethod
    def _process_bin(bin_meta):
        meta = tf.decode_raw(bin_meta, tf.uint8)
        origin, camera, data = tf.split(meta, [12, 64, 129600])
        origin, camera = [tf.bitcast(tf.reshape(raw, [-1, 4]), tf.float32) for raw in [origin, camera]]
        return origin, camera, data

    @staticmethod
    def _process_normal_image(normal_raw):
        img = tf.image.decode_png(normal_raw, channels=3, dtype=tf.uint16)
        img = tf.cast(img, tf.float32)
        img = (img / 65535 - 0.5) * 2
        return img

    def _read_meta(self, vox_scale=1):
        self.reader = tf.TFRecordReader()
        self.queue = tf.train.string_input_producer(self.tfrecord_list)
        _, serialized_example = self.reader.read(self.queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'img': tf.FixedLenFeature([], tf.string),
                                               'bin': tf.FixedLenFeature([], tf.string),
                                               'alias': tf.FixedLenFeature([], tf.string),
                                               'normal': tf.FixedLenFeature([], tf.string),
                                           })
        depth = SceneReader._process_depth_image(features['img'])
        normal = SceneReader._process_normal_image(features['normal'])
        # TODO: A potential risk here
        depth, normal = [tf.image.resize_images(img, [480, 640]) for img in [depth, normal]]
        # img = tf.transpose(img, [2, 0, 1])
        # origin, camera, _, label = libs.decode_scene(features['bin'], dataVoxSize=data_vox_size,
        #                                              labelVoxSize=label_vox_size, segClassMap=conf.seg_class_map,
        #                                              segClassWeight=conf.seg_class_weight)
        origin, camera, label = SceneReader._process_bin(features['bin'])
        vox_detail, fov = libs.scene_base_info(vox_scale=vox_scale)
        scene_info = tf.concat([origin, camera, vox_detail, fov], axis=-1)
        return depth, normal, scene_info, label

    def _batch_data(self, inputs):
        min_samples = 100
        if self.shuffle:
            batch_inputs = tf.train.shuffle_batch(
                inputs,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                capacity=3 * self.batch_size + min_samples,
                min_after_dequeue=min_samples
            )
        else:
            batch_inputs = tf.train.batch(
                inputs,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                capacity=3 * self.batch_size
                # capacity=3 * self.batch_size + min_samples
            )

        return batch_inputs

    def get_inputs(self, scale=1):
        with tf.name_scope('inputs') as _:
            inputs = self._read_meta(scale)
            return self._batch_data(inputs)
