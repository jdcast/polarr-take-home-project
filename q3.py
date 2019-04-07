#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2

import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if x.device_type == 'GPU'])
    print(tf.test.is_gpu_available())
    print(tf.test.gpu_device_name())

    config = tf.ConfigProto(
        # device_count = {'GPU': 0}
        allow_soft_placement=True,
        # log_device_placement=True
    )
    with tf.Session(config=config) as sess:
        test_img = cv2.imread("/home/nightrider/polarr-take-home-project/q3_test_img.png")
        print("image shape: {}".format(test_img.shape))
        test_img = cv2.resize(test_img, (224, 224))
        print("resized image shape: {}".format(test_img.shape))
        cv2.imshow("resized image", test_img)
        cv2.waitKey(3000)

        # load meta graph and restore weights
        new_saver = tf.train.import_meta_graph('/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception/model.ckpt-906808.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./mobilenet_width_1.0_preprocessing_same_as_inception'))

        # names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # count = 0
        # for op in sess.graph.get_operations():
        #     print(op.name)
        #     print(op.values())
        #     print("=========")
        #     count+=1
        #     if count == 100:
        #         assert(False)

        graph = tf.get_default_graph()
        # placeholders = [op for op in graph.get_operations()]

        input = graph.get_tensor_by_name("conv_1:0")
        # predictions = graph.get_tensor_by_name("Predictions:0")

        # sess.run([predictions], feed_dict={input: })