#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2

import numpy as np
import tensorflow as tf

from imagenet_dict import get_imagenet_labels


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
        image_path = "/home/nightrider/polarr-take-home-project/q3_test_img.png"
        test_img = cv2.imread("/home/nightrider/polarr-take-home-project/q3_test_img.png")
        # print("image shape: {}".format(test_img.shape))
        # test_img = cv2.resize(test_img, (224, 224))
        # print("resized image shape: {}".format(test_img.shape))
        # cv2.imshow("resized image", test_img)
        # cv2.waitKey(3000)

        # Generate batch of images using our single image.
        # This was my only way to feed in data since the only input layer I could find was a batch layer.
        image_data = tf.constant(test_img)
        img_data_png = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
        img_data_png = tf.image.resize_image_with_crop_or_pad(img_data_png, 224, 224)
        image_data = img_data_png.eval().reshape(-1, 224, 224, 3)

        for i in range(0, 127):
            next_data = tf.constant(test_img)
            next_data_png = tf.image.convert_image_dtype(next_data, dtype=tf.float32)
            next_data_png = tf.image.resize_image_with_crop_or_pad(next_data_png, 224, 224)
            next_data = next_data_png.eval().reshape(-1, 224, 224, 3)
            image_data = tf.concat(values=[image_data, next_data], axis=0)

        print("image_data shape: {}".format(image_data.shape))
        image_data = sess.run(image_data) # image_data.eval(session=sess)

        # load meta graph and restore weights
        graph = tf.get_default_graph()
        new_saver = tf.train.import_meta_graph('/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception/model.ckpt-906808.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./mobilenet_width_1.0_preprocessing_same_as_inception'))

        # file_write = tf.summary.FileWriter(logdir="/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception",
        #                                    graph=sess.graph)
        # file_write.close()

        # part 5 of question
        # write first 3 layer names/shapes to file
        count = 0
        with open("/home/nightrider/polarr-take-home-project/q3_parts_5.txt", "w") as file:
            for op in sess.graph.get_operations():
                if 'conv_ds_2/pw_batch_norm/moving_variance' in op.name:
                    print('hit 4th layer')
                    assert(False)
                print(op.name)
                print(op.values())
                print("=========")
                file.write("{}\n".format(op.name))
                file.write("{}\n".format(op.values()))
                file.write("========\n")


        # gather predictions
        input = graph.get_tensor_by_name("clone_0/fifo_queue_Dequeue:0")
        softmax_tensor = graph.get_tensor_by_name("clone_0/MobileNet/Predictions/Softmax:0")
        predictions = sess.run(softmax_tensor, {input: image_data}) # batch:0, distort_image/ExpandDims_4:0
        predictions = np.squeeze(predictions)
        print('predictions: ', predictions[0])
        print(predictions[0].shape)

        # report top 5 predictions/labels
        top_5_predictions = predictions[0].argsort()[-5:][::-1]
        top_5_probabilities = predictions[0][top_5_predictions]
        print('top_5 predictions: {}'.format(top_5_probabilities))

        imagenet_labels_dict = get_imagenet_labels()
        top_5_prediction_names = [imagenet_labels_dict[i] for i in top_5_predictions]
        print('top_5 labels: {}'.format(top_5_prediction_names))