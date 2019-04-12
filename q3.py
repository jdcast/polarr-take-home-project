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
        image_data = sess.run(image_data)

        # load meta graph and restore weights
        graph = tf.get_default_graph()
        new_saver = tf.train.import_meta_graph('/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception/model.ckpt-906808.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./mobilenet_width_1.0_preprocessing_same_as_inception'))

        # file_write = tf.summary.FileWriter(logdir="/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception",
        #                                    graph=sess.graph)
        # file_write.close()

        ### part 4 ###
        # gather predictions
        input = graph.get_tensor_by_name("clone_0/fifo_queue_Dequeue:0")
        softmax_tensor = graph.get_tensor_by_name("clone_0/MobileNet/Predictions/Softmax:0")
        predictions = sess.run(softmax_tensor, {input: image_data}) # batch:0, distort_image/ExpandDims_4:0
        predictions = np.squeeze(predictions)
        # print('predictions: ', predictions[0])
        print('predictions shape: {}'.format(predictions[0].shape))

        # report top 5 predictions/labels
        top_5_predictions = predictions[0].argsort()[-5:][::-1]
        top_5_probabilities = predictions[0][top_5_predictions]
        print('top_5 predictions: {}'.format(top_5_probabilities))

        imagenet_labels_dict = get_imagenet_labels()
        top_5_prediction_names = [imagenet_labels_dict[i] for i in top_5_predictions]
        print('top_5 labels: {}'.format(top_5_prediction_names))

        ### part 5 ###
        # write first 3 layer names/shapes to file
        # count = 0
        # with open("/home/nightrider/polarr-take-home-project/q3_parts_5_result.txt", "w") as file:
        #     for op in sess.graph.get_operations():
        #         if 'conv_ds_2/pw_batch_norm/moving_variance' in op.name:
        #             print('hit 4th layer')
        #             assert(False)
        #         print(op.name)
        #         print(op.values())
        #         print("=========")
        #         file.write("{}\n".format(op.name))
        #         file.write("{}\n".format(op.values()))
        #         file.write("========\n")

        ### part 6 ###
        # References:
        # 1) https://stackoverflow.com/questions/49536856/tensorflow-how-to-merge-batchnorm-into-convolution-for-faster-inference
        # 2) https://forums.developer.apple.com/thread/65821
        # 3) https://arxiv.org/pdf/1502.03167v3.pdf
        W = graph.get_tensor_by_name('MobileNet/conv_1/weights:0')
        b = 0
        gamma = 1
        beta = graph.get_tensor_by_name('MobileNet/conv_1/batch_norm/beta:0')
        m = graph.get_tensor_by_name('MobileNet/conv_1/batch_norm/moving_mean:0')
        var = graph.get_tensor_by_name('MobileNet/conv_1/batch_norm/moving_variance:0')

        W_new = gamma * W / var
        b_new = (gamma * m / var) + beta

        W = sess.run(W)
        beta = sess.run(beta)
        m = sess.run(m)
        var = sess.run(var)
        W_new = sess.run(W_new)
        b_new = sess.run(b_new)

        # print('W: {}'.format(W))
        # print("b_new: {}".format(b_new))
        # print("W_new: {}".format(W_new))

        # write new weights and biases of conv_1 layer to file
        # with open("/home/nightrider/polarr-take-home-project/q3_parts_6_result.txt", "w") as file:
        #     file.write("new conv_1 biases:\n{}\n\n\n\n\n".format(b_new))
        #     file.write("new conv_1 weights:\n{}\n".format(W_new))
        #     file.write("========\n")

        # sanity check
        print('W shape: {}'.format(W.shape))
        print('W_new shape: {}'.format(W_new.shape))
        # print("W_new - W: {}".format(W_new - W))

        ### part 7 ###
        # For this part, we squash any labels reported by the softmax layer that don't belong to the set of labels,
        # ["pickup", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate", "pitcher",
        # "plane"], to a label called "other".  Since we are using 11 labels instead of the original 1001, we will change
        # the reported output distribution and consequently the top-5 labels (by probability) reported.
        target_labels = ['pickup, pickup truck', "pier", "piggy bank, penny bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel",
                         "pirate, pirate ship", "pitcher, ewer", "plane, carpenter's plane, woodworking plane"]
        new_predictions = {}

        other_prob = 0
        for key in imagenet_labels_dict:
            label = imagenet_labels_dict[key]
            if label not in target_labels:
                other_prob += predictions[0][key]
            else:
                new_predictions[label] = predictions[0][key]
        new_predictions["other"] = other_prob

        new_predictions = [(label, prob) for label, prob in new_predictions.items()]
        new_predictions = sorted(new_predictions, key=lambda tup: tup[1], reverse=True)

        # print("new_predictions: {}".format(new_predictions))

        # report new top 5 predictions/labels
        print('new top_5 predictions: {}'.format(new_predictions[:5]))