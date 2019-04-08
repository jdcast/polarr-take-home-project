#!/usr/bin/env python2
# -*- coding: utf-8 -*-

def convert_pbtxt_to_pb():
    from google.protobuf import text_format

    with open('/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception/graph.pbtxt') as f:
        text_graph = f.read()
    graph_def = text_format.Parse(text_graph, tf.GraphDef())
    tf.train.write_graph(graph_def,
                         "/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception",
                         'graph.pb',
                         as_text=False)

if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.python.platform import gfile

    convert_pbtxt_to_pb()

    with tf.Session() as sess:
        model_filename ='/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception/graph.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    LOGDIR='/home/nightrider/polarr-take-home-project/mobilenet_width_1.0_preprocessing_same_as_inception'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
