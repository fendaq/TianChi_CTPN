

from tensorflow.contrib.slim import nets
from tensorflow.contrib import slim

import tensorflow as tf
import numpy as np

def conv_feat(input):
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        _, endpoint = nets.resnet_v2.resnet_v2_50(
            inputs=input,
            is_training=True,
            output_stride=8
        )

    return endpoint['resnet_v2_50/block4']

def define_graph(graph):

    with graph.as_default():

        x = tf.placeholder(tf.float32, shape=[None,None,None,3],name='input')
        print(x.name)
        w = tf.placeholder(tf.int32, shape=[None,],name='width')

        feat = conv_feat(x)
        print(feat.name)






if __name__ == '__main__':
    ckpt_path = 'ckpts/resnet_v2_50.ckpt'
    graph = tf.Graph()
    define_graph(graph)
    with graph.as_default():
        print(tf.get_variable_scope())
        restore = tf.train.Saver(tf.trainable_variables())
        x = graph.get_tensor_by_name('input:0')
        feature_map = graph.get_tensor_by_name('resnet_v2_50/block4/unit_3/bottleneck_v2/add:0')
        print(feature_map.shape)
        print(feature_map.shape.as_list())
        with tf.Session(graph=graph) as sess:
            restore.restore(sess,ckpt_path)
            feat = sess.run(feature_map,feed_dict={x:np.ones([1,513,512,3])})
            print(feat.shape)



