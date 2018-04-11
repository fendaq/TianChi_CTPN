from model import Model

import tensorflow as tf
import tensorflow.contrib.slim as slim


model = Model()

graph = tf.Graph()

model.define_graph(graph,False)


slim.get_variables_to_restore()


def get_variables_to_train(include=None, exclude=None):
    if include is None:
        # 包括所有的变量
        vars_to_include = tf.trainable_variables()
    else:
        if not isinstance(include,(list,tuple)):
            raise TypeError('include 必须是一个list或者tuple')
        vars_to_include = []
        for scope in include:
            vars_to_include += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

    vars_to_exclude = set()
    if exclude is not None:
        if not isinstance(exclude,(list,tuple)):
            raise TypeError('exclude 必须是一个list或者tuple')
        for scope in exclude:
            vars_to_exclude |= set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope))

    return [v for v in vars_to_include if v not in vars_to_exclude]




with graph.as_default():


    vars_to_train = get_variables_to_train(exclude=['resnet_v2_50/conv1','resnet_v2_50/block1','resnet_v2_50/block2'])

    for var in vars_to_train:
        print(var.name)


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # with tf.Graph().as_default():
# #
# #     x = tf.constant(1.0)
# #
# #     with tf.variable_scope('layer_1'):
# #         w1 = tf.Variable(tf.ones([1]),name='w')
# #         print('w1.name:',w1.name)
# #         b1 = tf.Variable(tf.zeros([1]),name='b')
# #
# #
# #     with tf.variable_scope('layer_2'):
# #         w2 = tf.Variable(tf.ones([1]),name='w')
# #         b2 = tf.Variable(tf.zeros([1]),name='b')
# #
# #
# #
# #     pre = (w1*x+b1)*w2 + b2
# #     y = tf.constant(23.0)
# #
# #     loss = tf.square(pre-y)
# #
# #     global_step = tf.train.create_global_step()
# #
# #
# #     train_list = tf.trainable_variables()
# #     print(tf.GraphKeys.TRAINABLE_VARIABLES)
# #     print(tf.get_collection('trainable_variables'))
# #     tf.trainable_variables()
# #
# #     for var in train_list:
# #         print(var.name)
# #
# #     # var_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='layer_2')
# #     train_op= tf.train.GradientDescentOptimizer(0.01).minimize(loss,global_step=global_step,var_list=tf.trainable_variables())
# #
# #
# #
# #
# #     # with tf.Session() as sess:
# #     #     sess.run(tf.global_variables_initializer())
# #     #     for i in range(1200):
# #     #         w_1,b_1,w_2,b_2,pre_ = sess.run([w1,b1,w2,b2,pre])
# #     #         _,step,loss_ = sess.run([train_op,global_step,loss])
# #     #
# #     #         print('Iter : %d loss : %.4f pre: %.5f' % (step, loss_, pre_))
# #     #         print('w1 : %.4f w2 : %.4f b1 : %.4f b2 : %.4f '% (w_1[0],w_2[0],b_1[0],b_2[0]))
