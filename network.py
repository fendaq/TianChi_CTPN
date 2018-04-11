from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2

import tensorflow as tf



def resnet_layers(input, output_stride=8, training=True):

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        _, endpoint = resnet_v2.resnet_v2_50(inputs=input,
                                            output_stride=output_stride,
                                            is_training=training)

        conv_feat = tf.squeeze(endpoint['resnet_v2_50/block4'], axis=0, name='conv_feat')

        return conv_feat


def rnn_detect_layers(conv_feat, sequence_length, num_anchors):
    with tf.variable_scope("RNN_module"):
        #
        # Input features is [batchSize paddedSeqLen numFeatures]
        #
        #
        rnn_size = 256  # 256, 512
        fc_size = 512  # 256, 384, 512
        #
        #
        # Transpose to time-major order for efficiency
        #  --> [paddedSeqLen batchSize numFeatures]
        #
        # 这里总是出错
        rnn_sequence = tf.transpose(conv_feat, perm=[1, 0, 2], name='time_major')
        rnn1 = rnn_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
        rnn2 = rnn_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
        # rnn3 = rnn_layer(rnn2, sequence_length, rnn_size, 'bdrnn3')
        #
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        #
        rnn_feat = tf.layers.dense(rnn2, fc_size,
                                   activation=tf.nn.relu,
                                   kernel_initializer=weight_initializer,
                                   bias_initializer=bias_initializer,
                                   name='rnn_feat')
        #
        # out
        #
        rnn_cls = tf.layers.dense(rnn_feat, num_anchors * 2,
                                  activation=tf.nn.sigmoid,
                                  kernel_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  name='text_cls')
        #
        rnn_ver = tf.layers.dense(rnn_feat, num_anchors * 2,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  name='text_ver')
        #
        rnn_hor = tf.layers.dense(rnn_feat, num_anchors * 2,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  name='text_hor')
        #
        # dense operates on last dim
        #

        #
        rnn_cls = tf.transpose(rnn_cls, perm=[1, 0, 2], name='rnn_cls')
        rnn_ver = tf.transpose(rnn_ver, perm=[1, 0, 2], name='rnn_ver')
        rnn_hor = tf.transpose(rnn_hor, perm=[1, 0, 2], name='rnn_hor')
    #
    return rnn_cls, rnn_ver, rnn_hor




# INPUT = tf.constant(1.0, shape=[1, 512, 512, 3])
# OUTPUT = resnet_layers(INPUT,output_stride=8)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     o = sess.run(OUTPUT)
#     print(o.shape)


'''
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                              sequence_length = None, # 输入序列的实际长度（可选，默认为输入序列的最大长度）
                                                      # sequence_length must be a vector of length batch_size
                              initial_state_fw = None,  # 前向的初始化状态（可选）
                              initial_state_bw = None,  # 后向的初始化状态（可选）
                              dtype = None, # 初始化和输出的数据类型（可选）
                              parallel_iterations = None,
                              swap_memory = False,
                              time_major = False, 
                              # 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`. 
                              # 如果为false, tensor的形状必须为`[batch_size, max_time, depth]`. 
                              scope = None)
返回值：一个(outputs, output_states)的元组
其中，
1. outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。
假设time_major = false, tensor的shape为[batch_size, max_time, depth]。
实验中使用tf.concat(outputs, 2)将其拼接。
2. output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
output_state_fw和output_state_bw的类型为LSTMStateTuple。
LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
'''


def rnn_layer(input_sequence, sequence_length, rnn_size, scope):
    '''build bidirectional (concatenated output) lstm layer'''
    #
    # time_major = True
    #
    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    #
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=weight_initializer)
    #
    # Include?
    # cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw,
    #                                         input_keep_prob=dropout_rate )
    # cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw,
    #                                         input_keep_prob=dropout_rate )

    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length=sequence_length,
                                                    time_major=True,
                                                    dtype=tf.float32,
                                                    scope=scope)

    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    #
    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')
    # rnn_output_stack = rnn_output[0] + rnn_output[1]

    return rnn_output_stack
    #


def detect_loss(rnn_cls, rnn_ver, rnn_hor, target_cls, target_ver, target_hor):
    #
    # loss_cls
    #
    rnn_cls_posi = rnn_cls * target_cls
    rnn_cls_neg = rnn_cls - rnn_cls_posi
    #
    pow_posi = tf.square(rnn_cls_posi - target_cls)
    pow_neg = tf.square(rnn_cls_neg)
    #
    mod_posi = tf.pow(pow_posi / 0.24, 5)  # 0.3, 0.2,     0.5,0.4
    mod_neg = tf.pow(pow_neg / 0.24, 5)  # 0.7, 0.6,
    mod_con = tf.pow(0.25 / 0.2, 5)
    #
    num_posi = tf.reduce_sum(target_cls) / 2
    num_neg = tf.reduce_sum(target_cls + 1) / 2 - num_posi * 2
    #
    loss_cls_posi = tf.reduce_sum(pow_posi * mod_posi) / 2
    loss_cls_neg = tf.reduce_sum(pow_neg * mod_neg) / 2
    #
    loss_cls = loss_cls_posi / num_posi + loss_cls_neg / num_neg
    #
    # loss reg
    #
    rnn_ver_posi = rnn_ver * target_cls
    rnn_hor_posi = rnn_hor * target_cls
    #
    rnn_ver_neg = rnn_ver - rnn_ver_posi
    rnn_hor_neg = rnn_hor - rnn_hor_posi
    #
    pow_ver_posi = tf.square(rnn_ver_posi - target_ver)
    pow_hor_posi = tf.square(rnn_hor_posi - target_hor)
    #
    pow_ver_neg = tf.square(rnn_ver_neg)
    pow_hor_neg = tf.square(rnn_hor_neg)
    #
    loss_ver_posi = tf.reduce_sum(pow_ver_posi * mod_con) / num_posi
    loss_hor_posi = tf.reduce_sum(pow_hor_posi * mod_con) / num_posi
    #
    loss_ver_neg = tf.reduce_sum(pow_ver_neg * mod_neg) / num_neg
    loss_hor_neg = tf.reduce_sum(pow_hor_neg * mod_neg) / num_neg
    #
    loss_ver = loss_ver_posi + loss_ver_neg
    loss_hor = loss_hor_posi + loss_hor_neg
    #

    #
    loss = tf.add(loss_cls, loss_ver + loss_hor, name='loss')
    #

    #
    return loss
    #


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