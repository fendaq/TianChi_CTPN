import network
import random,os
import model_detect_data
import tensorflow as tf
import numpy as np
import model_comm_meta as meta
import tensorflow.contrib.slim as slim

from scipy import misc

TRAINING_STEPS = 1000000
LEARNING_RATE_BASE = 0.00001
DECAY_RATE = 0.9
DECAY_STAIRCASE = True
DECAY_STEPS = 2000
MOMENTUM = 0.9
anchor_heights = [12, 24, 36, 48, 64, 80, 96]

class Model():

    def __init__(self, fine_tune=False):

        self.config = tf.ConfigProto()
        self.fine_tune = fine_tune

    @staticmethod
    def define_graph(graph, fine_tune):
        with graph.as_default():

            x = tf.placeholder(tf.float32, [1,None,None,3], name='x-input')

            sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')

            print(fine_tune)
            print(not fine_tune)

            conv_feat = network.resnet_layers(input=x,output_stride=8, training= not fine_tune)

            rnn_cls, rnn_ver, rnn_hor = network.rnn_detect_layers(conv_feat,
                                                                  sequence_length,
                                                                  len(anchor_heights))



            t_cls = tf.placeholder(tf.float32, (None, None, None), name = 'c-input')
            t_ver = tf.placeholder(tf.float32, (None, None, None), name = 'v-input')
            t_hor = tf.placeholder(tf.float32, (None, None, None), name = 'h-input')

            loss = network.detect_loss(rnn_cls, rnn_ver, rnn_hor, t_cls, t_ver, t_hor)

            global_step = tf.train.get_or_create_global_step()
            #
            # Update batch norm stats [http://stackoverflow.com/questions/43234667]
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #
            with tf.control_dependencies(extra_update_ops):
                #
                learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           tf.train.get_global_step(),
                                                           DECAY_STEPS,
                                                           DECAY_RATE,
                                                           staircase = DECAY_STAIRCASE,
                                                           name = 'learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                                   beta1 = MOMENTUM,name='train_op').minimize(loss,global_step=global_step)


    def restore(self, sess, ckpt_path):

        restorer = tf.train.Saver(self.graph.get_collection('model_variables'))
        restorer.restore(sess, ckpt_path)



    def train(self, data_path='validation/', ):

        list_path = model_detect_data.getFilesInDirect(data_path)


        self.graph = tf.Graph()
        self.define_graph(self.graph, self.fine_tune)


        with self.graph.as_default():

            self.config.gpu_options.per_process_gpu_memory_fraction = 0.90

            with tf.Session(config=self.config) as sess:

                if self.fine_tune == True:
                    self.restore(sess, 'model/ckpts/resnet_v2_50.ckpt')

                tf.global_variables_initializer().run()

                x = self.graph.get_tensor_by_name('x-input:0')

                t_cls = self.graph.get_tensor_by_name('c-input:0')
                t_ver = self.graph.get_tensor_by_name('v-input:0')
                t_hor = self.graph.get_tensor_by_name('h-input:0')

                rnn_cls = self.graph.get_tensor_by_name('rnn_cls:0')
                rnn_ver = self.graph.get_tensor_by_name('rnn_ver:0')
                rnn_hor = self.graph.get_tensor_by_name('rnn_hor:0')

                loss = self.graph.get_tensor_by_name('loss:0')
                train_op = self.graph.get_operation_by_name('train_op')
                global_step = self.graph.get_tensor_by_name('global_step:0')
                learning_rate = self.graph.get_tensor_by_name('learning_rate:0')
                conv_feat = self.graph.get_tensor_by_name('conv_feat:0')
                sequence_length = self.graph.get_tensor_by_name('sequence_length:0')

                total_loss = 0
                print('开始训练 ...')
                for i in range(1, TRAINING_STEPS):

                    img_file = random.choice(list_path)
                    img_data, rate = model_detect_data.transform_image(img_file)

                    feat = sess.run(conv_feat,feed_dict={x: [img_data],})
                    feat_size = (feat.shape[0], feat.shape[1])

                    img_data, target_cls, target_ver, target_hor = \
                    model_detect_data.getImageAndTargets(img_file, anchor_heights, feat_size)

                    feed_dict = {x: img_data,
                                 t_cls: target_cls,
                                 t_ver: target_ver,
                                 t_hor: target_hor,
                                 sequence_length:np.ones([feat_size[0]])*64}



                    _, loss_value, step, lr = sess.run(
                        [   train_op,
                            loss,
                            global_step,
                            learning_rate,
                        ],  feed_dict)

                    total_loss += loss_value
                    if  i % 500 == 0:
                        print('step: %d, loss: %g, lr: %g, ' %
                              (step, total_loss/500, lr))
                        total_loss = 0


                    if  i % 2000 == 0:
                        path = 'result/' + str(i) + '/'
                        if not os.path.exists(path): os.mkdir(path)
                        show_list = model_detect_data.getFilesInDirect('result/')
                        count = 0
                        for show_path in show_list:
                            img_data, rate = model_detect_data.transform_image(show_path)
                            feat = sess.run(conv_feat, feed_dict={
                                x: [img_data],
                            })
                            feat_size = (feat.shape[0], feat.shape[1])

                            img_data, target_cls, target_ver, target_hor = model_detect_data.getImageAndTargets(show_path, meta.anchor_heights, feat_size)


                            feed_dict = {x: img_data,
                                         t_cls: target_cls,
                                         t_ver: target_ver,
                                         t_hor: target_hor,
                                         sequence_length: np.ones([feat_size[0]]) * 64}
                            #
                            r_cls, r_ver, r_hor = sess.run([rnn_cls, rnn_ver, rnn_hor], feed_dict)
                            #

                            #
                            # image

                            #

                            file_target = path + str(count) + '.png'
                            count += 1
                            img, rate = model_detect_data.transform_image(show_path)
                            misc.imsave(file_target, img)
                            # trans
                            text_bbox = model_detect_data.transResults(r_cls, r_ver, r_hor, anchor_heights, meta.threshold)
                            #
                            model_detect_data.drawTextBox(file_target, text_bbox)
                            #
                        #
                            print('validation finished')



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
    model = Model(fine_tune=True)
    model.train()